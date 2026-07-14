"""Tests for the subject-aware pipeline's public behavior."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from pcp_project.pipeline import SubjectPipeline, accepts_param

NO_TARGET = object()


class WindowRows(BaseEstimator, TransformerMixin):
    """Duplicate each subject row and attach window-level metadata."""

    def fit(self, X, y=NO_TARGET, groups=None, token=None):
        self.fit_target_ = y
        self.fit_groups_ = groups
        self.token_ = token
        return self

    def transform(self, X, y=NO_TARGET, groups=None):
        self.transform_target_ = y
        subject_ids = np.arange(len(X)) if groups is None else np.asarray(groups)
        states = np.tile([0, 1], len(X))
        return np.repeat(X, 2, axis=0), (np.repeat(subject_ids, 2), states)


class PlainPassThrough(BaseEstimator, TransformerMixin):
    """Represent an ordinary sklearn transformer with generic kwargs."""

    def fit(self, X, **kwargs):
        self.fit_kwargs_ = kwargs
        return self

    def transform(self, X, **kwargs):
        self.transform_kwargs_ = kwargs
        return X


class GroupMean(BaseEstimator, TransformerMixin):
    """Collapse window rows back to one row per subject."""

    def fit(self, X, y=NO_TARGET, groups=None):
        self.fit_target_ = y
        self.fit_groups_ = groups
        return self

    def transform(self, X, y=NO_TARGET, groups=None):
        self.transform_target_ = y
        subject_ids = np.asarray(groups[0])
        subjects = list(dict.fromkeys(subject_ids.tolist()))
        means = [
            np.asarray(X)[subject_ids == subject].mean(axis=0) for subject in subjects
        ]
        return np.asarray(means), np.asarray(subjects)


class GroupAwareClassifier(ClassifierMixin, BaseEstimator):
    """Record the subject IDs received by the final estimator."""

    def fit(self, X, y, groups=None):
        self.fit_groups_ = np.asarray(groups).copy()
        self.classes_ = np.unique(y)
        return self

    def predict(self, X, groups=None, offset=0.0):
        self.predict_groups_ = np.asarray(groups).copy()
        return (np.asarray(X)[:, 0] + offset >= 0).astype(int)


class PairClassifier(ClassifierMixin, BaseEstimator):
    """Return one prediction for each adjacent pair of input rows."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return (np.asarray(X).reshape(-1, 2).mean(axis=1) > 0).astype(int)


class GroupAwareCluster(BaseEstimator):
    """Record groups passed specifically to ``fit_predict``."""

    def fit(self, X, y=None):
        return self

    def fit_predict(self, X, y=None, groups=None):
        self.groups_ = np.asarray(groups).copy()
        return np.zeros(len(X), dtype=int)


def make_pipeline(*, mask=None, classifier=None):
    return SubjectPipeline(
        [
            ("window", WindowRows()),
            ("plain", PlainPassThrough()),
            ("aggregate", GroupMean()),
            ("classifier", classifier or LogisticRegression()),
        ],
        mask=mask,
    )


def test_metadata_stays_aligned_as_rows_expand_and_collapse():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    subject_ids = np.array([10, 11, 12, 13])
    weights = np.arange(1, 5, dtype=float)

    pipeline = make_pipeline(mask=subject_ids).fit(
        X,
        y,
        window__token="routed",
        classifier__sample_weight=weights,
    )

    window = pipeline.named_steps.window
    aggregate = pipeline.named_steps.aggregate
    np.testing.assert_array_equal(window.fit_target_, y)
    np.testing.assert_array_equal(window.transform_target_, y)
    np.testing.assert_array_equal(window.fit_groups_, subject_ids)
    assert window.token_ == "routed"
    assert pipeline.named_steps.plain.fit_kwargs_ == {}
    assert pipeline.named_steps.plain.transform_kwargs_ == {}
    assert aggregate.fit_target_ is NO_TARGET
    assert aggregate.transform_target_ is NO_TARGET
    assert isinstance(aggregate.fit_groups_, tuple)

    np.testing.assert_allclose(pipeline.transform(X), X)
    np.testing.assert_array_equal(pipeline.predict(X), y)
    assert pipeline.predict_proba(X).shape == (4, 2)
    assert pipeline.decision_function(X).shape == (4,)
    assert pipeline.score(X, y, sample_weight=weights) == 1.0
    np.testing.assert_allclose(make_pipeline(mask=subject_ids).fit_transform(X, y), X)


def test_fresh_input_metadata_reaches_the_final_estimator():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    original_ids = np.array([10, 11, 12, 13])
    replacement_ids = np.array([20, 21, 22, 23])
    pipeline = make_pipeline(
        mask=original_ids,
        classifier=GroupAwareClassifier(),
    ).fit(X, y)

    np.testing.assert_array_equal(
        pipeline.named_steps.classifier.fit_groups_, original_ids
    )
    np.testing.assert_array_equal(pipeline.predict((X, replacement_ids), offset=0.0), y)
    np.testing.assert_array_equal(
        pipeline.named_steps.classifier.predict_groups_, replacement_ids
    )
    np.testing.assert_array_equal(pipeline.mask, original_ids)

    pipeline.fit(X, y)
    np.testing.assert_array_equal(pipeline.predict((X, replacement_ids)), y)


def test_passthrough_and_transformer_endings():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    skipped = SubjectPipeline(
        [
            ("none", None),
            ("skip", "passthrough"),
            ("classifier", LogisticRegression()),
        ]
    ).fit(X, y)
    np.testing.assert_array_equal(skipped.transform(X), X)

    scaled = SubjectPipeline([("scale", StandardScaler())]).fit(X, y)
    np.testing.assert_allclose(scaled.transform(X).mean(axis=0), 0.0, atol=1e-12)
    assert not hasattr(scaled, "predict_proba")
    assert not hasattr(scaled, "decision_function")
    with pytest.raises(AttributeError, match="predict"):
        scaled.predict(X)

    extractor = SubjectPipeline(
        [("identity", FunctionTransformer()), ("output", None)]
    ).fit(X, y)
    np.testing.assert_array_equal(extractor.transform(X), X)
    assert not hasattr(extractor, "predict_proba")
    with pytest.raises(AttributeError, match="predict"):
        extractor.predict(X)

    tuple_X = ((-1.0,), (1.0,))
    tuple_y = np.array([0, 1])
    plain = SubjectPipeline([("classifier", LogisticRegression())]).fit(
        tuple_X, tuple_y
    )
    np.testing.assert_array_equal(plain.predict(tuple_X), tuple_y)


def test_grouped_scoring_accepts_row_or_subject_targets():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    row_targets = np.array([0, 0, 1, 1])
    subject_targets = np.array([0, 1])
    groups = np.array(["b", "b", "a", "a"])
    pipeline = SubjectPipeline([("classifier", LogisticRegression())]).fit(
        X, row_targets
    )

    assert pipeline.score(X, row_targets, groups=groups) == 1.0
    assert pipeline.score(X, subject_targets, groups=groups) == 1.0
    assert (
        pipeline.score(
            X,
            row_targets,
            groups=groups,
            sample_weight=np.ones(2),
        )
        == 1.0
    )
    assert (
        pipeline.score(
            X,
            np.array([0.0, np.nan, 1.0, 1.0]),
            groups=groups,
        )
        == 1.0
    )

    grouped_predictor = SubjectPipeline([("classifier", PairClassifier())]).fit(
        X, row_targets
    )
    assert grouped_predictor.score(X, subject_targets, groups=groups) == 1.0

    with pytest.raises(ValueError, match="different levels"):
        pipeline.score(X, subject_targets)
    with pytest.raises(ValueError, match="no valid labels"):
        pipeline.score(
            X,
            np.array([np.nan, np.nan, 1.0, 1.0]),
            groups=groups,
        )


def test_sklearn_estimators_and_fit_predict_remain_compatible():
    X = np.array([[-1.0], [1.0]])
    y = np.array([0, 1])

    nested = SubjectPipeline(
        [
            ("inner", Pipeline([("scale", StandardScaler())])),
            ("classifier", LogisticRegression()),
        ]
    ).fit(X, y)
    np.testing.assert_array_equal(nested.predict(X), y)

    clusters = SubjectPipeline(
        [
            ("scale", StandardScaler()),
            ("cluster", KMeans(n_clusters=2, n_init=1, random_state=0)),
        ]
    ).fit_predict(X)
    assert clusters.shape == y.shape

    group_ids = np.array(["a", "b"])
    group_aware = SubjectPipeline(
        [("cluster", GroupAwareCluster())],
        mask=group_ids,
    )
    assert group_aware.fit_predict(X).shape == y.shape
    np.testing.assert_array_equal(group_aware.named_steps.cluster.groups_, group_ids)

    assert accepts_param(lambda *, groups: None, "groups")
    assert accepts_param(lambda **kwargs: kwargs, "groups")
    assert not accepts_param(lambda X: X, "groups")


def test_pipeline_contract_errors():
    X = np.array([[-1.0], [1.0]])
    y = np.array([0, 1])

    empty = SubjectPipeline([]).fit(X, y)
    assert empty.is_fitted_

    pipeline = SubjectPipeline([("classifier", LogisticRegression())])
    with pytest.raises(ValueError, match="step__param"):
        pipeline.fit(X, y, sample_weight=np.ones(2))
    with pytest.raises(ValueError, match="Unknown step"):
        pipeline.fit(X, y, missing__value=1)

    unfitted = SubjectPipeline([("classifier", LogisticRegression())])
    with pytest.raises(Exception, match="not fitted"):
        unfitted.transform(X)
    with pytest.raises(Exception, match="not fitted"):
        unfitted.predict(X)

    with pytest.raises(ValueError, match="requires y"):
        pipeline.fit(X, y).score(X)
