"""Tests for classification helpers and the complete pipeline."""

import numpy as np
import pytest
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from pcp_project.classify import (
    FeatureConsolidator,
    StateAwareMDM,
    StateTangentFeatures,
    build_classification_pipeline,
    cross_validate_subjects,
    make_subject_folds,
    mean_label_balanced_accuracy,
    nested_search_subjects,
)


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Small test classifier that records which subject rows it saw."""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y):
        X = np.asarray(X)
        self.seen_subjects_ = X[:, 1].astype(int)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return (np.asarray(X)[:, 0] > self.threshold).astype(int)


@pytest.fixture
def window_rows():
    features = np.array(
        [
            [0.0, 0.0],
            [20.0, 20.0],
            [10.0, 10.0],
            [30.0, 30.0],
            [2.0, 2.0],
            [50.0, 50.0],
        ]
    )
    subjects = np.array(["s2", "s1", "s2", "s1", "s2", "s1"])
    states = np.array([0, 0, 1, 1, 0, 1])
    return features, (subjects, states)


def test_feature_aggregation(window_rows):
    features, metadata = window_rows
    expected = {
        "mean": [[4.0, 4.0], [100 / 3, 100 / 3]],
        "mean_of_state_means": [[5.5, 5.5], [30.0, 30.0]],
        "concat": [[1.0, 1.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]],
        "stack": [
            [[1.0, 1.0], [10.0, 10.0]],
            [[20.0, 20.0], [40.0, 40.0]],
        ],
    }

    for strategy, values in expected.items():
        result, subjects = FeatureConsolidator(strategy).fit_transform(
            features, groups=metadata
        )
        np.testing.assert_allclose(result, values)
        np.testing.assert_array_equal(subjects, ["s2", "s1"])

    custom = FeatureConsolidator("concat", state_order=(1, 0)).fit(
        features, groups=metadata
    )
    np.testing.assert_array_equal(custom.state_order_, [1, 0])
    assert custom.n_features_in_ == 2


def test_feature_consolidator_validation(window_rows):
    features, metadata = window_rows
    with pytest.raises(ValueError, match="feature matrix"):
        FeatureConsolidator().fit(np.ones(3), groups=metadata)

    invalid_metadata = [
        None,
        (np.arange(6)[:, None], np.zeros(6)),
    ]
    for invalid in invalid_metadata:
        with pytest.raises(ValueError):
            FeatureConsolidator().fit(features, groups=invalid)

    bad_aggregators = [
        FeatureConsolidator("bad"),
        FeatureConsolidator("concat", state_order=()),
        FeatureConsolidator("concat", state_order=(0, 0)),
    ]
    for aggregator in bad_aggregators:
        with pytest.raises(ValueError):
            aggregator.fit(features, groups=metadata)

    with pytest.raises(Exception, match="not fitted"):
        FeatureConsolidator().transform(features, groups=metadata)

    mean = FeatureConsolidator().fit(features, groups=metadata)
    assert not hasattr(mean, "state_order_")
    with pytest.raises(ValueError, match="expected"):
        mean.transform(np.ones((len(features), 3)), groups=metadata)

    covariance_rows = np.ones((len(features), 2, 2))
    with pytest.raises(ValueError, match="two-dimensional"):
        FeatureConsolidator("concat").fit(covariance_rows, groups=metadata)

    state_aware = FeatureConsolidator("mean_of_state_means").fit(
        features, groups=metadata
    )
    unknown_states = metadata[1].copy()
    unknown_states[2] = 2
    with pytest.raises(ValueError, match="unknown window states"):
        state_aware.transform(features, groups=(metadata[0], unknown_states))

    only_state_zero = metadata[1] == 0
    for strategy in ["concat", "stack"]:
        aggregator = FeatureConsolidator(strategy).fit(features, groups=metadata)
        with pytest.raises(ValueError, match="every subject"):
            aggregator.transform(
                features[only_state_zero],
                groups=(metadata[0][only_state_zero], metadata[1][only_state_zero]),
            )


def test_stack_covariances_by_subject_and_state(window_rows):
    features, metadata = window_rows
    covariances = np.asarray(
        [np.diag([value + 1, value + 2]) for value in features[:, 0]]
    )

    result, subjects = FeatureConsolidator("stack").fit_transform(
        covariances, groups=metadata
    )

    assert result.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(subjects, ["s2", "s1"])
    np.testing.assert_allclose(result[0, 0], covariances[[0, 4]].mean(axis=0))
    np.testing.assert_allclose(result[0, 1], covariances[2])


def test_build_classification_pipeline():
    pipeline = build_classification_pipeline()
    expected_steps = (
        "filter select epoch covariance features aggregate scale classifier"
    )
    assert list(pipeline.named_steps) == expected_steps.split()
    assert pipeline.named_steps.epoch.length == 512
    assert pipeline.named_steps.epoch.step_size == 512
    assert isinstance(pipeline.named_steps.features, TangentSpace)
    assert isinstance(pipeline.named_steps.classifier, LogisticRegression)

    classifier = DummyClassifier(strategy="stratified", random_state=7)
    custom = build_classification_pipeline(
        classifier,
        states=("eyes_closed",),
        window_seconds=1,
        step_seconds=0.25,
        covariance="lwf",
        aggregation="concat",
        state_order=(1,),
        sfreq=32,
    )
    assert custom.named_steps.classifier is classifier
    assert custom.named_steps.epoch.length == 32
    assert custom.named_steps.epoch.step_size == 8
    assert custom.named_steps.aggregate.strategy == "concat"

    identity = build_classification_pipeline(
        classifier="passthrough", feature_transform="identity"
    )
    assert identity.named_steps.features == "passthrough"
    assert identity.named_steps.scale == "passthrough"

    with pytest.raises(ValueError, match="explicit final estimator"):
        build_classification_pipeline(feature_transform="identity")

    with pytest.raises(ValueError, match="feature_transform"):
        build_classification_pipeline(feature_transform="bad")

    for name in ["window_seconds", "step_seconds", "sfreq"]:
        with pytest.raises(ValueError, match=name):
            build_classification_pipeline(**{name: 0})


def test_state_aware_mdm():
    scales = [1.0, 1.2, 4.0, 4.2]
    covariances = np.asarray(
        [
            [np.diag([scale, scale + 0.4]), np.diag([scale + 0.2, scale + 0.8])]
            for scale in scales
        ]
    )
    labels = np.array(["absent", "absent", "present", "present"])

    model = StateAwareMDM().fit(covariances, labels)
    state_distances = np.stack(
        [
            state_model.transform(covariances[:, state])
            for state, state_model in enumerate(model.state_models_)
        ],
        axis=1,
    )

    assert clone(model).metric == "riemann"
    assert model.transform(covariances).shape == (4, 2)
    np.testing.assert_allclose(
        model.transform(covariances),
        np.sqrt(np.sum(state_distances**2, axis=1)),
    )
    np.testing.assert_array_equal(model.predict(covariances), labels)
    np.testing.assert_allclose(
        model.decision_function(covariances),
        model.transform(covariances)[:, 0] - model.transform(covariances)[:, 1],
    )

    multilabels = np.column_stack([labels == "present", labels == "absent"])
    multilabel_model = OneVsRestClassifier(StateAwareMDM()).fit(
        covariances, multilabels
    )
    np.testing.assert_array_equal(multilabel_model.predict(covariances), multilabels)

    multiclass_covariances = np.concatenate(
        [covariances, covariances[:2] * 2.0], axis=0
    )
    multiclass_labels = np.array([0, 0, 1, 1, 2, 2])
    multiclass_model = StateAwareMDM().fit(multiclass_covariances, multiclass_labels)
    np.testing.assert_allclose(
        multiclass_model.decision_function(multiclass_covariances),
        -multiclass_model.transform(multiclass_covariances),
    )

    with pytest.raises(Exception, match="not fitted"):
        StateAwareMDM().transform(covariances)

    invalid_shapes = [covariances[0], np.empty((2, 2, 2, 3))]
    for invalid in invalid_shapes:
        with pytest.raises(ValueError, match="subjects, states, channels"):
            StateAwareMDM().fit(invalid, labels)

    for invalid_labels in [labels[:, None], labels[:-1]]:
        with pytest.raises(ValueError, match="one label per subject"):
            StateAwareMDM().fit(covariances, invalid_labels)

    with pytest.raises(ValueError, match="fitted state covariance shape"):
        model.transform(covariances[:, :1])


def test_state_tangent_features():
    scales = [1.0, 1.2, 4.0, 4.2]
    covariances = np.asarray(
        [
            [np.diag([scale, scale + 0.4]), np.diag([scale + 0.2, scale + 0.8])]
            for scale in scales
        ]
    )
    features = StateTangentFeatures().fit(covariances)
    assert clone(features).metric == "riemann"
    assert features.transform(covariances).shape == (4, 6)

    with pytest.raises(Exception, match="not fitted"):
        StateTangentFeatures().transform(covariances)
    with pytest.raises(ValueError, match="subjects, states, channels"):
        StateTangentFeatures().fit(covariances[0])
    with pytest.raises(ValueError, match="fitted state covariance shape"):
        features.transform(covariances[:, :1])


def test_subject_folds_and_evaluation_helpers():
    X = np.arange(16, dtype=float).reshape(8, 2)
    y = np.array([0, 1] * 4)
    groups = np.array([f"s{i}" for i in range(len(y))])

    folds = make_subject_folds(y, groups, n_splits=4, random_state=7)
    for train, test in folds:
        assert set(groups[train]).isdisjoint(groups[test])
    assert len(make_subject_folds(y, groups, n_splits=2, random_state=None)) == 2

    multilabel_y = np.column_stack([y[:4], 1 - y[:4]])
    multilabel_groups = groups[:4]
    for random_state in [0, None]:
        folds = make_subject_folds(
            multilabel_y, multilabel_groups, n_splits=2, random_state=random_state
        )
        assert len(folds) == 2

    scores = cross_validate_subjects(
        DummyClassifier(strategy="most_frequent"),
        X,
        y,
        groups,
        scoring="accuracy",
        n_splits=2,
        n_jobs=1,
    )
    assert len(scores["test_score"]) == 2

    multilabel_X = X[:4]
    multilabel_classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    multilabel_scores = cross_validate_subjects(
        multilabel_classifier,
        multilabel_X,
        multilabel_y,
        multilabel_groups,
        cv=make_subject_folds(multilabel_y, multilabel_groups, n_splits=2),
        scoring={"macro_f1": "f1_macro", "subset_accuracy": "accuracy"},
    )
    assert multilabel_scores["test_macro_f1"].shape == (2,)
    assert multilabel_scores["test_subset_accuracy"].shape == (2,)

    default_multilabel_scores = cross_validate_subjects(
        multilabel_classifier,
        multilabel_X,
        multilabel_y,
        multilabel_groups,
        cv=make_subject_folds(multilabel_y, multilabel_groups, n_splits=2),
    )
    assert default_multilabel_scores["test_score"].shape == (2,)
    assert np.isfinite(default_multilabel_scores["test_score"]).all()


def test_mean_label_balanced_accuracy():
    y_true = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    y_pred = np.array([[0, 1], [0, 1], [1, 1], [0, 0]])
    assert mean_label_balanced_accuracy(y_true, y_pred) == pytest.approx(0.75)

    invalid_pairs = [
        (y_true[:, 0], y_pred[:, 0]),
        (y_true, y_pred[:-1]),
        (np.empty((4, 0)), np.empty((4, 0))),
    ]
    for actual, predicted in invalid_pairs:
        with pytest.raises(ValueError, match="matching non-empty 2-D"):
            mean_label_balanced_accuracy(actual, predicted)


def test_nested_search_keeps_outer_test_subjects_held_out():
    y = np.array([0, 1] * 6)
    subject_ids = np.arange(len(y))
    X = np.column_stack([np.where(y == 1, 1.0, -1.0), subject_ids])
    folds = make_subject_folds(y, subject_ids, n_splits=3, random_state=7)

    result = nested_search_subjects(
        ThresholdClassifier(),
        {"threshold": [-2.0, 0.0, 2.0]},
        X,
        y,
        subject_ids,
        cv=folds,
        inner_splits=2,
        scoring="accuracy",
        n_jobs=1,
    )

    np.testing.assert_allclose(result["test_score"], 1.0)
    np.testing.assert_allclose(result["best_inner_score"], 1.0)
    assert result["best_params"] == [{"threshold": 0.0}] * len(folds)
    assert len(result["estimator"]) == len(folds)
    for (train, test), estimator in zip(folds, result["estimator"], strict=True):
        assert set(estimator.seen_subjects_) == set(subject_ids[train])
        assert set(estimator.seen_subjects_).isdisjoint(subject_ids[test])


def make_subjects(rng):
    targets = np.array([0, 1] * 4)
    states = np.repeat([0, 1, 0, 1], 64)
    subjects = []
    for target in targets:
        recording = rng.normal(size=(4, 256))
        recording[0] *= 1 + 3 * target
        subjects.append((recording, states.copy()))
    return subjects, targets


def test_classification_pipeline_end_to_end(rng):
    subjects, y = make_subjects(rng)
    pipeline = build_classification_pipeline(
        sfreq=32,
        frequency_bands=((4.0, 15.0),),
        window_seconds=1,
    ).fit(subjects, y)

    assert pipeline.transform(subjects).shape == (8, 10)
    np.testing.assert_array_equal(pipeline.predict(subjects), y)
    assert pipeline.predict_proba(subjects).shape == (8, 2)
    assert pipeline.predict_log_proba(subjects).shape == (8, 2)
    assert pipeline.decision_function(subjects).shape == y.shape

    multilabel_y = np.column_stack([y, 1 - y, np.roll(y, 2)])
    multilabel_pipeline = build_classification_pipeline(
        classifier=OneVsRestClassifier(LogisticRegression(max_iter=1000)),
        sfreq=32,
        frequency_bands=((4.0, 15.0),),
        window_seconds=1,
    ).fit(subjects, multilabel_y)
    np.testing.assert_array_equal(multilabel_pipeline.predict(subjects), multilabel_y)
    assert multilabel_pipeline.predict_proba(subjects).shape == multilabel_y.shape
    assert 0.0 <= multilabel_pipeline.score(subjects, multilabel_y) <= 1.0

    mdm_pipeline = build_classification_pipeline(
        classifier=StateAwareMDM(),
        aggregation="stack",
        feature_transform="identity",
        sfreq=32,
        frequency_bands=((4.0, 15.0),),
        window_seconds=1,
    ).fit(subjects, y)
    assert mdm_pipeline.named_steps.classifier.state_shape_ == (2, 4, 4)
    assert mdm_pipeline.transform(subjects).shape == (8, 2)
    np.testing.assert_array_equal(mdm_pipeline.predict(subjects), y)


def test_subject_evaluation_validation():
    invalid_fold_inputs = [
        (np.ones((2, 1, 1)), ["a", "b"], {}, "one entry per subject"),
        (np.empty((4, 0)), ["a", "b", "c", "d"], {}, "one entry per subject"),
        ([0, 1], ["a", "a"], {}, "one unique ID"),
        (
            [0, 0, 0, 1],
            ["a", "b", "c", "d"],
            {"n_splits": 2},
            "each class",
        ),
        ([[0], [1]], ["a", "b"], {}, "each label class"),
        (
            [[0, 0], [1, 0], [0, 0], [1, 1]],
            ["a", "b", "c", "d"],
            {"n_splits": 2},
            "each label class",
        ),
        ([0, 1], ["a", "b"], {"n_splits": 1}, "at least 2"),
    ]
    for y, groups, kwargs, message in invalid_fold_inputs:
        with pytest.raises(ValueError, match=message):
            make_subject_folds(y, groups, **kwargs)

    valid_y = np.array([0, 1, 0, 1])
    valid_groups = np.array(["s0", "s1", "s2", "s3"])

    estimator = DummyClassifier()
    X = np.arange(8, dtype=float).reshape(4, 2)
    with pytest.raises(ValueError, match="X, y, and groups"):
        cross_validate_subjects(estimator, X[:-1], valid_y, valid_groups, n_splits=2)
    with pytest.raises(ValueError, match="at least one fold"):
        cross_validate_subjects(estimator, X, valid_y, valid_groups, cv=[])
    with pytest.raises(ValueError, match="overlap"):
        cross_validate_subjects(
            estimator,
            X,
            valid_y,
            valid_groups,
            cv=[(np.array([0, 1]), np.array([1, 2]))],
        )
