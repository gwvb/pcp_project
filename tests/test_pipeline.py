# tests/test_pipeline.py

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError

from pipeline import SubjectPipeline


class RecordingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add=0, name="transformer", return_tuple=False):
        self.add = add
        self.name = name
        self.return_tuple = return_tuple

    def fit(self, X, y=None, **kwargs):
        self.fit_called_ = True
        self.fit_X_ = np.array(X, copy=True)
        self.fit_y_ = None if y is None else np.array(y, copy=True)
        return self

    def transform(self, X, y=None, groups=None, **kwargs):
        self.transform_called_ = True
        self.transform_X_ = np.array(X, copy=True)
        if groups is not None:
            return np.array(X, copy=True) + self.add, np.array(groups)
        return np.array(X, copy=True) + self.add

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        self.fit_transform_called_ = True
        self.fit_transform_X_ = np.array(X, copy=True)
        self.fit_transform_y_ = None if y is None else np.array(y, copy=True)
        self.fit_transform_sample_weight_ = sample_weight
        self.fit_transform_kwargs_ = kwargs

        self.fit(X, y)

        if self.return_tuple:
            Xt, new_mask = self.transform(X, y, kwargs.get("groups", None))
            return Xt, new_mask
        Xt = self.transform(X, y)
        return Xt


class RecordingEstimator(BaseEstimator, ClassifierMixin):
    def fit(self, X, y, **kwargs):
        self.fit_called_ = True
        self.fit_X_ = np.array(X, copy=True)
        self.fit_y_ = np.array(y, copy=True)
        self.fit_kwargs_ = kwargs
        self.classes_ = np.unique(y)
        return self

    def predict(self, X, **kwargs):
        self.predict_called_ = True
        self.predict_X_ = np.array(X, copy=True)
        self.predict_kwargs_ = kwargs
        return np.repeat(self.classes_[0], len(X))


class ProbaEstimator(BaseEstimator, ClassifierMixin):
    """Final estimator with predict_proba/predict_log_proba/decision_function."""

    def fit(self, X, y, **kwargs):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X, **kwargs):
        return np.repeat(self.classes_[0], len(X))

    def predict_proba(self, X, groups=None, **kwargs):
        self.predict_proba_X_ = np.array(X, copy=True)
        self.predict_proba_groups_ = groups
        return np.tile([0.7, 0.3], (len(X), 1))

    def predict_log_proba(self, X, groups=None, **kwargs):
        self.predict_log_proba_X_ = np.array(X, copy=True)
        self.predict_log_proba_groups_ = groups
        return np.log(np.tile([0.7, 0.3], (len(X), 1)))

    def decision_function(self, X, groups=None, **kwargs):
        self.decision_function_X_ = np.array(X, copy=True)
        self.decision_function_groups_ = groups
        return np.ones(len(X))


@pytest.fixture
def Xy():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    y = np.array([0, 1, 0])
    return X, y


@pytest.fixture
def mask():
    return np.array([True, True, True])


@pytest.fixture
def simple_pipeline(mask):
    return SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1")),
            ("t2", RecordingTransformer(add=2, name="t2")),
            ("clf", RecordingEstimator()),
        ],
        mask=mask,
    )


def test_fit_returns_self(simple_pipeline, Xy):
    X, y = Xy
    result = simple_pipeline.fit(X, y)
    assert result is simple_pipeline


def test_predict_before_fit_raises(simple_pipeline, Xy):
    X, _ = Xy
    with pytest.raises(NotFittedError):
        simple_pipeline.predict(X)


def test_transform_before_fit_raises(simple_pipeline, Xy):
    X, _ = Xy
    with pytest.raises(NotFittedError):
        simple_pipeline.transform(X)


def test_fit_applies_transformers_in_order_and_passes_transformed_X_to_final_estimator(
    simple_pipeline, Xy
):
    X, y = Xy
    simple_pipeline.fit(X, y)

    expected_after_t1 = X + 1
    expected_after_t2 = X + 1 + 2

    np.testing.assert_array_equal(simple_pipeline.named_steps["t1"].fit_transform_X_, X)
    np.testing.assert_array_equal(
        simple_pipeline.named_steps["t2"].fit_transform_X_, expected_after_t1
    )
    np.testing.assert_array_equal(
        simple_pipeline.named_steps["clf"].fit_X_, expected_after_t2
    )
    np.testing.assert_array_equal(simple_pipeline.named_steps["clf"].fit_y_, y)


def test_predict_transforms_with_non_final_steps_then_calls_final_estimator(
    simple_pipeline, Xy
):
    X, y = Xy
    simple_pipeline.fit(X, y)
    preds = simple_pipeline.predict(X)

    expected_Xt = X + 1 + 2

    np.testing.assert_array_equal(simple_pipeline.named_steps["t1"].transform_X_, X)
    np.testing.assert_array_equal(simple_pipeline.named_steps["t2"].transform_X_, X + 1)
    np.testing.assert_array_equal(
        simple_pipeline.named_steps["clf"].predict_X_, expected_Xt
    )
    assert preds.shape == (len(X),)


def test_fit_forwards_step_params_to_transformer(mask, Xy):
    X, y = Xy
    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1")),
            ("clf", RecordingEstimator()),
        ],
        mask=mask,
    )

    pipe._fit(X, y, t1__alpha=123)

    assert pipe.named_steps["t1"].fit_transform_kwargs_["alpha"] == 123


def test_fit_forwards_step_params_to_final_estimator(mask, Xy):
    X, y = Xy
    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1")),
            ("clf", RecordingEstimator()),
        ],
        mask=mask,
    )

    Xt, yt, final_fit_params = pipe._fit(
        X, y, clf__sample_weight=np.array([1.0, 2.0, 3.0])
    )

    np.testing.assert_array_equal(Xt, X + 1)
    np.testing.assert_array_equal(yt, y)
    np.testing.assert_array_equal(
        final_fit_params["sample_weight"], np.array([1.0, 2.0, 3.0])
    )


def test_fit_updates_X(Xy):
    X, y = Xy
    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1", return_tuple=False)),
            ("clf", RecordingEstimator()),
        ]
    )
    pipe.fit(X, y)
    np.testing.assert_array_equal(pipe.named_steps["clf"].fit_X_, X + 1)


def test_transform_updates_mask(mask, Xy):
    X, y = Xy
    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1", return_tuple=True)),
            ("clf", RecordingEstimator()),
        ],
        mask=mask,
    )
    pipe.fit(X, y)
    _ = pipe.transform(X, y)
    np.testing.assert_array_equal(pipe.mask, np.array(mask))


def test_refit_overwrites_state(mask, Xy):
    X, y = Xy
    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1")),
            ("clf", RecordingEstimator()),
        ],
        mask=mask,
    )

    pipe.fit(X, y)
    first_fit_X = pipe.named_steps["clf"].fit_X_.copy()

    X2 = X + 100
    pipe.fit(X2, y)
    second_fit_X = pipe.named_steps["clf"].fit_X_

    assert not np.allclose(first_fit_X, second_fit_X)
    np.testing.assert_array_equal(second_fit_X, X2 + 1)


def test_mask_is_stored(mask):
    pipe = SubjectPipeline(
        steps=[("clf", RecordingEstimator())],
        mask=mask,
    )
    assert pipe.mask is mask


def test_fit_accepts_nan_padded_X(mask):
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, np.nan],
            [np.nan, np.nan],
        ]
    )
    y = np.array([0, 1, 0])

    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1")),
            ("clf", RecordingEstimator()),
        ],
        mask=mask[: len(X)],
    )

    result = pipe.fit(X, y)
    assert result is pipe


def test_predict_accepts_nan_padded_X(mask):
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, np.nan],
            [np.nan, np.nan],
        ]
    )
    y = np.array([0, 1, 0])

    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1")),
            ("clf", RecordingEstimator()),
        ],
        mask=mask[: len(X)],
    )

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (len(X),)


def test_nan_values_are_forwarded_to_final_estimator(mask):
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, np.nan],
            [np.nan, np.nan],
        ]
    )
    y = np.array([0, 1, 0])

    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=0, name="t1")),
            ("clf", RecordingEstimator()),
        ],
        mask=mask[: len(X)],
    )

    pipe.fit(X, y)

    assert np.isnan(pipe.named_steps["clf"].fit_X_[1, 1])
    assert np.isnan(pipe.named_steps["clf"].fit_X_[2, 0])
    assert np.isnan(pipe.named_steps["clf"].fit_X_[2, 1])


def test_fit_private_with_no_steps_returns_inputs_unchanged(Xy):
    X, y = Xy
    pipe = SubjectPipeline(steps=[])

    Xt, yt, final_fit_params = pipe._fit(X, y)

    np.testing.assert_array_equal(Xt, X)
    np.testing.assert_array_equal(yt, y)
    assert final_fit_params == {}


def test_fit_private_skips_passthrough_intermediate_steps(mask, Xy):
    X, y = Xy
    pipe = SubjectPipeline(
        steps=[
            ("skip", "passthrough"),
            ("t1", RecordingTransformer(add=1, name="t1")),
            ("clf", RecordingEstimator()),
        ],
        mask=mask,
    )

    Xt, yt, final_fit_params = pipe._fit(X, y)

    np.testing.assert_array_equal(Xt, X + 1)
    np.testing.assert_array_equal(yt, y)


def test_predict_proba_not_available_without_final_predict_proba(mask, Xy):
    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1")),
            ("clf", RecordingEstimator()),
        ],
        mask=mask,
    )
    assert not hasattr(pipe, "predict_proba")


def test_predict_proba_before_fit_raises(mask, Xy):
    X, _ = Xy
    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1")),
            ("clf", ProbaEstimator()),
        ],
        mask=mask,
    )
    with pytest.raises(NotFittedError):
        pipe.predict_proba(X)


def test_decision_function_calls_final_estimator_with_groups(mask, Xy):
    X, y = Xy
    final = ProbaEstimator()
    pipe = SubjectPipeline(
        steps=[("t1", RecordingTransformer(add=1, name="t1")), ("clf", final)],
        mask=mask,
    )
    pipe.fit(X, y)
    scores = pipe.decision_function(X)

    np.testing.assert_array_equal(final.decision_function_X_, X + 1)
    np.testing.assert_array_equal(final.decision_function_groups_, mask)
    assert scores.shape == (len(X),)


def test_predict_proba_calls_final_estimator_with_groups(mask, Xy):
    X, y = Xy
    final = ProbaEstimator()
    pipe = SubjectPipeline(
        steps=[("t1", RecordingTransformer(add=1, name="t1")), ("clf", final)],
        mask=mask,
    )
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)

    np.testing.assert_array_equal(final.predict_proba_X_, X + 1)
    np.testing.assert_array_equal(final.predict_proba_groups_, mask)
    assert proba.shape == (len(X), 2)


def test_predict_log_proba_calls_final_estimator_with_groups(mask, Xy):
    X, y = Xy
    final = ProbaEstimator()
    pipe = SubjectPipeline(
        steps=[("t1", RecordingTransformer(add=1, name="t1")), ("clf", final)],
        mask=mask,
    )
    pipe.fit(X, y)
    log_proba = pipe.predict_log_proba(X)

    np.testing.assert_array_equal(final.predict_log_proba_X_, X + 1)
    np.testing.assert_array_equal(final.predict_log_proba_groups_, pipe.mask)
    assert log_proba.shape == (len(X), 2)


def test_transform_skips_passthrough_intermediate_steps(mask, Xy):
    X, y = Xy
    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1")),
            ("skip", "passthrough"),
            ("clf", RecordingEstimator()),
        ],
        mask=mask,
    )
    pipe.fit(X, y)
    X_t, pipe.mask = pipe.transform(X, y)
    np.testing.assert_array_equal(X_t, X + 1)


def test_transform_calls_intermediate_transformer_without_groups_param(mask, Xy):
    X, y = Xy
    intermediate = RecordingTransformer(add=3)
    pipe = SubjectPipeline(
        steps=[("t1", intermediate), ("clf", RecordingEstimator())],
        mask=mask,
    )
    pipe.fit(X, y)
    X_t, pipe.mask = pipe.transform(X, y)

    assert intermediate.transform_called_
    np.testing.assert_array_equal(X_t, X + 3)
    np.testing.assert_array_equal(pipe.mask, mask)
