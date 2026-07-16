# tests/test_subject_pipeline.py
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError

from pcp_project.pipeline import SubjectPipeline


class RecordingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add=0, name="transformer", return_tuple=False):
        self.add = add
        self.name = name
        self.return_tuple = return_tuple

    # dimension of X and Y don't change after fitting a model on the data
    # fit log regression etc
    def fit(self, X, y=None, **kwargs):
        self.fit_called_ = True
        self.fit_X_ = np.array(X, copy=True)
        self.fit_y_ = None if y is None else np.array(y, copy=True)
        return self

    # transform X
    def transform(self, X, y=None, groups=None):
        self.transform_called_ = True
        self.transform_X_ = np.array(X, copy=True)
        if groups is not None:
            return np.array(X, copy=True) + self.add, np.array(groups)
        return np.array(X, copy=True) + self.add

    # return transformed X
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


# if we have 2 channels 3 timeseries
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


# mask returns 1,1,1 for 3 timeseries, meaning all 3 belong to eyes closed.
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


# fit should return self
def test_fit_returns_self(simple_pipeline, Xy):
    X, y = Xy
    result = simple_pipeline.fit(X, y)
    assert result is simple_pipeline


# calling predict before fit raises error
def test_predict_before_fit_raises(simple_pipeline, Xy):
    X, _ = Xy
    # if the code inside pytest raises NotFittedError, pytest catches it, test passes
    with pytest.raises(NotFittedError):
        # X is unfitted X
        simple_pipeline.predict(X)


# calling transform before fit raises notfitted error.
# pipeline.fit(X) fits and transforms the data sequentially
def test_transform_before_fit_raises(simple_pipeline, Xy):
    X, _ = Xy
    with pytest.raises(NotFittedError):
        # first fit, then transform X
        simple_pipeline.transform(X)


#
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


# test an empty pipeline, steps = []
# directly sets self.is_fitted_ = True, returns itself
# predict raises AttributeError


def test_empty_pipeline(Xy):
    X, y = Xy
    pipe = SubjectPipeline(steps=[])

    # fitting returns self
    assert pipe.fit(X, y) is pipe

    # Transforming returns the input data
    np.testing.assert_array_equal(pipe.transform(X), X)

    # Predicting raises AttributeError as there is no final estimator
    with pytest.raises(
        AttributeError, match="The final step does not implement predict"
    ):
        pipe.predict(X)


# in scikit-learn it is common to use pipelines for data processing.
# without final classifier. final step set to None or "passthrough"
# final estimator(classifier) doesn't exist. end goal is not a
# yes or no, but a transformed data


# pipeline succesfully fitted
# pipeline can transform correctly
# calling predict raises Attributeerror
@pytest.mark.parametrize("final_step", [None, "passthrough"])
def test_pipeline_with_none_final_step(final_step, Xy):
    X, y = Xy
    pipe = SubjectPipeline(
        steps=[
            ("t1", RecordingTransformer(add=1, name="t1")),
            ("clf", final_step),
        ]
    )

    # fit works
    assert pipe.fit(X, y) is pipe

    # transform runs transformers
    np.testing.assert_array_equal(pipe.transform(X), X + 1)

    # predict raises error
    with pytest.raises(
        AttributeError, match="The final step does not implement predict"
    ):
        pipe.predict(X)


# test invalid fit parameter format
# if a parameter is passed without step__param syntax,
# it raises value error
def test_fit_invalid_parameters_raise_value_error(simple_pipeline, Xy):
    X, y = Xy

    # Parameter without double underscores '__'
    # _fit function splits each parameter name by "__"
    with pytest.raises(
        ValueError, match="Fit parameters must use the step__param format"
    ):
        simple_pipeline.fit(X, y, invalid_param_name=True)

    # Parameter with an unknown step name "non_existent_step"
    # _fit function inside pipeline iterates through steps,
    # splits param name from "__", so "non_existent_step__param"
    # becomes "non_existent_step", and if it isn't in step names ["t1,"t2]
    # then throws value error
    with pytest.raises(ValueError, match="Unknown step name in fit parameters"):
        simple_pipeline.fit(X, y, non_existent_step__param=True)
