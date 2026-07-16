"""Tests for the EEG estimators."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyriemann.estimation import Covariances
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from pcp_project._helpers import (
    _declares_param,
    _final_estimator_has,
    _map_recording_pairs,
    _metadata_kwargs,
    _selected_runs,
    _split_input,
    _subject_collection,
    _transform_one,
    _window_subjects,
)
from pcp_project.data import (
    balanced_subject_ids,
    binary_target,
    list_subject_ids,
    load_labels,
    load_subject,
)
from pcp_project.estimators import (
    BandPassFilter,
    BatchCovariances,
    MeanProbabilityAggregator,
    NotchFilter,
    SlidingWindow,
    StateSelector,
)

N_CHANNELS = 4
N_SAMPLES = 600
ESTIMATORS = ["lwf", "oas", "scm"]


@pytest.fixture
def recording(rng):
    X = rng.standard_normal((N_CHANNELS, N_SAMPLES))
    states = np.repeat([0, 1], N_SAMPLES // 2)
    return X, states


@pytest.fixture
def subject_collection():
    first = np.vstack([np.arange(12), np.arange(12) + 20.0])
    second = np.vstack([np.arange(12) + 100.0, np.arange(12) + 120.0])
    return [
        (first, np.repeat([0, 1, 0], 4)),
        (second, np.repeat([1, 0], 6)),
    ]


##########################################################################
# Tests of StateSelector Class
##########################################################################


def test_state_selector(recording, subject_collection):
    X, states = recording

    # 1
    selected, selected_states = StateSelector("eyes_closed").fit_transform(
        X, groups=states
    )
    assert selected.shape == (N_CHANNELS, N_SAMPLES // 2)
    np.testing.assert_array_equal(selected_states, np.ones(N_SAMPLES // 2))

    # 2
    named_states = np.where(states == 0, "eyes_open", "eyes_closed")
    _, selected_names = StateSelector("eyes_closed").fit_transform(
        X, groups=named_states
    )
    assert set(selected_names) == {"eyes_closed"}

    # 3
    np.testing.assert_array_equal(StateSelector().fit_transform(X), X)
    unchanged, unchanged_states = StateSelector().fit_transform((X, states))
    np.testing.assert_array_equal(unchanged, X)
    np.testing.assert_array_equal(unchanged_states, states)

    # 4
    object_array = np.empty(2, dtype=object)
    object_array[:] = subject_collection
    runs = StateSelector([0]).fit_transform(object_array)
    assert [len(subject_runs) for subject_runs in runs] == [2, 1]

    # 5
    np.testing.assert_array_equal(runs[0][0][0][0], [0, 1, 2, 3])
    np.testing.assert_array_equal(runs[0][1][0][0], [8, 9, 10, 11])

    # 6
    all_runs = StateSelector([0, 1]).fit_transform(subject_collection)
    assert [int(run_states[0]) for _, run_states in all_runs[0]] == [0, 1, 0]

    # 7
    one_sample, one_state = StateSelector([1]).fit_transform(
        np.ones((2, 1)),
        groups=np.array([1]),
    )
    assert one_sample.shape == (2, 1)
    np.testing.assert_array_equal(one_state, [1])

    # 8
    selector = StateSelector([1]).fit(X)
    with pytest.raises(ValueError, match="groups are required"):
        selector.transform(X)

    # 9
    alternate = 1 - states
    _, chosen = selector.transform((X, states), groups=alternate)
    np.testing.assert_array_equal(chosen, np.ones(N_SAMPLES // 2))


# 10
def test_state_selector_not_modify_input(recording):
    X, states = recording
    original = X.copy()
    selector = StateSelector([0])
    _ = selector.fit_transform(X, groups=states)
    np.testing.assert_array_equal(X, original)


# 11
def test_state_selector_transform_before_fit_raises_error(recording):
    X, _ = recording
    selector = StateSelector([0])
    with pytest.raises((ValueError, AttributeError, NotFittedError)):
        selector.transform(X)


# 12
def test_state_selector_fit_returns_self(recording):
    X, _ = recording
    selector = StateSelector([0])
    result = selector.fit(X)
    assert result is selector


# 13
def test_state_selector_fitted_attribute_exists(recording):
    X, _ = recording
    selector = StateSelector([0])
    assert not hasattr(selector, "fitted_")
    selector.fit(X)
    assert hasattr(selector, "fitted_")


##########################################################################
# Tests of BandPassFilter Class
##########################################################################


@pytest.fixture
def eeg_signal():
    """Create deterministic fake EEG data.

    The shape follows our project convention:
    (n_channels, n_samples).
    """
    rng = np.random.default_rng(0)
    return rng.normal(size=(61, 1000))


@pytest.fixture
def eeg_signal_float32():
    """Create fake EEG data with float32 dtype."""
    rng = np.random.default_rng(1)
    return rng.normal(size=(61, 1000)).astype(np.float32)


@pytest.fixture
def eeg_signal_with_nan(eeg_signal):
    """Create EEG data with a NaN interval."""
    X = eeg_signal.copy()
    X[:, 100:200] = np.nan
    return X


def test_bandpass_output_shape(eeg_signal):
    """Output shape must be the same as input shape."""
    filt = BandPassFilter(frequency_bands=[[5, 10]])

    result = filt.fit_transform(eeg_signal)

    assert result.shape == eeg_signal.shape


def test_bandpass_output_is_float64(eeg_signal_float32):
    """Output must be float64 even if input is float32."""
    filt = BandPassFilter(frequency_bands=[[5, 10]])

    result = filt.fit_transform(eeg_signal_float32)

    assert result.dtype == np.float64


def test_bandpass_single_band(eeg_signal):
    """Filter must work with one frequency band."""
    filt = BandPassFilter(frequency_bands=[[8, 13]])

    result = filt.fit_transform(eeg_signal)

    assert result.shape == eeg_signal.shape
    assert result.dtype == np.float64


def test_bandpass_multiple_bands(eeg_signal):
    """Filter must work with multiple frequency bands."""
    filt = BandPassFilter(frequency_bands=[[5, 10], [13, 35]])

    result = filt.fit_transform(eeg_signal)

    assert result.shape == eeg_signal.shape
    assert result.dtype == np.float64


def test_bandpass_default_sfreq():
    """Default sampling frequency must be 256.0."""
    filt = BandPassFilter(frequency_bands=[[5, 10]])

    assert filt.sfreq == 256.0


def test_bandpass_fit_returns_self(eeg_signal):
    """fit() must return self for scikit-learn compatibility."""
    filt = BandPassFilter(frequency_bands=[[5, 10]])

    result = filt.fit(eeg_signal)

    assert result is filt


def test_bandpass_fitted_attribute_exists(eeg_signal):
    """fit() must create the fitted_ attribute."""
    filt = BandPassFilter(frequency_bands=[[5, 10]])

    filt.fit(eeg_signal)

    assert hasattr(filt, "fitted_")
    assert filt.fitted_ is True


def test_bandpass_transform_before_fit_raises_error(eeg_signal):
    """transform() before fit() must raise NotFittedError."""
    filt = BandPassFilter(frequency_bands=[[5, 10]])

    with pytest.raises(NotFittedError):
        filt.transform(eeg_signal)


def test_bandpass_does_not_modify_input(eeg_signal):
    """Filtering must not modify the original input array in place."""
    original = eeg_signal.copy()
    filt = BandPassFilter(frequency_bands=[[5, 10]])

    _ = filt.fit_transform(eeg_signal)

    np.testing.assert_array_equal(eeg_signal, original)


def test_bandpass_nan_mask_handling(eeg_signal_with_nan):
    """NaN columns must stay NaN in the output."""
    filt = BandPassFilter(frequency_bands=[[5, 10]])

    result = filt.fit_transform(eeg_signal_with_nan)

    assert result.shape == eeg_signal_with_nan.shape

    # The NaN interval must stay NaN.
    assert np.isnan(result[:, 100:200]).all()

    # Valid parts should not contain NaN.
    assert not np.isnan(result[:, :100]).any()
    assert not np.isnan(result[:, 200:]).any()


def test_bandpass_all_nan_returns_unchanged_shape():
    """All-NaN input must be handled safely."""
    X_all_nan = np.full((61, 1000), fill_value=np.nan)

    filt = BandPassFilter(frequency_bands=[[5, 10]])
    result = filt.fit_transform(X_all_nan)

    assert result.shape == X_all_nan.shape
    assert result is not X_all_nan
    assert np.isnan(result).all()


def test_bandpass_get_params():
    """get_params() must return constructor parameters."""
    filt = BandPassFilter(
        frequency_bands=[[5, 10], [13, 35]],
        sfreq=512.0,
    )

    params = filt.get_params()

    assert params["frequency_bands"] == [[5, 10], [13, 35]]
    assert params["sfreq"] == 512.0


def test_bandpass_preserves_groups_metadata(eeg_signal):
    """If groups are passed, they must be returned unchanged."""
    groups = np.arange(eeg_signal.shape[1])

    filt = BandPassFilter(frequency_bands=[[5, 10]])
    X_filtered, groups_out = filt.fit_transform(eeg_signal, groups=groups)

    assert X_filtered.shape == eeg_signal.shape
    np.testing.assert_array_equal(groups_out, groups)


def test_bandpass_accepts_tuple_input(eeg_signal):
    """Tuple input must be interpreted as (recording, groups)."""
    groups = np.arange(eeg_signal.shape[1])

    filt = BandPassFilter(frequency_bands=[[5, 10]])
    X_filtered, groups_out = filt.fit_transform((eeg_signal, groups))

    assert X_filtered.shape == eeg_signal.shape
    np.testing.assert_array_equal(groups_out, groups)


def test_bandpass_collection_of_recording_pairs():
    """A collection of recording pairs must be filtered item by item."""
    rng = np.random.default_rng(2)

    X1 = rng.normal(size=(61, 1000))
    X2 = rng.normal(size=(61, 1500))

    groups1 = np.zeros(X1.shape[1], dtype=int)
    groups2 = np.ones(X2.shape[1], dtype=int)

    collection = [
        (X1, groups1),
        (X2, groups2),
    ]

    filt = BandPassFilter(frequency_bands=[[5, 10]])
    result = filt.fit_transform(collection)

    assert isinstance(result, list)
    assert len(result) == 2

    X1_filtered, groups1_out = result[0]
    X2_filtered, groups2_out = result[1]

    assert X1_filtered.shape == X1.shape
    assert X2_filtered.shape == X2.shape

    np.testing.assert_array_equal(groups1_out, groups1)
    np.testing.assert_array_equal(groups2_out, groups2)


def test_bandpass_removes_frequency_outside_band():
    """A 5-10 Hz filter should strongly reduce a 50 Hz signal."""
    sfreq = 256.0
    n_samples = int(4 * sfreq)
    t = np.arange(n_samples) / sfreq

    signal_50_hz = np.sin(2 * np.pi * 50 * t)
    X = np.tile(signal_50_hz, (61, 1))

    filt = BandPassFilter(
        frequency_bands=[[5, 10]],
        sfreq=sfreq,
    )
    result = filt.fit_transform(X)

    # Ignore filter edge effects.
    middle = slice(256, -256)

    assert np.var(result[:, middle]) < 0.01 * np.var(X[:, middle])


def test_bandpass_preserves_frequency_inside_band():
    """A 5-10 Hz filter should preserve an 8 Hz signal."""
    sfreq = 256.0
    n_samples = int(4 * sfreq)
    t = np.arange(n_samples) / sfreq

    signal_8_hz = np.sin(2 * np.pi * 8 * t)
    X = np.tile(signal_8_hz, (61, 1))

    filt = BandPassFilter(
        frequency_bands=[[5, 10]],
        sfreq=sfreq,
    )
    result = filt.fit_transform(X)

    # Ignore filter edge effects.
    middle = slice(256, -256)

    correlation = np.corrcoef(
        result[0, middle],
        signal_8_hz[middle],
    )[0, 1]

    assert correlation > 0.95


def test_bandpass_explicit_groups_override_tuple_groups(eeg_signal):
    tuple_groups = np.arange(eeg_signal.shape[1])
    explicit_groups = np.arange(eeg_signal.shape[1]) + 100

    filt = BandPassFilter(frequency_bands=[[5, 10]])
    X_filtered, groups_out = filt.fit_transform(
        (eeg_signal, tuple_groups),
        groups=explicit_groups,
    )

    assert X_filtered.shape == eeg_signal.shape
    np.testing.assert_array_equal(groups_out, explicit_groups)


##########################################################################
# Tests of NotchFilter Class
##########################################################################


def test_notch_output_properties(recording):
    X, _ = recording
    filt = NotchFilter(freqs=50.0)
    result = filt.fit_transform(X)
    assert result.shape == X.shape
    assert result.dtype == np.float64


def test_notch_fit_lifecycle(recording):
    X, _ = recording
    filt = NotchFilter(freqs=50.0)
    result = filt.fit(X)
    assert result is filt
    assert hasattr(filt, "fitted_")


def test_notch_does_not_modify_input(recording):
    X, _ = recording
    original = X.copy()
    filt = NotchFilter(freqs=50.0)
    _ = filt.fit_transform(X)
    np.testing.assert_array_equal(X, original)


def test_notch_transform_before_fit_raises_error(recording):
    X, _ = recording
    filt = NotchFilter(freqs=50.0)
    with pytest.raises(NotFittedError):
        filt.transform(X)


def test_notch_removes_frequency():
    sfreq = 250.0
    t = np.arange(0, 15, 1 / sfreq)
    pure_50hz = np.sin(2 * np.pi * 50 * t).reshape(1, -1)

    filt = NotchFilter(freqs=50.0, sfreq=sfreq)
    filtered = filt.fit_transform(pure_50hz)
    assert np.var(filtered) < 0.05 * np.var(pure_50hz)


def test_notch_filter(recording, monkeypatch):
    X, states = recording
    X = X.copy()
    X[:, -2:] = np.nan
    calls = []

    def fake_notch(values, **kwargs):
        calls.append(kwargs)
        return values + 1

    monkeypatch.setattr("pcp_project.estimators.mne.filter.notch_filter", fake_notch)
    # 1
    filtered, output_states = NotchFilter(
        freqs=[50, 200], notch_widths=[2, 8]
    ).fit_transform(X, groups=states)
    np.testing.assert_allclose(filtered[:, :-2], X[:, :-2] + 1)
    assert np.isnan(filtered[:, -2:]).all()
    # 2
    np.testing.assert_array_equal(output_states, states)
    np.testing.assert_array_equal(calls[0]["freqs"], [50])
    np.testing.assert_array_equal(calls[0]["notch_widths"], [2])
    # 3
    _, tuple_states = NotchFilter().fit_transform((X, states))
    np.testing.assert_array_equal(tuple_states, states)
    alternate_states = 1 - states
    _, output_states = NotchFilter().fit_transform((X, states), groups=alternate_states)
    np.testing.assert_array_equal(output_states, alternate_states)
    # 4
    NotchFilter(freqs=[50, 100], notch_widths=2).fit_transform(X)
    np.testing.assert_array_equal(calls[-1]["notch_widths"], [2])

    np.testing.assert_equal(NotchFilter(sfreq=32).fit_transform(X), X)
    all_nan = np.full_like(X, np.nan)
    np.testing.assert_equal(NotchFilter().fit_transform(all_nan), all_nan)


##########################################################################
# Tests of BatchCovariances Class
##########################################################################


def assert_matches_pyriemann(
    X,
    estimator,
    *,
    assume_centered=True,
    rtol=1e-6,
    atol=1e-8,
):
    ours = BatchCovariances(
        estimator=estimator,
        assume_centered=assume_centered,
    ).fit_transform(X)

    theirs = Covariances(
        estimator=estimator,
        assume_centered=assume_centered,
    ).fit_transform(X)

    assert np.allclose(ours, theirs, rtol=rtol, atol=atol)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize(
    "shape,dtype,seed,assume_centered,rtol,atol",
    [
        ((5, 61, 10), np.float64, 42, True, 1e-6, 1e-8),
        ((10, 61, 20), np.float64, 42, True, 1e-6, 1e-8),
        ((20, 61, 15), np.float64, 42, True, 1e-6, 1e-8),
        ((50, 61, 8), np.float64, 42, True, 1e-6, 1e-8),
        ((10, 61, 20), np.float64, 42, False, 1e-6, 1e-8),
        ((25, 61, 18), np.float32, 1234, True, 1e-5, 1e-6),
    ],
)
def test_batch_covariances_matches_pyriemann(
    estimator,
    shape,
    dtype,
    seed,
    assume_centered,
    rtol,
    atol,
):
    """Batch implementation should match pyRiemann across representative settings."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(shape).astype(dtype)

    assert_matches_pyriemann(
        X,
        estimator,
        assume_centered=assume_centered,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("seed", range(3))
def test_batch_covariances_matches_pyriemann_random(estimator, seed):
    """Multiple random realizations should agree."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((10, 61, 20))

    assert_matches_pyriemann(X, estimator)


@pytest.mark.parametrize(
    "shape,expected_shape",
    [
        ((17, 61, 11), (17, 61, 61)),
        ((5, 1, 20), (5, 1, 1)),
    ],
)
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_batch_covariances_output_shape(
    estimator,
    shape,
    expected_shape,
):
    """Output covariance matrices should have the expected shape."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal(shape)

    cov = BatchCovariances(
        estimator=estimator,
        assume_centered=True,
    ).fit_transform(X)

    assert cov.shape == expected_shape


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_batch_covariances_invalid_dimension(estimator):
    """Invalid input dimensions should raise a ValueError."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((61, 20))

    with pytest.raises(ValueError, match="shape"):
        BatchCovariances(
            estimator=estimator,
            assume_centered=True,
        ).fit_transform(X)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_batch_covariances_warns_on_single_sample(estimator):
    """A warning should be emitted when only one sample is provided."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 61, 1))

    with pytest.warns(UserWarning, match="Only one sample"):
        BatchCovariances(
            estimator=estimator,
            assume_centered=True,
        ).fit_transform(X)


def test_batch_covariances_invalid_estimator():
    """An unknown estimator name should raise a ValueError."""
    with pytest.raises(ValueError, match="Invalid method"):
        BatchCovariances(estimator="not_a_real_estimator")


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_batch_covariances_accepts_tuple_input(estimator):
    """Tuple inputs should use the first element as the data."""
    rng = np.random.default_rng(42)

    X = rng.standard_normal((10, 61, 20))
    y = np.arange(10)

    ours_tuple = BatchCovariances(
        estimator=estimator,
        assume_centered=True,
    ).fit_transform((X, y))

    ours_array = BatchCovariances(
        estimator=estimator,
        assume_centered=True,
    ).fit_transform(X)

    assert np.allclose(ours_tuple, ours_array)


##########################################################################
# Tests of MeanProbabilityAggregator Class
##########################################################################


def test_fit_sets_attribute():
    """Verify that fit sets the fitted_ attribute and returns self."""
    aggregator = MeanProbabilityAggregator()
    assert not hasattr(aggregator, "fitted_")

    returned_estimator = aggregator.fit(X=np.array([0.1, 0.2]), y=None)

    assert hasattr(aggregator, "fitted_")
    assert aggregator.fitted_ is True
    assert returned_estimator is aggregator


def test_not_fitted_raises_error():
    """Verify that calling predict or transform before fit raises NotFittedError."""
    aggregator = MeanProbabilityAggregator()
    values = np.array([0.1, 0.2])
    groups = np.array(["s1", "s1"])

    with pytest.raises(NotFittedError):
        aggregator.predict_proba(values, groups=groups)

    with pytest.raises(NotFittedError):
        aggregator.predict(values, groups=groups)

    with pytest.raises(NotFittedError):
        aggregator.transform(values, groups=groups)


def test_missing_groups_raises_value_error():
    """Verify that a ValueError is raised when groups are not provided in any format."""
    aggregator = MeanProbabilityAggregator()
    aggregator.fit(None)
    values = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="groups must be provided"):
        aggregator.transform(values, groups=None)


def test_aggregation_order_and_values_1d():
    """Verify aggregation on 1D arrays, ensuring correct averaging and first-appearance ordering."""
    # "s2" appears first, then "s1"
    groups = np.array(["s2", "s1", "s2", "s1"])
    values = np.array([10.0, 1.0, 20.0, 3.0])

    aggregator = MeanProbabilityAggregator()
    aggregator.fit(None)

    # Expected order: s2 (mean of 10 and 20 -> 15), then s1 (mean of 1 and 3 -> 2)
    expected = np.array([15.0, 2.0])
    result = aggregator.transform(values, groups=groups)

    np.testing.assert_allclose(result, expected)


def test_aggregation_order_and_values_multi_dim():
    """Verify aggregation handles multi-dimensional arrays (windows, features/classes)."""
    # 4 windows, 2 classes/features
    values = np.array(
        [
            [0.8, 0.2],  # s2
            [0.1, 0.9],  # s1
            [0.6, 0.4],  # s2
            [0.3, 0.7],  # s1
        ]
    )
    groups = np.array(["s2", "s1", "s2", "s1"])

    aggregator = MeanProbabilityAggregator()
    aggregator.fit(None)

    # Expected:
    # s2 mean: [(0.8 + 0.6)/2, (0.2 + 0.4)/2] = [0.7, 0.3]
    # s1 mean: [(0.1 + 0.3)/2, (0.9 + 0.7)/2] = [0.2, 0.8]
    expected = np.array([[0.7, 0.3], [0.2, 0.8]])
    result = aggregator.transform(values, groups=groups)

    np.testing.assert_allclose(result, expected)


def test_tuple_input_for_pipeline():
    """Verify that passing (X, groups) as a tuple works correctly for pipelines."""
    values = np.array([10.0, 1.0, 20.0, 3.0])
    groups = np.array(["s2", "s1", "s2", "s1"])

    aggregator = MeanProbabilityAggregator()
    aggregator.fit(None)

    # Passing tuple, omitting explicit groups arg
    result = aggregator.transform((values, groups))
    expected = np.array([15.0, 2.0])

    np.testing.assert_allclose(result, expected)


def test_explicit_groups_overrides_tuple_groups():
    """Verify that explicit groups parameter overrides groups bundled inside a tuple."""
    values = np.array([10.0, 20.0, 30.0, 40.0])
    bundled_groups = np.array(["s1", "s1", "s2", "s2"])
    explicit_groups = np.array(["g1", "g2", "g1", "g2"])

    aggregator = MeanProbabilityAggregator()
    aggregator.fit(None)

    # Should use explicit_groups:
    # g1 mean (indices 0, 2): (10 + 30) / 2 = 20
    # g2 mean (indices 1, 4): (20 + 40) / 2 = 30
    result = aggregator.transform((values, bundled_groups), groups=explicit_groups)
    expected = np.array([20.0, 30.0])

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "threshold, expected_labels",
    [
        (0.5, [1, 0]),  # 0.7 >= 0.5 (1), 0.4 < 0.5 (0)
        (0.8, [0, 0]),  # Both below 0.8
        (0.3, [1, 1]),  # Both above 0.3
    ],
)
def test_predict_thresholding(threshold, expected_labels):
    """Verify that predict thresholds averaged probabilities correctly based on threshold."""
    # Mean values: s2 = 0.7, s1 = 0.4
    values = np.array([0.8, 0.3, 0.6, 0.5])
    groups = np.array(["s2", "s1", "s2", "s1"])

    aggregator = MeanProbabilityAggregator(threshold=threshold)
    aggregator.fit(None)

    predictions = aggregator.predict(values, groups=groups)
    np.testing.assert_array_equal(predictions, expected_labels)


def test_fit_transform_combines_steps():
    """Verify fit_transform successfully fits the model and returns transformed outputs."""
    values = np.array([10.0, 1.0, 20.0, 3.0])
    groups = np.array(["s2", "s1", "s2", "s1"])

    aggregator = MeanProbabilityAggregator()
    result = aggregator.fit_transform(values, groups=groups)

    assert aggregator.fitted_ is True
    expected = np.array([15.0, 2.0])
    np.testing.assert_allclose(result, expected)


##########################################################################
# Tests of SlidingWindow Class
##########################################################################


def test_sliding_window_basics():
    X = np.arange(8, dtype=float).reshape(2, 4)
    labels = np.array([0, 1, 1, 2])
    expected_labels = {"majority": 1, "first": 0, "last": 2}
    for strategy, expected in expected_labels.items():
        windows, window_labels = SlidingWindow(
            length=4, step_size=4, label_strategy=strategy
        ).fit_transform(X, groups=labels)
        assert windows.shape == (1, 2, 4)
        np.testing.assert_array_equal(window_labels, [expected])

    X = np.arange(12, dtype=float).reshape(2, 6)
    groups = np.array(["a"] * 3 + ["b"] * 3)
    window = SlidingWindow(length=3, step_size=3)
    assert window.fit_transform(X).shape == (2, 2, 3)
    _, direct_groups = window.fit_transform(X, groups=groups)
    np.testing.assert_array_equal(direct_groups, ["a", "b"])

    _, tuple_groups = window.fit_transform((X, groups))
    np.testing.assert_array_equal(tuple_groups, ["a", "b"])
    alternate_groups = np.array(["b"] * 3 + ["a"] * 3)
    _, output_groups = window.fit_transform((X, groups), groups=alternate_groups)
    np.testing.assert_array_equal(output_groups, ["b", "a"])


def test_sliding_window_subject_collection(subject_collection):
    selected_runs = StateSelector([0, 1]).fit_transform(subject_collection)
    window = SlidingWindow(length=3, step_size=3)
    windows, (subject_ids, states) = window.fit_transform(selected_runs)
    direct_windows, direct_metadata = window.fit_transform(subject_collection)

    np.testing.assert_array_equal(
        windows[:, 0],
        [
            [0, 1, 2],
            [4, 5, 6],
            [8, 9, 10],
            [100, 101, 102],
            [103, 104, 105],
            [106, 107, 108],
            [109, 110, 111],
        ],
    )
    np.testing.assert_array_equal(subject_ids, [0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(states, [0, 1, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(direct_windows, windows)
    np.testing.assert_array_equal(direct_metadata, (subject_ids, states))

    short_then_valid = [
        [
            (np.ones((2, 2)), np.zeros(2, dtype=int)),
            (np.ones((2, 3)), np.ones(3, dtype=int)),
        ]
    ]
    valid_windows, (_, valid_states) = window.fit_transform(short_then_valid)
    assert valid_windows.shape == (1, 2, 3)
    np.testing.assert_array_equal(valid_states, [1])


def test_sliding_window_padding():
    X = np.arange(10, dtype=float).reshape(2, 5)
    labels = np.array([0, 0, 1, 1, 2])
    for policy, last_window in [("zero", [3, 4, 0, 0]), ("edge", [3, 4, 4, 4])]:
        windows, window_labels = SlidingWindow(
            length=4, step_size=3, padding_policy=policy
        ).fit_transform(X, groups=labels)
        np.testing.assert_array_equal(windows[-1, 0], last_window)
        np.testing.assert_array_equal(window_labels, [0, 2])

    short = np.arange(4, dtype=float).reshape(2, 2)
    assert SlidingWindow(4, padding_policy="zero").fit_transform(short).shape == (
        1,
        2,
        4,
    )
    complete = np.arange(12, dtype=float).reshape(2, 6)
    assert SlidingWindow(3, 3, padding_policy="edge").fit_transform(complete).shape == (
        2,
        2,
        3,
    )
    with pytest.raises(ValueError, match="shorter"):
        SlidingWindow(4).fit_transform(short)


def test_sliding_window_validation(recording, subject_collection):
    X, states = recording
    invalid_windows = [
        SlidingWindow(length=0),
        SlidingWindow(step_size=0),
        SlidingWindow(padding_policy="bad"),
        SlidingWindow(label_strategy="bad"),
    ]
    for window in invalid_windows:
        with pytest.raises(ValueError):
            window.fit(X)

    numeric_list = [[0.0, 1.0], [2.0, 3.0]]
    np.testing.assert_array_equal(
        StateSelector().fit_transform(numeric_list), numeric_list
    )


##########################################################################
# Tests for data.py
##########################################################################


@pytest.fixture
def mock_data_environment():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        labels_data = {
            "EEG_ID": ["sub_01", "sub_02", "sub_03", "sub_04"],
            "SCID5_CV_Depression": [1, 0, 0, 0],
            "SCID5_CV_OCD": [0, 0, 1, 0],
            "SCID5_CV_Tic_TrichoDerma_Hoarding": [0, 0, 0, 0],
            "SCID5_CV_SAD": [0, 0, 0, 0],
            "SCID5_CV_PHOB": [0, 0, 0, 0],
            "SCID5_CV_PANIC": [0, 0, 0, 0],
            "SCID5_CV_AGORA": [0, 0, 0, 0],
            "SCID5_CV_GAD": [0, 0, 0, 0],
            "SCID5_CV_PTSD": [0, 0, 0, 0],
            "SCID5_CV_Soma_Health": [0, 0, 0, 0],
            "SCID5_CV_Separation": [0, 0, 0, 0],
            "SCID5_CV_Sleep": [0, 0, 0, 0],
            "SCID5_CV_Bodydysmorphia": [0, 0, 0, 0],
            "SCID5_CV_Eating": [0, 0, 0, 0],
            "SCID5_CV_Anxiety_OCD_etc": [0, 0, 0, 0],
            "SCID5_CV_Eating_Bodydysmorphia": [0, 0, 0, 0],
            "SCID5_CV_ADHD_Explosive": [0, 0, 0, 0],
        }
        df = pd.DataFrame(labels_data)
        df.to_csv(tmp_path / "labels_reduced.csv", index=False)

        for sub_id in ["sub_01", "sub_02", "sub_03", "sub_04"]:
            mock_x = np.random.randn(
                100, 4
            )  # shape (n_samples, n_channels) -> بعداً T می‌شود
            mock_y = np.random.randint(0, 2, size=100)
            np.savez(tmp_path / f"{sub_id}.npz", X=mock_x, y=mock_y)

        yield tmp_path


def test_load_labels(mock_data_environment):
    csv_path = mock_data_environment / "labels_reduced.csv"
    df = load_labels(csv_path)
    assert df.index.name == "EEG_ID"
    assert "sub_01" in df.index


def test_list_subject_ids(mock_data_environment):
    subjects = list_subject_ids(mock_data_environment)
    assert subjects == ["sub_01", "sub_02", "sub_03", "sub_04"]


def test_binary_target(mock_data_environment):
    df = load_labels(mock_data_environment / "labels_reduced.csv")
    subject_ids = ["sub_01", "sub_02", "sub_03"]

    target_any = binary_target(df, subject_ids)
    np.testing.assert_array_equal(target_any, [1, 0, 1])

    target_dep = binary_target(df, subject_ids, diagnosis="SCID5_CV_Depression")
    np.testing.assert_array_equal(target_dep, [1, 0, 0])


def test_balanced_subject_ids_success(mock_data_environment):
    chosen = balanced_subject_ids(mock_data_environment, n_subjects=4)
    assert len(chosen) == 4
    assert chosen == ["sub_01", "sub_02", "sub_03", "sub_04"]

    chosen_random = balanced_subject_ids(
        mock_data_environment, n_subjects=2, random_state=42
    )
    assert len(chosen_random) == 2


def test_balanced_subject_ids_exceptions(mock_data_environment):
    with pytest.raises(ValueError, match="n_subjects must be positive"):
        balanced_subject_ids(mock_data_environment, n_subjects=0)

    with pytest.raises(ValueError, match="n_subjects must be even"):
        balanced_subject_ids(mock_data_environment, n_subjects=3)

    with pytest.raises(ValueError, match="need at least"):
        balanced_subject_ids(mock_data_environment, n_subjects=10)


def test_load_subject(mock_data_environment):
    rec, states = load_subject("sub_01", mock_data_environment)
    assert rec.shape == (4, 100)
    assert states.shape == (100,)


##########################################################################
# Tests for Private Metadata-Routing & Helpers (_helpers.py)
##########################################################################


def test_declares_param():
    def dummy_func(x, y, groups=None):
        pass

    assert _declares_param(dummy_func, "groups") is True
    assert _declares_param(dummy_func, "z") is False


def test_final_estimator_has():
    class DummyPipeline:
        def __init__(self, final):
            self._final_estimator = final

    pipe_none = DummyPipeline(None)
    check_func = _final_estimator_has("predict")
    with pytest.raises(AttributeError, match="The final step does not implement"):
        check_func(pipe_none)

    class DummyEstimator:
        pass

    pipe_no_method = DummyPipeline(DummyEstimator())
    with pytest.raises(AttributeError, match="does not implement predict"):
        check_func(pipe_no_method)

    class ValidEstimator:
        def predict(self):
            pass

    pipe_valid = DummyPipeline(ValidEstimator())
    assert check_func(pipe_valid) is True


def test_split_input():
    X = np.ones((5, 10))
    groups = np.array([1, 2, 3, 4, 5])

    res_x, res_groups = _split_input((X, groups), default_mask=None)
    np.testing.assert_array_equal(res_x, X)
    np.testing.assert_array_equal(res_groups, groups)

    res_x, res_groups = _split_input(X, default_mask=groups)
    np.testing.assert_array_equal(res_x, X)
    np.testing.assert_array_equal(res_groups, groups)


def test_metadata_kwargs():
    def method_with_y_and_groups(X, y, groups):
        pass

    def method_without_them(X):
        pass

    X = np.ones((5, 10))
    y = np.arange(5)
    groups = np.arange(5)

    kwargs = _metadata_kwargs(method_with_y_and_groups, X, y, groups)
    assert "y" in kwargs
    assert "groups" in kwargs

    kwargs_empty = _metadata_kwargs(method_without_them, X, y, groups)
    assert kwargs_empty == {}

    kwargs_mismatch = _metadata_kwargs(
        method_with_y_and_groups, X, np.arange(2), groups
    )
    assert "y" not in kwargs_mismatch


def test_transform_one():
    class CustomTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None, groups=None):
            if groups is not None:
                return X, groups, "extra_meta"
            return X

    transformer = CustomTransformer()
    X = np.ones((5, 10))
    groups = np.arange(5)

    res, out_groups = _transform_one(transformer, X, y=None, groups=groups)
    np.testing.assert_array_equal(res, X)
    np.testing.assert_array_equal(out_groups, groups)

    class SimpleTransformer(BaseEstimator, TransformerMixin):
        def transform(self, X):
            return X

    res_simple, out_groups_simple = _transform_one(
        SimpleTransformer(), X, y=None, groups=groups
    )
    np.testing.assert_array_equal(res_simple, X)
    np.testing.assert_array_equal(out_groups_simple, groups)


def test_subject_collection():
    recording = (np.ones((2, 10)), np.zeros(10))

    assert _subject_collection([recording]) == [recording]

    obj_arr = np.empty(1, dtype=object)
    obj_arr[0] = recording
    assert _subject_collection(obj_arr) == [recording]

    assert _subject_collection("not_a_collection") is None
    assert _subject_collection([123, "invalid_item"]) is None


def test_selected_runs():
    recording = np.arange(20).reshape(2, 10)
    sample_states = np.repeat([0, 1], 5)
    subject = (recording, sample_states)

    runs = _selected_runs(subject, states=None)
    assert len(runs) == 2
    np.testing.assert_array_equal(runs[0][0], recording[:, :5])
    np.testing.assert_array_equal(runs[1][0], recording[:, 5:])

    runs_filtered = _selected_runs(subject, states="eyes_closed")
    assert len(runs_filtered) == 1
    np.testing.assert_array_equal(runs_filtered[0][1], np.ones(5))


def test_map_recording_pairs():
    recording1 = (np.ones((2, 5)), np.zeros(5))
    recording2 = (np.ones((2, 5)), np.ones(5))
    collection = [recording1, [recording2]]

    def dummy_op(pair):
        rec, states = pair
        return rec + 1, states

    mapped = _map_recording_pairs(collection, dummy_op)
    assert len(mapped) == 2
    np.testing.assert_array_equal(mapped[0][0], np.ones((2, 5)) + 1)
    np.testing.assert_array_equal(mapped[1][0][0], np.ones((2, 5)) + 1)


def test_window_subjects_with_valid_padding_filter():
    class DummyWindow:
        def __init__(self):
            self.length = 10
            self.padding_policy = "valid"

        def transform(self, X, groups=None):
            return np.ones((1, 2, self.length)), np.array([99])

    window = DummyWindow()
    short_run = (np.ones((2, 5)), np.zeros(5))
    valid_run = (np.ones((2, 12)), np.zeros(12))

    collection = [short_run, valid_run]

    windows, (sub_ids, states) = _window_subjects(window, collection)

    assert len(windows) == 1
    np.testing.assert_array_equal(sub_ids, [1])
    np.testing.assert_array_equal(states, [99])
