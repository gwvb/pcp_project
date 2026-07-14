"""Tests for the EEG estimators."""

import numpy as np
import pytest
from sklearn.covariance import empirical_covariance, ledoit_wolf, oas

from pcp_project.estimators import (
    BandPassFilter,
    BatchCovariances,
    MeanProbabilityAggregator,
    NotchFilter,
    SlidingWindow,
    StateSelector,
    batch_empirical_covariance,
    batch_ledoit_wolf,
    batch_ledoit_wolf_shrinkage,
    batch_oas,
)

N_CHANNELS = 4
N_SAMPLES = 600


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


def test_state_selector(recording, subject_collection):
    X, states = recording
    selected, selected_states = StateSelector("eyes_closed").fit_transform(
        X, groups=states
    )
    assert selected.shape == (N_CHANNELS, N_SAMPLES // 2)
    np.testing.assert_array_equal(selected_states, np.ones(N_SAMPLES // 2))

    named_states = np.where(states == 0, "eyes_open", "eyes_closed")
    _, selected_names = StateSelector("eyes_closed").fit_transform(
        X, groups=named_states
    )
    assert set(selected_names) == {"eyes_closed"}

    np.testing.assert_array_equal(StateSelector().fit_transform(X), X)
    unchanged, unchanged_states = StateSelector().fit_transform((X, states))
    np.testing.assert_array_equal(unchanged, X)
    np.testing.assert_array_equal(unchanged_states, states)

    object_array = np.empty(2, dtype=object)
    object_array[:] = subject_collection
    runs = StateSelector([0]).fit_transform(object_array)
    assert [len(subject_runs) for subject_runs in runs] == [2, 1]
    np.testing.assert_array_equal(runs[0][0][0][0], [0, 1, 2, 3])
    np.testing.assert_array_equal(runs[0][1][0][0], [8, 9, 10, 11])

    all_runs = StateSelector([0, 1]).fit_transform(subject_collection)
    assert [int(run_states[0]) for _, run_states in all_runs[0]] == [0, 1, 0]

    one_sample, one_state = StateSelector([1]).fit_transform(
        np.ones((2, 1)),
        groups=np.array([1]),
    )
    assert one_sample.shape == (2, 1)
    np.testing.assert_array_equal(one_state, [1])

    selector = StateSelector([1]).fit(X)
    with pytest.raises(ValueError, match="groups are required"):
        selector.transform(X)

    alternate = 1 - states
    _, chosen = selector.transform((X, states), groups=alternate)
    np.testing.assert_array_equal(chosen, np.ones(N_SAMPLES // 2))


def test_bandpass_filter(recording, subject_collection, monkeypatch):
    monkeypatch.setattr(
        "pcp_project.estimators.signal.sosfiltfilt",
        lambda _, values, axis: values + 1.0,
    )
    X, states = recording
    X = X.copy()
    X[:, -2:] = np.nan
    filtered, output_states = BandPassFilter([[5, 10]]).fit_transform((X, states))
    np.testing.assert_allclose(filtered[:, :-2], X[:, :-2] + 1)
    assert np.isnan(filtered[:, -2:]).all()
    np.testing.assert_array_equal(output_states, states)

    alternate_states = 1 - states
    _, output_states = BandPassFilter([[5, 10]]).fit_transform(
        (X, states), groups=alternate_states
    )
    np.testing.assert_array_equal(output_states, alternate_states)

    all_nan = np.full_like(X, np.nan)
    np.testing.assert_equal(BandPassFilter([[5, 10]]).fit_transform(all_nan), all_nan)

    bandpass = BandPassFilter([[5, 10]])
    filtered_subjects = bandpass.fit_transform(subject_collection)
    np.testing.assert_allclose(filtered_subjects[0][0], subject_collection[0][0] + 1)

    selected_runs = StateSelector([0]).fit_transform(subject_collection)
    filtered_runs = bandpass.fit_transform(selected_runs)
    assert [len(runs) for runs in filtered_runs] == [2, 1]
    np.testing.assert_allclose(filtered_runs[0][0][0], selected_runs[0][0][0] + 1)


def test_notch_filter(recording, monkeypatch):
    X, states = recording
    X = X.copy()
    X[:, -2:] = np.nan
    calls = []

    def fake_notch(values, **kwargs):
        calls.append(kwargs)
        return values + 1

    monkeypatch.setattr("pcp_project.estimators.mne.filter.notch_filter", fake_notch)
    filtered, output_states = NotchFilter(
        freqs=[50, 200], notch_widths=[2, 8]
    ).fit_transform(X, groups=states)
    np.testing.assert_allclose(filtered[:, :-2], X[:, :-2] + 1)
    assert np.isnan(filtered[:, -2:]).all()
    np.testing.assert_array_equal(output_states, states)
    np.testing.assert_array_equal(calls[0]["freqs"], [50])
    np.testing.assert_array_equal(calls[0]["notch_widths"], [2])

    _, tuple_states = NotchFilter().fit_transform((X, states))
    np.testing.assert_array_equal(tuple_states, states)
    alternate_states = 1 - states
    _, output_states = NotchFilter().fit_transform((X, states), groups=alternate_states)
    np.testing.assert_array_equal(output_states, alternate_states)

    NotchFilter(freqs=[50, 100], notch_widths=2).fit_transform(X)
    np.testing.assert_array_equal(calls[-1]["notch_widths"], [2])

    np.testing.assert_equal(NotchFilter(sfreq=32).fit_transform(X), X)
    all_nan = np.full_like(X, np.nan)
    np.testing.assert_equal(NotchFilter().fit_transform(all_nan), all_nan)


def test_covariance_helpers_match_sklearn(rng):
    for n_channels in [1, N_CHANNELS]:
        X = rng.normal(loc=2.0, size=(3, n_channels, 100))
        original = X.copy()

        for assume_centered in [False, True]:
            empirical = batch_empirical_covariance(X, assume_centered=assume_centered)
            oas_covariance, oas_shrinkage = batch_oas(
                X, assume_centered=assume_centered
            )
            lw_covariance, lw_shrinkage = batch_ledoit_wolf(
                X, assume_centered=assume_centered, block_size=1
            )
            expected_empirical = np.stack(
                [
                    empirical_covariance(window.T, assume_centered=assume_centered)
                    for window in X
                ]
            )
            expected_oas = [
                oas(window.T, assume_centered=assume_centered) for window in X
            ]
            expected_lw = [
                ledoit_wolf(window.T, assume_centered=assume_centered) for window in X
            ]

            np.testing.assert_allclose(empirical, expected_empirical)
            np.testing.assert_allclose(
                oas_covariance, [item[0] for item in expected_oas]
            )
            np.testing.assert_allclose(
                oas_shrinkage, [item[1] for item in expected_oas]
            )
            np.testing.assert_allclose(lw_covariance, [item[0] for item in expected_lw])
            np.testing.assert_allclose(lw_shrinkage, [item[1] for item in expected_lw])

        np.testing.assert_array_equal(X, original)


def test_batch_covariance_estimators_and_aliases(rng, monkeypatch):
    X = rng.normal(size=(3, N_CHANNELS, 20))
    references = {
        "scm": lambda values: empirical_covariance(values.T),
        "oas": lambda values: oas(values.T)[0],
        "lwf": lambda values: ledoit_wolf(values.T)[0],
    }
    for name, reference in references.items():
        result = BatchCovariances(name, block_size=1).fit_transform(
            (X, np.arange(len(X)))
        )
        np.testing.assert_allclose(result, [reference(window) for window in X])

    aliases = {"empirical": "scm", "lw": "lwf", "ledoit_wolf": "lwf"}
    for alias, canonical in aliases.items():
        expected = BatchCovariances(canonical).fit_transform(X)
        np.testing.assert_allclose(BatchCovariances(alias).fit_transform(X), expected)

    centered = X - X.mean(axis=2, keepdims=True)
    np.testing.assert_allclose(
        batch_ledoit_wolf_shrinkage(centered, block_size=1),
        batch_ledoit_wolf_shrinkage(centered, block_size=1000),
    )

    with pytest.warns(UserWarning, match="Only one sample"):
        covariance, shrinkage = batch_ledoit_wolf(np.ones((1, 2, 1)))
    assert np.isfinite(covariance).all()
    assert np.isfinite(shrinkage).all()

    calls = []

    def record_output(self, **kwargs):
        calls.append(kwargs)
        return self

    monkeypatch.setattr(BatchCovariances, "_set_output", record_output, raising=False)
    BatchCovariances()

    assert calls == [{"transform": "bypass"}]


def test_mean_probability_aggregator():
    values = np.arange(16, dtype=float).reshape(4, 2, 2)
    groups = np.array(["s2", "s1", "s2", "s1"])
    aggregator = MeanProbabilityAggregator()
    result = aggregator.fit_transform((values, groups))
    expected = np.stack([values[[0, 2]].mean(axis=0), values[[1, 3]].mean(axis=0)])
    np.testing.assert_allclose(result, expected)

    explicit_groups = np.array(["s2", "s2", "s1", "s1"])
    result = aggregator.transform((values, groups), groups=explicit_groups)
    np.testing.assert_allclose(
        result, [values[:2].mean(axis=0), values[2:].mean(axis=0)]
    )

    with pytest.raises(ValueError, match="groups must be provided"):
        aggregator.transform(values)


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
