"""EEG preprocessing and covariance estimators used by the project."""

import warnings

import mne
import numpy as np
import pyriemann
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _helpers

# FIX(ref): Resolve advertised state names locally; the reference compared names
# directly with numeric sample codes and could select nothing.
# FIX(ref): Collection helpers preserve variable-length subjects, contiguous
# runs, window subject/state metadata, and first-seen aggregation order.
from ._helpers import (
    STATE_NAME_TO_CODE,
    _map_recording_pairs,
    _selected_runs,
    _subject_collection,
    _window_subjects,
)

_is_recording_pair = _helpers._is_recording_pair
_is_run_list = _helpers._is_run_list
_recording_pair = _helpers._recording_pair
_state_values = _helpers._state_values


class StateSelector(BaseEstimator):
    """Keep selected recording states without joining separate state runs."""

    def __init__(self, states=None):
        self.states = states

    def fit(self, X, y=None):
        """Validate no data-dependent parameters and mark the selector fitted."""
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Select samples or contiguous runs matching ``states``."""
        check_is_fitted(self, "fitted_")

        # FIX(ref): Select collections run by run so disjoint states and
        # variable-length subject boundaries remain intact.
        collection = _subject_collection(X)
        if collection is not None:
            return [_selected_runs(subject, self.states) for subject in collection]

        if isinstance(X, tuple):
            if len(X) > 1 and groups is None:
                groups = X[1]
            X = X[0]

        X_copied = np.asarray(X, dtype=np.float64)

        # FIX(ref): Require local metadata for requested states, preserve a
        # one-sample vector, and accept numeric codes or their advertised names.
        if groups is None:
            if self.states is not None:
                raise ValueError("groups are required when states are selected")
            return X_copied
        eye_states = np.asarray(groups)
        if self.states is None:
            return X_copied, eye_states

        raw_states = np.atleast_1d(self.states)
        states = [
            STATE_NAME_TO_CODE[state] if isinstance(state, str) else int(state)
            for state in raw_states
        ]
        mask = np.isin(eye_states, states) | np.isin(eye_states, raw_states)

        X_selected = X_copied[:, mask]
        groups_selected = eye_states[mask]

        return X_selected, groups_selected

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        """Fit the selector and transform while forwarding state metadata."""
        self.fit(X, y)
        return self.transform(X, y, groups=groups)


# FIX(ref): Support direct array-like recordings and subject/run collections
# while preserving bare-array versus metadata-pair returns, including all-NaN.
class BandPassFilter(BaseEstimator, TransformerMixin):
    """Apply Butterworth band-pass filters to EEG recordings."""

    def __init__(self, frequency_bands, sfreq=256.0):
        self.frequency_bands = frequency_bands
        self.sfreq = sfreq

    def fit(self, X, y=None, groups=None):
        """Mark the stateless filter as fitted."""
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Filter one recording or a collection while preserving metadata."""
        check_is_fitted(self, "fitted_")

        collection = _subject_collection(X)
        if collection is not None:
            return _map_recording_pairs(collection, self.transform)

        if isinstance(X, tuple):
            if len(X) > 1 and groups is None:
                groups = X[1]
            X = X[0]

        X = np.asarray(X, dtype=np.float64)

        mask = ~np.isnan(X).any(axis=0)
        X_valid = X[:, mask]

        if X_valid.shape[1] == 0:
            return X.copy() if groups is None else (X.copy(), groups)

        filters = np.array(
            [
                signal.butter(
                    N=5,
                    Wn=[low, high],
                    btype="bandpass",
                    fs=self.sfreq,
                    output="sos",
                )
                for low, high in self.frequency_bands
            ]
        )

        filtered_valid = np.stack(
            [signal.sosfiltfilt(f, X_valid, axis=-1) for f in filters]
        )
        summed_valid = filtered_valid.sum(axis=0)

        X_filtered = np.full_like(X, fill_value=np.nan)
        X_filtered[:, mask] = summed_valid

        return X_filtered if groups is None else (X_filtered, groups)

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        """Fit and filter while forwarding recording metadata."""
        self.fit(X, y, groups=groups)
        return self.transform(X, y, groups=groups)


# FIX(ref): Keep usable notch frequencies and matching widths, skip empty MNE
# calls, and preserve bare-array versus metadata-pair returns.
class NotchFilter(BaseEstimator, TransformerMixin):
    """Remove line-noise frequencies with MNE's notch filter."""

    def __init__(self, freqs=50.0, sfreq=256.0, notch_widths=None, n_jobs=None):
        self.freqs = freqs
        self.sfreq = sfreq
        self.notch_widths = notch_widths
        self.n_jobs = n_jobs

    def fit(self, X, y=None, groups=None):
        """Mark the stateless filter as fitted."""
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Filter one recording and preserve optional sample metadata."""
        check_is_fitted(self, "fitted_")

        if isinstance(X, tuple):
            if len(X) > 1 and groups is None:
                groups = X[1]
            X = X[0]

        X_copied = np.asarray(X, dtype=np.float64)

        mask = ~np.isnan(X_copied).any(axis=0)
        X_valid = X_copied[:, mask]

        freqs_array = np.asarray(np.atleast_1d(self.freqs), dtype=float)
        usable = freqs_array < self.sfreq / 2
        freqs_array = freqs_array[usable]
        notch_widths = self.notch_widths
        if notch_widths is not None:
            notch_widths = np.asarray(np.atleast_1d(notch_widths), dtype=float)
            if len(notch_widths) == len(usable):
                notch_widths = notch_widths[usable]
        if X_valid.shape[1] == 0 or len(freqs_array) == 0:
            return X_copied.copy() if groups is None else (X_copied.copy(), groups)

        X_filtered_valid = mne.filter.notch_filter(
            X_valid,
            Fs=self.sfreq,
            freqs=freqs_array,
            notch_widths=notch_widths,
            n_jobs=self.n_jobs,
            method="fir",
            phase="zero",
            verbose=False,
        )

        X_filtered = np.full_like(X_copied, fill_value=np.nan)
        X_filtered[:, mask] = X_filtered_valid

        return X_filtered if groups is None else (X_filtered, groups)

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        """Fit and filter while forwarding recording metadata."""
        self.fit(X, y, groups=groups)
        return self.transform(X, y, groups=groups)


# FIX(ref): Normalize SCM/OAS/LWF aliases and dispatch the requested estimator
# instead of always running LWF or silently replacing NaNs with zeros.
class BatchCovariances(pyriemann.estimation.Covariances):
    """Estimate a covariance matrix for every EEG window."""

    def __init__(self, estimator="scm", assume_centered=False, block_size=1000):
        super().__init__(estimator=estimator)
        self.assume_centered = assume_centered
        self.block_size = block_size

        if hasattr(self, "_set_output"):
            self._set_output(transform="bypass")

    def fit(self, X, y=None):
        """Remember the covariance estimator name and mark the estimator fitted."""
        self.estimator_ = _covariance_name(self.estimator)
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Transform EEG windows into covariance matrices."""
        check_is_fitted(self, "fitted_")

        if isinstance(X, tuple):
            X_data = X[0]
        else:
            X_data = X

        X_copied = np.asarray(X_data, dtype=float)
        if self.estimator_ == "scm":
            return batch_empirical_covariance(
                X_copied, assume_centered=self.assume_centered
            )
        if self.estimator_ == "oas":
            return batch_oas(X_copied, assume_centered=self.assume_centered)[0]
        covmats, _ = batch_ledoit_wolf(
            X_copied,
            assume_centered=self.assume_centered,
            block_size=self.block_size,
        )
        return covmats


# FIX(ref): Implement centered SCM, true OAS, and chunked LWF with per-matrix
# traces, nonmutating inputs, guarded shrinkage, and valid one-feature cases.
def batch_empirical_covariance(X, *, assume_centered=False):
    """Compute empirical covariance matrices for a batch of windows."""
    X = np.asarray(X, dtype=float)
    if not assume_centered:
        X = X - X.mean(axis=2, keepdims=True)
    return X @ X.transpose(0, 2, 1) / X.shape[2]


def batch_oas(X, *, assume_centered=False):
    """Compute OAS covariances and shrinkage values for each window."""
    X = np.asarray(X, dtype=float)
    if not assume_centered:
        X = X - X.mean(axis=2, keepdims=True)
    n_features, n_samples = X.shape[1:]
    empirical = batch_empirical_covariance(X, assume_centered=True)
    if n_features == 1:
        return empirical, np.zeros(len(X), dtype=float)
    alpha = np.mean(empirical**2, axis=(1, 2))
    mu = np.trace(empirical, axis1=1, axis2=2) / n_features
    numerator = alpha + mu**2
    denominator = (n_samples + 1.0) * (alpha - mu**2 / n_features)
    shrinkage = np.ones(len(X), dtype=float)
    np.divide(numerator, denominator, out=shrinkage, where=denominator != 0)
    shrinkage = np.clip(shrinkage, 0.0, 1.0)
    covariance = (1.0 - shrinkage)[:, None, None] * empirical
    diagonal = np.arange(n_features)
    covariance[:, diagonal, diagonal] += (shrinkage * mu)[:, None]
    return covariance, shrinkage


def batch_ledoit_wolf_shrinkage(X, block_size=1000):
    """Compute one Ledoit-Wolf shrinkage value per window."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] == 1:
        return np.zeros(len(X), dtype=float)
    shrinkages = []
    for start in range(0, len(X), block_size):
        block = X[start : start + block_size].astype(float, copy=True)
        n_matrices, n_features, n_samples = block.shape
        block = block.transpose(0, 2, 1)

        X2 = block**2
        emp_cov_trace = X2.sum(axis=1) / n_samples
        mu = emp_cov_trace.sum(axis=1) / n_features

        Xt = block.transpose(0, 2, 1)
        XtX2 = X2.transpose(0, 2, 1) @ X2
        beta_ = XtX2.sum(axis=(1, 2))

        XtX = Xt @ block
        delta_ = (XtX**2).sum(axis=(1, 2)) / n_samples**2

        beta = (beta_ / n_samples - delta_) / (n_features * n_samples)
        delta = (
            delta_ - 2.0 * mu * emp_cov_trace.sum(axis=1) + n_features * mu**2
        ) / n_features
        beta = np.minimum(beta, delta)
        shrinkage = np.zeros(n_matrices, dtype=float)
        np.divide(beta, delta, out=shrinkage, where=delta > 0)
        shrinkages.append(np.clip(shrinkage, 0.0, 1.0))
    return np.concatenate(shrinkages)


def batch_ledoit_wolf(X, *, assume_centered=False, block_size=1000):
    """Compute Ledoit-Wolf covariances and shrinkage values for each window."""
    X = np.array(X, dtype=float, copy=True)

    if X.shape[2] == 1:
        warnings.warn("Only one sample available.", stacklevel=2)

    if not assume_centered:
        X -= np.mean(X, axis=2, keepdims=True)

    n_features = X.shape[1]
    shrinkages = batch_ledoit_wolf_shrinkage(X, block_size=block_size)
    emp_cov = batch_empirical_covariance(X, assume_centered=True)
    mu = np.trace(emp_cov, axis1=1, axis2=2) / n_features

    shrunk_cov = (1.0 - shrinkages)[:, None, None] * emp_cov
    i = np.arange(n_features)
    shrunk_cov[:, i, i] += (shrinkages * mu)[:, None]
    return shrunk_cov, shrinkages


# FIX(ref): Unpack tuple metadata before conversion, preserve trailing axes and
# first-seen subject order, and forward groups through fit_transform.
class MeanProbabilityAggregator(BaseEstimator, TransformerMixin):
    """Average aligned window-level values within each subject."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Mark the stateless aggregator as fitted."""
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Return one mean probability array per subject in first-seen order."""
        check_is_fitted(self, "fitted_")

        if isinstance(X, tuple):
            if len(X) > 1 and groups is None:
                groups = X[1]
            X = X[0]
        X = np.asarray(X, dtype=float)

        if groups is None:
            raise ValueError("groups must be provided to aggregate per subject")

        groups = np.asarray(groups)
        _, first_indices, inverse = np.unique(
            groups, return_index=True, return_inverse=True
        )
        ordered_groups = np.argsort(first_indices)
        aggregated = np.array(
            [X[inverse == group].mean(axis=0) for group in ordered_groups]
        )
        return aggregated

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        """Fit and aggregate while forwarding subject metadata."""
        self.fit(X, y)
        return self.transform(X, y, groups=groups)


# FIX(ref): Window collections one contiguous run at a time, align zero/edge
# padding with labels, and return a bare window array when metadata is absent.
class SlidingWindow(BaseEstimator, TransformerMixin):
    """Split recordings into fixed-length windows."""

    def __init__(
        self,
        length=200,
        step_size=50,
        padding_policy="valid",
        label_strategy="majority",
    ):
        self.length = length
        self.step_size = step_size
        self.padding_policy = padding_policy
        self.label_strategy = label_strategy

    def fit(self, X, y=None):
        """Validate window, padding, and label-strategy parameters."""
        if self.length <= 0 or self.step_size <= 0:
            raise ValueError("Length and step_size must be positive integers.")
        if self.padding_policy not in ["valid", "zero", "edge"]:
            raise ValueError(f"Unknown padding_policy: {self.padding_policy}")
        if self.label_strategy not in ["majority", "last", "first"]:
            raise ValueError(f"Unknown label_strategy: {self.label_strategy}")

        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Create windows and optional window-level metadata labels."""
        check_is_fitted(self, "fitted_")

        collection = _subject_collection(X)
        if collection is not None:
            return _window_subjects(self, collection)

        if isinstance(X, tuple):
            if len(X) > 1 and groups is None:
                groups = X[1]
            X = X[0]

        n_channels, n_samples = X.shape
        groups_arr = None if groups is None else np.asarray(groups)

        if n_samples < self.length and self.padding_policy == "valid":
            raise ValueError(
                f"Data length ({n_samples}) is shorter than window length "
                f"({self.length})."
            )

        remainder = (n_samples - self.length) % self.step_size

        if (
            n_samples < self.length or remainder != 0
        ) and self.padding_policy != "valid":
            pad_size = (
                self.length - n_samples
                if n_samples < self.length
                else self.step_size - remainder
            )
            if self.padding_policy == "zero":
                X = np.pad(
                    X,
                    ((0, 0), (0, pad_size)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                X = np.pad(X, ((0, 0), (0, pad_size)), mode="edge")
            if groups_arr is not None:
                groups_arr = np.pad(groups_arr, (0, pad_size), mode="edge")
            n_samples = X.shape[1]

        start_idx = np.arange(0, n_samples - self.length + 1, self.step_size)
        indexer = start_idx[:, None] + np.arange(self.length)
        X_windows = X[:, indexer].transpose(1, 0, 2)

        if groups_arr is None:
            return X_windows

        n_windows = len(start_idx)
        groups_windows = np.empty(n_windows, dtype=groups_arr.dtype)
        for i, start in enumerate(start_idx):
            w_groups = groups_arr[start : start + self.length]
            if self.label_strategy == "majority":
                vals, counts = np.unique(w_groups, return_counts=True)
                groups_windows[i] = vals[np.argmax(counts)]
            elif self.label_strategy == "last":
                groups_windows[i] = w_groups[-1]
            else:
                groups_windows[i] = w_groups[0]
        return X_windows, groups_windows

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        """Fit and create windows while forwarding sample metadata."""
        self.fit(X, y)
        return self.transform(X, y, groups=groups)


# FIX(ref): Central alias normalization keeps every advertised covariance name
# on the corrected SCM/OAS/LWF dispatch path.
def _covariance_name(estimator):
    aliases = {
        "scm": "scm",
        "empirical": "scm",
        "oas": "oas",
        "lw": "lwf",
        "lwf": "lwf",
        "ledoit_wolf": "lwf",
    }
    return aliases[estimator.lower()]
