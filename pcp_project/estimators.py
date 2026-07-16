"""EEG preprocessing and covariance estimators used by the project."""

import warnings

import mne
import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _helpers
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


###########################################################################################
# BandPassFilter
###########################################################################################
class BandPassFilter(BaseEstimator, TransformerMixin):
    """Apply Butterworth band-pass filters to EEG recordings.

    This transformer keeps only selected frequency bands from an EEG
    recording. A single recording is expected to have shape
    ``(n_channels, n_samples)``. The output has the same shape as the input.

    Parameters
    ----------
    frequency_bands : list of list of float
        Frequency ranges to keep, in Hz. For example, ``[[5, 10], [13, 35]]``
        keeps activity between 5-10 Hz and 13-35 Hz.
    sfreq : float, default=256.0
        Sampling frequency of the EEG recording in Hz.

    Notes
    -----
    The filter is stateless. It does not learn data-dependent parameters in
    :meth:`fit`; the filtering is performed in :meth:`transform`.

    Examples
    --------
    >>> import numpy as np
    >>> from pcp_project.estimators import BandPassFilter
    >>> X = np.random.randn(61, 1000)
    >>> filt = BandPassFilter(frequency_bands=[[5, 10]], sfreq=256.0)
    >>> X_filtered = filt.fit_transform(X)
    >>> X_filtered.shape
    (61, 1000)
    """

    def __init__(self, frequency_bands, sfreq=256.0):
        self.frequency_bands = frequency_bands
        self.sfreq = sfreq

    def fit(self, X, y=None, groups=None):
        """Mark the filter as fitted.

        The band-pass filter is stateless and does not learn parameters from
        the data. This method exists for scikit-learn compatibility.

        Parameters
        ----------
        X : array-like
            EEG recording. Ignored during fitting.
        y : None, default=None
            Ignored. Present for scikit-learn compatibility.
        groups : array-like, default=None
            Optional metadata. Ignored during fitting.

        Returns
        -------
        self : BandPassFilter
            The fitted transformer.
        """
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Apply the band-pass filter to EEG data.

        The input can be a single EEG recording, a collection of subject
        recordings, or a tuple ``(X, groups)``. Metadata in ``groups`` is preserved
        and returned unchanged.

        Parameters
        ----------
        X : array-like, tuple, or collection
            EEG data. A single recording should have shape
            ``(n_channels, n_samples)``. A tuple is interpreted as
            ``(recording, groups)``.
        y : None, default=None
            Ignored. Present for scikit-learn compatibility.
        groups : array-like, default=None
            Optional metadata such as sample states.

        Returns
        -------
        ndarray or tuple or collection
            Filtered EEG data with the same structure as the input. If metadata is
            provided, the output is returned as ``(X_filtered, groups)``.
        """
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


###########################################################################################
# NotchFilter
###########################################################################################
class NotchFilter(BaseEstimator, TransformerMixin):
    """Remove line-noise frequencies with MNE's notch filter.

    This filter applies a notch filter to eliminate power line noise or specific
    interference frequencies (e.g., 50 Hz or 60 Hz). It handles missing values
    (NaNs), filters out invalid frequencies above the Nyquist limit, and preserves
    sample-level metadata if provided.

    Parameters
    ----------
    freqs : float or list of float, default=50.0
        Frequencies to attenuate.
    sfreq : float, default=256.0
        Sampling frequency of the EEG signal in Hz.
    notch_widths : float or array-like, optional
        Width of each notch. If None, MNE's default width is used.
    n_jobs : int, optional
        Number of jobs to run in parallel.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a dummy EEG signal with 2 channels and 2000 samples
    >>> # Channel 0 has a simulated 50 Hz sine wave
    >>> sfreq = 256.0
    >>> n_samples = 2000
    >>> t = np.arange(n_samples) / sfreq
    >>> signal_50hz = np.sin(2 * np.pi * 50 * t)
    >>> X = np.vstack([signal_50hz, np.zeros(n_samples)])
    >>> groups = np.zeros(n_samples, dtype=int)
    >>> groups[n_samples // 2:] = 1  # Half eye-closed, half eye-open
    >>>
    >>> # Initialize and run the NotchFilter
    >>> filter_50 = NotchFilter(freqs=50.0, sfreq=sfreq)
    >>> X_filtered, out_groups = filter_50.fit_transform(X, groups=groups)
    >>>
    >>> # The shape of the output signal remains unchanged
    >>> X_filtered.shape
    (2, 2000)
    >>>
    >>> # Metadata groups are successfully forwarded
    >>> np.array_equal(out_groups, groups)
    True
    """

    def __init__(self, freqs=50.0, sfreq=256.0, notch_widths=None, n_jobs=None):
        self.freqs = freqs
        self.sfreq = sfreq
        self.notch_widths = notch_widths
        self.n_jobs = n_jobs

    def fit(self, X, y=None, groups=None):
        """Mark the filter as fitted.

        The notch filter is stateless and does not learn parameters from
        the data. This method exists for scikit-learn compatibility.

        Parameters
        ----------
        X : array-like
            EEG recording. Ignored during fitting.
        y : None, default=None
            Ignored. Present for scikit-learn compatibility.
        groups : array-like, default=None
            Optional metadata. Ignored during fitting.

        Returns
        -------
        self : NotchFilter
            The fitted transformer.
        """
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Apply the notch filter to EEG data.

        The input can be a single EEG recording, a collection of subject
        recordings, or a tuple ``(X, groups)``. Metadata in ``groups`` is preserved
        and returned unchanged. NaNs (missing values) in the signal are handled
        gracefully and restored in the final filtered output.

        Parameters
        ----------
        X : array-like, tuple, or collection
            EEG data. A single recording should have shape
            ``(n_channels, n_samples)``. A tuple is interpreted as
            ``(recording, groups)``.
        y : None, default=None
            Ignored. Present for scikit-learn compatibility.
        groups : array-like, default=None
            Optional metadata such as sample states.

        Returns
        -------
        ndarray or tuple or collection
            Filtered EEG data with the same structure as the input. If metadata is
            provided, the output is returned as ``(X_filtered, groups)``.
        """
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


###########################################################################################
# StateSelector
###########################################################################################


class StateSelector(BaseEstimator):
    """Select specific recording states from EEG signals.

    This estimator filters EEG timepoints or contiguous runs based on the
    requested states. It supports both single-subject arrays and multi-subject
    collections (lists of runs) while preserving recording boundaries.

    Parameters
    ----------
    states : list, str, int or None, default=None
        The recording states to retain. Can be numeric codes (e.g., 1) or
        their corresponding string names (e.g., 'eyes_closed').
        If None, all states are kept.

    Attributes
    ----------
    fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a dummy EEG signal with 2 channels and 5 samples
    >>> X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
    ...               [6.0, 7.0, 8.0, 9.0, 10.0]])
    >>> # Eye states for these 5 samples (0: open, 1: closed)
    >>> groups = np.array([0, 0, 1, 1, 0])
    >>> # Select times when eyes are closed (code 1)
    >>> selector = StateSelector(states=[1])
    >>> selector.fit(X)
    StateSelector(states=[1])
    >>> X_sel, g_sel = selector.transform(X, groups=groups)
    >>> print(X_sel)
    [[3. 4.]
     [8. 9.]]
    >>> print(g_sel)
    [1 1]
    """

    def __init__(self, states=None):
        self.states = states

    def fit(self, X, y=None):
        """Mark the state selector as fitted.

        This estimator is stateless and does not learn any parameters from the
        training data during fitting. This method is provided to comply with the
        scikit-learn transformer API.

        Parameters
        ----------
        X : array-like or collection
            EEG data. Ignored during fitting.
        y : None, default=None
            Ignored. Present for scikit-learn compatibility.

        Returns
        -------
        self : StateSelector
            The fitted transformer.
        """
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Select times or runs matching the target states.

        This method extracts only those EEG timepoints where the recording state
        matches the configured interest. If a collection of multi-subject runs
        is passed, each run is processed individually. If a tuple ``(X, groups)``
        is provided, metadata is unpacked and sliced synchronously with the data.

        Parameters
        ----------
        X : array-like, tuple, or collection
            EEG data. A single recording should have shape
            ``(n_channels, n_samples)``. A tuple is interpreted as
            ``(recording, groups)``.
        y : None, default=None
            Ignored. Present for scikit-learn compatibility.
        groups : array-like, default=None
            State metadata (e.g., eye-state codes or strings) representing
            the recording state of each time sample. Required if target states
            are selected.

        Returns
        -------
        ndarray or tuple or list
            Selected EEG samples. If input is a collection, returns a list of
            filtered runs.
            If metadata is provided, returns ``(X_selected, groups_selected)``.

        Raises
        ------
        ValueError
            If specific target states are requested but `groups` metadata is not
            provided.
        """
        check_is_fitted(self, "fitted_")

        collection = _subject_collection(X)
        if collection is not None:
            return [_selected_runs(subject, self.states) for subject in collection]

        if isinstance(X, tuple):
            if len(X) > 1 and groups is None:
                groups = X[1]
            X = X[0]

        X_copied = np.asarray(X, dtype=np.float64)

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


###########################################################################################
# SlidingWindow
###########################################################################################
class SlidingWindow(BaseEstimator, TransformerMixin):
    """Split recordings into fixed-length windows.

    This transformer slices 2D EEG recordings of shape (n_channels, n_samples)
    into 3D window arrays of shape (n_windows, n_channels, window_length).

    Parameters
    ----------
    length : int, default=200
        The length of each window in data samples.
    step_size : int, default=50
        The sliding window step size in samples.
    padding_policy : {"valid", "zero", "edge"}, default="valid"
        The policy for padding incomplete trailing segments.
            "valid": Discard trailing samples that do not form a complete window.
            "zero": Pad with zeros up to the next valid window boundary.
            "edge": Pads with the edge value of array.
    label_strategy : {"majority", "last", "first"}, default="majority"
        The method used to merge sample states into one window state.
            "majority": Assign the label that occurs most frequently in the window.
            "last": Assign the label of the final sample in the window.
            "first": Assign the label of the starting sample in the window.

    Attributes
    ----------
    fitted_ : bool
        True after fit() has been called.
    """

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
        """Validate window, padding, and label-strategy parameters.

        This estimator is stateless and does not learn any parameters from the
        training data during fitting. This method is provided to comply with the
        scikit-learn.

        Parameters
        ----------
        X : array-like or collection
            EEG recording or subject collection. Ignored during fitting.
        y : None, default=None
            Ignored. Present for scikit-learn compatibility.

        Returns
        -------
        self : SlidingWindow
            The fitted transformer instance.

        Raises
        ------
        ValueError
            If ``length`` or ``step_size`` is non-positive, or if ``padding_policy``
            or ``label_strategy`` is set to an unsupported value.
        """
        if self.length <= 0 or self.step_size <= 0:
            raise ValueError("Length and step_size must be positive integers.")
        if self.padding_policy not in ["valid", "zero", "edge"]:
            raise ValueError(f"Unknown padding_policy: {self.padding_policy}")
        if self.label_strategy not in ["majority", "last", "first"]:
            raise ValueError(f"Unknown label_strategy: {self.label_strategy}")

        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Create windows and optional window-level metadata labels.

        Parameters
        ----------
        X : array-like, tuple, or collection
            EEG data. A single recording should have shape
            (n_channels, n_samples). A tuple is interpreted as
            (recording, groups).
        y : None, default=None
            Ignored. Present for scikit-learn compatibility.
        groups : array-like, default=None
            State metadata (e.g., eye-state codes or strings) representing
            the recording state of each time sample.

        Returns
        -------
        ndarray, tuple, or list
            Returns a 3D array of windows when given a single recording without
            metadata. When ``groups`` are provided, returns
            ``(X_windows, groups_windows)``. When the input is a collection of
            recordings, returns the corresponding collection of windowed outputs.

        Raises
        ------
        ValueError
            If the signal length is shorter than ``length`` and ``padding_policy``
            is set to ``"valid"``.
        """
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


###########################################################################################
# BatchCovariances
###########################################################################################


def batch_empirical_covariance(X, assume_centered):
    """Compute the empirical covariance of several matrices.

    Parameters
    ----------
    X : ndarray of shape (n_matrices, n_features, n_samples)
        Data from which to compute the batched covariance estimate.

    Returns
    -------
    covariance : ndarray of shape (n_matrices, n_features, n_features)
    """
    if not assume_centered:
        X = X - X.mean(axis=2, keepdims=True)
    return X @ X.transpose(0, 2, 1) / X.shape[2]


def batch_ledoit_wolf_shrinkage(X):
    """Estimate the Ledoit Wolf shrinkage parameter for several matrices.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_features, n_samples)

    Returns
    -------
    shrinkage : ndarray, shape (n_matrices,)
    """
    n_matrices, n_features, n_samples = X.shape
    X = X.astype(float, copy=True).transpose(0, 2, 1)

    X2 = X**2
    emp_cov_trace = X2.sum(axis=1) / n_samples
    mu = emp_cov_trace.sum(axis=1) / n_features

    Xt = X.transpose(0, 2, 1)
    XtX2 = X2.transpose(0, 2, 1) @ X2
    beta_ = XtX2.sum(axis=(1, 2))

    XtX = Xt @ X
    delta_ = (XtX**2).sum(axis=(1, 2)) / n_samples**2

    beta = (beta_ / n_samples - delta_) / (n_features * n_samples)
    delta = (
        delta_ - 2.0 * mu * emp_cov_trace.sum(axis=1) + n_features * mu**2
    ) / n_features
    beta = np.minimum(beta, delta)

    shrinkage = np.zeros(n_matrices, dtype=float)
    np.divide(beta, delta, out=shrinkage, where=delta > 0)
    shrinkage = np.clip(shrinkage, 0.0, 1.0)
    return shrinkage


def batch_ledoit_wolf(X, *, assume_centered):
    """Estimate batch Ledoit-Wolf covariance matrices.

    Parameters
    ----------
    X : ndarray of shape (n_matrices, n_features, n_samples)
        Input data.
    assume_centered : bool, default=False
        If False, center each signal before estimating the covariance.

    Returns
    -------
    covariance : ndarray of shape (n_matrices, n_features, n_features)
        Estimated covariance matrices.
    shrinkage : ndarray of shape (n_matrices,)
        Ledoit-Wolf shrinkage coefficients.
    """
    if not assume_centered:
        X -= np.mean(X, axis=2, keepdims=True)

    n_features = X.shape[1]
    shrinkages = batch_ledoit_wolf_shrinkage(X)
    emp_cov = batch_empirical_covariance(X, assume_centered)
    mu = np.linalg.trace(emp_cov) / n_features

    shrunk_cov = (1.0 - shrinkages)[:, None, None] * emp_cov
    i = np.arange(n_features)
    shrunk_cov[:, i, i] += (shrinkages * mu)[:, None]
    return shrunk_cov, shrinkages


def batch_oas(X, *, assume_centered=False):
    """Estimate batch OAS covariance matrices.

    Parameters
    ----------
    X : ndarray of shape (n_matrices, n_features, n_samples)
        Input data.
    assume_centered : bool, default=False
        If False, center each signal before estimating the covariance.

    Returns
    -------
    covariance : ndarray of shape (n_matrices, n_features, n_features)
        Estimated covariance matrices.
    shrinkage : ndarray of shape (n_matrices,)
        OAS shrinkage coefficients.
    """
    n_matrices, n_features, n_samples = X.shape

    if not assume_centered:
        X -= np.mean(X, axis=2, keepdims=True)

    emp_cov = batch_empirical_covariance(X, assume_centered)

    alpha = np.mean(emp_cov**2, axis=(1, 2))

    mu = np.linalg.trace(emp_cov) / n_features
    mu_squared = mu**2

    numerator = alpha + mu_squared
    denominator = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = np.ones(len(X), dtype=float)
    np.divide(numerator, denominator, out=shrinkage, where=denominator != 0)
    shrinkage = np.clip(shrinkage, 0.0, 1.0)

    shrunk_cov = (1.0 - shrinkage[:, None, None]) * emp_cov
    i = np.arange(n_features)
    shrunk_cov[:, i, i] += (shrinkage * mu)[:, None]

    return shrunk_cov, shrinkage


class BatchCovariances(BaseEstimator, TransformerMixin):
    """Estimate covariance matrices for a batch of signals.

    Parameters
    ----------
    estimator : {"scm", "lwf", "oas"}, default="scm"
        Covariance estimator to use.
    **kwds
        Additional keyword arguments passed to the covariance estimator.
    """

    _COVARIANCE_METHODS = {
        "scm": batch_empirical_covariance,
        "lwf": batch_ledoit_wolf,
        "oas": batch_oas,
    }

    def __init__(self, estimator="scm", assume_centered=False):
        if estimator not in self._COVARIANCE_METHODS.keys():
            raise ValueError(
                f"Invalid method: '{estimator}'. "
                f"Available methods: {list(self._COVARIANCE_METHODS.keys())}"
            )
        self.estimator = estimator
        self.assume_centered = assume_centered

    def fit(self, X, y=None, groups=None):
        """No fitting necessary, just for compatibility with sk-learn."""
        self.fitted_ = True
        return self

    def transform(self, X, y=None, groups=None):
        """Estimate covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_features, n_samples)
            Multi-channel time-series.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_features, n_features)
            Covariance matrices.
        """
        check_is_fitted(self, "fitted_")

        if isinstance(X, tuple):
            X_data = X[0]
        else:
            X_data = X

        if X_data.ndim != 3:
            raise ValueError(
                f"X must have shape (n_matrices, n_features, n_samples), got {X.shape}"
            )

        if X_data.shape[2] == 1:
            warnings.warn(
                "Only one sample available. You may want to reshape your data array",
                stacklevel=2,
            )

        X_copied = X_data.copy()

        covariance_method = self._COVARIANCE_METHODS[self.estimator]
        covmats = covariance_method(X_copied, assume_centered=self.assume_centered)
        return covmats[0] if type(covmats) is tuple else covmats


###########################################################################################
# MeanProbabilityAggregator
###########################################################################################
class MeanProbabilityAggregator(BaseEstimator, TransformerMixin):
    """Aggregate window-level probabilities into one prediction per subject.

    Designed for the case where a single call handles windows from
    *multiple* subjects at once (e.g. an entire held-out test set). Each
    window's prediction is averaged together with every other window
    belonging to the same subject, and the result is returned as either
    a probability vector or a thresholded binary label vector, with one
    entry per subject.

    Parameters
    ----------
    threshold : float, default=0.5
        Decision threshold applied to the mean probability when producing
        binary labels via ``predict``.

    Attributes
    ----------
    fitted_ : bool
        Set to True after ``fit`` is called. This estimator is stateless
        (it has nothing to learn from training data), so ``fit`` only
        marks readiness for ``predict`` / ``predict_proba`` / ``transform``.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Mark the stateless aggregator as fitted.

        Parameters
        ----------
        X : array-like
            Ignored. Present for sklearn API compatibility.
        y : array-like, default=None
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        self : MeanProbabilityAggregator
        """
        self.fitted_ = True
        return self

    def _aggregate(self, X, groups):
        """Average window-level values within each subject.

        Parameters
        ----------
        X : array-like of shape (n_windows,) or (n_windows, n_classes)
            Window-level predictions, possibly from multiple subjects
            stacked together.
        groups : array-like of shape (n_windows,)
            Subject identifier for each row of ``X``. Rows sharing the
            same identifier are averaged together.

        Returns
        -------
        ndarray of shape (n_subjects,) or (n_subjects, n_classes)
            One averaged row per subject, ordered by first appearance
            of that subject's identifier in ``groups``.
        """
        if isinstance(X, tuple):
            # Allows groups to travel bundled with X (e.g. (X, groups))
            # through call sites, such as a plain Pipeline.transform(X),
            # that have no dedicated channel for extra metadata.
            if len(X) > 1 and groups is None:
                groups = X[1]
            X = X[0]

        X = np.asarray(X, dtype=float)

        if groups is None:
            raise ValueError(
                "groups must be provided to aggregate per subject "
                "(one identifier per row of X)"
            )
        groups = np.asarray(groups)

        # first_indices[i]: position where the i-th sorted-unique label
        # first appears in `groups`. inverse[j]: which sorted-unique
        # label row j of X belongs to.
        _, first_indices, inverse = np.unique(
            groups, return_index=True, return_inverse=True
        )
        # Reorder sorted-unique label indices by first appearance, so
        # subjects come out in the order they first showed up in the
        # data rather than alphabetically.
        ordered_groups = np.argsort(first_indices)

        aggregated = np.array(
            [X[inverse == group].mean(axis=0) for group in ordered_groups]
        )
        return aggregated

    def predict_proba(self, X, groups=None):
        """Return the mean probability vector for each subject.

        Parameters
        ----------
        X : array-like of shape (n_windows,) or (n_windows, n_classes), \
                or tuple (X, groups)
            Window-level predictions, possibly spanning multiple subjects.
        groups : array-like of shape (n_windows,), default=None
            Subject identifier for each row of ``X``. Required unless
            bundled into ``X`` as a tuple.

        Returns
        -------
        ndarray of shape (n_subjects,) or (n_subjects, n_classes)
            Mean probability per subject, ordered by first appearance.
        """
        check_is_fitted(self, "fitted_")
        return self._aggregate(X, groups)

    def predict(self, X, groups=None):
        """Return the binary label vector for each subject.

        Parameters
        ----------
        X : array-like of shape (n_windows,) or (n_windows, n_classes), \
                or tuple (X, groups)
            Window-level predictions, possibly spanning multiple subjects.
        groups : array-like of shape (n_windows,), default=None
            Subject identifier for each row of ``X``. Required unless
            bundled into ``X`` as a tuple.

        Returns
        -------
        ndarray of shape (n_subjects,) or (n_subjects, n_classes)
            1 where the mean probability meets or exceeds ``threshold``,
            else 0. One entry per subject, ordered by first appearance.
        """
        probabilities = self.predict_proba(X, groups=groups)
        return (probabilities >= self.threshold).astype(int)

    def transform(self, X, y=None, groups=None):
        """Alias for ``predict_proba``, for use inside sklearn Pipelines.

        Parameters
        ----------
        X : array-like of shape (n_windows,) or (n_windows, n_classes), \
                or tuple (X, groups)
            Window-level predictions, possibly spanning multiple subjects.
        y : array-like, default=None
            Ignored. Present for sklearn API compatibility.
        groups : array-like of shape (n_windows,), default=None
            Subject identifier for each row of ``X``. Required unless
            bundled into ``X`` as a tuple.

        Returns
        -------
        ndarray of shape (n_subjects,) or (n_subjects, n_classes)
            Mean probability per subject, ordered by first appearance.
        """
        return self.predict_proba(X, groups=groups)

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        """Fit (no-op) and aggregate in one call, forwarding ``groups``.

        Parameters
        ----------
        X : array-like of shape (n_windows,) or (n_windows, n_classes), \
                or tuple (X, groups)
            Window-level predictions, possibly spanning multiple subjects.
        y : array-like, default=None
            Ignored. Present for sklearn API compatibility.
        groups : array-like of shape (n_windows,), default=None
            Subject identifier for each row of ``X``. Required unless
            bundled into ``X`` as a tuple.
        **fit_params : dict
            Ignored. Accepted for sklearn Pipeline compatibility.

        Returns
        -------
        ndarray of shape (n_subjects,) or (n_subjects, n_classes)
            Mean probability per subject, ordered by first appearance.
        """
        self.fit(X, y)
        return self.transform(X, y, groups=groups)
