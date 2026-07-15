"""Private metadata-routing and recording-collection helpers."""

import inspect

import numpy as np

# Metadata-routing helpers
# ------------------------
# These functions keep groups local to one pipeline operation and only pass
# metadata to methods that explicitly declare support for it.


# FIX(ref): Route metadata only to concrete parameters; generic ``**kwargs``
# does not mean a nested sklearn estimator supports subject groups.
def _declares_param(func, param_name):
    return param_name in inspect.signature(func).parameters


def _final_estimator_has(method_name):
    def check(self):
        final = self._final_estimator
        if final in (None, "passthrough") or not hasattr(final, method_name):
            raise AttributeError(f"The final step does not implement {method_name}().")
        return True

    return check


def _split_input(X, default_mask):
    # FIX(ref): Distinguish ``(X, groups)`` from a valid two-row tuple while
    # keeping transformed metadata local to the current operation.
    is_metadata_pair = isinstance(X, tuple) and len(X) == 2 and np.ndim(X[0]) >= 2
    if is_metadata_pair:
        X, mask = X
    else:
        mask = default_mask
    return X, mask


def _metadata_kwargs(method, X, y, groups):
    # FIX(ref): Forward targets only while row-aligned and groups only when the
    # concrete method declares support for them.
    kwargs = {}
    if y is not None and len(X) == len(y) and _declares_param(method, "y"):
        kwargs["y"] = y
    if _declares_param(method, "groups"):
        kwargs["groups"] = groups
    return kwargs


def _transform_one(transformer, X, y, groups):
    # FIX(ref): Preserve opaque groups while guarding transform-time target routing.
    kwargs = _metadata_kwargs(transformer.transform, X, y, groups)
    result = transformer.transform(X, **kwargs)
    # FIX(ref): Update operation-local groups from tuple outputs and retain them
    # across array-only outputs without mutating ``self.mask``.
    if isinstance(result, tuple):
        return result[:2]
    return result, groups


# Recording-collection helpers
# ----------------------------
# Collections retain subject boundaries and contiguous state runs while the
# public estimators operate on ordinary recording/metadata pairs.

STATE_NAME_TO_CODE = {"eyes_open": 0, "eyes_closed": 1}


def _subject_collection(X):
    if isinstance(X, list):
        items = X
    elif isinstance(X, np.ndarray) and X.dtype == object and X.ndim == 1:
        items = X.tolist()
    else:
        return None

    if all(_is_recording_pair(item) or _is_run_list(item) for item in items):
        return items
    return None


def _is_recording_pair(value):
    return isinstance(value, tuple) and len(value) == 2


def _is_run_list(value):
    return isinstance(value, list) and all(_is_recording_pair(run) for run in value)


def _recording_pair(value):
    recording, sample_states = value
    recording = np.asarray(recording, dtype=np.float64)
    sample_states = np.asarray(sample_states)
    return recording, sample_states


def _state_values(states):
    if states is None:
        return None, None
    raw_states = np.atleast_1d(states)
    state_codes = np.asarray(
        [
            STATE_NAME_TO_CODE[state] if isinstance(state, str) else int(state)
            for state in raw_states
        ]
    )
    return raw_states, state_codes


def _selected_runs(subject, states):
    recording, sample_states = _recording_pair(subject)
    raw_states, state_codes = _state_values(states)
    starts = np.r_[0, np.flatnonzero(sample_states[1:] != sample_states[:-1]) + 1]
    stops = np.r_[starts[1:], len(sample_states)]
    runs = []
    for start, stop in zip(starts, stops, strict=True):
        state = sample_states[start]
        if states is None or state in raw_states or state in state_codes:
            runs.append((recording[:, start:stop], sample_states[start:stop]))
    return runs


def _map_recording_pairs(collection, operation):
    mapped = []
    for subject in collection:
        if _is_recording_pair(subject):
            mapped.append(operation(subject))
        else:
            mapped.append([operation(run) for run in subject])
    return mapped


def _window_subjects(window, collection):
    batches = []
    subject_ids = []
    window_states = []

    for subject_id, subject in enumerate(collection):
        source_runs = [subject] if _is_recording_pair(subject) else subject
        runs = [
            run
            for source_run in source_runs
            for run in _selected_runs(source_run, states=None)
        ]
        for recording, sample_states in runs:
            if len(sample_states) < window.length and window.padding_policy == "valid":
                continue
            run_windows, run_states = window.transform(
                recording,
                groups=sample_states,
            )
            batches.append(run_windows)
            subject_ids.append(np.full(len(run_windows), subject_id, dtype=int))
            window_states.append(run_states)

    return (
        np.concatenate(batches),
        (np.concatenate(subject_ids), np.concatenate(window_states)),
    )
