"""Build and evaluate the subject-level classifiers used in the examples."""

from __future__ import annotations

import numpy as np
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_is_fitted

from .estimators import (
    BandPassFilter,
    BatchCovariances,
    SlidingWindow,
    StateSelector,
)
from .pipeline import SubjectPipeline

__all__ = [
    "FeatureConsolidator",
    "StateAwareMDM",
    "StateTangentFeatures",
    "build_classification_pipeline",
    "cross_validate_subjects",
    "make_subject_folds",
    "mean_label_balanced_accuracy",
    "nested_search_subjects",
]

_DEFAULT_RANDOM_STATE = 0
_AGGREGATION_STRATEGIES = ("mean", "mean_of_state_means", "concat", "stack")


class FeatureConsolidator(BaseEstimator, TransformerMixin):
    """Collapse window feature rows to one row per subject.

    ``groups`` must be ``(window_subject_ids, window_states)``. Subject rows
    follow subject-ID first-appearance order, matching the input subject list.

    Parameters
    ----------
    strategy : {"mean", "mean_of_state_means", "concat", "stack"}, default="mean"
        ``mean`` averages every window belonging to a subject.
        ``mean_of_state_means`` first averages within each available state and
        then gives those state means equal weight. ``concat`` joins the state
        means in ``state_order`` and requires every subject to contain every
        configured state. ``stack`` keeps the same state means on a new axis,
        which is useful when the rows are covariance matrices.
    state_order : sequence, default=(0, 1)
        Permitted recording states and, for ``concat`` or ``stack``, their
        output order.

    Notes
    -----
    ``transform`` returns ``(features, subject_ids)`` so
    :class:`pcp_project.pipeline.SubjectPipeline` can keep metadata aligned
    after the row count changes from windows to subjects.
    """

    def __init__(self, strategy="mean", state_order=(0, 1)):
        self.strategy = strategy
        self.state_order = state_order

    def fit(self, X, y=None, groups=None):
        """Validate the feature and metadata contract."""
        features = _feature_array(X)
        _window_metadata(groups, len(features))
        if self.strategy not in _AGGREGATION_STRATEGIES:
            raise ValueError(f"strategy must be one of {_AGGREGATION_STRATEGIES}")
        if self.strategy != "mean":
            self.state_order_ = _state_order(self.state_order)
        if self.strategy == "concat" and features.ndim != 2:
            raise ValueError("concat requires two-dimensional feature rows")
        self.n_features_in_ = features.shape[1]
        self.feature_shape_in_ = features.shape[1:]
        return self

    def transform(self, X, y=None, groups=None):
        """Return one aggregated feature value per subject."""
        check_is_fitted(self, "feature_shape_in_")
        features = _feature_array(X)
        if features.shape[1:] != self.feature_shape_in_:
            raise ValueError(
                f"X has trailing shape {features.shape[1:]}, "
                f"expected {self.feature_shape_in_}"
            )
        subject_ids, states = _window_metadata(groups, len(features))
        if self.strategy != "mean":
            unknown = np.setdiff1d(np.unique(states), self.state_order_)
            if len(unknown):
                raise ValueError(f"unknown window states: {unknown.tolist()}")

        unique_subjects = list(dict.fromkeys(subject_ids.tolist()))
        rows = []
        for subject_id in unique_subjects:
            subject_mask = subject_ids == subject_id
            subject_features = features[subject_mask]
            if self.strategy == "mean":
                rows.append(subject_features.mean(axis=0))
                continue

            subject_states = states[subject_mask]
            state_means = [
                subject_features[subject_states == state].mean(axis=0)
                for state in self.state_order_
                if np.any(subject_states == state)
            ]
            if self.strategy in ("concat", "stack"):
                if len(state_means) != len(self.state_order_):
                    raise ValueError(
                        f"{self.strategy} requires every subject to contain every state"
                    )
                combine = np.concatenate if self.strategy == "concat" else np.stack
                rows.append(combine(state_means))
            else:
                rows.append(np.mean(state_means, axis=0))
        return np.asarray(rows), np.asarray(unique_subjects)

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        """Fit and aggregate while forwarding window metadata."""
        return self.fit(X, y=y, groups=groups).transform(X, y=y, groups=groups)


class StateAwareMDM(ClassifierMixin, BaseEstimator):
    """Fit one Riemannian class-centroid model per recording state.

    Parameters
    ----------
    metric : str or dict, default="riemann"
        Distance metric passed to :class:`pyriemann.classification.MDM`.

    Notes
    -----
    Input has shape ``(n_subjects, n_states, n_channels, n_channels)``. Each
    state gets its own MDM model. Prediction combines the state distances with
    the Euclidean norm and chooses the closest class.
    """

    def __init__(self, metric="riemann"):
        self.metric = metric

    def fit(self, X, y):
        """Fit one minimum-distance-to-mean model for each state."""
        X = _state_covariances(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must contain one label per subject")

        self.state_models_ = [
            MDM(metric=self.metric).fit(X[:, state], y) for state in range(X.shape[1])
        ]
        self.classes_ = self.state_models_[0].classes_
        self.state_shape_ = X.shape[1:]
        return self

    def transform(self, X):
        """Return the combined state distance to each class."""
        check_is_fitted(self, "state_models_")
        X = _state_covariances(X)
        if X.shape[1:] != self.state_shape_:
            raise ValueError("X does not match the fitted state covariance shape")

        state_distances = np.stack(
            [
                model.transform(X[:, state])
                for state, model in enumerate(self.state_models_)
            ],
            axis=1,
        )
        return np.sqrt(np.sum(state_distances**2, axis=1))

    def predict(self, X):
        """Predict the class with the smallest combined state distance."""
        distances = self.transform(X)
        return self.classes_[np.argmin(distances, axis=1)]

    def decision_function(self, X):
        """Return class preference scores derived from the state distances.

        Binary output is the distance to class 0 minus the distance to class 1.
        For more than two classes, the score for each class is its negative
        distance, so larger values still indicate a closer class mean.
        """
        distances = self.transform(X)
        if len(self.classes_) == 2:
            return distances[:, 0] - distances[:, 1]
        return -distances


class StateTangentFeatures(BaseEstimator, TransformerMixin):
    """Map each recording state's covariance to tangent space and concatenate.

    Consumes the ``(n_subjects, n_states, n_channels, n_channels)`` tensor from
    the ``stack`` aggregation and returns one tangent-space feature row per
    subject. Each state is projected at its own reference mean, mirroring the
    per-state models in :class:`StateAwareMDM`.

    Parameters
    ----------
    metric : str or dict, default="riemann"
        Tangent-space metric passed to :class:`pyriemann.tangentspace.TangentSpace`.
    """

    def __init__(self, metric="riemann"):
        self.metric = metric

    def fit(self, X, y=None):
        """Fit one tangent-space map for each recording state."""
        X = _state_covariances(X)
        self.state_maps_ = [
            TangentSpace(metric=self.metric).fit(X[:, state])
            for state in range(X.shape[1])
        ]
        self.state_shape_ = X.shape[1:]
        return self

    def transform(self, X):
        """Concatenate the per-state tangent vectors for each subject."""
        check_is_fitted(self, "state_maps_")
        X = _state_covariances(X)
        if X.shape[1:] != self.state_shape_:
            raise ValueError("X does not match the fitted state covariance shape")
        return np.concatenate(
            [
                state_map.transform(X[:, state])
                for state, state_map in enumerate(self.state_maps_)
            ],
            axis=1,
        )


def build_classification_pipeline(
    classifier=None,
    *,
    states=("eyes_open", "eyes_closed"),
    window_seconds=2.0,
    step_seconds=None,
    covariance="oas",
    aggregation="mean",
    state_order=(0, 1),
    feature_transform="tangent",
    sfreq=256.0,
    frequency_bands=((4.0, 15.0),),
    random_state: int | None = _DEFAULT_RANDOM_STATE,
) -> SubjectPipeline:
    """Build the raw-subject-to-classification pipeline.

    ``step_seconds=None`` uses adjacent, non-overlapping windows. Set an
    explicit positive hop duration to choose overlap or a wider stride.

    Parameters
    ----------
    classifier : sklearn classifier, optional
        Final subject-level classifier. Defaults to balanced logistic
        regression. Any cloneable sklearn classifier can be supplied; pass a
        multi-label estimator such as ``OneVsRestClassifier`` for a 2-D target.
    states : sequence of str or int
        Recording states retained before windowing.
    window_seconds : float, default=2.0
        Duration of each window.
    step_seconds : float, optional
        Hop between windows. By default it equals ``window_seconds``.
    covariance : {"scm", "oas", "lwf"}, default="oas"
        Window-level covariance estimator.
    aggregation : {"mean", "mean_of_state_means", "concat", "stack"}, default="mean"
        Feature aggregation performed before classification.
    state_order : sequence, default=(0, 1)
        Permitted state codes and concatenation or stacking order.
    feature_transform : {"tangent", "identity"}, default="tangent"
        Covariance-to-feature representation. ``identity`` retains covariance
        matrices and bypasses feature scaling. Supply a covariance-aware final
        estimator, or ``"passthrough"`` when extracting covariances.
    sfreq : float, default=256.0
        Recording sampling frequency in hertz.
    frequency_bands : sequence of (low, high)
        Pass bands in hertz whose filtered signals are summed.
    random_state : int or None, default=0
        Classifier random seed.

    Returns
    -------
    SubjectPipeline
        Unfitted pipeline from raw subject recordings to subject predictions.
    """
    length = _window_length(window_seconds, sfreq)
    step_size = (
        length
        if step_seconds is None
        else _window_length(step_seconds, sfreq, name="step_seconds")
    )
    if classifier is None:
        if feature_transform == "identity":
            raise ValueError("identity features require an explicit final estimator")
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
    feature_step = _feature_transform(feature_transform)
    scale_step = "passthrough" if feature_transform == "identity" else StandardScaler()
    steps = [
        ("filter", BandPassFilter(frequency_bands, sfreq=sfreq)),
        ("select", StateSelector(states=states)),
        (
            "epoch",
            SlidingWindow(
                length=length,
                step_size=step_size,
                padding_policy="valid",
            ),
        ),
        ("covariance", BatchCovariances(estimator=covariance)),
        ("features", feature_step),
        (
            "aggregate",
            FeatureConsolidator(strategy=aggregation, state_order=state_order),
        ),
        ("scale", scale_step),
        ("classifier", classifier),
    ]
    return SubjectPipeline(steps)


def make_subject_folds(
    y,
    groups,
    *,
    n_splits=3,
    random_state: int | None = _DEFAULT_RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create reusable cross-validation folds at the subject level.

    One-dimensional targets use stratified group folds. Scikit-learn does not
    provide a multi-label stratified group splitter, so two-dimensional
    targets use shuffled K-folds. In both cases each subject belongs to exactly
    one input row and cannot cross between train and test.

    Parameters
    ----------
    y : array-like of shape (n_subjects,) or (n_subjects, n_labels)
        Subject targets.
    groups : array-like of shape (n_subjects,)
        Unique subject IDs.
    n_splits : int, default=3
        Number of folds.
    random_state : int or None, default=0
        Seed used for shuffling. ``None`` disables shuffling.

    Returns
    -------
    list of (train_indices, test_indices) tuples
        Integer row indices for each fold.
    """
    y, groups = _subject_arrays(y, groups)
    _check_splits(y, n_splits)
    if y.ndim == 1:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=random_state is not None,
            random_state=random_state,
        )
        return list(splitter.split(np.zeros((len(y), 1)), y, groups))

    splitter = KFold(
        n_splits=n_splits,
        shuffle=random_state is not None,
        random_state=random_state,
    )
    return list(splitter.split(np.zeros((len(y), 1))))


def cross_validate_subjects(
    estimator,
    X,
    y,
    groups,
    *,
    cv=None,
    scoring=None,
    n_splits=3,
    random_state: int | None = _DEFAULT_RANDOM_STATE,
    n_jobs=None,
):
    """Cross-validate an estimator after checking the subject-level folds.

    Parameters
    ----------
    estimator : estimator
        Any estimator accepted by :func:`sklearn.model_selection.cross_validate`.
    X : array-like of shape (n_subjects, ...)
        One input row or recording pair per subject.
    y : array-like of shape (n_subjects,) or (n_subjects, n_labels)
        Subject targets.
    groups : array-like of shape (n_subjects,)
        Unique subject IDs.
    cv : iterable of folds or None, optional
        Saved ``(train_indices, test_indices)`` pairs. New folds are created when
        this is ``None``.
    scoring : str, callable, dict, sequence, or None, optional
        Scoring argument passed to scikit-learn. By default, one-dimensional
        targets use balanced accuracy and two-dimensional targets use the mean
        balanced accuracy across labels.
    n_splits : int, default=3
        Number of folds to create when ``cv`` is ``None``.
    random_state : int or None, default=0
        Seed used when new folds are created.
    n_jobs : int or None, optional
        Number of parallel jobs used by scikit-learn.

    Returns
    -------
    dict
        Timing and test-score arrays returned by scikit-learn.
    """
    y, groups = _subject_arrays(y, groups, n_subjects=len(X))
    folds = _subject_folds(y, groups, cv, n_splits, random_state)
    return cross_validate(
        estimator,
        X,
        y,
        groups=groups,
        scoring=_scoring_for_target(y, scoring),
        cv=folds,
        n_jobs=n_jobs,
        error_score="raise",
    )


def mean_label_balanced_accuracy(y_true, y_pred):
    """Return balanced accuracy averaged equally across target labels.

    This metric is intended for multi-label indicator targets. Each label is
    scored as its own binary classification problem before the scores are
    averaged, so labels with different prevalences have equal weight.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_subjects, n_labels)
        True and predicted multi-label indicator targets.

    Returns
    -------
    float
        Mean per-label balanced accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 2 or y_true.shape != y_pred.shape or y_true.shape[1] == 0:
        raise ValueError("y_true and y_pred must be matching non-empty 2-D arrays")
    return float(
        np.mean(
            [
                balanced_accuracy_score(y_true[:, label], y_pred[:, label])
                for label in range(y_true.shape[1])
            ]
        )
    )


def nested_search_subjects(
    estimator,
    param_grid,
    X,
    y,
    groups,
    *,
    cv=None,
    inner_splits=3,
    scoring=None,
    n_splits=3,
    random_state: int | None = _DEFAULT_RANDOM_STATE,
    n_jobs=None,
):
    """Tune and evaluate an estimator with nested subject-level folds.

    The inner search only sees the training subjects from its outer fold. Its
    selected estimator is then evaluated once on that fold's held-out subjects.
    This separation avoids using the reported test folds for model selection.

    Parameters
    ----------
    estimator : estimator
        Cloneable scikit-learn-compatible estimator.
    param_grid : dict or list of dict
        Parameter grid passed to :class:`sklearn.model_selection.GridSearchCV`.
    X : array-like of shape (n_subjects, ...)
        One input row or recording pair per subject.
    y : array-like of shape (n_subjects,) or (n_subjects, n_labels)
        Subject targets.
    groups : array-like of shape (n_subjects,)
        Unique subject IDs.
    cv : iterable of folds or None, optional
        Saved outer ``(train_indices, test_indices)`` pairs. New folds are
        created when this is ``None``.
    inner_splits : int, default=3
        Number of subject-level folds used by each parameter search.
    scoring : str, callable, or None, optional
        Search and evaluation scorer. The default follows
        :func:`cross_validate_subjects`.
    n_splits : int, default=3
        Number of outer folds to create when ``cv`` is ``None``.
    random_state : int or None, default=0
        Seed used when folds are created.
    n_jobs : int or None, optional
        Number of parallel jobs used within each grid search.

    Returns
    -------
    dict
        Arrays named ``test_score`` and ``best_inner_score``, plus lists named
        ``best_params`` and ``estimator``. Each returned estimator is fitted on
        the complete training portion of its outer fold.
    """
    y, groups = _subject_arrays(y, groups, n_subjects=len(X))
    folds = _subject_folds(y, groups, cv, n_splits, random_state)
    scoring = _scoring_for_target(y, scoring)
    test_scores = []
    inner_scores = []
    best_params = []
    estimators = []

    for train_index, test_index in folds:
        X_train = _safe_indexing(X, train_index)
        y_train = _safe_indexing(y, train_index)
        groups_train = _safe_indexing(groups, train_index)
        inner_folds = make_subject_folds(
            y_train,
            groups_train,
            n_splits=inner_splits,
            random_state=random_state,
        )
        search = GridSearchCV(
            clone(estimator),
            param_grid,
            scoring=scoring,
            cv=inner_folds,
            refit=True,
            n_jobs=n_jobs,
            error_score="raise",
        )
        search.fit(X_train, y_train, groups=groups_train)

        test_scores.append(
            search.score(_safe_indexing(X, test_index), _safe_indexing(y, test_index))
        )
        inner_scores.append(search.best_score_)
        best_params.append(search.best_params_)
        estimators.append(search.best_estimator_)

    return {
        "test_score": np.asarray(test_scores),
        "best_inner_score": np.asarray(inner_scores),
        "best_params": best_params,
        "estimator": estimators,
    }


def _scoring_for_target(y, scoring):
    if scoring is not None:
        return scoring
    if y.ndim == 1:
        return "balanced_accuracy"
    return make_scorer(mean_label_balanced_accuracy)


def _feature_array(X):
    features = np.asarray(X, dtype=float)
    if features.ndim not in (2, 3):
        raise ValueError("X must be a feature matrix or covariance batch")
    return features


def _state_covariances(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 4 or X.shape[2] != X.shape[3]:
        raise ValueError("X must have shape (subjects, states, channels, channels)")
    return X


def _window_metadata(groups, n_rows):
    if not isinstance(groups, tuple) or len(groups) != 2:
        raise ValueError("groups must be (window_subject_ids, window_states)")
    subject_ids, states = (np.asarray(values) for values in groups)
    if (
        subject_ids.ndim != 1
        or states.ndim != 1
        or len(subject_ids) != n_rows
        or len(states) != n_rows
    ):
        raise ValueError("window metadata must have one entry per feature row")
    return subject_ids, states


def _state_order(state_order):
    states = np.asarray(state_order)
    if states.ndim != 1 or len(states) == 0:
        raise ValueError("state_order must be a non-empty one-dimensional sequence")
    if len(np.unique(states)) != len(states):
        raise ValueError("state_order must not contain duplicates")
    return states


def _subject_arrays(y, groups, n_subjects=None):
    y = np.asarray(y)
    groups = np.asarray(groups)
    if (
        y.ndim not in (1, 2)
        or (y.ndim == 2 and y.shape[1] == 0)
        or groups.ndim != 1
        or len(y) != len(groups)
    ):
        raise ValueError("y and groups must contain one entry per subject")
    if n_subjects is not None and len(y) != n_subjects:
        raise ValueError("X, y, and groups must contain one entry per subject")
    if len(dict.fromkeys(groups.tolist())) != len(groups):
        raise ValueError("groups must contain one unique ID per subject")
    return y, groups


def _subject_folds(y, groups, cv, n_splits, random_state):
    folds = (
        make_subject_folds(
            y,
            groups,
            n_splits=n_splits,
            random_state=random_state,
        )
        if cv is None
        else list(cv)
    )
    if not folds:
        raise ValueError("cv must contain at least one fold")

    checked = []
    for train_index, test_index in folds:
        train_index = np.asarray(train_index)
        test_index = np.asarray(test_index)
        if not set(groups[train_index]).isdisjoint(groups[test_index]):
            raise ValueError("train and test subjects overlap in a fold")
        checked.append((train_index, test_index))
    return checked


def _feature_transform(name):
    if name == "identity":
        return "passthrough"
    if name == "tangent":
        return TangentSpace()
    raise ValueError("feature_transform must be 'tangent' or 'identity'")


def _window_length(seconds, sfreq, *, name="window_seconds"):
    if not seconds > 0:
        raise ValueError(f"{name} must be a positive number")
    if not sfreq > 0:
        raise ValueError("sfreq must be a positive number")
    return max(1, int(round(seconds * sfreq)))


def _check_splits(y, n_splits):
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if y.ndim == 1:
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2 or counts.min() < n_splits:
            raise ValueError("each class must contain at least n_splits subjects")
        return

    for target in y.T:
        _, counts = np.unique(target, return_counts=True)
        if len(counts) < 2 or counts.min() < n_splits:
            raise ValueError("each label class must contain at least n_splits subjects")
