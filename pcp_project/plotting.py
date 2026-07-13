"""Plots used to explain covariance features and classifier results."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array

__all__ = [
    "plot_covariance_ellipses",
    "plot_covariance_heatmaps",
    "plot_covariance_space",
    "plot_feature_clusters",
    "plot_mdm_distances",
    "plot_state_trajectories",
]

_STATE_COLORS = ("#4C78A8", "#D99B2B", "#7A9E5B", "#D66BA0")
_STATE_MARKERS = ("o", "s", "^", "D")


def plot_covariance_ellipses(
    windows_by_state: Mapping,
    *,
    channel_names=("Channel 1", "Channel 2"),
    n_std=2.0,
) -> Figure:
    """Show two-channel sample clouds and their covariance ellipses.

    Each value in ``windows_by_state`` must have shape ``(2, n_samples)``.
    The ellipse is centered on the sample mean and extends ``n_std`` standard
    deviations along each principal covariance direction.

    Parameters
    ----------
    windows_by_state : mapping
        State names mapped to two-channel sample arrays.
    channel_names : sequence of str, default=("Channel 1", "Channel 2")
        Labels for the two axes.
    n_std : float, default=2.0
        Radius of the ellipse in standard deviations.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one panel per state.
    """
    items = _mapping_items(windows_by_state, "windows_by_state")
    channel_names = _channel_names(channel_names, 2)
    try:
        n_std = float(n_std)
    except (TypeError, ValueError) as error:
        raise ValueError("n_std must be a positive finite number") from error
    if not np.isfinite(n_std) or n_std <= 0:
        raise ValueError("n_std must be a positive finite number")

    windows = []
    for state, values in items:
        values = np.asarray(values, dtype=float)
        if (
            values.ndim != 2
            or values.shape[0] != 2
            or values.shape[1] < 2
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"windows for {state!r} must have finite shape (2, n_samples)"
            )
        windows.append((state, values))

    combined = np.hstack([values for _, values in windows])
    limits = [_padded_limits(combined[row]) for row in range(2)]
    figure, axes = plt.subplots(
        1,
        len(windows),
        figsize=(5 * len(windows), 4.5),
        squeeze=False,
        constrained_layout=True,
    )

    for index, (axis, (state, values)) in enumerate(
        zip(axes.ravel(), windows, strict=True)
    ):
        color = _STATE_COLORS[index % len(_STATE_COLORS)]
        covariance = np.cov(values)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0)
        principal = eigenvectors[:, order[0]]
        angle = np.degrees(np.arctan2(principal[1], principal[0]))
        ellipse = Ellipse(
            xy=values.mean(axis=1),
            width=2 * n_std * np.sqrt(eigenvalues[0]),
            height=2 * n_std * np.sqrt(eigenvalues[1]),
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.18,
            linewidth=2,
        )
        axis.scatter(
            values[0],
            values[1],
            color=color,
            alpha=0.28,
            s=14,
            edgecolors="none",
        )
        axis.add_patch(ellipse)
        axis.set(
            title=_display_name(state),
            xlabel=channel_names[0],
            ylabel=channel_names[1],
            xlim=limits[0],
            ylim=limits[1],
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)

    figure.suptitle("Two-channel samples and covariance ellipses")
    return figure


def plot_covariance_heatmaps(
    covariances_by_state: Mapping,
    *,
    channel_names=None,
) -> Figure:
    """Plot state covariance matrices with one shared color scale.

    Parameters
    ----------
    covariances_by_state : mapping
        State names mapped to square, symmetric covariance matrices.
    channel_names : sequence of str or None, optional
        Channel labels. Numbered labels are used when this is omitted.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one heatmap per state.
    """
    items = _mapping_items(covariances_by_state, "covariances_by_state")
    matrices = []
    matrix_size = None
    for state, values in items:
        matrix = np.asarray(values, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T)
        ):
            raise ValueError(f"covariance for {state!r} must be finite and symmetric")
        if matrix_size is not None and matrix.shape[0] != matrix_size:
            raise ValueError("all covariance matrices must have the same shape")
        matrix_size = matrix.shape[0]
        matrices.append((state, matrix))

    names = _channel_names(channel_names, matrix_size)
    color_limit = max(np.max(np.abs(matrix)) for _, matrix in matrices)
    color_limit = color_limit if color_limit > 0 else 1.0
    figure, axes = plt.subplots(
        1,
        len(matrices),
        figsize=(5 * len(matrices), 4.5),
        squeeze=False,
        constrained_layout=True,
    )
    tick_positions = (
        np.arange(matrix_size)
        if matrix_size <= 12
        else np.unique(np.linspace(0, matrix_size - 1, 9, dtype=int))
    )

    image = None
    for axis, (state, matrix) in zip(axes.ravel(), matrices, strict=True):
        image = axis.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
            interpolation="nearest",
        )
        axis.set_title(_display_name(state))
        axis.set_xticks(
            tick_positions,
            [names[position] for position in tick_positions],
            rotation=45,
            ha="right",
        )
        axis.set_yticks(
            tick_positions,
            [names[position] for position in tick_positions],
        )

    figure.colorbar(image, ax=axes.ravel().tolist(), label="Covariance", shrink=0.82)
    figure.suptitle("Covariance matrices")
    return figure


def plot_covariance_space(
    covariances_by_state: Mapping,
    *,
    channel_names=("Channel 1", "Channel 2"),
) -> Figure:
    """Plot two-channel covariances in variance coordinates.

    The two variances set the horizontal and vertical coordinates. Color shows
    the remaining unique covariance entry, and marker shape denotes state.

    Parameters
    ----------
    covariances_by_state : mapping
        State names mapped to arrays of shape ``(n_windows, 2, 2)``.
    channel_names : sequence of str, default=("Channel 1", "Channel 2")
        Names of the two channels.

    Returns
    -------
    matplotlib.figure.Figure
        Figure showing the covariance matrices in variance coordinates.
    """
    items = _mapping_items(covariances_by_state, "covariances_by_state")
    channel_names = _channel_names(channel_names, 2)
    batches = []
    for state, values in items:
        matrices = np.asarray(values, dtype=float)
        if (
            matrices.ndim != 3
            or matrices.shape[0] == 0
            or matrices.shape[1:] != (2, 2)
            or not np.isfinite(matrices).all()
            or not np.allclose(matrices, matrices.transpose(0, 2, 1))
            or np.any(np.linalg.eigvalsh(matrices) <= 0)
        ):
            raise ValueError(
                f"covariances for {state!r} must have finite SPD shape (n, 2, 2)"
            )
        batches.append((state, matrices))

    off_diagonal = np.concatenate([matrices[:, 0, 1] for _, matrices in batches])
    color_limit = np.max(np.abs(off_diagonal))
    color_limit = color_limit if color_limit > 0 else 1.0
    figure, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)

    scatter = None
    for index, (state, matrices) in enumerate(batches):
        variances = np.diagonal(matrices, axis1=1, axis2=2)
        marker = _STATE_MARKERS[index % len(_STATE_MARKERS)]
        scatter = axis.scatter(
            variances[:, 0],
            variances[:, 1],
            c=matrices[:, 0, 1],
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
            marker=marker,
            edgecolor="#333333",
            linewidth=0.4,
            label=_display_name(state),
        )

    axis.set(
        title="Variance coordinates",
        xlabel=f"Variance of {channel_names[0]}",
        ylabel=f"Variance of {channel_names[1]}",
    )
    axis.grid(alpha=0.2)
    axis.legend(title="State")
    figure.colorbar(
        scatter,
        ax=axis,
        label=f"Covariance: {channel_names[0]} × {channel_names[1]}",
        shrink=0.82,
    )
    figure.suptitle("Each point is one two-channel covariance matrix")
    return figure


def plot_mdm_distances(
    distances,
    labels,
    *,
    classes,
    class_names=None,
) -> Figure:
    """Plot binary MDM distances and the nearest-centroid decision boundary.

    Parameters
    ----------
    distances : array-like of shape (n_subjects, 2)
        Distance from each subject to the two class means.
    labels : array-like of shape (n_subjects,)
        True class labels.
    classes : array-like of shape (2,)
        Class values in the same order as the distance columns.
    class_names : sequence of str or None, optional
        Display names for the two classes.

    Returns
    -------
    matplotlib.figure.Figure
        Distance plot with incorrect predictions marked.
    """
    distances = np.asarray(distances, dtype=float)
    labels = np.asarray(labels)
    classes = np.asarray(classes)
    if (
        distances.ndim != 2
        or distances.shape[1] != 2
        or distances.shape[0] == 0
        or not np.isfinite(distances).all()
        or np.any(distances < 0)
    ):
        raise ValueError("distances must have finite non-negative shape (n, 2)")
    if labels.ndim != 1 or len(labels) != len(distances):
        raise ValueError("labels must contain one value per distance row")
    if classes.shape != (2,) or len(np.unique(classes)) != 2:
        raise ValueError("classes must contain two distinct values")
    if not np.isin(labels, classes).all():
        raise ValueError("labels must contain only values listed in classes")

    if class_names is None:
        names = [str(value) for value in classes]
    else:
        names = list(class_names)
        if len(names) != 2:
            raise ValueError("class_names must contain two names")

    predictions = classes[np.argmin(distances, axis=1)]
    incorrect = predictions != labels
    limit = 1.05 * distances.max()
    limit = limit if limit > 0 else 1.0
    figure, axis = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    for index, class_value in enumerate(classes):
        mask = labels == class_value
        axis.scatter(
            distances[mask, 0],
            distances[mask, 1],
            color=_STATE_COLORS[index],
            marker=_STATE_MARKERS[index],
            edgecolor="white",
            s=65,
            label=names[index],
        )
    if np.any(incorrect):
        axis.scatter(
            distances[incorrect, 0],
            distances[incorrect, 1],
            color="#222222",
            marker="x",
            s=65,
            linewidth=1.4,
            label="Incorrect",
        )

    axis.plot([0, limit], [0, limit], color="#444444", linestyle="--")
    axis.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel=f"Distance to {names[0]} mean",
        ylabel=f"Distance to {names[1]} mean",
        title="Held-out MDM distances",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(title="Actual class")
    return figure


def plot_feature_clusters(
    features_by_state: Mapping,
    labels,
    *,
    subject_ids=None,
) -> Figure:
    """Plot one standardized PCA panel per recording state.

    ``features_by_state`` maps state names to feature matrices whose rows share
    the order of ``labels``. Optional ``subject_ids`` are written beside each
    point.

    Parameters
    ----------
    features_by_state : mapping
        State names mapped to ``(n_subjects, n_features)`` arrays.
    labels : array-like of shape (n_subjects,)
        Labels used to color the points.
    subject_ids : array-like of shape (n_subjects,) or None, optional
        Labels written beside the points.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one PCA panel per state.
    """
    feature_sets, labels, subject_ids = _plot_inputs(
        features_by_state, labels, subject_ids
    )
    classes, colors = _class_colors(labels)
    figure, axes = plt.subplots(
        1,
        len(feature_sets),
        figsize=(6 * len(feature_sets), 5),
        squeeze=False,
    )

    for axis, (state, features) in zip(axes.ravel(), feature_sets, strict=True):
        embedding, explained = _pca_embedding(features)
        for class_label in classes:
            mask = labels == class_label
            axis.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                color=colors[class_label],
                label=str(class_label),
                alpha=0.85,
            )
        _annotate(axis, embedding, subject_ids)
        _style_axis(axis, state, explained)

    axes[0, 0].legend(title="Diagnosis")
    figure.suptitle("Subject-level EEG features by recording state")
    figure.tight_layout()
    return figure


def plot_state_trajectories(
    features_by_state: Mapping,
    labels,
    *,
    subject_ids=None,
) -> Figure:
    """Plot paired subject movement between two states in one PCA space.

    ``features_by_state`` must contain two matrices in trajectory order. Color
    denotes ``labels``, marker shape denotes state, and optional ``subject_ids``
    are placed at trajectory midpoints.

    Parameters
    ----------
    features_by_state : mapping
        Two state names mapped to ``(n_subjects, n_features)`` arrays.
    labels : array-like of shape (n_subjects,)
        Labels used to color the trajectories.
    subject_ids : array-like of shape (n_subjects,) or None, optional
        Labels written at trajectory midpoints.

    Returns
    -------
    matplotlib.figure.Figure
        Shared PCA view with a line joining each subject's two states.
    """
    feature_sets, labels, subject_ids = _plot_inputs(
        features_by_state, labels, subject_ids
    )
    if len(feature_sets) != 2:
        raise ValueError("features_by_state must contain exactly two states")
    if feature_sets[0][1].shape[1] != feature_sets[1][1].shape[1]:
        raise ValueError("state feature matrices must have the same columns")

    classes, colors = _class_colors(labels)
    stacked = np.vstack([features for _, features in feature_sets])
    embedding, explained = _pca_embedding(stacked)
    state_embeddings = np.split(embedding, 2)

    figure, axis = plt.subplots(figsize=(8, 6))
    for subject_index, class_label in enumerate(labels):
        points = np.vstack(
            [state_embedding[subject_index] for state_embedding in state_embeddings]
        )
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=colors[class_label],
            alpha=0.3,
        )

    for state_index, ((state, _), state_embedding) in enumerate(
        zip(feature_sets, state_embeddings, strict=True)
    ):
        for class_label in classes:
            mask = labels == class_label
            axis.scatter(
                state_embedding[mask, 0],
                state_embedding[mask, 1],
                color=colors[class_label],
                marker=_STATE_MARKERS[state_index],
                label=f"{_display_name(state)} — {class_label}",
            )

    _annotate(axis, np.mean(state_embeddings, axis=0), subject_ids)
    _style_axis(axis, "Paired state trajectories", explained)
    axis.legend(title="State — diagnosis")
    figure.suptitle("Within-subject feature shifts between recording states")
    figure.tight_layout()
    return figure


def _plot_inputs(features_by_state, labels, subject_ids):
    if not isinstance(features_by_state, Mapping) or not features_by_state:
        raise ValueError("features_by_state must be a non-empty mapping")

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if len(labels) < 2:
        raise ValueError("labels must contain at least two subjects")

    feature_sets = []
    for state, values in features_by_state.items():
        features = check_array(values, dtype=float, ensure_min_features=2)
        if features.shape[0] != len(labels):
            raise ValueError(f"features for {state!r} must have one row per subject")
        feature_sets.append((str(state), features))

    if subject_ids is not None:
        subject_ids = np.asarray(subject_ids)
        if subject_ids.shape != labels.shape:
            raise ValueError("subject_ids must contain one entry per subject")
    return feature_sets, labels, subject_ids


def _class_colors(labels):
    classes = list(dict.fromkeys(labels.tolist()))
    palette = plt.get_cmap("tab10")
    return classes, {
        class_label: palette(index % 10) for index, class_label in enumerate(classes)
    }


def _pca_embedding(features):
    standardized = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(standardized)
    explained = 100 * np.nan_to_num(pca.explained_variance_ratio_)
    return embedding, explained


def _annotate(axis, coordinates, subject_ids):
    if subject_ids is None:
        return
    for point, subject_id in zip(coordinates, subject_ids, strict=True):
        axis.annotate(
            str(subject_id),
            point,
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )


def _style_axis(axis, title, explained):
    axis.set_title(_display_name(title), loc="left")
    axis.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    axis.grid(alpha=0.25)


def _display_name(value):
    return value.replace("_", " ").title()


def _mapping_items(values, name):
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must be a non-empty mapping")
    return [(str(label), data) for label, data in values.items()]


def _channel_names(names, size):
    if names is None:
        return tuple(f"Channel {index + 1}" for index in range(size))
    names = tuple(str(name) for name in names)
    if len(names) != size:
        raise ValueError(f"channel_names must contain {size} names")
    return names


def _padded_limits(values):
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = 0.05 * span if span > 0 else max(0.05 * abs(lower), 0.5)
    return lower - padding, upper + padding
