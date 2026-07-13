"""Tests for subject-level feature plots."""

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from pcp_project.plotting import (  # noqa: E402
    plot_covariance_ellipses,
    plot_covariance_heatmaps,
    plot_covariance_space,
    plot_feature_clusters,
    plot_mdm_distances,
    plot_state_trajectories,
)


@pytest.fixture
def plot_data():
    rng = np.random.default_rng(4)
    features = {
        "eyes_open": rng.normal(size=(8, 4)),
        "eyes_closed": rng.normal(size=(8, 4)),
    }
    labels = np.array(["control", "patient"] * 4)
    subject_ids = np.array([f"s{index}" for index in range(8)])
    return features, labels, subject_ids


def test_feature_plots(plot_data):
    features, labels, subject_ids = plot_data
    clusters = plot_feature_clusters(features, labels, subject_ids=subject_ids)
    trajectories = plot_state_trajectories(features, labels, subject_ids=subject_ids)

    assert isinstance(clusters, Figure) and isinstance(trajectories, Figure)
    assert len(clusters.axes) == 2
    assert "PC1" in clusters.axes[0].get_xlabel()
    assert {text.get_text() for text in clusters.axes[0].texts} == set(subject_ids)
    assert {text.get_text() for text in clusters.axes[0].get_legend().texts} == {
        "control",
        "patient",
    }
    assert len(trajectories.axes[0].lines) == len(labels)
    assert len(trajectories.axes[0].get_legend().texts) == 4
    assert {text.get_text() for text in trajectories.axes[0].texts} == set(subject_ids)
    assert not plot_feature_clusters(features, labels).axes[0].texts
    plt.close("all")


def test_feature_plot_validation(plot_data):
    invalid_inputs = [
        ({}, [0, 1], None),
        ({"open": np.ones((2, 2))}, [[0, 1]], None),
        ({"open": np.ones((1, 2))}, [0], None),
        ({"open": np.ones((3, 2))}, [0, 1], None),
        ({"open": np.ones((2, 2))}, [0, 1], [["s0", "s1"]]),
    ]
    for features, labels, subject_ids in invalid_inputs:
        with pytest.raises(ValueError):
            plot_feature_clusters(features, labels, subject_ids=subject_ids)

    features, labels, _ = plot_data
    with pytest.raises(ValueError, match="exactly two states"):
        plot_state_trajectories({"open": features["eyes_open"]}, labels)

    features["eyes_closed"] = np.ones((len(labels), 3))
    with pytest.raises(ValueError, match="same columns"):
        plot_state_trajectories(features, labels)


@pytest.fixture
def covariance_plot_data():
    rng = np.random.default_rng(12)
    open_window = rng.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[2.0, 0.8], [0.8, 1.0]],
        size=80,
    ).T
    closed_window = rng.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[1.0, -0.4], [-0.4, 1.6]],
        size=80,
    ).T
    covariance_batches = {
        "eyes_open": np.array(
            [
                [[2.0, 0.8], [0.8, 1.0]],
                [[1.8, 0.5], [0.5, 1.2]],
            ]
        ),
        "eyes_closed": np.array(
            [
                [[1.0, -0.4], [-0.4, 1.6]],
                [[1.2, -0.2], [-0.2, 1.4]],
            ]
        ),
    }
    return {
        "windows": {"eyes_open": open_window, "eyes_closed": closed_window},
        "batches": covariance_batches,
    }


def test_covariance_teaching_plots(covariance_plot_data):
    windows = covariance_plot_data["windows"]
    batches = covariance_plot_data["batches"]

    ellipses = plot_covariance_ellipses(
        windows,
        channel_names=("F1 amplitude", "F2 amplitude"),
    )
    heatmaps = plot_covariance_heatmaps(
        {state: matrices.mean(axis=0) for state, matrices in batches.items()}
    )
    space = plot_covariance_space(batches, channel_names=("F1", "F2"))

    assert isinstance(ellipses, Figure)
    assert len(ellipses.axes) == 2
    assert all(
        sum(isinstance(patch, Ellipse) for patch in axis.patches) == 1
        for axis in ellipses.axes
    )
    assert ellipses.axes[0].get_xlabel() == "F1 amplitude"

    assert isinstance(heatmaps, Figure)
    assert len(heatmaps.axes) == 3
    assert all(len(axis.images) == 1 for axis in heatmaps.axes[:2])
    assert (
        heatmaps.axes[0].images[0].get_clim() == heatmaps.axes[1].images[0].get_clim()
    )
    assert heatmaps.axes[0].get_xticklabels()[0].get_text() == "Channel 1"

    assert isinstance(space, Figure)
    assert len(space.axes) == 2
    assert len(space.axes[0].collections) == 2
    assert space.axes[0].get_legend().get_title().get_text() == "State"
    plt.close("all")


def test_covariance_plot_edge_cases():
    constant = {"constant": np.ones((2, 4))}
    zero = {"zero": np.zeros((2, 2))}
    diagonal = {"diagonal": np.repeat(np.eye(2)[None, :, :], 2, axis=0)}
    large = {"large": np.eye(13)}

    assert isinstance(plot_covariance_ellipses(constant), Figure)
    assert plot_covariance_heatmaps(zero).axes[0].images[0].get_clim() == (-1.0, 1.0)
    assert len(plot_covariance_heatmaps(large).axes[0].get_xticks()) == 9
    assert isinstance(plot_covariance_space(diagonal), Figure)
    plt.close("all")


def test_covariance_ellipse_validation():
    valid_windows = {"open": np.ones((2, 3))}
    for n_std in ["wide", 0, np.inf]:
        with pytest.raises(ValueError, match="n_std"):
            plot_covariance_ellipses(valid_windows, n_std=n_std)

    invalid_windows = [
        {},
        {"open": np.ones((3, 4))},
        {"open": np.ones((2, 1))},
        {"open": np.array([[1.0, np.nan], [1.0, 2.0]])},
    ]
    for windows in invalid_windows:
        with pytest.raises(ValueError):
            plot_covariance_ellipses(windows)

        if windows:
            with pytest.raises(ValueError, match="channel_names"):
                plot_covariance_ellipses(windows, channel_names=("only one",))


def test_covariance_heatmap_validation():
    invalid_matrices = [
        {"open": np.ones(2)},
        {"open": np.ones((2, 3))},
        {"open": np.empty((0, 0))},
        {"open": np.array([[1.0, np.nan], [np.nan, 1.0]])},
        {"open": np.array([[1.0, 0.2], [0.1, 1.0]])},
    ]
    for matrices in invalid_matrices:
        with pytest.raises(ValueError, match="finite and symmetric"):
            plot_covariance_heatmaps(matrices)

    with pytest.raises(ValueError, match="same shape"):
        plot_covariance_heatmaps({"open": np.eye(2), "closed": np.eye(3)})
    with pytest.raises(ValueError, match="channel_names"):
        plot_covariance_heatmaps({"open": np.eye(2)}, channel_names=("one",))


def test_covariance_space_validation():
    invalid_batches = [
        {"open": np.ones((2, 2))},
        {"open": np.empty((0, 2, 2))},
        {"open": np.ones((2, 3, 3))},
        {"open": np.array([[[1.0, np.nan], [np.nan, 1.0]]])},
        {"open": np.array([[[1.0, 0.2], [0.1, 1.0]]])},
        {"open": np.array([[[1.0, 2.0], [2.0, 1.0]]])},
    ]
    for batches in invalid_batches:
        with pytest.raises(ValueError, match="SPD shape"):
            plot_covariance_space(batches)

    with pytest.raises(ValueError, match="non-empty mapping"):
        plot_covariance_space([])
    with pytest.raises(ValueError, match="channel_names"):
        plot_covariance_space(
            {"open": np.repeat(np.eye(2)[None, :, :], 2, axis=0)},
            channel_names=("one",),
        )


def test_mdm_distance_plot_marks_errors():
    distances = np.array([[1.0, 2.0], [2.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    labels = np.array([0, 1, 0, 1])
    figure = plot_mdm_distances(
        distances,
        labels,
        classes=np.array([0, 1]),
        class_names=("No diagnosis", "Diagnosis"),
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes[0].collections) == 3
    assert len(figure.axes[0].lines) == 1
    assert figure.axes[0].get_xlabel() == "Distance to No diagnosis mean"
    assert "Incorrect" in {
        text.get_text() for text in figure.axes[0].get_legend().get_texts()
    }
    plt.close("all")


def test_mdm_distance_plot_without_errors_or_names():
    figure = plot_mdm_distances(
        np.zeros((2, 2)),
        np.zeros(2, dtype=int),
        classes=[0, 1],
    )
    assert len(figure.axes[0].collections) == 2
    assert figure.axes[0].get_xlim() == (0.0, 1.0)
    plt.close("all")


def test_mdm_distance_validation():
    invalid_inputs = [
        (np.ones(2), [0, 1], [0, 1], None, "distances"),
        (np.ones((0, 2)), [], [0, 1], None, "distances"),
        (np.ones((2, 3)), [0, 1], [0, 1], None, "distances"),
        (np.array([[1.0, np.nan]]), [0], [0, 1], None, "distances"),
        (np.array([[-1.0, 1.0]]), [0], [0, 1], None, "distances"),
        (np.ones((2, 2)), [[0, 1]], [0, 1], None, "labels"),
        (np.ones((2, 2)), [0], [0, 1], None, "labels"),
        (np.ones((2, 2)), [0, 1], [0], None, "classes"),
        (np.ones((2, 2)), [0, 1], [0, 0], None, "classes"),
        (np.ones((2, 2)), [0, 2], [0, 1], None, "labels"),
        (np.ones((2, 2)), [0, 1], [0, 1], ["one"], "class_names"),
    ]
    for distances, labels, classes, class_names, message in invalid_inputs:
        with pytest.raises(ValueError, match=message):
            plot_mdm_distances(
                distances,
                labels,
                classes=classes,
                class_names=class_names,
            )
