"""Tests for the small dataset helpers."""

import numpy as np
import pytest

from pcp_project.data import (
    LABEL_COLUMNS,
    SUBJECT_ID_COLUMN,
    balanced_subject_ids,
    binary_target,
    list_subject_ids,
    load_labels,
    load_subject,
)


def test_labels_and_targets(synthetic_data_dir):
    assert list_subject_ids(synthetic_data_dir) == [
        "AD_001",
        "AD_002",
        "AD_003",
        "AD_004",
    ]

    labels = load_labels(synthetic_data_dir / "labels_reduced.csv")
    assert labels.index.name == SUBJECT_ID_COLUMN

    subject_ids = ["AD_004", "AD_001", "AD_002"]
    assert binary_target(labels, subject_ids).tolist() == [0, 1, 0]
    assert binary_target(labels, subject_ids, LABEL_COLUMNS[0]).tolist() == [0, 1, 0]

    recording, states = load_subject("AD_001", synthetic_data_dir)
    assert recording.dtype == np.float64

    with np.load(synthetic_data_dir / "AD_001.npz") as npz:
        np.testing.assert_array_equal(recording, npz["X"].T)
        np.testing.assert_array_equal(states, npz["y"].reshape(-1))


def test_balanced_subject_ids(synthetic_data_dir):
    np.savez(
        synthetic_data_dir / "UNLABELED.npz",
        X=np.zeros((10, 4)),
        y=np.zeros((1, 10), dtype=int),
    )

    assert balanced_subject_ids(synthetic_data_dir, 4) == [
        "AD_001",
        "AD_002",
        "AD_003",
        "AD_004",
    ]
    assert set(balanced_subject_ids(synthetic_data_dir, 2)) == {"AD_001", "AD_002"}

    selected = balanced_subject_ids(synthetic_data_dir, 4, random_state=0)
    assert set(selected) == {"AD_001", "AD_002", "AD_003", "AD_004"}

    with pytest.raises(ValueError, match="even"):
        balanced_subject_ids(synthetic_data_dir, 3)
    with pytest.raises(ValueError, match="positive"):
        balanced_subject_ids(synthetic_data_dir, 0)
    with pytest.raises(ValueError, match="each class"):
        balanced_subject_ids(synthetic_data_dir, 10)
