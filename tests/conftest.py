"""Shared synthetic-data fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcp_project.data import LABEL_COLUMNS, SUBJECT_ID_COLUMN

N_CHANNELS = 4
N_SAMPLES = 600


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random generator for reproducible synthetic data."""
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_data_dir(tmp_path, rng):
    """Create four short recordings and an aligned labels CSV."""
    subject_ids = ["AD_001", "AD_002", "AD_003", "AD_004"]
    rows = []
    for i, sid in enumerate(subject_ids):
        X = rng.standard_normal((N_SAMPLES, N_CHANNELS))
        y = np.zeros((1, N_SAMPLES), dtype=np.int64)
        y[0, N_SAMPLES // 2 :] = 1
        np.savez(tmp_path / f"{sid}.npz", X=X, y=y)

        label_values = {col: 0 for col in LABEL_COLUMNS}
        if i % 2 == 0:
            label_values[LABEL_COLUMNS[0]] = 1
        rows.append({SUBJECT_ID_COLUMN: sid, **label_values})

    pd.DataFrame(rows).to_csv(tmp_path / "labels_reduced.csv", index=False)
    return tmp_path
