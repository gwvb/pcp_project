"""Load the EEG recordings and diagnosis labels used in the assignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SFREQ = 256.0
SUBJECT_ID_COLUMN = "EEG_ID"

LABEL_COLUMNS = (
    "SCID5_CV_Depression",
    "SCID5_CV_OCD",
    "SCID5_CV_Tic_TrichoDerma_Hoarding",
    "SCID5_CV_SAD",
    "SCID5_CV_PHOB",
    "SCID5_CV_PANIC",
    "SCID5_CV_AGORA",
    "SCID5_CV_GAD",
    "SCID5_CV_PTSD",
    "SCID5_CV_Soma_Health",
    "SCID5_CV_Separation",
    "SCID5_CV_Sleep",
    "SCID5_CV_Bodydysmorphia",
    "SCID5_CV_Eating",
    "SCID5_CV_Anxiety_OCD_etc",
    "SCID5_CV_Eating_Bodydysmorphia",
    "SCID5_CV_ADHD_Explosive",
)


def load_labels(labels_path: str | Path) -> pd.DataFrame:
    """Load the diagnosis table and index it by subject ID.

    Parameters
    ----------
    labels_path : path-like
        Path to ``labels_reduced.csv``.

    Returns
    -------
    pandas.DataFrame
        Label table indexed by the ``EEG_ID`` column.
    """
    return pd.read_csv(labels_path).set_index(SUBJECT_ID_COLUMN)


def list_subject_ids(data_dir: str | Path) -> list[str]:
    """Return the sorted subject IDs that have an ``.npz`` recording file.

    Parameters
    ----------
    data_dir : path-like
        Directory containing the recording files.

    Returns
    -------
    list of str
        File stems in alphabetical order.
    """
    return sorted(path.stem for path in Path(data_dir).glob("*.npz"))


def binary_target(
    labels: pd.DataFrame,
    subject_ids: list[str],
    diagnosis: str | None = None,
) -> np.ndarray:
    """Build a binary target in the same order as ``subject_ids``.

    Parameters
    ----------
    labels : pandas.DataFrame
        Label table returned by :func:`load_labels`.
    subject_ids : list of str
        Subjects to include, in the required output order.
    diagnosis : str or None, optional
        Label column to predict. If omitted, a subject is positive when any of
        the 17 diagnosis columns is positive.

    Returns
    -------
    numpy.ndarray of shape (n_subjects,)
        Integer zeros and ones.
    """
    selected = labels.loc[subject_ids]
    if diagnosis is None:
        target = selected.loc[:, LABEL_COLUMNS].sum(axis=1) > 0
    else:
        target = selected.loc[:, diagnosis]
    return target.astype(int).to_numpy()


def balanced_subject_ids(
    data_dir: str | Path,
    n_subjects: int,
    *,
    diagnosis: str | None = None,
    random_state: int | None = None,
) -> list[str]:
    """Choose equally many positive and negative subjects.

    Parameters
    ----------
    data_dir : path-like
        Directory containing the recording files and ``labels_reduced.csv``.
    n_subjects : int
        Total number of subjects to return. It must be positive and even.
    diagnosis : str or None, optional
        Label column used to define the positive class. If omitted, any recorded
        diagnosis counts as positive.
    random_state : int or None, optional
        Seed for random sampling. If omitted, the first sorted subjects in each
        class are used.

    Returns
    -------
    list of str
        Selected IDs in sorted order.
    """
    if n_subjects <= 0:
        raise ValueError(f"n_subjects must be positive, got {n_subjects}")
    if n_subjects % 2:
        raise ValueError(
            f"n_subjects must be even for a balanced split, got {n_subjects}"
        )

    data_dir = Path(data_dir)
    labels = load_labels(data_dir / "labels_reduced.csv")
    available = [
        subject_id
        for subject_id in list_subject_ids(data_dir)
        if subject_id in labels.index
    ]
    target = binary_target(labels, available, diagnosis)

    positive = [
        subject_id
        for subject_id, value in zip(available, target, strict=True)
        if value == 1
    ]
    negative = [
        subject_id
        for subject_id, value in zip(available, target, strict=True)
        if value == 0
    ]
    n_each = n_subjects // 2

    if len(positive) < n_each or len(negative) < n_each:
        raise ValueError(
            f"need at least {n_each} subjects in each class, "
            f"got {len(positive)} disorder and {len(negative)} no-disorder"
        )

    if random_state is None:
        chosen = positive[:n_each] + negative[:n_each]
    else:
        rng = np.random.default_rng(random_state)
        chosen = [
            *rng.choice(positive, size=n_each, replace=False),
            *rng.choice(negative, size=n_each, replace=False),
        ]
    return sorted(str(subject_id) for subject_id in chosen)


def load_subject(
    subject_id: str,
    data_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one full-resolution recording and its sample states.

    Parameters
    ----------
    subject_id : str
        File stem of the subject recording.
    data_dir : path-like
        Directory containing ``<subject_id>.npz``.

    Returns
    -------
    recording : numpy.ndarray of shape (n_channels, n_samples)
        Channels-first floating-point EEG data.
    states : numpy.ndarray of shape (n_samples,)
        Sample-level recording states.
    """
    with np.load(Path(data_dir) / f"{subject_id}.npz") as npz:
        recording = np.asarray(npz["X"], dtype=float).T
        states = np.asarray(npz["y"]).reshape(-1)
    return recording, states
