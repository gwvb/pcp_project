"""End-to-end tests for the public subject-classification workflow."""

import numpy as np

from pcp_project.classify import (
    build_classification_pipeline,
    cross_validate_subjects,
    make_subject_folds,
)
from pcp_project.data import (
    balanced_subject_ids,
    binary_target,
    load_labels,
    load_subject,
)


def test_recordings_can_be_loaded_and_cross_validated(synthetic_data_dir):
    subject_ids = balanced_subject_ids(synthetic_data_dir, n_subjects=4)
    labels = load_labels(synthetic_data_dir / "labels_reduced.csv")
    targets = binary_target(labels, subject_ids)
    recordings = [
        load_subject(subject_id, synthetic_data_dir) for subject_id in subject_ids
    ]

    pipeline = build_classification_pipeline(
        sfreq=32,
        frequency_bands=((4.0, 12.0),),
        window_seconds=2,
    )
    folds = make_subject_folds(targets, subject_ids, n_splits=2, random_state=7)
    scores = cross_validate_subjects(
        pipeline,
        recordings,
        targets,
        subject_ids,
        cv=folds,
        n_jobs=1,
    )

    assert scores["test_score"].shape == (2,)
    assert np.isfinite(scores["test_score"]).all()

    fitted_pipeline = pipeline.fit(recordings, targets)
    assert fitted_pipeline.predict(recordings).shape == targets.shape
