Getting started
===============

Installation
------------

The project requires Python 3.12 or newer.  Create the full environment with
`uv`_:

.. code-block:: console

   uv sync --all-extras --group dev --group docs

.. _uv: https://docs.astral.sh/uv/

Data files
----------

The EEG data is not stored in the repository.  Put ``labels_reduced.csv`` and
the subject ``.npz`` files in ``data/data-20260528``, or set
``PCP_DATA_DIR`` to another directory:

.. code-block:: console

   export PCP_DATA_DIR=/path/to/data-20260528

The examples use only subjects that have both a recording file and a row in the
label table.

Input shape
-----------

One subject is represented by ``(recording, sample_states)``:

* ``recording`` has shape ``(n_channels, n_samples)``;
* ``sample_states`` has shape ``(n_samples,)``;
* state ``0`` means eyes open and state ``1`` means eyes closed;
* ``X`` is a list with one recording pair per subject;
* ``y`` contains one binary or multi-label target per subject; and
* ``groups`` contains one unique subject ID per subject.

The data helpers load that representation and create a binary target:

.. code-block:: python

   from pcp_project.data import (
       binary_target,
       list_subject_ids,
       load_labels,
       load_subject,
   )

   data_dir = "data/data-20260528"
   labels = load_labels(f"{data_dir}/labels_reduced.csv")
   subject_ids = [
       subject_id
       for subject_id in list_subject_ids(data_dir)
       if subject_id in labels.index
   ]

   X = [load_subject(subject_id, data_dir) for subject_id in subject_ids]
   y = binary_target(labels, subject_ids, diagnosis="SCID5_CV_SAD")
   groups = subject_ids

Evaluation
----------

Build the pipeline and create reusable subject-level folds:

.. code-block:: python

   from pcp_project.classify import (
       build_classification_pipeline,
       cross_validate_subjects,
       make_subject_folds,
   )

   pipeline = build_classification_pipeline(
       window_seconds=2.0,
       covariance="oas",
       aggregation="mean",
       random_state=7,
   )
   folds = make_subject_folds(y, groups, n_splits=3, random_state=7)
   results = cross_validate_subjects(
       pipeline,
       X,
       y,
       groups,
       cv=folds,
       scoring="balanced_accuracy",
   )

Reuse the same ``folds`` when comparing models.  That way each model is tested
on the same subjects.  When selecting parameters, use nested subject folds:

.. code-block:: python

   from pcp_project.classify import nested_search_subjects

   search_results = nested_search_subjects(
       pipeline,
       {"classifier__C": [0.01, 0.1, 1.0]},
       X,
       y,
       groups,
       cv=folds,
       inner_splits=3,
   )

The outer test rows are used only for ``search_results["test_score"]``.  For a
two-dimensional target, both evaluation helpers default to mean per-label
balanced accuracy.

Project checks
--------------

.. code-block:: console

   uv run ruff format --check .
   uv run ruff check .
   uv run pytest --cov=pcp_project --cov-report=term-missing --cov-fail-under=100
   uv run sphinx-build -W --keep-going -b html docs docs/_build/html
