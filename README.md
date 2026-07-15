# PCP EEG Pipeline

This repository is a class project about subject-level classification from EEG
recordings. It extends the usual scikit-learn pipeline idea to data that changes
shape during preprocessing:

```text
subjects -> state runs -> windows -> covariance matrices -> features
         -> one row per subject -> classifier
```

The main challenge is keeping subjects, recording states, and labels aligned as
one recording becomes many windows and then one subject-level feature row. The
project is meant to demonstrate that workflow clearly.

## Setup

The project requires Python 3.12 or newer and uses
[`uv`](https://docs.astral.sh/uv/) to manage the environment.

```bash
uv sync --all-extras --group dev --group docs
```

## Data

The EEG data is not included in the repository. By default, the notebooks look
for this directory:

```text
data/data-20260528/
```

You can use another location by setting `PCP_DATA_DIR`:

```bash
export PCP_DATA_DIR=/path/to/data-20260528
```

The directory should contain `labels_reduced.csv` and one `.npz` file per
subject. Each recording file contains:

- `X`: EEG samples stored as `(n_samples, n_channels)`;
- `y`: one recording-state code per sample (`0` for eyes open and `1` for eyes
  closed).

`load_subject` converts a file to the representation used by the pipeline:
`(recording, sample_states)`, where `recording` has shape
`(n_channels, n_samples)`.

## Quick example

`X` contains one recording pair per subject, `y` contains one target per
subject, and `groups` contains one unique subject ID per row.

```python
from pcp_project.classify import (
    build_classification_pipeline,
    make_subject_folds,
    nested_search_subjects,
)

pipeline = build_classification_pipeline(
    window_seconds=2.0,
    covariance="oas",
)
folds = make_subject_folds(y, groups, n_splits=3, random_state=7)
scores = nested_search_subjects(
    pipeline,
    {"classifier__C": [0.01, 0.1, 1.0]},
    X,
    y,
    groups,
    cv=folds,
    inner_splits=3,
)
```

The outer folds provide the reported scores. Parameter selection happens only
within each outer training fold, and raw subjects are split before filtering or
windowing. This keeps all windows from a subject in the same fold.

## Notebooks

The notebooks are numbered in the order we recommend reading them:

1. `0-dataset-exploration.ipynb` introduces the data, windowing, covariance
   matrices, tangent-space features, and MDM classification.
2. `1-logistic-reg-binary-classifier.ipynb` tunes and evaluates a binary
   tangent-space logistic regression pipeline with nested subject folds.
3. `2-mdm-binary-classifier.ipynb` tunes and evaluates the state-aware MDM
   covariance metric with nested subject folds.
4. `3-multilabel-classification.ipynb` predicts several diagnosis labels with
   one-vs-rest logistic regression.

Start Jupyter Lab with:

```bash
uv run jupyter lab
```

The notebooks use the locally available subjects, so exact cohort counts and
scores are reported in the notebooks rather than repeated here.

## Package layout

- `pcp_project.data` loads recordings and creates targets.
- `pcp_project.estimators` contains state selection, filtering, windowing, and
  batched covariance estimators.
- `pcp_project.pipeline` contains the metadata-aware `SubjectPipeline`.
- `pcp_project.classify` contains feature aggregation, classifiers, and
  subject-level cross-validation helpers.
- `pcp_project.plotting` contains the plots used in the teaching notebooks.

The Sphinx documentation in `docs/` gives the data contract, classification
choices, notebook guide, and API reference.

The code is released under the BSD 3-Clause license; see `LICENSE`.

## Checks

Run the same checks as CI:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest --cov=pcp_project --cov-report=term-missing --cov-fail-under=100
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```
