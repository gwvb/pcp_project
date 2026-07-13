Notebook guide
==============

The notebooks are numbered so they can be read in order.

``0-dataset-exploration.ipynb``
   Introduces the label table and EEG recordings, then walks through state
   selection, filtering, windows, covariance matrices, tangent space, and a
   small MDM example.

``1-logistic-reg-binary-classifier.ipynb``
   Selects the tangent metric and logistic regularization strength with nested
   subject folds, then compares held-out scores with a dummy classifier.

``2-mdm-binary-classifier.ipynb``
   Selects the metric for :class:`pcp_project.classify.StateAwareMDM` with
   nested subject folds and plots held-out class distances.

``3-multilabel-classification.ipynb``
   Predicts several diagnosis columns with one-vs-rest logistic regression and
   reports both per-label and combined scores.

The notebooks use ``data/data-20260528`` unless ``PCP_DATA_DIR`` is set.  Start
Jupyter Lab from the repository root:

.. code-block:: console

   uv run jupyter lab

The binary and multi-label notebooks process recordings one at a time before
cross-validation to avoid keeping the full raw dataset in memory.  Only
subject-local, label-free preprocessing is done at that stage.  Tangent-space
maps, scaling, and classifiers are fitted inside the cross-validation folds.

The binary notebook outputs are cleared when their search settings change so
old fixed-model scores are not mistaken for nested-search results.  Run the
notebooks with the local data to produce current values.
