Classification workflow
=======================

Pipeline stages
---------------

The default classifier built by
:func:`pcp_project.classify.build_classification_pipeline` applies these steps:

#. band-pass filter each subject's recording;
#. keep the requested eyes-open or eyes-closed runs;
#. make fixed-length windows inside each run;
#. estimate one covariance matrix per window;
#. map each covariance matrix to tangent-space features;
#. average the window features for each subject;
#. standardize the subject feature rows; and
#. fit logistic regression.

All of these steps stay inside one estimator.  Cross-validation therefore
splits the raw subject rows first and fits each learned step on the training
subjects only.

Windows
-------

``window_seconds`` sets the window length.  ``step_seconds`` sets the distance
between the start of two neighboring windows.  If ``step_seconds`` is omitted,
the windows are adjacent and do not overlap.

Windows are made separately inside each contiguous state run.  A window never
crosses from eyes open to eyes closed.  The classification builder uses the
``valid`` padding policy, so a short remainder at the end of a run is skipped.

Combining windows
-----------------

:class:`pcp_project.classify.FeatureConsolidator` changes the data from many
window rows to one row per subject.  It supports four strategies:

``mean``
   Average all windows belonging to the subject.

``mean_of_state_means``
   Average within each available state, then average the state means.  Each
   state gets the same weight even if one state contains more windows.

``concat``
   Concatenate the state means in ``state_order``.  Every subject must contain
   every requested state.

``stack``
   Keep the state means on a separate axis.  With covariance input, one subject
   has shape ``(n_states, n_channels, n_channels)``.

The default path uses ``mean`` after tangent-space mapping.  The notebooks also
use ``stack`` with ``feature_transform="identity"`` to keep covariance matrices
intact.  Those stacked matrices can be passed to either:

* :class:`pcp_project.classify.StateAwareMDM`, which combines one Riemannian MDM
  model per recording state; or
* :class:`pcp_project.classify.StateTangentFeatures`, which makes a separate
  tangent-space map for each state and joins the resulting feature vectors.

For example:

.. code-block:: python

   from pcp_project.classify import StateAwareMDM, build_classification_pipeline

   pipeline = build_classification_pipeline(
       classifier=StateAwareMDM(),
       aggregation="stack",
       feature_transform="identity",
   )

Evaluation notes
----------------

Use :func:`pcp_project.classify.make_subject_folds` and
:func:`pcp_project.classify.cross_validate_subjects` for the examples.  They
check that each input row has a unique subject ID and that no subject appears on
both sides of a fold.  If ``scoring`` is omitted, binary targets use balanced
accuracy.  Two-dimensional targets use balanced accuracy for each label and
then average the label scores, so the documented default also works for
multi-label estimators.

Nested parameter search
-----------------------

Use :func:`pcp_project.classify.nested_search_subjects` when parameters are
selected from the data.  Each outer fold is reserved for evaluation.  A fresh
set of subject-level folds is made from that outer fold's training subjects for
the grid search:

.. code-block:: python

   from pcp_project.classify import nested_search_subjects

   results = nested_search_subjects(
       pipeline,
       {"classifier__C": [0.01, 0.1, 1.0]},
       X,
       y,
       groups,
       cv=folds,
       inner_splits=3,
   )

``results["test_score"]`` contains the outer scores.  The selected parameters,
inner scores, and fitted outer-fold estimators are available as
``best_params``, ``best_inner_score``, and ``estimator``.  Keeping model
selection inside the outer training fold avoids evaluating on the same data
used to choose the model.

For comparisons in this assignment:

* reuse the same saved folds for every model;
* include a simple dummy classifier;
* report each fold as well as the mean.
