"""Build the pipeline."""

import inspect

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._helpers import (
    _declares_param,
    _final_estimator_has,
    _metadata_kwargs,
    _split_input,
    _transform_one,
)


class SubjectPipeline(Pipeline):
    """
    A custom scikit-learn Pipeline.

    Supports subject-level/group-level scoring
    by aggregating window-level or sample-level predictions.

    Parameters
    ----------
        steps (list of tuple) : List of (name, transform) tuples.
        mask (array-like, optional) : Optional mask filtering for a desired state.
    """

    @staticmethod
    def accepts_param(func, param_name):
        """Return whether a callable accepts a named argument or ``**kwargs``."""
        sig = inspect.signature(func)
        params = sig.parameters
        return param_name in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    def __init__(self, steps, mask=None):
        super().__init__(steps)
        self.mask = mask

    def fit(self, X, y=None, **fit_params):
        """
        Fit all the transformers, then fit the final estimator.

        Parameters
        ----------
        X (array) : Training data. (n_subj,n_trials,n_channels)
        y (array) : Training targets. (n_subject,n_trials), optional

        **fit_params (dict) : Parameters passed to the fit method of each step.

        Returns
        -------
        self : This fitted pipeline.
        """
        if len(self.steps) == 0:
            self.is_fitted_ = True
            return self

        Xt, yt, final_fit_params = self._fit(X, y, **fit_params)

        # FIX(ref): Preserve transformed groups for group-aware final estimators.
        final_groups = self.__dict__.pop("_fit_groups_")
        if self._final_estimator not in (None, "passthrough"):
            if _declares_param(self._final_estimator.fit, "groups"):
                final_fit_params.setdefault("groups", final_groups)
            self._final_estimator.fit(Xt, yt, **final_fit_params)

        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """
        Apply transforms sequentially.

        Parameters
        ----------
        X (array) : Data to transform. (n_subj,n_trials,n_channels)

        Returns
        -------
        Xt (array) : Transformed data. (n_subj,n_trials,n_channels)
        """
        check_is_fitted(self, "is_fitted_")

        if len(self.steps) == 0:
            return X

        Xt = X
        for _, _, transform in self._iter(with_final=False):
            if transform in (None, "passthrough"):
                continue

            if self.accepts_param(transform.transform, "groups"):
                Xt = transform.transform(Xt, y, groups=self.mask)
            else:
                Xt = transform.transform(Xt, y)

        final = self._final_estimator
        if final not in (None, "passthrough") and hasattr(final, "transform"):
            if self.accepts_param(final.transform, "groups"):
                Xt = final.transform(Xt, y, groups=self.mask)
            else:
                Xt = final.transform(Xt, y)
        return Xt

    def predict(self, X, **predict_params):
        """
        Transform the input data and make predictions using the final estimator.

        Parameters
        ----------
        X (array) : Data to predict. (n_subj,n_trials,n_channels)

        **predict_params (dict) : Parameters for prediction step.

        Returns
        -------
        Call of predict method of final estimator.

        """
        check_is_fitted(self, "is_fitted_")

        if len(self.steps) == 0 or self._final_estimator in (None, "passthrough"):
            raise AttributeError("The final step does not implement predict().")

        Xt = X
        for _, _, transform in self._iter(with_final=False):
            if transform in (None, "passthrough"):
                continue
            Xt = transform.transform(Xt)

        return self._final_estimator.predict(Xt, **predict_params)

        # FIX(ref): Override sklearn's incompatible private ``_fit`` invocation.

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the pipeline and transform the original input."""
        return self.fit(X, y, **fit_params).transform(X, y)

        # FIX(ref): Use the local metadata flow and route groups against
        # ``fit_predict`` itself rather than the final estimator's ``fit`` method.

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        """Fit all steps and return the final estimator's training predictions."""
        Xt, yt, final_fit_params = self._fit(X, y, **fit_params)
        final_groups = self.__dict__.pop("_fit_groups_")
        if _declares_param(self._final_estimator.fit_predict, "groups"):
            final_fit_params.setdefault("groups", final_groups)
        predictions = self._final_estimator.fit_predict(
            Xt,
            yt,
            **final_fit_params,
        )
        self.is_fitted_ = True
        return predictions

    # FIX(ref): Restore sklearn's conditional probability and decision methods.
    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_params):
        """Predict class probabilities while preserving transformed metadata."""
        return self._call_final("predict_proba", X, predict_params)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **predict_params):
        """Predict log-probabilities while preserving transformed metadata."""
        return self._call_final("predict_log_proba", X, predict_params)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X, **predict_params):
        """Compute decision scores while preserving transformed metadata."""
        return self._call_final("decision_function", X, predict_params)

    def _fit(self, X, y=None, **fit_params):
        """
        Fit the pipeline except the last step.

        Difference to the sci-kit learn Pipeline class method _fit is that the mask
        attribute is potentially manipulated by the transformer and, in case, updated.

        Parameters
        ----------
        X (array) : Training data. (n_subj,n_trials,n_channels)
        y (array) : Training targets. (n_subject,n_trials), optional

        **fit_params (dict) : Parameters passed to the fit method of each step.

        Returns
        -------
        Xt, yt, final_fit_params (tuple) : Transformed data and final fit parameters
        for classification step.
        """
        self.steps = list(self.steps)

        if len(self.steps) == 0:
            return X, y, {}

        self._validate_steps()

        Xt, yt = X, y

        fit_params_steps = {
            name: {} for name, step in self.steps if step not in (None, "passthrough")
        }

        for pname, pval in fit_params.items():
            if "__" not in pname:
                raise ValueError(
                    f"Fit parameters must use the step__param format, got {pname!r}."
                )
            step, param = pname.split("__", 1)
            if step not in fit_params_steps:
                raise ValueError(f"Unknown step name in fit parameters: {step!r}")
            fit_params_steps[step][param] = pval

        for _, name, transformer in self._iter(with_final=False):
            if transformer in (None, "passthrough"):
                continue
            step_params = fit_params_steps.get(name, {})

            # FIX(ref): Route metadata separately to ``fit`` and ``transform``.
            # FIX(ref): Do not pass stale subject targets into row-level fits.
            fit_kwargs = _metadata_kwargs(transformer.fit, Xt, yt, self.mask)
            fit_kwargs.update(step_params)
            transformer.fit(Xt, **fit_kwargs)

            if self.accepts_param(transformer.fit_transform, "groups"):
                result = transformer.fit_transform(
                    Xt, yt, groups=self.mask, **step_params
                )
            else:
                result = transformer.fit_transform(Xt, yt, **step_params)

            if isinstance(result, tuple):
                Xt, self.mask = result
            else:
                Xt = result

        # Keep the final metadata local to this fit call.
        self._fit_groups_ = self.mask
        final_name = self.steps[-1][0]
        final_fit_params = fit_params_steps.get(final_name, {})

        return Xt, yt, final_fit_params

    def _transform_before_final(self, X, y):
        # FIX(ref): Replay metadata changes during inference instead of reusing
        # groups produced by the preceding fit call.
        Xt, mask = _split_input(X, self.mask)
        for _, _, transform in self._iter(with_final=False):
            Xt, mask = _transform_one(transform, Xt, y, mask)
        return Xt, mask

    def _call_final(self, method_name, X, method_params):
        check_is_fitted(self, "is_fitted_")
        if len(self.steps) == 0 or self._final_estimator in (None, "passthrough"):
            raise AttributeError(f"The final step does not implement {method_name}().")

        Xt, mask = self._transform_before_final(X, None)
        method = getattr(self._final_estimator, method_name)
        if _declares_param(method, "groups"):
            method_params.setdefault("groups", mask)
        return method(Xt, **method_params)
