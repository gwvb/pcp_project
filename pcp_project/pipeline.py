"""A scikit-learn pipeline that preserves metadata across shape changes."""

import inspect

import numpy as np
from sklearn.metrics import accuracy_score
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


def accepts_param(func, param_name):
    """Return whether a callable accepts a named argument or ``**kwargs``."""
    sig = inspect.signature(func)
    params = sig.parameters
    return param_name in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


class SubjectPipeline(Pipeline):
    """Run transforms that may return ``X`` or ``(X, groups)``."""

    def __init__(self, steps, mask=None):
        super().__init__(steps)
        self.mask = mask

    def fit(self, X, y=None, **fit_params):
        """Fit every pipeline step while preserving aligned metadata."""
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

    def transform(self, X, y=None):
        """Transform input data through every available transformer."""
        check_is_fitted(self, "is_fitted_")

        Xt, mask = self._transform_before_final(X, y)
        final = self._final_estimator
        if final not in (None, "passthrough") and hasattr(final, "transform"):
            Xt, _ = _transform_one(final, Xt, y, mask)

        return Xt

    def predict(self, X, **predict_params):
        """Predict one target per subject using the fitted final estimator."""
        return self._call_final("predict", X, predict_params)

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

    def score(self, X, y=None, sample_weight=None, groups=None):
        """Return accuracy, optionally after grouped majority voting."""
        check_is_fitted(self, "is_fitted_")

        if y is None:
            raise ValueError("Score requires y for supervised evaluation.")

        y_pred = np.asarray(self.predict(X))
        y_true = np.asarray(y)

        # FIX(ref): Make grouped voting reachable and accept either row-level or
        # already-grouped targets and predictions.
        if groups is None:
            if y_pred.shape[0] != y_true.shape[0]:
                raise ValueError(
                    "Prediction and target are on different levels. "
                    "Provide groups for aggregation or align predict(X) with y."
                )
            return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

        groups = np.asarray(groups)

        if len(y_true) == len(groups):
            y_true = self._majority_vote_by_group(y_true, groups)

        if len(y_pred) == len(groups):
            y_pred = self._majority_vote_by_group(y_pred, groups)

        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    def _majority_vote_by_group(self, values, groups):
        values = np.asarray(values)
        groups = np.asarray(groups)

        unique_groups = []
        grouped_values = []

        for g in groups:
            if g not in unique_groups:
                unique_groups.append(g)

        for g in unique_groups:
            vals = values[groups == g]
            vals = vals[~self._is_nan_label_array(vals)]
            if len(vals) == 0:
                raise ValueError(
                    f"Group {g!r} contains no valid labels after filtering."
                )
            grouped_values.append(self._majority_vote(vals))

        return np.asarray(grouped_values)

    @staticmethod
    def _majority_vote(values):
        values = np.asarray(values)
        uniq, counts = np.unique(values, return_counts=True)
        return uniq[np.argmax(counts)]

    @staticmethod
    def _is_nan_label_array(values):
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.floating):
            return np.isnan(values)
        return np.zeros(values.shape, dtype=bool)

    def _fit(self, X, y=None, **fit_params):
        """Fit the pipeline except the last step."""
        self.steps = list(self.steps)

        self._validate_steps()

        # FIX(ref): Keep transformed groups local so repeated calls start cleanly.
        Xt, mask = _split_input(X, self.mask)
        yt = y

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
            step_params = fit_params_steps.get(name, {})

            # FIX(ref): Route metadata separately to ``fit`` and ``transform``.
            # FIX(ref): Do not pass stale subject targets into row-level fits.
            fit_kwargs = _metadata_kwargs(transformer.fit, Xt, yt, mask)
            fit_kwargs.update(step_params)
            transformer.fit(Xt, **fit_kwargs)
            Xt, mask = _transform_one(transformer, Xt, yt, mask)
        final_name = self.steps[-1][0]
        final_fit_params = fit_params_steps.get(final_name, {})
        # Keep the final metadata local to this fit call.
        self._fit_groups_ = mask

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
