import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import inspect


class SubjectPipeline(Pipeline):
    """
    A custom scikit-learn Pipeline that supports subject-level/group-level scoring
    by aggregating window-level or sample-level predictions.
    """

    def __init__(self, steps, mask=None):
        """
        Parameters
        steps (list of tuple) : List of (name, transform) tuples.
        mask (array-like, optional) : Optional mask filtering for a desired state.
        """
        super().__init__(steps)
        self.mask = mask

    def fit(self, X, y=None, **fit_params):
        """
        Fit all the transformers, then fit the final estimator.

        Parameters
        X (array) : Training data. (n_subj,n_trials,n_channels)
        y (array) : Training targets. (n_subject,n_trials), optional

        **fit_params (dict) : Parameters passed to the fit method of each step.

        Returns
        self : This fitted pipeline.
        """
        if len(self.steps) == 0:
            self.is_fitted_ = True
            return self

        Xt, yt, final_fit_params = self._fit(X, y, **fit_params)

        if self._final_estimator not in (None, "passthrough"):
            self._final_estimator.fit(Xt, yt, **final_fit_params)

        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """
        Apply transforms sequentially.

        Parameters
        X (array) : Data to transform. (n_subj,n_trials,n_channels)

        Returns
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
        X (array) : Data to predict. (n_subj,n_trials,n_channels)

        **predict_params (dict) : Parameters for prediction step.

        Returns
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


    def _fit(self, X, y=None, **fit_params):
        """Fit the pipeline except the last step. Difference to the sci-kit learn Pipeline class method _fit is that
        the mask attribute is potentially manipulated by the transformer and, in case, updated.

        Parameters
        X (array) : Training data. (n_subj,n_trials,n_channels)
        y (array) : Training targets. (n_subject,n_trials), optional

        **fit_params (dict) : Parameters passed to the fit method of each step.

        Returns
        Xt, yt, final_fit_params (tuple) : Transformed data and final fit parameters for classification step.
        """
        self.steps = list(self.steps)

        if len(self.steps) == 0:
            return X, y, {}

        self._validate_steps()

        Xt, yt = X, y

        fit_params_steps = {
            name: {}
            for name, step in self.steps
            if step not in (None, "passthrough")
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
            if self.accepts_param(transformer.fit_transform, "groups"):
                result = transformer.fit_transform(Xt, yt, groups=self.mask, **step_params)
            else:
                result = transformer.fit_transform(Xt, yt, **step_params)

            if isinstance(result, tuple):
                Xt, self.mask = result
            else:
                Xt = result

        final_name = self.steps[-1][0]
        final_fit_params = fit_params_steps.get(final_name, {})

        return Xt, yt, final_fit_params

    @staticmethod
    def accepts_param(func, param_name):
        """ Checks if a function takes a certain parameter.

        Parameters
        func (function) : Function to inspect.
        param_name (str) : Parameter name to check.

        Returns
        (bool) : If parameter in function.
        """
        sig = inspect.signature(func)
        params = sig.parameters
        return (
                param_name in params
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        )