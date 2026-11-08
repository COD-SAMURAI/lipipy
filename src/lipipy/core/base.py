"""Base classes for estimators.

lipipy follows the familiar scikit-learn conventions so users can switch between
the two easily:

* hyperparameters are set in ``__init__`` and never changed by ``fit``;
* learned attributes end with a trailing underscore (``coef_``);
* ``fit`` returns ``self``.
"""

from __future__ import annotations

import inspect

import numpy as np


class NotFittedError(RuntimeError):
    """Raised when an estimator is used before ``fit`` has been called."""


class BaseEstimator:
    """Minimal estimator base: parameter introspection and a readable repr."""

    @classmethod
    def _param_names(cls) -> list[str]:
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters if p != "self"]

    def get_params(self) -> dict:
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params) -> BaseEstimator:
        valid = set(self._param_names())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for {type(self).__name__}")
            setattr(self, key, value)
        return self

    def _check_is_fitted(self, attribute: str) -> None:
        if not hasattr(self, attribute):
            raise NotFittedError(
                f"{type(self).__name__} is not fitted yet. Call 'fit' before using it."
            )

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class RegressorMixin:
    """Adds an R^2 ``score`` method to regressors."""

    def score(self, X, y) -> float:
        from lipipy.core.metrics import r2_score

        return r2_score(np.asarray(y), self.predict(X))
