"""Evaluation metrics."""

from __future__ import annotations

import numpy as np


def mean_squared_error(y_true, y_pred) -> float:
    r"""Mean squared error: :math:`\frac{1}{n}\sum_i (y_i - \hat y_i)^2`."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred) -> float:
    r"""Coefficient of determination: :math:`1 - SS_{res} / SS_{tot}`."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def accuracy_score(y_true, y_pred) -> float:
    """Fraction of predictions that exactly match the labels."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
