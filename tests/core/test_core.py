import numpy as np
import pytest

from lipipy.core import NotFittedError, check_array, check_X_y
from lipipy.core.metrics import accuracy_score, mean_squared_error, r2_score
from lipipy.ml import LinearRegression


def test_check_array_rejects_1d_and_nan():
    with pytest.raises(ValueError, match="2D"):
        check_array([1.0, 2.0])
    with pytest.raises(ValueError, match="NaN"):
        check_array([[1.0, np.nan]])


def test_check_X_y_length_mismatch():
    with pytest.raises(ValueError, match="samples"):
        check_X_y([[1.0], [2.0]], [1.0])


def test_metrics_match_sklearn(rng):
    sk = pytest.importorskip("sklearn.metrics")
    y, p = rng.normal(size=50), rng.normal(size=50)
    assert np.isclose(mean_squared_error(y, p), sk.mean_squared_error(y, p))
    assert np.isclose(r2_score(y, p), sk.r2_score(y, p))
    a, b = rng.integers(0, 3, 50), rng.integers(0, 3, 50)
    assert np.isclose(accuracy_score(a, b), sk.accuracy_score(a, b))


def test_params_and_repr():
    m = LinearRegression(solver="gd")
    assert m.get_params()["solver"] == "gd"
    m.set_params(solver="lstsq")
    assert "solver='lstsq'" in repr(m)
    with pytest.raises(ValueError):
        m.set_params(nope=1)


def test_not_fitted():
    with pytest.raises(NotFittedError):
        LinearRegression().predict([[1.0]])
