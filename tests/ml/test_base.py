"""ml._base mixins and the public surface of lipipy.ml."""

import numpy as np
import pytest

import lipipy.ml as ml
from lipipy.core import registry
from lipipy.ml import ClassifierMixin, TransformerMixin


def test_classifier_mixin_score_is_accuracy():
    class Echo(ClassifierMixin):
        def predict(self, X):
            return np.asarray(X)[:, 0].astype(int)

    assert Echo().score([[1.0], [0.0], [1.0], [1.0]], [1, 1, 1, 0]) == 0.5


def test_transformer_mixin_fit_transform_calls_fit_then_transform():
    calls = []

    class T(TransformerMixin):
        def fit(self, X, y=None):
            calls.append(("fit", y))
            return self

        def transform(self, X):
            calls.append(("transform", None))
            return np.asarray(X) * 2

    out = T().fit_transform([[1.0]], y="target")
    assert out.tolist() == [[2.0]]
    assert calls == [("fit", "target"), ("transform", None)]


@pytest.mark.parametrize(
    "name",
    [
        "LinearRegression",
        "Ridge",
        "KNeighborsClassifier",
        "KNeighborsRegressor",
        "KMeans",
        "PCA",
        "GaussianNB",
    ],
)
def test_estimators_exported_registered_and_documented(name):
    cls = getattr(ml, name)
    assert name in ml.__all__
    info = registry.get(f"ml.{name}")
    assert info.obj is cls and info.domain == "ml"
    assert info.complexity.time.startswith("O(") and info.references
    for section in ("Parameters", "Complexity", "Examples"):
        assert section in cls.__doc__, f"{name} docstring lacks {section}"


def test_get_set_params_roundtrip():
    km = ml.KMeans(n_clusters=3, random_state=0)
    assert km.get_params()["n_clusters"] == 3
    assert km.set_params(n_clusters=5).n_clusters == 5
    assert "KMeans(n_clusters=5" in repr(km)
