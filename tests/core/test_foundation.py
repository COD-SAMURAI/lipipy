"""Tests for the layer-0 services: registry, complexity, backend, random, imports."""

import json

import numpy as np
import pytest

import lipipy
from lipipy.core import (
    Complexity,
    LipipyError,
    MissingDependencyError,
    RegistryError,
    backend_name,
    check_random_state,
    get_namespace,
    is_available,
    register,
    registry,
    require,
    spawn,
    to_numpy,
)
from lipipy.ml import LinearRegression

# -- complexity -----------------------------------------------------------------


def test_complexity_validates_and_formats():
    c = Complexity(time="O(n log n)", space="O(1)", notes="in place")
    assert str(c) == "time O(n log n), space O(1) (in place)"
    assert c.to_dict()["space"] == "O(1)"
    with pytest.raises(ValueError, match="O\\(...\\)"):
        Complexity(time="n^2", space="O(1)")


# -- registry -------------------------------------------------------------------


def test_linear_regression_is_registered():
    info = registry.get("ml.LinearRegression")
    assert info.obj is LinearRegression
    assert LinearRegression.__lipipy_info__ is info
    assert info.status == "stable"
    assert info.complexity.time.startswith("O(")
    assert info.import_path == "lipipy.ml.linear.linear_regression.LinearRegression"
    assert info.summary == "Ordinary least squares linear regression."
    assert "ml.LinearRegression" in registry
    assert len(registry) >= 1


def test_find_filters():
    names = [i.name for i in registry.find(task="regression")]
    assert "ml.LinearRegression" in names
    assert registry.find(domain="ml", text="least squares")
    assert not registry.find(domain="ml", task="no-such-task")
    assert not registry.find(backend="no-such-backend")


def test_to_json_is_machine_readable():
    data = json.loads(registry.to_json(domain="ml"))
    entry = next(d for d in data if d["name"] == "ml.LinearRegression")
    assert set(entry) >= {"summary", "complexity", "references", "import_path", "tasks"}


def test_register_rejects_duplicates_and_bad_metadata():
    c = Complexity(time="O(n)", space="O(1)")

    @register(domain="test", complexity=c, name="test.dup")
    def first(x):
        """First."""
        return x

    try:
        with pytest.raises(RegistryError, match="already registered"):

            @register(domain="test", complexity=c, name="test.dup")
            def second(x):
                return x
    finally:
        registry._entries.pop("test.dup")

    with pytest.raises(ValueError, match="status"):
        register(domain="test", complexity=c, status="beta")
    with pytest.raises(TypeError):
        register(domain="test", complexity="O(n)")


def test_plugins_can_be_disabled(monkeypatch):
    import sys

    # `lipipy.core.registry` is the Registry instance; the module lives in sys.modules
    registry_module = sys.modules["lipipy.core.registry"]

    loaded = []

    class FakeEntryPoint:
        def load(self):
            loaded.append(True)

    monkeypatch.setattr(registry_module, "entry_points", lambda group: [FakeEntryPoint()])
    fresh = registry_module.Registry()
    monkeypatch.setenv("LIPIPY_DISABLE_PLUGINS", "1")
    fresh.discover()
    assert loaded == []

    fresh = registry_module.Registry()
    monkeypatch.delenv("LIPIPY_DISABLE_PLUGINS")
    fresh.discover()
    assert loaded == [True]


def test_unknown_name_raises_registry_error():
    with pytest.raises(RegistryError, match="No algorithm named"):
        registry.get("ml.DoesNotExist")
    assert issubclass(RegistryError, LipipyError)


def test_top_level_registry_alias():
    assert lipipy.registry is registry


# -- random ---------------------------------------------------------------------


def test_check_random_state():
    assert np.allclose(check_random_state(1).random(3), check_random_state(1).random(3))
    g = np.random.default_rng(0)
    assert check_random_state(g) is g
    assert isinstance(check_random_state(None), np.random.Generator)
    assert isinstance(check_random_state(np.random.SeedSequence(3)), np.random.Generator)
    for bad in (1.5, "1", True):
        with pytest.raises(TypeError):
            check_random_state(bad)


def test_spawn_is_reproducible_and_independent():
    a = [g.random() for g in spawn(7, 4)]
    b = [g.random() for g in spawn(7, 4)]
    assert a == b
    assert len(set(a)) == 4
    with pytest.raises(ValueError):
        spawn(0, -1)


# -- backend --------------------------------------------------------------------


def test_numpy_namespace():
    xp = get_namespace(np.ones(2), [1.0, 2.0], None)
    assert backend_name(xp) == "numpy"
    assert float(xp.sum(xp.asarray([1.0, 2.0]))) == 3.0
    assert isinstance(to_numpy([1, 2]), np.ndarray)


def test_foreign_array_without_compat(monkeypatch):
    import lipipy.core.backend as backend

    class FakeArray:
        pass

    monkeypatch.setattr(backend, "_HAS_COMPAT", False)
    with pytest.raises(MissingDependencyError, match="lipipy\\[backends\\]"):
        backend.get_namespace(FakeArray())


@pytest.mark.skipif(not is_available("torch"), reason="torch not installed")
def test_torch_namespace():  # pragma: no cover - runs only with torch
    import torch

    xp = get_namespace(torch.ones(2))
    assert backend_name(xp) == "torch"


# -- optional imports -----------------------------------------------------------


def test_require_gives_install_hint():
    assert require("numpy", extra="core") is np
    with pytest.raises(MissingDependencyError, match="pip install 'lipipy\\[quantum\\]'"):
        require("definitely_not_a_module_xyz", extra="quantum", purpose="parity tests")
    assert not is_available("definitely_not_a_module_xyz")
