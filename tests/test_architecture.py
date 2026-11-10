"""Enforce the layer rules in ARCHITECTURE.md §3.

A subpackage may import only from subpackages in a strictly lower layer.
Adding a new subpackage requires adding it to LAYERS (and to lipipy.SUBPACKAGES).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import lipipy

SRC = Path(lipipy.__file__).parent

LAYERS = {
    "core": 0,
    "numerics": 1,
    "bench": 1,
    "ml": 2,
    "ai": 2,
    "dl": 2,
    "quantum": 2,
    "finance": 2,
    "rl": 3,
    "research": 3,
    "interpret": 3,
    "agents": 4,
}

HEAVY_DEPENDENCIES = ("torch", "jax", "cupy", "qiskit", "sklearn", "matplotlib", "scipy")


def _subpackage_of(module: str) -> str | None:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "lipipy" else None


def _imports(path: Path, root: Path) -> list[tuple[str, int]]:
    """All absolute lipipy modules imported by a file, with line numbers."""
    module = "lipipy." + ".".join(path.relative_to(root).with_suffix("").parts)
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    package = package.removesuffix(".__init__")
    found = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # resolve relative imports
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                target = f"{base}.{node.module}" if node.module else base
                found += [(target, node.lineno)]
            elif node.module:
                found += [(node.module, node.lineno)]
    return [(m, line) for m, line in found if m == "lipipy" or m.startswith("lipipy.")]


def test_every_subpackage_has_a_layer():
    on_disk = {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert on_disk == set(LAYERS), "add new subpackages to LAYERS in tests/test_architecture.py"
    assert set(lipipy.SUBPACKAGES) == set(LAYERS), "keep lipipy.SUBPACKAGES in sync with LAYERS"


def find_layer_violations(root: Path) -> list[str]:
    """Return one message per import that breaks the layer rule under ``root``."""
    violations = []
    for path in sorted(root.rglob("*.py")):
        if path.parent == root:
            continue  # top-level lipipy/__init__.py may expose everything lazily
        source_pkg = path.relative_to(root).parts[0]
        for target, line in _imports(path, root):
            target_pkg = _subpackage_of(target)
            if target_pkg is None or target_pkg == source_pkg:
                continue  # bare 'import lipipy' or same package
            if LAYERS[target_pkg] >= LAYERS[source_pkg]:
                violations.append(
                    f"{path.relative_to(root).as_posix()}:{line} {source_pkg} (layer "
                    f"{LAYERS[source_pkg]}) imports {target} (layer {LAYERS[target_pkg]})"
                )
    return violations


def test_imports_respect_layers():
    violations = find_layer_violations(SRC)
    assert not violations, "Layer violations:\n" + "\n".join(violations)


def test_layer_checker_catches_violations(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "ml").mkdir()
    (tmp_path / "core" / "bad.py").write_text("from lipipy.ml import LinearRegression\n")
    (tmp_path / "ml" / "bad.py").write_text("import lipipy.finance\n")
    (tmp_path / "ml" / "ok.py").write_text("from lipipy.core import register\nfrom . import bad\n")
    found = find_layer_violations(tmp_path)
    assert found == [
        "core/bad.py:1 core (layer 0) imports lipipy.ml (layer 2)",
        "ml/bad.py:1 ml (layer 2) imports lipipy.finance (layer 2)",
    ]


def test_import_lipipy_is_light():
    code = (
        "import sys, lipipy;"
        "subs = sorted({m.split('.')[1] for m in sys.modules"
        "  if m.startswith('lipipy.') and not m.startswith('lipipy.core')});"
        f"heavy = [m for m in {HEAVY_DEPENDENCIES!r} if m in sys.modules];"
        "print(subs, heavy)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[] []", out.stdout


def test_lazy_subpackage_access():
    assert lipipy.ml.LinearRegression.__name__ == "LinearRegression"
    assert "quantum" in dir(lipipy)
    with pytest.raises(AttributeError):
        lipipy.not_a_subpackage  # noqa: B018
