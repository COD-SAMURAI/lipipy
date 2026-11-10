"""Print the latest version, licence and Python requirement of packages from PyPI.

Used to refresh the tables in docs/dependencies.md. Standard library only.
Usage: ``python scripts/dependency_report.py numpy scipy torch ...``
(with no arguments, a default list of every package the docs mention).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import urllib.request

DEFAULT = (
    "numpy array-api-compat torch jax cupy-cuda12x numba gymnasium pandas pyarrow mcp "
    "qiskit qiskit-aer pennylane scipy array-api-strict scikit-learn statsmodels networkx "
    "QuantLib arch cvxpy pytest pytest-cov hypothesis pytest-benchmark ruff mypy pre-commit "
    "nbconvert ipykernel matplotlib mkdocs-material mkdocstrings hatchling build maturin "
    "pip-audit bandit detect-secrets pip-licenses cyclonedx-bom"
).split()


def fetch(name: str) -> str:
    url = f"https://pypi.org/pypi/{name}/json"  # fixed https host; name from argv/DEFAULT
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:  # nosec B310 - https only
            info = json.load(resp)["info"]
    except Exception as exc:  # noqa: BLE001 - report and continue
        return f"{name:22} ERROR {exc}"
    licence = info.get("license_expression") or next(
        (c.split(" :: ")[-1] for c in info.get("classifiers", []) if c.startswith("License ::")),
        (info.get("license") or "?")[:30],
    )
    python = info.get("requires_python") or "?"
    return f"{name:22} {info['version']:14} {licence[:40]:40} py{python}"


def main(names: list[str]) -> None:
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        for line in pool.map(fetch, names or DEFAULT):
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
