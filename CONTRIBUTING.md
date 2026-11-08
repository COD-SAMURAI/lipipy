# Contributing to lipipy

Thanks for helping! By contributing you agree your work is licensed under the
Apache License 2.0 (see LICENSE). lipipy values **correctness and readability over feature count**.

## Setup

```bash
git clone https://github.com/COD-SAMURAI/lipipy && cd lipipy
make dev        # creates .venv and installs lipipy[dev]
make check      # lint + format check + tests (same as CI)
```

## Definition of done for a new algorithm

A pull request that adds an algorithm should include all of the following.
Use `src/lipipy/ml/linear/linear_regression.py` as the template.

- [ ] **Implementation** in the right subpackage, using only NumPy for `ml/` (PyTorch is allowed in `dl/` and `research/`).
- [ ] **Docstring with the math**: the model, the objective, how it is solved, and a reference (textbook or paper).
- [ ] **scikit-learn style API**: hyperparameters in `__init__`, `fit` returns `self`, learned attributes end in `_`.
- [ ] **Parity test** against a reference implementation (scikit-learn / PyTorch / the paper's code) in `tests/`.
- [ ] **Edge-case tests**: bad inputs, unfitted use, degenerate data.
- [ ] **Notebook** in `notebooks/` that walks through the derivation and compares with the reference.
- [ ] **Docs page** in `docs/algorithms/` and an entry in `mkdocs.yml`.
- [ ] Exported from the subpackage `__init__.py`.

## Style

- `ruff` handles linting and formatting (line length 100).
- Prefer clear code to clever code. A reader should be able to match each line to the math.
- Short variable names like `X`, `y` and `w` are fine when they match the math.

## Research implementations

For `research/`, link the paper (arXiv ID) and the official code if any, say what
you simplified, and include a small reproduction of one result from the paper.
