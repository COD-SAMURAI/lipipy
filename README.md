# lipipy

**Readable, verified-from-scratch implementations of machine learning, deep learning
and modern research architectures.**

Every algorithm in lipipy is:

1. **Readable**: plain NumPy/PyTorch, with the math written in the docstring.
2. **Verified**: tested for numerical parity against a reference library
   (scikit-learn, PyTorch), so you can trust it as a baseline.
3. **Explained**: paired with a notebook that walks through the derivation.

Use it to learn how things work, as a clean baseline in research, or as a
starting point for your own variant of an architecture.

## Install

```bash
make dev                     # creates .venv and installs lipipy with dev tools
make test                    # runs the test suite
```

Or by hand:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # add ",dl" for the PyTorch modules
```

## Quick start

```python
import numpy as np
from lipipy.ml import LinearRegression

X = np.random.randn(100, 3)
y = X @ np.array([1.5, -2.0, 0.5]) + 4.0

model = LinearRegression().fit(X, y)  # closed form (SVD least squares)
model_gd = LinearRegression(solver="gd").fit(X, y)  # gradient descent, for learning
print(model.coef_, model.intercept_, model.score(X, y))
```

## Layout

| Package | What goes there | Status |
|---|---|---|
| `lipipy.core` | base classes, validation, metrics | started |
| `lipipy.ml` | classic ML in NumPy: linear models, trees, SVM, clustering, PCA, boosting | `LinearRegression` |
| `lipipy.ai` | classical AI: search, logic, constraint satisfaction, planning, causality | planned |
| `lipipy.dl` | autograd, layers, attention, CNN/RNN, transformer, optimizers | planned |
| `lipipy.research` | SSMs/Mamba, mixture-of-experts, diffusion, KANs + benchmark harness | planned |
| `lipipy.rl` | bandits, DQN, PPO | planned |
| `lipipy.agents` | LLM agent loops, tool use, reasoning, AGI research directions | planned |
| `lipipy.interpret` | probing, attribution, mechanistic interpretability | planned |

Each subpackage's `__init__.py` holds the full catalog of algorithms planned for it,
marked `[x]` when implemented and `[ ]` when planned. Run `help(lipipy.ml)` to see it.

Other top-level folders: `tests/` (parity tests), `notebooks/` (one walkthrough per
algorithm), `benchmarks/` (reproducible comparisons), `docs/` (MkDocs site).

## Mathematics formula notes

[`notebooks/math/`](notebooks/math/README.md) is a 25-part formula reference that goes in
order from first-semester engineering mathematics (calculus, linear algebra, differential
equations, transforms, probability) to the equations behind modern AI (transformers,
diffusion, RLHF/DPO, state-space models) and learning theory toward AGI. Each file states
its level, typical semester and prerequisites. Start with
[`00_index.ipynb`](notebooks/math/00_index.ipynb).

## Roadmap

- **v0.1**: about 10 classic ML algorithms in `ml/`, each verified against scikit-learn; first PyPI release.
- **v0.2**: a tiny autograd engine and core DL layers in `dl/`, verified against PyTorch.
- **v0.3**: first research architectures and a shared training/benchmark harness.
- **Later**: RL, agents and interpretability tracks.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). It includes the checklist every new algorithm
must pass. Good first issues are labelled on GitHub.

## Origin

lipipy started as the signature library of **Anveshan**, the applied research and
AI/ML club at The National Institute of Engineering, Mysore.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
