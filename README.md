# lipipy

**Readable, verified, complexity-documented implementations of the algorithms behind
computing, from numerical methods to ML, deep learning, AI agents, quantum computing
and quantitative finance.**

Every algorithm in lipipy is:

1. **Readable**: plain NumPy/PyTorch, with the math written in the docstring.
2. **Verified**: tested for numerical parity against a reference library
   (scikit-learn, PyTorch), so you can trust it as a baseline.
3. **Measured**: declares its time and space complexity, checked empirically by
   `lipipy.bench`.
4. **Explained**: paired with a notebook that walks through the derivation.

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

# every algorithm is in a searchable catalog with its complexity
import lipipy

info = lipipy.registry.get("ml.LinearRegression")
print(info.complexity)  # time O(n d^2 + d^3), space O(n d) (...)
print(lipipy.registry.to_json())  # machine-readable catalog, e.g. for AI agents

# check how a function actually scales
from lipipy.bench import estimate_complexity

est = estimate_complexity(
    np.sort, lambda n, rng: rng.random(n), sizes=(10**5, 2 * 10**5, 4 * 10**5)
)
print(est.summary())  # time ≈ O(n^1.05), space ≈ O(n^1.00)
```

## Layout

lipipy is layered: a package imports only from lower layers, and a test enforces it.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

| Layer | Package | What goes there | Status |
|---|---|---|---|
| 0 | `lipipy.core` | registry, complexity metadata, backends, randomness, errors, validation | started |
| 1 | `lipipy.numerics` | linear algebra, optimization, integration, ODEs, FFT, statistics, autodiff | special functions (`erf`, normal cdf/ppf, stable softmax), streaming moments, LU / Cholesky / QR, GD / Newton / BFGS / L-BFGS, Brent and Newton roots |
| 1 | `lipipy.bench` | empirical time/space complexity measurement | `estimate_complexity` |
| 2 | `lipipy.ml` | classic ML in NumPy: linear models, trees, SVM, clustering, PCA, boosting | `LinearRegression`, `Ridge`, k-nearest neighbours, `KMeans`, `PCA`, `GaussianNB`, metrics |
| 2 | `lipipy.ai` | classical AI: search, logic, constraint satisfaction, planning, causality | planned |
| 2 | `lipipy.dl` | autograd, layers, attention, CNN/RNN, transformer, optimizers | planned |
| 2 | `lipipy.quantum` | circuits, statevector / density-matrix / MPS simulators, Grover, QFT, VQE, QAOA | gates, statevector simulator, `Circuit`, QFT, Grover, Deutsch-Jozsa, Bernstein-Vazirani, phase estimation |
| 2 | `lipipy.finance` | returns, risk, portfolios, option pricing, backtesting (research only) | planned |
| 3 | `lipipy.rl` | bandits, DQN, PPO, SAC, MCTS | planned |
| 3 | `lipipy.research` | SSMs/Mamba, mixture-of-experts, diffusion, KANs + benchmark harness | planned |
| 3 | `lipipy.interpret` | probing, attribution, mechanistic interpretability | planned |
| 4 | `lipipy.agents` | LLM agent loops, tool use, reasoning, AGI research directions | planned |

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

## Dependencies and security

The only required dependency is NumPy. Everything else is an optional extra or used only
in tests; see [docs/dependencies.md](docs/dependencies.md) for the full list, licences and
what lipipy builds from scratch. Security policy and reporting: [SECURITY.md](SECURITY.md).
Run `make security` for the same checks CI runs.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). It includes the checklist every new algorithm
must pass. Good first issues are labelled on GitHub.

## Origin

lipipy started as the signature library of **Anveshan**, the applied research and
AI/ML club at The National Institute of Engineering, Mysore.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
