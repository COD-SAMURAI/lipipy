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
pip install -e ".[dev]"      # from a clone, with test tools
pip install -e ".[dl]"       # adds PyTorch for the deep learning modules
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
| `lipipy.dl` | autograd, layers, attention, CNN/RNN, transformer, optimizers | planned |
| `lipipy.research` | SSMs/Mamba, mixture-of-experts, diffusion, KANs + benchmark harness | planned |
| `lipipy.rl` | bandits, DQN, PPO | planned |
| `lipipy.agents` | LLM agent loops, tool use, reasoning | planned |
| `lipipy.interpret` | probing, attribution, mechanistic interpretability | planned |

Other top-level folders: `tests/` (parity tests), `notebooks/` (one walkthrough per
algorithm), `benchmarks/` (reproducible comparisons), `docs/` (MkDocs site).

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

MIT
