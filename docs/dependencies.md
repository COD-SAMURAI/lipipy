# Dependencies and from-scratch inventory

Status: researched **2026-11-11** against PyPI. Refresh the version tables with
`make deps-report` before each release (see §7).

This document answers two questions for every part of lipipy:

1. **What do we depend on?** Third-party packages by role: runtime, optional
   extras, reference libraries used only in tests, and tooling.
2. **What do we write ourselves?** The from-scratch code each layer needs, and the
   order to build it in.

It complements [ARCHITECTURE.md](../ARCHITECTURE.md) (§3 layers, §6 complexity) and
[SECURITY.md](../SECURITY.md).

---

## 1. Dependency principles

1. **One required runtime dependency: NumPy.** `pip install lipipy` must stay small and
   import fast. Everything else is an optional extra or test-only.
2. **Extras unlock backends and domains, never core features.** A feature that needs an
   extra imports it through `lipipy.core.require(module, extra=...)`, which fails with an
   install hint.
3. **Reference libraries are for verification only.** scikit-learn, SciPy, Qiskit,
   QuantLib and others are used in parity tests to check our implementations. lipipy
   never wraps them as its own implementation.
4. **Permissive licences only.** Runtime and extras: MIT, BSD, Apache-2.0, ISC, PSF,
   NCSA, Zlib. MPL-2.0 is acceptable for development tools only. GPL/AGPL/SSPL or
   proprietary packages are not allowed in any extra.
5. **Lower bounds, no upper caps** in `pyproject.toml` (a library must not cap its users'
   versions). Reproducibility comes from CI and lock files, not from pins in the package.
6. **No network or data-vendor packages as dependencies.** Market-data downloaders,
   scrapers and LLM provider SDKs stay out of the package; examples may show them.

## 2. Python version support

| Fact (PyPI, 2026-11) | Requires |
|---|---|
| NumPy 2.5, SciPy 1.18, JAX 0.12, NetworkX 3.7 | Python ≥ 3.12 |
| PyTorch 2.15, scikit-learn 1.9, pandas 3.1, Qiskit Aer 0.18, PennyLane 0.45, cvxpy 1.9 | Python ≥ 3.11 |

**Decision:** `requires-python = ">=3.11"` and CI on 3.11, 3.12, 3.13 and 3.14.
Python 3.10 is dropped because most extras can no longer be installed on it. We follow
the Scientific Python [SPEC 0](https://scientific-python.org/specs/spec-0000/) direction
(support Python versions for about three years) but move one step behind it, so users
on 3.11 are not stranded. Revisit at each minor release; the next step is 3.12.

## 3. Runtime and optional extras

| Extra | Package | Latest (2026-11) | Licence | Why |
|---|---|---|---|---|
| *(required)* | numpy | 2.5.4 | BSD-3-Clause (+ bundled permissive) | arrays, `linalg` (LAPACK), FFT, RNG |
| `backends` | array-api-compat | 1.16.0 | MIT | accept CuPy / PyTorch / JAX arrays through one namespace |
| `dl` | torch | 2.15.0 | Apache-2.0 | `lipipy.dl` modules, GPU autograd |
| `jax` *(planned)* | jax | 0.12.0.1 | Apache-2.0 | JAX backend, `jit` path for hotspots |
| `gpu` *(planned)* | cupy-cuda12x | 14.2.0 | MIT | NumPy-compatible GPU arrays |
| `jit` *(planned)* | numba | 0.68.1 | BSD-2-Clause | optional JIT for loops that cannot be vectorized |
| `rl` *(planned)* | gymnasium | 1.4.0 | MIT | standard environments; lipipy defines a compatible protocol so this stays optional |
| `data` *(planned)* | pandas / pyarrow | 3.1.1 / 26.0.1 | BSD-3 / Apache-2.0 | DataFrame input convenience in `finance` and `ml`; arrays remain the core format |
| `agents` *(planned)* | mcp | 2.3.0 | MIT | `lipipy-mcp` server exposing the registry to AI agents |
| `quantum` | qiskit | 2.6.2 | Apache-2.0 | parity tests and circuit import/export only |

Agents stay **provider-neutral**: lipipy defines a small `LLM` protocol; users plug in
any SDK (Anthropic, OpenAI, local models). No provider SDK becomes a dependency.

## 4. Reference libraries (tests only)

Installed through the `test-reference` extra (CI) or individually. Each test that uses
one calls `pytest.importorskip`, so the suite still runs without it.

| Domain | Reference | Version | Licence | Verifies |
|---|---|---|---|---|
| numerics | scipy | 1.18.1 | BSD-3 | linalg, optimize, integrate, special functions, FFT, stats |
| numerics / backends | array-api-strict | 2.6.1 | BSD-3 | our code uses only Array API standard functions |
| ml | scikit-learn | 1.9.1 | BSD-3 | estimators and metrics |
| ml / finance | statsmodels | 0.15.0 | BSD-3 | ARIMA, regression statistics, tests |
| ai / graphs | networkx | 3.7 | BSD-3 | graph search, shortest paths, spectral methods |
| dl | torch | 2.15.0 | Apache-2.0 | layers, losses, optimizers, autograd |
| quantum | qiskit, qiskit-aer | 2.6.2, 0.18.2 | Apache-2.0 | statevectors, circuits, noise models |
| quantum | pennylane | 0.45.1 | Apache-2.0 | parameter-shift gradients, VQE/QAOA |
| finance | QuantLib | 1.44 | BSD-3 | option pricing, yield curves |
| finance | arch | 8.0.0 | NCSA | GARCH models |
| finance | cvxpy | 1.9.3 | Apache-2.0 | portfolio optimization optima |
| rl | gymnasium | 1.4.0 | MIT | environment semantics |

## 5. Development, documentation, release and security tooling

| Role | Package / service | Version | Licence |
|---|---|---|---|
| Tests | pytest, pytest-cov | 9.2.0, 7.1.0 | MIT |
| Property tests *(planned)* | hypothesis | 6.170 | MPL-2.0 (dev only) |
| Benchmarks *(planned)* | pytest-benchmark | 5.3.1 | BSD-2 |
| Lint + format | ruff | 0.17.4 | MIT |
| Types *(planned)* | mypy | 2.4.1 | MIT |
| Git hooks *(planned)* | pre-commit | 4.7.0 | MIT |
| Notebooks | nbconvert, ipykernel, matplotlib | 7.17, 7.4, 3.11 | BSD / PSF |
| Docs | mkdocs-material, mkdocstrings | 9.8.1, 1.0.8 | MIT / ISC |
| Build | hatchling, build | 1.33.0, 1.6.1 | MIT |
| Publish | `pypa/gh-action-pypi-publish` with PyPI trusted publishing (OIDC, no API tokens) | v1.14.3 | BSD-3 |
| Native kernels *(future)* | maturin + PyO3 (Rust) | 1.16.0 | MIT / Apache-2.0 |
| Dependency audit | pip-audit | 2.10.1 | Apache-2.0 |
| Code security scan | bandit; GitHub CodeQL | 1.9.4; v4.39 | Apache-2.0; MIT |
| Secret scan | detect-secrets; GitHub secret scanning + push protection | 1.5.0 | Apache-2.0 |
| Licence audit | pip-licenses | 5.5.5 | MIT |
| SBOM *(release)* | cyclonedx-bom | 8.0.1 | Apache-2.0 |
| Dependency updates | Dependabot (pip + GitHub Actions) | built in | – |

**Considered and rejected**

| Package | Reason |
|---|---|
| yfinance and other market-data downloaders | network access, data-vendor terms; examples only, never a dependency or used in tests |
| TA-Lib | C library build step; indicators are simple to implement from scratch |
| pandas / polars as required | heavy; accept any array-like, convert at the boundary |
| LLM provider SDKs in core | lock-in; provider-neutral protocol instead |
| import-linter | duplicates `tests/test_architecture.py`, which needs no extra dependency |
| solvers with commercial licences (Gurobi, MOSEK) | not permissively licensed |

## 6. From-scratch inventory and build order

What lipipy writes itself. **Policy:** `numerics` contains readable from-scratch
implementations (verified against SciPy/NumPy). Domain algorithms may call NumPy's
LAPACK-backed primitives (`np.linalg.lstsq`, `svd`, `eigh`) for speed and stability,
and must say so in their docstring. Each item lists what it builds on.

### 6.1 Cross-cutting infrastructure (layer 0, `core`)

| Component | Status | Builds on |
|---|---|---|
| Registry, `Complexity`, backends, random, errors, optional imports | done | numpy |
| Estimator protocols (`Estimator`, `Transformer`, `Classifier` as `typing.Protocol`) | next | `core.base` |
| Result dataclasses with `.to_dict()` (`OptimizeResult`, `FitResult`, …) | next | dataclasses |
| Safe serialization: `save`/`load` via `.npz` with `allow_pickle=False` + JSON metadata | planned | numpy |
| Synthetic dataset generators (regression, blobs, moons, time series, price paths) | planned | `core.random` |
| Input-size guards from declared complexity (`ResourceLimitError`) | planned | registry |

### 6.2 Build order (critical path)

```
core ─► numerics.special ─► numerics.stats ─► numerics.linalg ─► numerics.optimize ─► numerics.autodiff
            │                      │                 │                  │                    │
            │                      │                 │                  ├─► ml (logistic, SVM, …)
            │                      │                 ├─► quantum (statevector, gates)        └─► dl (teaching autograd)
            │                      └─────────────────┴─► finance (risk, portfolio, pricing)
            └─► numerics.fft / integrate / ode ─► finance (PDE pricing), research (SSM kernels)
```

### 6.3 Per-layer scratch code

**numerics (layer 1)** — the foundation every domain reuses

| Order | Module | Contents | Key complexity |
|---|---|---|---|
| 1 | `special` | stable `logsumexp`, `softmax`, `sigmoid`, `log1pexp`; vectorized `erf`/`erfc` (NumPy has none) and normal `cdf`/`ppf` | O(n) |
| 2 | `stats` | Welford mean/variance (streaming), covariance, quantiles, distributions (pdf/cdf/sample) | O(n d²) for covariance |
| 3 | `linalg` | triangular solves, LU with pivoting, Cholesky, Householder QR, power/QR eigen iteration, CG, GMRES | O(d³) dense, O(nnz·T) iterative |
| 4 | `optimize` | Armijo/Wolfe line search, gradient descent, Newton, BFGS, L-BFGS, Nelder-Mead, projected gradient, simplex, QP | per-iteration O(d) to O(d³) |
| 5 | `autodiff` | forward mode (dual numbers), reverse-mode tape for scalars then tensors, gradient check | ≈ 2–4× forward cost |
| 6 | `fft` | radix-2 Cooley-Tukey, real FFT, FFT convolution | O(n log n) |
| 7 | `integrate`, `ode`, `sde` | Simpson, Gauss-Legendre, adaptive quadrature; RK4, RK45, implicit Euler; Euler-Maruyama | O(T) steps |
| 8 | `interpolate`, `sparse` | cubic splines; COO/CSR with matvec | O(n); O(nnz) |

**ml (layer 2)**: metrics and preprocessing → linear models (logistic uses
`optimize.lbfgs`) → kNN with chunked distances (memory O(n·chunk), not O(n²)) →
trees (sorted-split search O(n d log n)) → ensembles → clustering (k-means++) →
PCA via SVD → probabilistic models (EM, HMM).

**ai (layer 2)**: problem/state protocols → BFS/DFS/UCS/A* with a binary heap
(`heapq`) → CSP backtracking with AC-3 → DPLL/CDCL SAT → minimax/alpha-beta → MCTS.

**quantum (layer 2)**

| Order | Component | Notes |
|---|---|---|
| 1 | `states` | basis/product states, normalization checks, Bloch coordinates |
| 2 | `gates` | unitary matrices; `is_unitary` check (‖U†U − I‖ < tol) |
| 3 | `statevector` | apply a k-qubit gate by reshaping the state to `(2,)*q` and `np.tensordot` / `einsum` on the target axes: O(2^q · 2^k) time, no 2^q × 2^q matrix; `max_qubits` guard |
| 4 | `measure` | probabilities, shots via `rng.choice`, Pauli-string expectations |
| 5 | `Circuit` | fluent builder, parameters, depth/gate counts |
| 6 | algorithms | QFT, Grover, phase estimation, Deutsch-Jozsa, teleportation |
| 7 | variational | parameter shift, VQE and QAOA using `numerics.optimize` |
| 8 | `density_matrix`, noise, `mps`, stabilizer | O(4^q); O(q χ²); polynomial |

**finance (layer 2)**

| Order | Component | Builds on | Notes |
|---|---|---|---|
| 1 | returns, volatility, Sharpe/Sortino, drawdown | `stats` | pure functions over `(T,)` or `(T, A)` arrays |
| 2 | VaR / CVaR, covariance shrinkage (Ledoit-Wolf) | `stats`, `linalg` | |
| 3 | Black-Scholes and Greeks, implied volatility | `special.ndtr`, `optimize` (Brent/Newton) | |
| 4 | binomial trees, Monte Carlo (antithetic, control variates) | `core.random` | chunk over paths: memory O(chunk·T) |
| 5 | Markowitz, min-variance, risk parity, Black-Litterman | `linalg`, `optimize` (QP) | |
| 6 | vectorized backtester → event-driven backtester | all of the above | time-indexed data view prevents look-ahead; costs required |

**dl (layer 2)**: thin modules on PyTorch (layers, losses, optimizers written out
explicitly) plus a teaching path on `numerics.autodiff` for a NumPy-only MLP.

**rl (layer 3)**: `Env` protocol compatible with Gymnasium → tabular methods → bandits →
DQN/PPO on `dl`.

**research, interpret (layer 3)** and **agents (layer 4)**: build on the layers above;
agents add the provider-neutral `LLM` protocol, tool schemas generated from the registry
and type hints, and the MCP server.

## 7. Keeping this document current

- Dependabot opens weekly PRs for GitHub Actions and Python tooling.
- Before each release: refresh the version tables from PyPI, run `make security`
  (pip-audit, bandit, secret scan, licence check) and update the Python support line.
- Adding a dependency requires: the role (§3–5), licence check against §1, why
  a from-scratch implementation is not appropriate, and a line in this file.
