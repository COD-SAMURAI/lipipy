# lipipy architecture

Status: **accepted, v1** (2026-11). This document is the contract for how lipipy grows.
Change it through a pull request that also adds a decision record (see §13).

---

## 1. What lipipy is

**One package of readable, verified, complexity-documented implementations of the
algorithms behind computing, from numerical methods to ML, deep learning, AI agents,
quantum computing and quantitative finance.**

Three promises hold for every public algorithm:

1. **Readable.** The code maps line by line to the math in its docstring.
2. **Verified.** Tested against a reference implementation (NumPy/SciPy, scikit-learn,
   PyTorch, Qiskit, QuantLib, a paper's numbers) or an analytic result.
3. **Measured.** Declared time and space complexity, checked empirically by
   `lipipy.bench`.

Non-goals: competing with PyTorch/JAX on raw speed, being a production trading
system, or wrapping other libraries without our own implementation.

## 2. Who it is for, and what each needs

| User | Needs | How the architecture serves it |
|---|---|---|
| **Students** | code that matches the textbook | one algorithm per module, math in docstrings, notebooks, `notebooks/math/` |
| **Researchers** | trusted baselines, easy modification, reproducibility, citations | parity tests, plain subclassable classes, explicit `random_state`, `references` metadata, benchmark harness |
| **Engineers** | stable API, typed, fast enough, few dependencies | layered packages, SemVer, numpy-only core, optional extras, backend dispatch |
| **AI agents / LLM tools** | discoverable, machine-readable, side-effect-free functions | central registry with JSON export, typed signatures, structured result objects, no printing |
| **Heavy computation / HPC** | big arrays, GPUs, predictable memory | Array API backends (NumPy, CuPy, PyTorch, JAX), declared space complexity, chunked/streaming variants |
| **Quantum** | simulators, circuits, algorithms with known costs | `lipipy.quantum` with statevector, density-matrix and tensor-network simulators; qubit-count guards |
| **Quant / trading research** | correct risk and pricing math, backtests without look-ahead bias | `lipipy.finance` with pure metric functions, pricing models and a causal backtest engine |

## 3. Layered package structure

```
                      ┌────────────────────────────────────────────┐
 Layer 4  interfaces  │ agents                                     │  compose everything
                      ├────────────────────────────────────────────┤
 Layer 3  composite   │ rl · research · interpret                  │  combine domains
                      ├────────────────────────────────────────────┤
 Layer 2  domains     │ ml · ai · dl · quantum · finance           │  independent of each other
                      ├────────────────────────────────────────────┤
 Layer 1  shared math │ numerics · bench                           │  primitives + measurement
                      ├────────────────────────────────────────────┤
 Layer 0  core        │ core (registry, backend, random, errors,   │  numpy only
                      │       validation, base classes, metrics)   │
                      └────────────────────────────────────────────┘
```

**Dependency rule:** a package may import only from packages in a **lower** layer.
Layer-2 domains never import each other. This rule is enforced by
`tests/test_architecture.py`, which parses every module's imports.

Why: domains stay installable and testable on their own, there are no import cycles,
and any layer-2 domain can later be split into its own distribution
(`lipipy-quantum`) without touching the others.

| Package | Layer | Hard deps | Extra |
|---|---|---|---|
| `core` | 0 | numpy | none |
| `numerics` | 1 | numpy | none |
| `bench` | 1 | numpy | none |
| `ml`, `ai`, `finance` | 2 | numpy | none |
| `quantum` | 2 | numpy | none (optional `[quantum]` for parity tests) |
| `dl` | 2 | torch | `[dl]` |
| `rl`, `research`, `interpret` | 3 | per algorithm | `[rl]`, `[dl]` |
| `agents` | 4 | per tool | `[agents]` |

New domains (planned: `timeseries`, `signal`, `graph`, `optim`, `causal`) enter at
layer 2 using the template in §12.

## 4. Core services (layer 0)

All of these exist from v0.1 and every other package uses them.

| Module | Responsibility | Key API |
|---|---|---|
| `core.registry` | catalog of every algorithm with metadata | `@register(...)`, `lipipy.registry.find(...)`, `.to_json()` |
| `core.complexity` | big-O metadata type and variable glossary | `Complexity(time=..., space=...)` |
| `core.backend` | Array API dispatch across NumPy / CuPy / PyTorch / JAX | `get_namespace(*arrays)`, `to_numpy(x)` |
| `core.random` | reproducible randomness | `check_random_state(seed)`, `spawn(rng, n)` |
| `core.imports` | optional dependencies with clear install hints | `require("torch", extra="dl")` |
| `core.errors` | one exception hierarchy | `LipipyError`, `NotFittedError`, `MissingDependencyError`, `ConvergenceWarning` |
| `core.validation` | input checking at public boundaries | `check_array`, `check_X_y` |
| `core.base` | estimator conventions | `BaseEstimator`, mixins |
| `core.metrics` | metrics shared across domains | `r2_score`, `accuracy_score`, … |

### 4.1 The registry

Every public algorithm is registered with a decorator:

```python
@register(
    domain="ml",
    tasks=("regression",),
    complexity=Complexity(time="O(n d^2 + d^3)", space="O(n d)"),
    references=("Hastie, Tibshirani & Friedman, ESL 2e, §3.2",),
    backends=("numpy",),
    status="stable",
)
class LinearRegression(...): ...
```

The registry is the single source of truth that powers:

- **discovery**: `lipipy.registry.find(task="classification")`
- **documentation**: algorithm tables generated from metadata
- **agents**: `lipipy.registry.to_json()` gives an LLM a machine-readable catalog
- **benchmarks**: `lipipy.bench` iterates over registered algorithms
- **citations**: `references` metadata
- **plugins**: third-party packages register through the `lipipy.plugins` entry-point group

Lookups by name are O(1) (dict). `find` is O(R) over R registered algorithms, which is
fine for R in the thousands. Discovery imports all subpackages once (lazily, on first
`find`) and skips any whose optional dependencies are missing.

### 4.2 Backends

lipipy code is written against the
[Python Array API standard](https://data-apis.org/array-api/) where practical:

```python
xp = get_namespace(X)  # numpy, cupy, torch or jax.numpy
mean = xp.mean(X, axis=0)  # runs on the caller's device
```

- With only NumPy installed, everything works on NumPy.
- With `lipipy[backends]` (array-api-compat), CuPy/PyTorch/JAX arrays are accepted
  and results come back as the same array type on the same device.
- Algorithms that need features outside the standard (e.g. `lstsq`) declare
  `backends=("numpy",)` in the registry and convert with `to_numpy`.

### 4.3 Randomness

No global random state. Every stochastic API takes `random_state` (None, int,
`SeedSequence` or `Generator`) and calls `check_random_state`. Parallel workers get
independent streams from `spawn(rng, n)` (built on `SeedSequence.spawn`), so results
do not depend on how many workers ran.

### 4.4 Import cost

`import lipipy` loads only `core`. Subpackages load lazily on first attribute access
(`lipipy.ml`), and heavy dependencies (torch, qiskit) load only inside the algorithms
that need them. A test enforces that `import lipipy` does not import any subpackage
or heavy dependency.

## 5. API design: three tiers, one convention set

| Tier | Shape | Used for | Example |
|---|---|---|---|
| **Functional** | pure function, arrays in → arrays/result out | numerics, metrics, finance math, quantum gates | `finance.sharpe_ratio(returns)` |
| **Estimator** | `fit` / `predict` / `transform` / `score` | ML, time series, anomaly detection | `ml.LinearRegression().fit(X, y)` |
| **System** | composable objects with explicit state | DL modules, RL env/agent, quantum circuits, backtests, agents | `quantum.Circuit(3).h(0).cx(0, 1)` |

Conventions for all tiers:

- **Inputs** are validated once at the public boundary; internals trust them.
- **Outputs** keep the caller's backend and dtype (float64 by default for numerics).
- **Rich results** are frozen dataclasses with `.to_dict()` (JSON-friendly for agents),
  never tuples longer than two.
- **Hyperparameters** live in `__init__`; learned state ends with `_`; `fit` returns `self`.
- **Determinism**: same inputs + same `random_state` ⇒ same outputs.
- **No side effects**: no printing, no global state changes, logging via `logging` only.
- **Typing**: every public function is fully annotated (checked by a type checker in CI).
- **Docstrings**: NumPy style with sections *Parameters, Returns, Complexity, References,
  Examples*; the example runs as a doctest.

## 6. Time and space complexity policy

### 6.1 Declaring complexity

Every registered algorithm declares `Complexity(time=..., space=...)` using the shared
variable glossary, so costs are comparable across domains:

| Symbol | Meaning | Symbol | Meaning |
|---|---|---|---|
| `n` | samples / rows | `d` | features / dimension |
| `k` | clusters, classes, neighbours | `T` | iterations or time steps |
| `L` | sequence length | `h` | hidden width |
| `V` | vertices | `E` | edges |
| `q` | qubits | `g` | gates |
| `χ` | bond dimension (tensor networks) | `A` | assets |
| `P` | paths / Monte Carlo samples | `B` | batch size |

Space counts **extra** memory beyond the inputs and outputs.

### 6.2 Verifying complexity

`lipipy.bench.estimate_complexity(func, make_input, sizes)` times the function over
growing input sizes, measures peak allocation with `tracemalloc`, and fits the
log-log slope:

```
time ≈ O(n^1.02), space ≈ O(n^1.00)
```

Selected algorithms get complexity tests (marked `complexity`, run in a separate CI
job) that fail if the measured exponent exceeds the declared one by a margin. This
catches accidental quadratic behaviour (a Python loop, an n×n temporary) before release.

### 6.3 Performance rules

1. **Vectorize first.** No Python loop over `n` when an array operation exists.
2. **Never materialize what you can stream.** If a naive version allocates O(n²)
   (pairwise distances, attention scores), provide a chunked version with O(n·c) memory
   and declare which one is the default.
3. **Numerically stable formulas** over textbook ones: `lstsq` over explicit inverse,
   log-sum-exp, Welford's variance, Kahan summation where error accumulates,
   Cholesky over `inv` for SPD systems.
4. **Explicit dtype policy:** float64 default for numerics and finance; float32 and
   bfloat16 allowed where declared (DL).
5. **Copies are visible:** functions that mutate inputs say so in their name
   (`_inplace`) and docstring.
6. **Escalation path for speed**, only for profiled hotspots:
   NumPy → optional JIT (Numba/JAX) → native kernels in `lipipy._native` (Rust via PyO3)
   with a pure-Python reference implementation kept as the tested baseline.

### 6.4 Domain-specific cost notes

| Domain | Watch out for | Policy |
|---|---|---|
| Quantum statevector | memory `16·2^q` bytes (complex128): 30 qubits ≈ 16 GiB | refuse above a configurable `max_qubits` (default 26), point users to MPS (`O(q·χ²)` memory) |
| Kernel methods / GPs | `O(n²)` memory, `O(n³)` time | declare it; offer Nyström / random features |
| Attention | `O(L²)` scores | chunked / online-softmax implementation |
| Backtesting | `O(T·A)` | vectorized engine for research, event-driven engine for path-dependent logic |
| Monte Carlo pricing | `O(P·T)` | batch paths, antithetic / control variates, chunk over `P` |
| Graph algorithms | dense adjacency `O(V²)` | sparse (CSR) by default |

## 7. Domain blueprints

Each domain package follows the same internal shape:

```
lipipy/<domain>/
├── __init__.py          # catalog docstring + public exports
├── _base.py             # domain base classes / protocols
├── <topic>/             # one subpackage per topic (linear/, trees/, …)
│   └── <algorithm>.py   # one algorithm per module, math in the module docstring
└── datasets.py          # tiny synthetic generators for tests and docs (optional)
```

**`numerics`**: linear algebra, root finding, optimization, integration, ODE/PDE solvers,
FFT, special functions, statistics, sparse matrices, a reference autodiff engine. The
layer every other domain builds on instead of re-deriving primitives.

**`quantum`**: `Circuit` builder → simulator backends (`statevector`, `density_matrix`,
`mps`) → `measure`/`expectation`. Algorithms (Grover, QFT, phase estimation, VQE, QAOA,
quantum kernels) use `numerics.optimize` for classical loops. Parity tests against
Qiskit/PennyLane through the optional `[quantum]` extra.

**`finance`**: pure functions for returns and risk metrics; portfolio optimizers on
`numerics.optimize`; pricing models (Black-Scholes, binomial, Monte Carlo, Heston);
a backtest engine whose data access is time-indexed so a strategy can only see data
up to time *t* (look-ahead bias is impossible by construction). For research and
education only, not investment advice.

**`ml`, `ai`, `dl`, `rl`, `research`, `agents`, `interpret`**: catalogs already listed in
each package's `__init__.py`.

## 8. Interfaces for AI agents

Agents should be able to use lipipy without reading the source:

1. `lipipy.registry.to_json()` gives every algorithm's name, summary, domain, tasks,
   complexity, references and import path.
2. Typed signatures + NumPy docstrings let an agent framework build a tool schema
   automatically (planned helper: `lipipy.agents.tools.schema(func)`).
3. Results are dataclasses with `.to_dict()`, so outputs serialize to JSON.
4. Planned: an MCP server (`lipipy-mcp`) exposing registry search, algorithm docs and
   safe, bounded computations (input-size limits derived from declared complexity).

## 9. Extensibility (plugins)

A third-party package adds algorithms without forking lipipy:

```toml
# their pyproject.toml
[project.entry-points."lipipy.plugins"]
my_algos = "my_package.lipipy_plugin"
```

The module calls `@register(...)` on import; `lipipy.registry` discovers it via
`importlib.metadata.entry_points(group="lipipy.plugins")`. Plugin names are prefixed with
the plugin's domain, and duplicate names are rejected.

## 10. Quality gates

| Gate | Tool | Runs |
|---|---|---|
| Lint + format | ruff | every PR |
| Unit + doctest | pytest | every PR, Python 3.10–3.13 |
| Reference parity | pytest + sklearn / torch / qiskit (skipped if absent) | every PR |
| Architecture rules | `tests/test_architecture.py` | every PR |
| Property tests | hypothesis (planned) | every PR |
| Complexity checks | `pytest -m complexity` | nightly / pre-release |
| Benchmarks | `benchmarks/` scripts | pre-release, results committed |
| Notebooks execute | `make notebooks` | pre-release |
| Types | mypy / pyright, strict on `core` (planned) | every PR |
| Docs build | mkdocs | on `main` |

## 11. Versioning and stability

- Semantic versioning. Before 1.0, minor versions may break APIs, with a changelog entry.
- Every registered algorithm has a `status`: `experimental` → `stable` → `deprecated`.
  Only `stable` APIs carry the compatibility promise.
- Deprecations warn for two minor releases before removal.
- Distribution: a single `lipipy` package with extras. A domain moves into its own
  distribution only when its dependencies conflict with others; the layer rule makes
  that move mechanical.

## 12. Adding things: templates

**A new algorithm**
1. Module in the right domain/topic, math in the module docstring.
2. `@register(...)` with complexity, references, backends and status.
3. Parity test + edge-case tests; complexity test if it is performance-sensitive.
4. Notebook and docs page (see CONTRIBUTING.md checklist).

**A new domain**
1. Pick the layer (normally 2) and add the package to `LAYERS` in
   `tests/test_architecture.py`.
2. Create `lipipy/<domain>/__init__.py` with the catalog docstring.
3. Add an extra in `pyproject.toml` if it needs new dependencies.
4. Add a decision record explaining why it is a separate domain.

## 13. Decision records

Significant decisions live in `docs/adr/NNNN-title.md` (context, decision,
consequences). Initial decisions captured by this document:

| # | Decision |
|---|---|
| 0001 | Python with a NumPy-only core; heavy deps behind extras |
| 0002 | Strict layering with domains independent at layer 2, enforced by tests |
| 0003 | Central registry with complexity metadata as the source of truth |
| 0004 | Array API standard for multi-backend support |
| 0005 | No global RNG; explicit `random_state` everywhere |
| 0006 | Native code only for profiled hotspots, always with a Python reference |

## 14. Roadmap

| Phase | Scope | Exit criteria |
|---|---|---|
| **0 Foundation** (this PR) | architecture doc; registry, complexity, backend, random, imports, errors; `bench.estimate_complexity`; layer test; domain stubs for numerics, quantum, finance | tests pass; `LinearRegression` registered and complexity-checked |
| **1 Numerics + classic ML** | `numerics` core primitives; ~10 `ml` algorithms | v0.1 on PyPI, docs site live |
| **2 Quantum + finance basics** | statevector simulator, gates, Grover/QFT; returns, risk metrics, Black-Scholes, Markowitz, vectorized backtester | parity with Qiskit / closed-form results |
| **3 Deep learning** | autograd engine in `numerics`, `dl` layers on PyTorch, transformer | parity with PyTorch |
| **4 Research + agents** | modern architectures, benchmark harness, registry-based agent tools, MCP server | reproduced paper results, agent demo |
| **5 Scale** | GPU backends end-to-end, native kernels for hotspots, plugin ecosystem | measured speedups, first external plugin |
