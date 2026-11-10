# Security policy

## Supported versions

lipipy is pre-1.0. Security fixes go into the latest release and `main` only.

## Reporting a vulnerability

Please **do not open a public issue**. Report privately through GitHub:
**Security → Report a vulnerability** on
[github.com/COD-SAMURAI/lipipy](https://github.com/COD-SAMURAI/lipipy/security/advisories/new).

Include what is affected, how to reproduce it, and the impact you expect. You will
get an acknowledgement within 7 days and a fix or mitigation plan within 30 days for
confirmed issues. Credit is given in the advisory unless you prefer otherwise.

## What lipipy does to stay safe

| Area | Control |
|---|---|
| Dependencies | one required dependency (NumPy); permissive licences only; weekly Dependabot updates; `pip-audit` and GitHub dependency review on every PR |
| Code | `bandit` and GitHub CodeQL (`security-extended`) on every PR and weekly |
| Secrets | GitHub secret scanning with push protection; `detect-secrets` on every PR; no tokens in workflows (PyPI uses trusted publishing via OIDC) |
| CI supply chain | actions pinned to full commit SHAs; read-only `GITHUB_TOKEN` by default; credentials not persisted after checkout |
| Plugins | third-party code runs only from packages you installed (`lipipy.plugins` entry points); set `LIPIPY_DISABLE_PLUGINS=1` to skip them |
| Data loading | no `pickle` of untrusted data: arrays load with `allow_pickle=False`; PyTorch weights with `weights_only=True` |
| Resource limits | algorithms with exponential or quadratic memory (e.g. quantum statevectors) check sizes before allocating and raise `ResourceLimitError` |
| Network | the package makes no network calls; examples that download data say so |

Run the same checks locally with `make security`.

## Scope notes

`lipipy.finance` is for research and education. It is not a trading system, and its
output is not investment advice. `lipipy.agents` executes only the tools you register;
treat model output as untrusted input.
