"""Layer 1: numerical primitives shared by every domain.

Domains build on these instead of re-deriving primitives, so a better solver here
improves every algorithm that uses it. Each entry declares its complexity in the
registry. ``[x]`` = implemented, ``[ ]`` = planned.

Linear algebra
--------------
[x] Triangular solves, LU with partial pivoting, determinant, Cholesky, Householder QR
[ ] Givens QR
[x] Least squares via Householder QR
[ ] Least squares (normal equations, SVD), pseudo-inverse
[ ] Eigenvalues: power iteration, inverse iteration, QR algorithm, Lanczos, Arnoldi
[ ] SVD (one-sided Jacobi, randomized SVD)
[ ] Iterative solvers: Jacobi, Gauss-Seidel, SOR, conjugate gradient, GMRES
[ ] Matrix functions: expm (scaling and squaring), sqrtm, logm
[ ] Sparse matrices (COO, CSR, CSC) and sparse matvec

Roots and optimization
----------------------
[x] Secant, Newton-Raphson, Brent
[ ] Bisection, regula falsi
[x] Armijo line search, gradient descent, Newton, BFGS, L-BFGS
[ ] Wolfe line search
[ ] Conjugate gradient (nonlinear), trust region, Levenberg-Marquardt
[ ] Constrained: projected gradient, penalty / augmented Lagrangian, SQP
[ ] Linear programming (simplex, interior point), quadratic programming
[ ] Derivative-free: Nelder-Mead, golden section, CMA-ES

Calculus and differential equations
-----------------------------------
[ ] Finite differences, Richardson extrapolation
[ ] Quadrature: trapezoid, Simpson, Romberg, Gauss-Legendre, adaptive, Monte Carlo
[ ] ODE solvers: Euler, Heun, RK4, adaptive RK45 (Dormand-Prince), implicit Euler, BDF
[ ] SDE solvers: Euler-Maruyama, Milstein
[ ] PDE: finite-difference heat / wave / Laplace (explicit, Crank-Nicolson)
[ ] Interpolation: Lagrange, Newton divided differences, cubic splines

Transforms and signals
----------------------
[ ] DFT, radix-2 FFT, real FFT, convolution via FFT
[ ] DCT, wavelet transform (Haar, Daubechies)

Statistics and probability
--------------------------
[x] Streaming moments (Welford / Chan merge), covariance, correlation
[ ] Quantiles, histograms, kernel density estimation
[x] Standard normal cdf / ppf
[ ] Distributions: pdf / cdf / sampling for the common families
[x] Special functions: erf, erfc, log-sum-exp, softmax, log-softmax, sigmoid, softplus
[ ] Special functions: gamma, beta
[ ] Random number generation helpers, quasi-random sequences (Sobol, Halton)

Automatic differentiation
-------------------------
[ ] Forward mode (dual numbers)
[ ] Reverse mode (tape-based scalar and tensor autograd), gradient checking
"""

from lipipy.numerics.linalg import (
    cholesky,
    cholesky_solve,
    det,
    lstsq_qr,
    lu,
    lu_solve,
    qr,
    solve_triangular,
)
from lipipy.numerics.optimize import (
    LineSearchResult,
    OptimizeResult,
    RootResult,
    brentq,
    line_search_armijo,
    minimize,
    minimize_bfgs,
    minimize_gd,
    minimize_lbfgs,
    minimize_newton,
    newton,
)
from lipipy.numerics.special import (
    erf,
    erfc,
    log_sigmoid,
    log_softmax,
    logsumexp,
    norm_cdf,
    norm_ppf,
    sigmoid,
    softmax,
    softplus,
)
from lipipy.numerics.stats import RunningMoments, correlation, covariance

__all__ = [
    "LineSearchResult",
    "OptimizeResult",
    "RootResult",
    "RunningMoments",
    "brentq",
    "cholesky",
    "cholesky_solve",
    "correlation",
    "covariance",
    "det",
    "erf",
    "erfc",
    "line_search_armijo",
    "log_sigmoid",
    "log_softmax",
    "logsumexp",
    "lstsq_qr",
    "lu",
    "lu_solve",
    "minimize",
    "minimize_bfgs",
    "minimize_gd",
    "minimize_lbfgs",
    "minimize_newton",
    "newton",
    "norm_cdf",
    "norm_ppf",
    "qr",
    "sigmoid",
    "softmax",
    "softplus",
    "solve_triangular",
]
