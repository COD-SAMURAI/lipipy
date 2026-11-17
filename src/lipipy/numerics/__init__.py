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
[x] Iterative solvers: conjugate gradient (Jacobi / user preconditioner), restarted GMRES
[ ] Iterative solvers: Jacobi, Gauss-Seidel, SOR
[ ] Matrix functions: expm (scaling and squaring), sqrtm, logm
[x] Sparse matrices: CSR (from dense / COO triplets), matvec and matmat
[ ] Sparse matrices: COO, CSC formats

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
[x] Quadrature: trapezoid, Simpson, Gauss-Legendre (Golub-Welsch), adaptive Gauss-Kronrod
    (G7-K15), Monte Carlo
[ ] Quadrature: Romberg
[x] ODE solvers: RK4, adaptive RK45 (Dormand-Prince, dense output), implicit Euler (Newton)
[ ] ODE solvers: Euler, Heun, BDF
[x] SDE solvers: Euler-Maruyama, Milstein
[ ] PDE: finite-difference heat / wave / Laplace (explicit, Crank-Nicolson)
[x] Interpolation: piecewise linear, cubic splines (not-a-knot, natural, clamped)
[ ] Interpolation: Lagrange, Newton divided differences

Transforms and signals
----------------------
[x] FFT: radix-2 Cooley-Tukey, Bluestein (any length), real FFT, convolution via FFT
[ ] DCT, wavelet transform (Haar, Daubechies)

Statistics and probability
--------------------------
[x] Streaming moments (Welford / Chan merge), covariance, correlation
[ ] Quantiles, histograms, kernel density estimation
[x] Standard normal cdf / ppf
[ ] Distributions: pdf / cdf / sampling for the common families
[x] Special functions: erf, erfc, log-sum-exp, softmax, log-softmax, sigmoid, softplus
[ ] Special functions: gamma, beta
[x] Quasi-random sequences: Sobol (21 dims), Halton (optional scrambling), L2-star discrepancy
[ ] Random number generation helpers

Automatic differentiation
-------------------------
[x] Forward mode (dual numbers), derivative, Jacobian-vector products
[x] Reverse mode (tensor autograd with broadcasting), grad / value_and_grad, gradient checking
"""

from lipipy.numerics.autodiff import (
    Dual,
    Tensor,
    check_grad,
    concatenate,
    derivative,
    grad,
    jvp,
    maximum,
    stack,
    value_and_grad,
    where,
)
from lipipy.numerics.fft import fft, fftconvolve, fftfreq, ifft, irfft, rfft, rfftfreq
from lipipy.numerics.integrate import (
    MonteCarloResult,
    QuadResult,
    gauss_legendre,
    gauss_legendre_nodes,
    monte_carlo,
    quad,
    simpson,
    trapezoid,
)
from lipipy.numerics.interpolate import CubicSpline, interp_linear
from lipipy.numerics.iterative import IterativeResult, conjugate_gradient, gmres
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
from lipipy.numerics.ode import ODEResult, solve_ode
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
from lipipy.numerics.qmc import halton, l2_star_discrepancy, sobol
from lipipy.numerics.sde import SDEResult, euler_maruyama, milstein
from lipipy.numerics.sparse import CSRMatrix
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
    "CSRMatrix",
    "CubicSpline",
    "Dual",
    "IterativeResult",
    "LineSearchResult",
    "MonteCarloResult",
    "ODEResult",
    "OptimizeResult",
    "QuadResult",
    "RootResult",
    "RunningMoments",
    "SDEResult",
    "Tensor",
    "brentq",
    "check_grad",
    "cholesky",
    "cholesky_solve",
    "concatenate",
    "conjugate_gradient",
    "correlation",
    "covariance",
    "derivative",
    "det",
    "erf",
    "erfc",
    "euler_maruyama",
    "fft",
    "fftconvolve",
    "fftfreq",
    "gauss_legendre",
    "gauss_legendre_nodes",
    "gmres",
    "grad",
    "halton",
    "ifft",
    "interp_linear",
    "irfft",
    "jvp",
    "l2_star_discrepancy",
    "line_search_armijo",
    "log_sigmoid",
    "log_softmax",
    "logsumexp",
    "lstsq_qr",
    "lu",
    "lu_solve",
    "maximum",
    "milstein",
    "minimize",
    "minimize_bfgs",
    "minimize_gd",
    "minimize_lbfgs",
    "minimize_newton",
    "monte_carlo",
    "newton",
    "norm_cdf",
    "norm_ppf",
    "qr",
    "quad",
    "rfft",
    "rfftfreq",
    "sigmoid",
    "simpson",
    "sobol",
    "softmax",
    "softplus",
    "solve_ode",
    "solve_triangular",
    "stack",
    "trapezoid",
    "value_and_grad",
    "where",
]
