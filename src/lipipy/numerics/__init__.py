"""Layer 1: numerical primitives shared by every domain.

Domains build on these instead of re-deriving primitives, so a better solver here
improves every algorithm that uses it. Each entry declares its complexity in the
registry. ``[x]`` = implemented, ``[ ]`` = planned.

Linear algebra
--------------
[ ] Gaussian elimination with partial pivoting, LU, Cholesky, QR (Householder, Givens)
[ ] Least squares (normal equations, QR, SVD), pseudo-inverse
[ ] Eigenvalues: power iteration, inverse iteration, QR algorithm, Lanczos, Arnoldi
[ ] SVD (one-sided Jacobi, randomized SVD)
[ ] Iterative solvers: Jacobi, Gauss-Seidel, SOR, conjugate gradient, GMRES
[ ] Matrix functions: expm (scaling and squaring), sqrtm, logm
[ ] Sparse matrices (COO, CSR, CSC) and sparse matvec

Roots and optimization
----------------------
[ ] Bisection, regula falsi, secant, Newton-Raphson, Brent
[ ] Line search (Armijo, Wolfe), gradient descent, Newton, BFGS, L-BFGS
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
[ ] Streaming moments (Welford), quantiles, histograms, kernel density estimation
[ ] Distributions: pdf / cdf / sampling for the common families
[ ] Special functions: gamma, beta, erf, log-sum-exp, softmax (numerically stable)
[ ] Random number generation helpers, quasi-random sequences (Sobol, Halton)

Automatic differentiation
-------------------------
[ ] Forward mode (dual numbers)
[ ] Reverse mode (tape-based scalar and tensor autograd), gradient checking
"""
