"""Classic machine learning implemented from scratch with NumPy only.

Catalog of algorithms. ``[x]`` = implemented, ``[ ]`` = planned.
Every entry must follow the checklist in CONTRIBUTING.md before it is ticked.

Supervised learning: regression
-------------------------------
[x] Linear regression (ordinary least squares, closed form + gradient descent)
[ ] Polynomial regression (feature expansion)
[ ] Ridge regression (L2)
[ ] Lasso regression (L1, coordinate descent)
[ ] Elastic net
[ ] Bayesian linear regression
[ ] Least angle regression (LARS)
[ ] Huber / robust regression, RANSAC
[ ] Quantile regression
[ ] Generalized linear models (Poisson, Gamma, Tweedie)
[ ] Isotonic regression
[ ] Kernel ridge regression
[ ] Gaussian process regression
[ ] Support vector regression (SVR)
[ ] k-nearest neighbours regression
[ ] Locally weighted regression (LOESS)
[ ] Splines and generalized additive models (GAMs)

Supervised learning: classification
-----------------------------------
[ ] Perceptron
[ ] Logistic regression (binary, multinomial / softmax)
[ ] Linear discriminant analysis (LDA), quadratic discriminant analysis (QDA)
[ ] Naive Bayes (Gaussian, multinomial, Bernoulli, complement)
[ ] k-nearest neighbours classifier (brute force, KD-tree, ball tree)
[ ] Support vector machines (hard/soft margin, SMO solver)
[ ] Kernel SVM (linear, polynomial, RBF, sigmoid kernels)
[ ] Gaussian process classification
[ ] Nearest centroid
[ ] One-vs-rest, one-vs-one, error-correcting output codes

Trees and ensembles
-------------------
[ ] Decision trees: ID3, C4.5, CART (Gini, entropy, MSE splits; pruning)
[ ] Bagging
[ ] Random forest
[ ] Extremely randomized trees (Extra-Trees)
[ ] AdaBoost (SAMME, SAMME.R)
[ ] Gradient boosting machines
[ ] XGBoost-style boosting (second-order, regularized)
[ ] LightGBM-style histogram boosting (GOSS, EFB)
[ ] CatBoost-style ordered boosting
[ ] Stacking and blending
[ ] Voting ensembles
[ ] Isolation forest

Unsupervised learning: clustering
---------------------------------
[ ] k-means, k-means++, mini-batch k-means
[ ] k-medoids (PAM)
[ ] Hierarchical / agglomerative clustering (single, complete, average, Ward)
[ ] DBSCAN, HDBSCAN, OPTICS
[ ] Mean shift
[ ] Spectral clustering
[ ] Gaussian mixture models (EM algorithm)
[ ] Affinity propagation
[ ] BIRCH
[ ] Fuzzy c-means
[ ] Self-organizing maps

Dimensionality reduction and representation
-------------------------------------------
[ ] Principal component analysis (PCA), incremental PCA
[ ] Kernel PCA
[ ] Probabilistic PCA, factor analysis
[ ] Singular value decomposition (truncated SVD / LSA)
[ ] Independent component analysis (FastICA)
[ ] Non-negative matrix factorization (NMF)
[ ] Linear discriminant analysis as a projection
[ ] Multidimensional scaling (MDS)
[ ] Isomap, locally linear embedding (LLE), Laplacian eigenmaps
[ ] t-SNE
[ ] UMAP
[ ] Random projections (Johnson-Lindenstrauss)
[ ] Dictionary learning, sparse coding

Probabilistic and graphical models
----------------------------------
[ ] Expectation-maximization (general)
[ ] Hidden Markov models (forward-backward, Viterbi, Baum-Welch)
[ ] Bayesian networks (variable elimination, belief propagation)
[ ] Markov random fields, conditional random fields (CRF)
[ ] Latent Dirichlet allocation (LDA topic model)
[ ] Kalman filter, extended / unscented Kalman filter
[ ] Particle filter (sequential Monte Carlo)
[ ] Markov chain Monte Carlo: Metropolis-Hastings, Gibbs, Hamiltonian Monte Carlo
[ ] Variational inference (mean-field, CAVI)
[ ] Dirichlet process mixtures

Time series and spatiotemporal
------------------------------
[ ] Moving averages, exponential smoothing, Holt-Winters
[ ] AR, MA, ARMA, ARIMA, SARIMA
[ ] VAR (vector autoregression)
[ ] GARCH
[ ] State-space models and Kalman smoothing
[ ] Dynamic time warping (DTW)
[ ] Kriging / Gaussian process spatial interpolation
[ ] Spatial autocorrelation (Moran's I)

Anomaly detection
-----------------
[ ] Z-score / IQR rules, Mahalanobis distance
[ ] One-class SVM
[ ] Local outlier factor (LOF)
[ ] Elliptic envelope (robust covariance)

Recommender systems
-------------------
[ ] Collaborative filtering (user / item kNN)
[ ] Matrix factorization (ALS, SGD, SVD++)
[ ] Factorization machines
[ ] Association rules (Apriori, FP-Growth)

Semi-supervised and other learning settings
-------------------------------------------
[ ] Label propagation / label spreading
[ ] Self-training, co-training
[ ] Active learning (uncertainty sampling, query-by-committee)
[ ] Multi-armed bandits (see lipipy.rl)
[ ] Online learning (perceptron, passive-aggressive, FTRL)

Model selection, evaluation and preprocessing
---------------------------------------------
[ ] Train/test split, k-fold, stratified, time-series cross-validation
[ ] Grid search, random search, Bayesian optimization (TPE, GP-based)
[ ] Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss,
    MSE, MAE, R^2, silhouette, adjusted Rand index, NMI
[ ] Calibration (Platt scaling, isotonic)
[ ] Scalers (standard, min-max, robust), one-hot and target encoding
[ ] Imputation (mean, kNN, iterative)
[ ] Feature selection (filter, wrapper, embedded, mutual information)
[ ] Class imbalance (SMOTE, class weights)
[ ] Explainability: permutation importance, partial dependence, LIME, SHAP
    (see lipipy.interpret)
"""

from lipipy.ml._base import ClassifierMixin, TransformerMixin
from lipipy.ml.linear import LinearRegression, Ridge
from lipipy.ml.metrics import (
    PrecisionRecallF1,
    adjusted_rand_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_f1,
)

__all__ = [
    "ClassifierMixin",
    "LinearRegression",
    "PrecisionRecallF1",
    "Ridge",
    "TransformerMixin",
    "adjusted_rand_score",
    "confusion_matrix",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_recall_f1",
]
