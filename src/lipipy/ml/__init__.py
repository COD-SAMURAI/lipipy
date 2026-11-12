"""Classic machine learning implemented from scratch with NumPy only.

Catalog of algorithms. ``[x]`` = implemented, ``[ ]`` = planned.
Every entry must follow the checklist in CONTRIBUTING.md before it is ticked.

Supervised learning: regression
-------------------------------
[x] Linear regression (ordinary least squares, closed form + gradient descent)
[ ] Polynomial regression (feature expansion)
[x] Ridge regression (L2)
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
[x] k-nearest neighbours regression
[ ] Locally weighted regression (LOESS)
[ ] Splines and generalized additive models (GAMs)

Supervised learning: classification
-----------------------------------
[ ] Perceptron
[x] Logistic regression (binary, multinomial / softmax)
[ ] Linear discriminant analysis (LDA), quadratic discriminant analysis (QDA)
[x] Naive Bayes: Gaussian
[ ] Naive Bayes: multinomial, Bernoulli, complement
[x] k-nearest neighbours classifier (brute force, chunked distances)
[ ] k-nearest neighbours with KD-tree / ball tree
[ ] Support vector machines (hard/soft margin, SMO solver)
[ ] Kernel SVM (linear, polynomial, RBF, sigmoid kernels)
[ ] Gaussian process classification
[ ] Nearest centroid
[ ] One-vs-rest, one-vs-one, error-correcting output codes

Trees and ensembles
-------------------
[x] Decision trees: CART (Gini, entropy, MSE splits)
[ ] Decision trees: ID3, C4.5, cost-complexity pruning
[ ] Bagging
[x] Random forest
[ ] Extremely randomized trees (Extra-Trees)
[ ] AdaBoost (SAMME, SAMME.R)
[x] Gradient boosting machines (squared error, binary log loss)
[ ] Gradient boosting: multiclass, Huber / quantile losses
[ ] XGBoost-style boosting (second-order, regularized)
[ ] LightGBM-style histogram boosting (GOSS, EFB)
[ ] CatBoost-style ordered boosting
[ ] Stacking and blending
[ ] Voting ensembles
[ ] Isolation forest

Unsupervised learning: clustering
---------------------------------
[x] k-means (Lloyd), k-means++
[ ] Mini-batch k-means
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
[x] Principal component analysis (PCA)
[ ] Incremental PCA
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
[x] Metrics: accuracy, R^2 (lipipy.core); confusion matrix, precision, recall, F1,
    MSE, MAE, adjusted Rand index
[ ] Metrics: ROC-AUC, PR-AUC, log loss, silhouette, NMI
[ ] Calibration (Platt scaling, isotonic)
[ ] Scalers (standard, min-max, robust), one-hot and target encoding
[ ] Imputation (mean, kNN, iterative)
[ ] Feature selection (filter, wrapper, embedded, mutual information)
[ ] Class imbalance (SMOTE, class weights)
[ ] Explainability: permutation importance, partial dependence, LIME, SHAP
    (see lipipy.interpret)
"""

from lipipy.ml._base import ClassifierMixin, TransformerMixin
from lipipy.ml.cluster import KMeans
from lipipy.ml.decomposition import PCA
from lipipy.ml.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from lipipy.ml.linear import LinearRegression, LogisticRegression, Ridge
from lipipy.ml.metrics import (
    PrecisionRecallF1,
    adjusted_rand_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_f1,
)
from lipipy.ml.naive_bayes import GaussianNB
from lipipy.ml.neighbors import KNeighborsClassifier, KNeighborsRegressor
from lipipy.ml.tree import DecisionTreeClassifier, DecisionTreeRegressor, Tree

__all__ = [
    "PCA",
    "ClassifierMixin",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GaussianNB",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "KMeans",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "LinearRegression",
    "LogisticRegression",
    "PrecisionRecallF1",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "Ridge",
    "TransformerMixin",
    "Tree",
    "adjusted_rand_score",
    "confusion_matrix",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_recall_f1",
]
