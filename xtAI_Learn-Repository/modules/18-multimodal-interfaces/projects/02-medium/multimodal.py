"""Core building blocks for P02 (medium) — early vs. late fusion, mutual disambiguation,
the correlation trap.

Module 18 — Multimodal Interfaces.

>>> YOUR TASK <<<
The data generation and the classifiers are given. Implement the FUSION functions
marked with `# TODO`. The rest (run.py, test_multimodal.py) then runs through.
The complete solution is in solution/.

Two experiments:
  (1) COMPLEMENTARITY: two modalities, each ambiguous on its own (~50 %),
      almost perfect together. Shows early vs. late fusion and the missing modality.
  (2) REDUNDANCY + THE CORRELATION TRAP: two modalities estimate the same thing;
      the fusion gain disappears when their noise is correlated.
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB


# --------------------------------------------------------------------------
# Dataset 1 — complementarity (XOR-like)   [given]
# --------------------------------------------------------------------------
def make_complementary(n_per_class=600, sigma=1.0, seed=0):
    """4 classes y in {0,1,2,3}. Bit1 = y>>1, bit0 = y&1.

    Modality A is informative ONLY about bit1 (it separates {0,1} from {2,3}),
    modality B is informative ONLY about bit0 (it separates {0,2} from {1,3}).
    => Each modality alone can reach at most 50 %. Together all 4 classes are
       separable.

    Returns: XA (n,1), XB (n,1), y (n,).
    """
    rng = np.random.default_rng(seed)
    classes = np.repeat(np.arange(4), n_per_class)
    bit1 = classes >> 1
    bit0 = classes & 1
    muA = np.array([-2.0, 2.0])
    muB = np.array([-2.0, 2.0])
    XA = (muA[bit1] + rng.normal(0, sigma, classes.size)).reshape(-1, 1)
    XB = (muB[bit0] + rng.normal(0, sigma, classes.size)).reshape(-1, 1)
    perm = rng.permutation(classes.size)
    return XA[perm], XB[perm], classes[perm]


# --------------------------------------------------------------------------
# Dataset 2 — redundancy with an adjustable noise correlation   [given]
# --------------------------------------------------------------------------
def make_redundant(n=6000, sep=1.0, sigma=1.0, rho=0.0, seed=0):
    """Binary classification. Both modalities measure the same class
    c in {-sep, +sep}, each plus Gaussian noise, correlated with rho.

      x_A = c + e_A,  x_B = c + e_B,  corr(e_A, e_B) = rho.

    rho=0  -> independent errors -> fusion helps maximally.
    rho=1  -> identical noise -> fusion brings NOTHING.

    Returns: XA (n,1), XB (n,1), y (n,) with y in {0,1}.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    c = np.where(y == 1, sep, -sep).astype(float)
    cov = sigma**2 * np.array([[1.0, rho], [rho, 1.0]])
    e = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    XA = (c + e[:, 0]).reshape(-1, 1)
    XB = (c + e[:, 1]).reshape(-1, 1)
    return XA, XB, y


# --------------------------------------------------------------------------
# One classifier per modality   [given]
# --------------------------------------------------------------------------
def fit_modality(X, y):
    """Trains a probabilistic classifier (GaussianNB) on ONE modality."""
    return GaussianNB().fit(X, y)


def predict_from_proba(proba):
    """The argmax per row.   [given]"""
    return np.asarray(proba).argmax(axis=1)


def accuracy(y_true, y_pred):
    """[given]"""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# --------------------------------------------------------------------------
# FUSION — the core   >>> THIS IS YOUR PART <<<
# --------------------------------------------------------------------------
def late_fusion_proba(proba_list, prior):
    """Late fusion via the Bayes product rule under conditional independence:

        P(y | z_A, z_B, ...) proportional to  prod_m P(y | z_m) / P(y)^(M-1)

    - Multiply the posterior matrices from proba_list element-wise.
    - Divide out the shared prior: it is in there M times but may count only
      ONCE -> divide by prior^(M-1) (M = the number of modalities).
    - Normalize every row so that it sums to 1.

    proba_list: a list of (n, K) arrays.  prior: a (K,) vector P(y).
    Returns:    the (n, K) fused, normalized posterior.

    A hint against numerical problems: guard against 0 with np.maximum(..., 1e-300).
    """
    # TODO: implement the Bayes product fusion
    raise NotImplementedError


def early_fusion_fit_predict(XA_tr, XB_tr, y_tr, XA_te, XB_te):
    """Early fusion: concatenate the feature vectors of both modalities
    (np.hstack), train ONE GaussianNB on them and return the predictions on
    the test data.

    >>> THIS IS YOUR PART <<<
    """
    # TODO: implement early fusion (hstack the features + one model)
    raise NotImplementedError
