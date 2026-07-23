"""Core building blocks for P02 (medium) — early vs. late fusion, mutual disambiguation,
the correlation trap. The complete solution.

Module 18 — Multimodal Interfaces.

Two experiments:
  (1) COMPLEMENTARITY: two modalities, each ambiguous on its own (~50 %),
      almost perfect together. Shows early vs. late fusion and the missing modality.
  (2) REDUNDANCY + THE CORRELATION TRAP: two modalities estimate the same thing;
      the fusion gain disappears when their noise is correlated.
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB


# --------------------------------------------------------------------------
# Dataset 1 — complementarity (XOR-like)
# --------------------------------------------------------------------------
def make_complementary(n_per_class=600, sigma=1.0, seed=0):
    """4 classes y in {0,1,2,3}. Bit1 = y>>1, bit0 = y&1.

    Modality A is informative ONLY about bit1 (it separates {0,1} from {2,3}),
    modality B is informative ONLY about bit0 (it separates {0,2} from {1,3}).
    => Each modality alone can reach at most 50 % (two classes share the same
       distribution). Together all 4 classes are separable.

    Returns: XA (n,1), XB (n,1), y (n,).
    """
    rng = np.random.default_rng(seed)
    classes = np.repeat(np.arange(4), n_per_class)
    bit1 = classes >> 1          # informative for A
    bit0 = classes & 1           # informative for B
    muA = np.array([-2.0, 2.0])  # the centres for bit1
    muB = np.array([-2.0, 2.0])  # the centres for bit0
    XA = (muA[bit1] + rng.normal(0, sigma, classes.size)).reshape(-1, 1)
    XB = (muB[bit0] + rng.normal(0, sigma, classes.size)).reshape(-1, 1)
    perm = rng.permutation(classes.size)
    return XA[perm], XB[perm], classes[perm]


# --------------------------------------------------------------------------
# Dataset 2 — redundancy with an adjustable noise correlation
# --------------------------------------------------------------------------
def make_redundant(n=6000, sep=1.0, sigma=1.0, rho=0.0, seed=0):
    """Binary classification. Both modalities measure the same class
    c in {-sep, +sep}, each plus Gaussian noise. The noise of the two
    modalities is correlated with the correlation coefficient rho.

      x_A = c + e_A,  x_B = c + e_B,  corr(e_A, e_B) = rho.

    rho=0  -> independent errors -> fusion helps maximally.
    rho=1  -> identical noise -> fusion brings NOTHING (both say the same wrong thing).

    Returns: XA (n,1), XB (n,1), y (n,) with y in {0,1}.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    c = np.where(y == 1, sep, -sep).astype(float)
    # a bivariate normal distribution for (e_A, e_B) with correlation rho
    cov = sigma**2 * np.array([[1.0, rho], [rho, 1.0]])
    e = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    XA = (c + e[:, 0]).reshape(-1, 1)
    XB = (c + e[:, 1]).reshape(-1, 1)
    return XA, XB, y


# --------------------------------------------------------------------------
# One classifier per modality
# --------------------------------------------------------------------------
def fit_modality(X, y):
    """Trains a probabilistic classifier (GaussianNB) on ONE modality.
    Returns the fitted model."""
    return GaussianNB().fit(X, y)


# --------------------------------------------------------------------------
# FUSION — the core (to be implemented in the stub version)
# --------------------------------------------------------------------------
def late_fusion_proba(proba_list, prior):
    """Late fusion via the Bayes product rule under conditional independence:

        P(y | z_A, z_B, ...) proportional to  prod_m P(y | z_m) / P(y)^(M-1)

    Every factor P(y | z_m) is the posterior of one modality (n, K).
    The shared prior P(y) may count only ONCE, which is why we divide by
    P(y)^(M-1). Afterwards normalize row-wise.

    proba_list: a list of (n, K) arrays (one posterior matrix per modality).
    prior:      a (K,) vector P(y).
    Returns:    the (n, K) fused, normalized posterior.
    """
    M = len(proba_list)
    prior = np.asarray(prior, dtype=float)
    # the product of the posteriors
    prod = np.ones_like(proba_list[0], dtype=float)
    for p in proba_list:
        prod = prod * p
    # divide out the shared prior (it is in there M-1 times too often)
    prod = prod / np.maximum(prior[None, :] ** (M - 1), 1e-300)
    # normalize row-wise
    denom = prod.sum(axis=1, keepdims=True)
    return prod / np.maximum(denom, 1e-300)


def predict_from_proba(proba):
    """The argmax per row."""
    return np.asarray(proba).argmax(axis=1)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# --------------------------------------------------------------------------
# Helpers for early fusion
# --------------------------------------------------------------------------
def early_fusion_fit_predict(XA_tr, XB_tr, y_tr, XA_te, XB_te):
    """Early fusion: concatenate the feature vectors and train ONE model."""
    Xtr = np.hstack([XA_tr, XB_tr])
    Xte = np.hstack([XA_te, XB_te])
    model = GaussianNB().fit(Xtr, y_tr)
    return model.predict(Xte)
