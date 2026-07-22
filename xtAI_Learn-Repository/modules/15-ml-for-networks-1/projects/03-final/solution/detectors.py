"""Detectors: supervised (misuse) vs. semi-supervised (anomaly).

The decisive difference is NOT the algorithm, it is WHAT THEY GET TO SEE:
  - SupervisedDetector : normal traffic + KNOWN attacks (with labels) -> models the bad
  - *AnomalyDetector   : ONLY normal traffic, no attack labels at all -> models the normal

All of them offer the same interface:  fit(...) / score(X) / predict(X) -> 1 = alarm, 0 = harmless.

IMPORTANT DESIGN DECISION - all anomaly detectors share ONE FPR budget:
Instead of leaving every method its own, incomparable threshold logic (`contamination`,
`nu`, `offset_` ...), we compute an anomaly score for every detector (higher = more anomalous)
and set the threshold to the (100 - target_fpr) percentile of the scores ON THE TRAINING NORMAL
TRAFFIC. This way every detector has ~target_fpr percent false alarms by construction, and the
zero-day recall becomes the only free quantity -> a FAIR comparison at the same price.

(As a side effect this avoids two real pitfalls I stumbled over while building:
 - LocalOutlierFactor.predict() is unusable here: on this data its `offset_` derails to
   ~-1.3e6, so an alarm is NEVER raised - even though the scores separate cleanly.
 - SGDOneClassSVM is LINEAR and separates nothing here (score normal 3.946 vs. smurf 3.959).
   Only with a kernel approximation (Nystroem) does it work.)
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance

TARGET_FPR_PERCENT = 1.0    # shared false alarm budget of all anomaly detectors


class SupervisedDetector:
    """The classic misuse detector: learns to recognize the KNOWN attacks."""
    name = "Supervised RF (knows the known attacks)"

    def __init__(self, random_state=0):
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                            random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X).astype(int)


class _AnomalyBase:
    """Scaffold: scale, fit on normal traffic ONLY, calibrate the threshold to the FPR budget."""
    name = "Anomaly"
    deduplicate = False         # overridden by LOF - see LOFDetector

    def __init__(self, target_fpr_percent=TARGET_FPR_PERCENT):
        self.target_fpr_percent = target_fpr_percent

    def fit(self, X_normal):
        if self.deduplicate:
            X_normal = X_normal.drop_duplicates()
        self.scaler = StandardScaler().fit(X_normal)
        Xs = self.scaler.transform(X_normal)
        self._fit_core(Xs)
        # threshold such that ~target_fpr_percent alarms arise on the training normal traffic
        train_scores = self._raw_score(Xs)
        self.threshold_ = float(np.percentile(train_scores, 100.0 - self.target_fpr_percent))
        return self

    def score(self, X):
        return self._raw_score(self.scaler.transform(X))

    def predict(self, X):
        return (self.score(X) > self.threshold_).astype(int)


class IsolationForestDetector(_AnomalyBase):
    """Isolates points through random splits: anomalies need fewer splits."""
    name = "Isolation forest"

    def __init__(self, random_state=0, **kw):
        super().__init__(**kw)
        self.random_state = random_state

    def _fit_core(self, Xs):
        self.model = IsolationForest(n_estimators=200, random_state=self.random_state).fit(Xs)

    def _raw_score(self, Xs):
        return -self.model.score_samples(Xs)       # higher = more anomalous


class MahalanobisDetector(_AnomalyBase):
    """Gaussian model of the normal traffic: alarm at a large Mahalanobis distance.

    The simplest method imaginable - and a fair yardstick: if a complex detector does not beat
    this one, it is not worth its effort.
    """
    name = "Mahalanobis Gaussian"

    def _fit_core(self, Xs):
        # tiny noise against a singular covariance (constant/duplicated columns)
        Xs = Xs + 1e-9 * np.random.default_rng(0).standard_normal(Xs.shape)
        self.cov = EmpiricalCovariance().fit(Xs)

    def _raw_score(self, Xs):
        return self.cov.mahalanobis(Xs)


class OneClassSVMDetector(_AnomalyBase):
    """One-class SVM with a Nystroem kernel approximation.

    The exact OneClassSVM is too slow at ~70k points (O(n^2)); the pure SGD variant is linear
    and separates NOTHING here. Nystroem approximates the RBF kernel and makes the fast SGD
    variant usable.
    """
    name = "One-class SVM (Nystroem)"

    def __init__(self, nu=0.01, gamma=0.1, n_components=200, random_state=0, **kw):
        super().__init__(**kw)
        self.nu, self.gamma = nu, gamma
        self.n_components, self.random_state = n_components, random_state

    def _fit_core(self, Xs):
        self.model = make_pipeline(
            Nystroem(gamma=self.gamma, n_components=self.n_components,
                     random_state=self.random_state),
            SGDOneClassSVM(nu=self.nu, random_state=self.random_state),
        ).fit(Xs)

    def _raw_score(self, Xs):
        return -self.model.decision_function(Xs)


class LOFDetector(_AnomalyBase):
    """Local outlier factor: compares the local density of a point with that of its neighbours.

    TWO pitfalls that made this detector completely unusable here at first:

    1. **Do not use predict().** On this data its `offset_` derails to ~-1.3e6, so an alarm is
       NEVER raised. We use score_samples() plus our own threshold.

    2. **Duplicates destroy LOF** (hence `deduplicate = True`). KDD99 is massively redundant
       (script 3.6): 13.4 % of the normal traffic are EXACT duplicates. For LOF that is fatal -
       if the k neighbours of a point are identical to it, their distance is 0, the local
       density goes to infinity and the LOF ratio explodes. Measured:
           with duplicates : training scores up to 1.55e9 -> 99 % threshold 1.09e6 -> recall 0.000
           deduplicated    : training scores up to 393    -> 99 % threshold 1.98   -> recall 1.000
       So the redundancy of the dataset does not merely inflate metrics, it paralyses entire
       classes of methods.
    """
    name = "Local outlier factor"
    deduplicate = True

    def __init__(self, n_neighbors=20, **kw):
        super().__init__(**kw)
        self.n_neighbors = n_neighbors

    def _fit_core(self, Xs):
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True,
                                        n_jobs=-1).fit(Xs)

    def _raw_score(self, Xs):
        return -self.model.score_samples(Xs)
