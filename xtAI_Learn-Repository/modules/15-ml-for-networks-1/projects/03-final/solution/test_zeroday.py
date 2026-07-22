"""Tests for the zero-day scenario and the detectors.

Call:  python test_zeroday.py      (~1-2 min: several detectors are trained)
"""
import numpy as np
import pandas as pd

from flow_data import ZeroDayScenario, load_flows, build_features, FEATURES
from detectors import (SupervisedDetector, IsolationForestDetector,
                       MahalanobisDetector, OneClassSVMDetector, LOFDetector)

# load and split once - that is expensive, so reuse it for all tests
SCEN = ZeroDayScenario(known_attacks=("smurf",), random_state=0)


# --------------------------- data / scenario ---------------------------
def test_bytes_are_decoded_correctly():
    df = load_flows()
    # the strip("b'") trap would turn 'back' into 'ack' and 'bgp' into 'gp'
    assert "back" in set(df["labels"])
    assert not any(l.startswith("b'") for l in df["labels"].unique())


def test_features_are_finite_and_numeric():
    X = build_features(load_flows())
    assert list(X.columns) == FEATURES
    assert np.all(np.isfinite(X.values))
    assert X.dtypes.map(lambda d: d.kind == "f").all()


def test_zeroday_attacks_are_NOT_in_training():
    # The core of the scenario: no zero-day type may be in the training data.
    assert "smurf" not in SCEN.zeroday
    assert "neptune" in SCEN.zeroday, "neptune has to be a zero-day"
    X_train, y = SCEN.supervised_training_data()
    # training set = normal + known attacks, nothing else
    assert len(X_train) == len(SCEN.normal_train) + len(SCEN.known_train)
    assert set(np.unique(y)) == {0.0, 1.0}
    assert y.sum() == len(SCEN.known_train)


def test_normal_train_and_test_are_disjoint():
    # At the value level disjointness is impossible (the dataset contains 13 % exact
    # duplicates, script 3.6) - what is testable is the disjointness of the INDICES.
    assert not set(SCEN.normal_train.index) & set(SCEN.normal_test.index)


def test_anomaly_detector_only_sees_normal_traffic():
    # fit() takes exactly ONE argument (normal traffic) - no y, no attacks
    det = MahalanobisDetector().fit(SCEN.normal_train)
    assert hasattr(det, "threshold_")


# --------------------------- calibration ---------------------------
def test_fpr_budget_is_respected():
    # By construction every anomaly detector should have ~1 % false alarms.
    for D in (IsolationForestDetector(), MahalanobisDetector(),
              OneClassSVMDetector(), LOFDetector()):
        det = D.fit(SCEN.normal_train)
        fpr = det.predict(SCEN.normal_test).mean()
        assert 0.002 < fpr < 0.03, f"{det.name}: FPR {fpr:.4f} misses the 1 % budget"


def test_lof_needs_deduplication():
    # The documented pitfall: with duplicates the LOF scores explode
    # (up to ~1.5e9) and so does the percentile threshold -> nothing is ever detected.
    det_ok = LOFDetector().fit(SCEN.normal_train)          # deduplicate = True
    assert det_ok.deduplicate is True
    assert det_ok.predict(SCEN.known_test).mean() > 0.9, "LOF should find smurf"

    det_broken = LOFDetector()
    det_broken.deduplicate = False                         # deliberately sabotaged
    det_broken.fit(SCEN.normal_train)
    assert det_broken.threshold_ > 100 * det_ok.threshold_, \
        "without deduplication the threshold should derail"


# --------------------------- the core statement ---------------------------
def test_supervised_detects_the_known_attack():
    X, y = SCEN.supervised_training_data()
    det = SupervisedDetector().fit(X, y)
    assert det.predict(SCEN.known_test).mean() > 0.95       # smurf: practically perfect
    assert det.predict(SCEN.normal_test).mean() < 0.01      # hardly any false alarms


def test_supervised_is_blind_to_zeroday():
    # THE core statement of the project: what it has never seen, it does not see.
    X, y = SCEN.supervised_training_data()
    det = SupervisedDetector().fit(X, y)
    recall_neptune = det.predict(SCEN.zeroday["neptune"]).mean()
    assert recall_neptune < 0.05, f"unexpectedly good: {recall_neptune}"


def test_anomaly_detector_finds_zeroday_without_ever_having_seen_attacks():
    det = IsolationForestDetector().fit(SCEN.normal_train)   # only normal traffic!
    assert det.predict(SCEN.zeroday["neptune"]).mean() > 0.8


def test_anomaly_beats_supervised_on_zeroday():
    X, y = SCEN.supervised_training_data()
    sup = SupervisedDetector().fit(X, y)
    ano = IsolationForestDetector().fit(SCEN.normal_train)
    zd = [a for a, Xa in SCEN.zeroday.items() if len(Xa) >= 10]
    r_sup = np.mean([sup.predict(SCEN.zeroday[a]).mean() for a in zd])
    r_ano = np.mean([ano.predict(SCEN.zeroday[a]).mean() for a in zd])
    assert r_ano > r_sup, (r_ano, r_sup)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("All tests passed.")
