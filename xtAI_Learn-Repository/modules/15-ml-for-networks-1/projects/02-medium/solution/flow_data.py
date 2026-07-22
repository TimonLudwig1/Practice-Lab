"""Loading and preprocessing the KDD Cup 99 flow data — infrastructure, fully given.

Classifying the dataset (script 3.6): KDD99 is outdated (1999), synthetic and TOO EASY.
We use it because it delivers real flow features with a realistic imbalance and no download
hurdle. It permits NO statement about real IDS quality.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99

CATEGORICAL = ["protocol_type", "service", "flag"]
LOG_COLUMNS = ["src_bytes", "dst_bytes", "duration"]

# Deliberately WEAK feature set: only volume/time/counters, none of the leaky
# KDD artifacts (serror_rate & co). Two reasons:
#   1. With all features a random forest reaches FPR = 0.0 -> there would be nothing to show.
#      That is an artifact of the dataset, not reality.
#   2. With ENCRYPTED traffic this is exactly what you really have: volume and timing
#      (script 2.1).
# This makes the detector realistically strong instead of unrealistically perfect.
FEATURES_META = ["duration", "src_bytes", "dst_bytes", "wrong_fragment",
                 "urgent", "hot", "count", "srv_count"]


def _bytes_to_str(column: pd.Series) -> pd.Series:
    """Decode bytes properly.

    Do NOT use `.str.strip("b'")`! strip() removes a character SET, not a prefix:
    b'back.' would become "ack" and b'bgp' would become "gp".
    """
    return column.apply(lambda v: v.decode() if isinstance(v, bytes) else str(v))


def load_flows(random_state: int = 0) -> pd.DataFrame:
    """Load and clean the KDD99 subset 'SA'. Returns a DataFrame with proper types."""
    data = fetch_kddcup99(subset="SA", percent10=True, as_frame=True,
                          random_state=random_state)
    df = data.frame.copy()
    df["labels"] = _bytes_to_str(df["labels"]).str.rstrip(".")
    numeric = [c for c in df.columns if c not in CATEGORICAL + ["labels"]]
    for c in CATEGORICAL:
        df[c] = _bytes_to_str(df[c])
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    assert "back" in set(df["labels"]), "label 'back' destroyed -> the strip trap!"
    return df


def build_xy(df: pd.DataFrame, features=None):
    """Design matrix X (log-transformed) and binary target y (1 = attack)."""
    features = FEATURES_META if features is None else features
    X = df[features].copy()
    for c in LOG_COLUMNS:
        if c in X.columns:
            X[c] = np.log1p(X[c])
    y = (df["labels"] != "normal").astype(int)
    return X.astype(float), y


def subsample_to_base_rate(y, scores, pi, rng):
    """Bring the test set to a target base rate `pi` by THINNING OUT THE ATTACKS
    (all normal flows are kept). Returns (y_new, scores_new), or None if there are too few
    attacks for the target rate.

    This simulates a more realistic network without changing the detector.
    """
    y = np.asarray(y)
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    n_pos = int(round(pi * len(neg) / (1 - pi)))
    if n_pos < 1 or n_pos > len(pos):
        return None
    idx = np.concatenate([neg, rng.choice(pos, n_pos, replace=False)])
    return y[idx], np.asarray(scores)[idx]
