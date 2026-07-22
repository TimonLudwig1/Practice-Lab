"""Load the flow data and build the zero-day scenario.

The core idea of the scenario: we pretend that one attack type appears NEWLY in the future.
The training set must therefore NEVER see it — neither as a label (supervised) nor at all
(the anomaly detector sees only normal traffic).

This is the only honest way to measure "does my IDS detect the unknown?". A random split would
be self-deception here (script 3.5: data leakage).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split

CATEGORICAL = ["protocol_type", "service", "flag"]
LOG_COLUMNS = ["src_bytes", "dst_bytes", "duration"]

# Deliberately WITHOUT the leaky KDD artifacts (serror_rate & co, script 3.6): volume, timing,
# counters and a few host context features. This corresponds to what you really get from flow
# records (also for encrypted traffic).
FEATURES = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
            "count", "srv_count", "dst_host_count", "dst_host_srv_count",
            "logged_in", "root_shell", "num_failed_logins"]


def _bytes_to_str(column: pd.Series) -> pd.Series:
    """Decode bytes properly.

    Do NOT use `.str.strip("b'")`! strip() removes a character SET, not a prefix:
    b'back.' would become "ack", b'bgp' would become "gp".
    """
    return column.apply(lambda v: v.decode() if isinstance(v, bytes) else str(v))


def load_flows(random_state: int = 0) -> pd.DataFrame:
    data = fetch_kddcup99(subset="SA", percent10=True, as_frame=True,
                          random_state=random_state)
    df = data.frame.copy()
    df["labels"] = _bytes_to_str(df["labels"]).str.rstrip(".")
    for c in CATEGORICAL:
        df[c] = _bytes_to_str(df[c])
    for c in [c for c in df.columns if c not in CATEGORICAL + ["labels"]]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    assert "back" in set(df["labels"]), "label 'back' destroyed -> the strip trap!"
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in LOG_COLUMNS:
        X[c] = np.log1p(X[c])
    return X.astype(float)


class ZeroDayScenario:
    """Splits the data so that certain attack types stay INVISIBLE during training.

    Attributes:
      normal_train / normal_test : normal traffic (only normal_train may be trained on)
      known_train / known_test   : attacks the supervised detector is allowed to know
      zeroday : dict {attack type -> feature matrix} of the *never seen* attacks
    """

    def __init__(self, known_attacks=("smurf",), test_size=0.3, random_state=0):
        df = load_flows(random_state=random_state)
        X = build_features(df)
        lab = df["labels"]
        self.labels = lab
        self.known_attacks = list(known_attacks)

        normal = X[lab == "normal"]
        self.normal_train, self.normal_test = train_test_split(
            normal, test_size=test_size, random_state=random_state)

        X_known = X[lab.isin(self.known_attacks)]
        self.known_train, self.known_test = train_test_split(
            X_known, test_size=test_size, random_state=random_state)

        self.zeroday = {
            a: X[lab == a] for a in sorted(lab.unique())
            if a != "normal" and a not in self.known_attacks
        }

    def supervised_training_data(self):
        """(X, y) for the supervised detector: normal + ONLY the known attacks."""
        X = pd.concat([self.normal_train, self.known_train])
        y = np.r_[np.zeros(len(self.normal_train)), np.ones(len(self.known_train))]
        return X, y

    def overview(self) -> str:
        z = {a: len(v) for a, v in self.zeroday.items()}
        return (f"Normal: {len(self.normal_train)} train / {len(self.normal_test)} test\n"
                f"Known attacks {self.known_attacks}: "
                f"{len(self.known_train)} train / {len(self.known_test)} test\n"
                f"Zero-day (never in training): {z}")
