"""Laden und Vorverarbeiten der KDD-Cup-99-Flowdaten — Infrastruktur, vollstaendig gegeben.

Datensatz-Einordnung (Skript 3.6): KDD99 ist veraltet (1999), synthetisch und ZU LEICHT.
Wir nutzen ihn, weil er ohne Download-Huerde echte Flow-Features mit realistischem
Ungleichgewicht liefert. Er erlaubt KEINE Aussage ueber reale IDS-Guete.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99

KATEGORIAL = ["protocol_type", "service", "flag"]
LOG_SPALTEN = ["src_bytes", "dst_bytes", "duration"]

# Bewusst SCHWACHER Featuresatz: nur Volumen/Zeit/Zaehler, keine der leaky
# KDD-Artefakte (serror_rate & Co). Zwei Gruende:
#   1. Mit allen Features erreicht ein Random Forest FPR = 0.0 -> es gaebe nichts zu zeigen.
#      Das ist ein Artefakt des Datensatzes, nicht Realitaet.
#   2. Bei VERSCHLUESSELTEM Verkehr hat man real genau das: Volumen und Timing (Skript 2.1).
# Damit ist der Detektor realistisch stark statt unrealistisch perfekt.
FEATURES_META = ["duration", "src_bytes", "dst_bytes", "wrong_fragment",
                 "urgent", "hot", "count", "srv_count"]


def _bytes_zu_str(spalte: pd.Series) -> pd.Series:
    """Bytes sauber dekodieren.

    NICHT `.str.strip("b'")` benutzen! strip() entfernt eine Zeichen-MENGE, kein Praefix:
    aus b'back.' wuerde "ack" und aus b'bgp' wuerde "gp".
    """
    return spalte.apply(lambda v: v.decode() if isinstance(v, bytes) else str(v))


def lade_flows(random_state: int = 0) -> pd.DataFrame:
    """KDD99-Subset 'SA' laden und saeubern. Rueckgabe: DataFrame mit sauberen Typen."""
    data = fetch_kddcup99(subset="SA", percent10=True, as_frame=True,
                          random_state=random_state)
    df = data.frame.copy()
    df["labels"] = _bytes_zu_str(df["labels"]).str.rstrip(".")
    numerisch = [c for c in df.columns if c not in KATEGORIAL + ["labels"]]
    for c in KATEGORIAL:
        df[c] = _bytes_zu_str(df[c])
    for c in numerisch:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    assert "back" in set(df["labels"]), "Label 'back' zerstoert -> strip-Falle!"
    return df


def baue_xy(df: pd.DataFrame, features=None):
    """Design-Matrix X (log-transformiert) und binaeres Ziel y (1 = Angriff)."""
    features = FEATURES_META if features is None else features
    X = df[features].copy()
    for c in LOG_SPALTEN:
        if c in X.columns:
            X[c] = np.log1p(X[c])
    y = (df["labels"] != "normal").astype(int)
    return X.astype(float), y


def subsample_zu_basisrate(y, scores, pi, rng):
    """Testmenge auf eine Ziel-Basisrate `pi` bringen, indem ANGRIFFE ausgeduennt werden
    (alle Normalflows bleiben). Rueckgabe: (y_neu, scores_neu) oder None, wenn zu wenige
    Angriffe fuer die Zielrate vorhanden sind.

    Damit simulieren wir ein realistischeres Netz, ohne den Detektor zu veraendern.
    """
    y = np.asarray(y)
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    n_pos = int(round(pi * len(neg) / (1 - pi)))
    if n_pos < 1 or n_pos > len(pos):
        return None
    idx = np.concatenate([neg, rng.choice(pos, n_pos, replace=False)])
    return y[idx], np.asarray(scores)[idx]
