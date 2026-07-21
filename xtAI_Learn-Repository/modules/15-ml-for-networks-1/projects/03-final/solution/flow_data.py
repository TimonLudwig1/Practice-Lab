"""Flow-Daten laden + Zero-Day-Szenario aufbauen.

Kernidee des Szenarios: Wir tun so, als kaeme ein Angriffstyp in der Zukunft NEU dazu.
Das Trainingsset darf ihn deshalb NIE sehen — weder als Label (supervised) noch ueberhaupt
(Anomaliedetektor sieht nur Normalverkehr).

Das ist die einzige ehrliche Art, "erkennt mein IDS Unbekanntes?" zu messen. Ein zufaelliger
Split waere hier Selbstbetrug (Skript 3.5: Data Leakage).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split

KATEGORIAL = ["protocol_type", "service", "flag"]
LOG_SPALTEN = ["src_bytes", "dst_bytes", "duration"]

# Bewusst OHNE die leaky KDD-Artefakte (serror_rate & Co, Skript 3.6): Volumen, Timing,
# Zaehler und ein paar Host-Kontextmerkmale. Entspricht dem, was man real (auch bei
# verschluesseltem Verkehr) aus Flow-Records bekommt.
FEATURES = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
            "count", "srv_count", "dst_host_count", "dst_host_srv_count",
            "logged_in", "root_shell", "num_failed_logins"]


def _bytes_zu_str(spalte: pd.Series) -> pd.Series:
    """Bytes sauber dekodieren.

    NICHT `.str.strip("b'")`! strip() entfernt eine Zeichen-MENGE, kein Praefix:
    aus b'back.' wuerde "ack", aus b'bgp' wuerde "gp".
    """
    return spalte.apply(lambda v: v.decode() if isinstance(v, bytes) else str(v))


def lade_flows(random_state: int = 0) -> pd.DataFrame:
    data = fetch_kddcup99(subset="SA", percent10=True, as_frame=True,
                          random_state=random_state)
    df = data.frame.copy()
    df["labels"] = _bytes_zu_str(df["labels"]).str.rstrip(".")
    for c in KATEGORIAL:
        df[c] = _bytes_zu_str(df[c])
    for c in [c for c in df.columns if c not in KATEGORIAL + ["labels"]]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    assert "back" in set(df["labels"]), "Label 'back' zerstoert -> strip-Falle!"
    return df


def baue_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in LOG_SPALTEN:
        X[c] = np.log1p(X[c])
    return X.astype(float)


class ZeroDaySzenario:
    """Teilt die Daten so, dass bestimmte Angriffstypen im Training UNSICHTBAR bleiben.

    Attribute:
      normal_train / normal_test : Normalverkehr (nur normal_train darf trainiert werden)
      bekannt_train / bekannt_test : Angriffe, die der supervised Detektor kennen darf
      zeroday : dict {Angriffstyp -> Feature-Matrix} der *nie gesehenen* Angriffe
    """

    def __init__(self, bekannte_angriffe=("smurf",), test_size=0.3, random_state=0):
        df = lade_flows(random_state=random_state)
        X = baue_features(df)
        lab = df["labels"]
        self.labels = lab
        self.bekannte_angriffe = list(bekannte_angriffe)

        normal = X[lab == "normal"]
        self.normal_train, self.normal_test = train_test_split(
            normal, test_size=test_size, random_state=random_state)

        X_bek = X[lab.isin(self.bekannte_angriffe)]
        self.bekannt_train, self.bekannt_test = train_test_split(
            X_bek, test_size=test_size, random_state=random_state)

        self.zeroday = {
            a: X[lab == a] for a in sorted(lab.unique())
            if a != "normal" and a not in self.bekannte_angriffe
        }

    def supervised_trainingsdaten(self):
        """(X, y) fuer den ueberwachten Detektor: Normal + NUR die bekannten Angriffe."""
        X = pd.concat([self.normal_train, self.bekannt_train])
        y = np.r_[np.zeros(len(self.normal_train)), np.ones(len(self.bekannt_train))]
        return X, y

    def uebersicht(self) -> str:
        z = {a: len(v) for a, v in self.zeroday.items()}
        return (f"Normal: {len(self.normal_train)} train / {len(self.normal_test)} test\n"
                f"Bekannte Angriffe {self.bekannte_angriffe}: "
                f"{len(self.bekannt_train)} train / {len(self.bekannt_test)} test\n"
                f"Zero-Day (nie im Training): {z}")
