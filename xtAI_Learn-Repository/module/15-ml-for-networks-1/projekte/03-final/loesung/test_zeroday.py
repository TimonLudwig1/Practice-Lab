"""Tests fuer das Zero-Day-Szenario und die Detektoren.

Aufruf:  python test_zeroday.py      (~1-2 min: trainiert mehrere Detektoren)
"""
import numpy as np
import pandas as pd

from flow_data import ZeroDaySzenario, lade_flows, baue_features, FEATURES
from detektoren import (SupervisedDetektor, IsolationForestDetektor,
                        MahalanobisDetektor, OneClassSVMDetektor, LOFDetektor)

# einmal laden und teilen - das ist teuer, also fuer alle Tests wiederverwenden
SZEN = ZeroDaySzenario(bekannte_angriffe=("smurf",), random_state=0)


# --------------------------- Daten / Szenario ---------------------------
def test_bytes_werden_korrekt_dekodiert():
    df = lade_flows()
    # die strip("b'")-Falle wuerde aus 'back' ein 'ack' und aus 'bgp' ein 'gp' machen
    assert "back" in set(df["labels"])
    assert not any(l.startswith("b'") for l in df["labels"].unique())


def test_features_sind_endlich_und_numerisch():
    X = baue_features(lade_flows())
    assert list(X.columns) == FEATURES
    assert np.all(np.isfinite(X.values))
    assert X.dtypes.map(lambda d: d.kind == "f").all()


def test_zeroday_angriffe_sind_NICHT_im_training():
    # Der Kern des Szenarios: kein Zero-Day-Typ darf in den Trainingsdaten stecken.
    assert "smurf" not in SZEN.zeroday
    assert "neptune" in SZEN.zeroday, "neptune muss ein Zero-Day sein"
    X_train, y = SZEN.supervised_trainingsdaten()
    # Trainingsmenge = Normal + bekannte Angriffe, sonst nichts
    assert len(X_train) == len(SZEN.normal_train) + len(SZEN.bekannt_train)
    assert set(np.unique(y)) == {0.0, 1.0}
    assert y.sum() == len(SZEN.bekannt_train)


def test_normal_train_und_test_sind_disjunkt():
    # Auf Werteebene ist Disjunktheit unmoeglich (der Datensatz enthaelt 13 % exakte
    # Duplikate, Skript 3.6) - pruefbar ist die Disjunktheit der INDIZES.
    assert not set(SZEN.normal_train.index) & set(SZEN.normal_test.index)


def test_anomaliedetektor_sieht_nur_normalverkehr():
    # fit() nimmt genau EIN Argument (Normalverkehr) - kein y, keine Angriffe
    det = MahalanobisDetektor().fit(SZEN.normal_train)
    assert hasattr(det, "schwelle_")


# --------------------------- Kalibrierung ---------------------------
def test_fpr_budget_wird_eingehalten():
    # Jeder Anomaliedetektor soll per Konstruktion ~1 % Fehlalarme haben.
    for D in (IsolationForestDetektor(), MahalanobisDetektor(),
              OneClassSVMDetektor(), LOFDetektor()):
        det = D.fit(SZEN.normal_train)
        fpr = det.predict(SZEN.normal_test).mean()
        assert 0.002 < fpr < 0.03, f"{det.name}: FPR {fpr:.4f} verfehlt das 1%-Budget"


def test_lof_braucht_deduplizierung():
    # Der dokumentierte Fallstrick: mit Duplikaten explodieren die LOF-Scores
    # (bis ~1.5e9) und die Perzentil-Schwelle -> nichts wird je erkannt.
    det_ok = LOFDetektor().fit(SZEN.normal_train)          # dedupliziert = True
    assert det_ok.dedupliziere is True
    assert det_ok.predict(SZEN.bekannt_test).mean() > 0.9, "LOF sollte smurf finden"

    det_kaputt = LOFDetektor()
    det_kaputt.dedupliziere = False                        # absichtlich sabotiert
    det_kaputt.fit(SZEN.normal_train)
    assert det_kaputt.schwelle_ > 100 * det_ok.schwelle_, \
        "ohne Deduplizierung muesste die Schwelle entgleisen"


# --------------------------- Die Kernaussage ---------------------------
def test_supervised_erkennt_bekannten_angriff():
    X, y = SZEN.supervised_trainingsdaten()
    det = SupervisedDetektor().fit(X, y)
    assert det.predict(SZEN.bekannt_test).mean() > 0.95      # smurf: quasi perfekt
    assert det.predict(SZEN.normal_test).mean() < 0.01       # kaum Fehlalarme


def test_supervised_ist_blind_fuer_zeroday():
    # DIE Kernaussage des Projekts: was er nie gesehen hat, sieht er nicht.
    X, y = SZEN.supervised_trainingsdaten()
    det = SupervisedDetektor().fit(X, y)
    recall_neptune = det.predict(SZEN.zeroday["neptune"]).mean()
    assert recall_neptune < 0.05, f"unerwartet gut: {recall_neptune}"


def test_anomaliedetektor_findet_zeroday_ohne_je_angriffe_gesehen_zu_haben():
    det = IsolationForestDetektor().fit(SZEN.normal_train)   # nur Normalverkehr!
    assert det.predict(SZEN.zeroday["neptune"]).mean() > 0.8


def test_anomalie_schlaegt_supervised_auf_zeroday():
    X, y = SZEN.supervised_trainingsdaten()
    sup = SupervisedDetektor().fit(X, y)
    ano = IsolationForestDetektor().fit(SZEN.normal_train)
    zd = [a for a, Xa in SZEN.zeroday.items() if len(Xa) >= 10]
    r_sup = np.mean([sup.predict(SZEN.zeroday[a]).mean() for a in zd])
    r_ano = np.mean([ano.predict(SZEN.zeroday[a]).mean() for a in zd])
    assert r_ano > r_sup, (r_ano, r_sup)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Starte {len(tests)} Tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("Alle Tests bestanden.")
