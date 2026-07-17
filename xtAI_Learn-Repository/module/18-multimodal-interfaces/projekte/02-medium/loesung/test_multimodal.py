"""Testsuite fuer P02 (medium). pytest ist in dieser Umgebung nicht installiert,
deshalb ein eigener __main__-Runner. Ausfuehren:
    /Users/.../.venv/bin/python test_multimodal.py
"""
import numpy as np
from multimodal import (make_complementary, make_redundant, fit_modality,
                        late_fusion_proba, predict_from_proba, accuracy,
                        early_fusion_fit_predict)


def test_complementary_shapes_and_ambiguity():
    XA, XB, y = make_complementary(n_per_class=500, sigma=1.0, seed=0)
    assert XA.shape == (2000, 1) and XB.shape == (2000, 1) and y.shape == (2000,)
    assert set(np.unique(y)) == {0, 1, 2, 3}
    # Jede Modalitaet allein darf hoechstens ~knapp ueber 50 % erreichen
    mA = fit_modality(XA, y)
    accA = accuracy(y, mA.predict(XA))
    assert accA < 0.60, f"Modalitaet A sollte allein mehrdeutig sein, ist aber {accA:.3f}"


def test_late_fusion_product_math():
    # Von Hand nachgerechnetes Mini-Beispiel: 2 Klassen, uniformer Prior
    pA = np.array([[0.8, 0.2]])
    pB = np.array([[0.6, 0.4]])
    prior = np.array([0.5, 0.5])
    fused = late_fusion_proba([pA, pB], prior)
    # Produkt/prior: [0.8*0.6/0.5, 0.2*0.4/0.5] = [0.96, 0.16] -> normiert
    num = np.array([0.8 * 0.6 / 0.5, 0.2 * 0.4 / 0.5])
    expected = num / num.sum()
    assert np.allclose(fused[0], expected, atol=1e-9)
    # Fusion ist konfidenter als jede Einzelmodalitaet (schaerfer auf Klasse 0)
    assert fused[0, 0] > pA[0, 0] and fused[0, 0] > pB[0, 0]


def test_late_fusion_normalized():
    rng = np.random.default_rng(3)
    pA = rng.dirichlet(np.ones(4), size=50)
    pB = rng.dirichlet(np.ones(4), size=50)
    fused = late_fusion_proba([pA, pB], np.full(4, 0.25))
    assert np.allclose(fused.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(fused >= 0)


def test_fusion_beats_single_on_complementary():
    XA, XB, y = make_complementary(n_per_class=800, sigma=1.0, seed=0)
    n = XA.shape[0]
    rng = np.random.default_rng(1)
    idx = rng.permutation(n); k = int(0.7 * n)
    tr, te = idx[:k], idx[k:]
    mA = fit_modality(XA[tr], y[tr]); mB = fit_modality(XB[tr], y[tr])
    pA = mA.predict_proba(XA[te]); pB = mB.predict_proba(XB[te])
    accA = accuracy(y[te], predict_from_proba(pA))
    accLate = accuracy(y[te], predict_from_proba(late_fusion_proba([pA, pB], np.full(4, 0.25))))
    assert accLate > accA + 0.30, f"Fusion {accLate:.3f} sollte A {accA:.3f} klar schlagen"
    assert accLate > 0.90


def test_mutual_disambiguation_independent():
    # Bei unabhaengigen Fehlern (rho=0) gilt die Mutual-Disambiguation-Signatur:
    # P(beide Modalitaeten falsch) ~ eps_A * eps_B (Produkt = Unabhaengigkeit).
    XA, XB, y = make_redundant(n=40000, sep=1.0, sigma=1.0, rho=0.0, seed=0)
    wrongA = (XA[:, 0] > 0).astype(int) != y
    wrongB = (XB[:, 0] > 0).astype(int) != y
    eA, eB = wrongA.mean(), wrongB.mean()
    both = np.mean(wrongA & wrongB)
    assert abs(both - eA * eB) < 0.01, f"P(beide falsch)={both:.4f} vs eps_A*eps_B={eA*eB:.4f}"
    # und die Mittelungs-Fusion ist klar besser als jede Einzelmodalitaet
    eF = np.mean(((XA[:, 0] + XB[:, 0]) / 2 > 0).astype(int) != y)
    assert eF < min(eA, eB) - 0.03


def test_correlation_kills_gain():
    # Bei hoher Korrelation bringt Fusion fast nichts mehr
    XA, XB, y = make_redundant(n=40000, sep=1.0, sigma=1.0, rho=0.97, seed=0)
    eA = np.mean((XA[:, 0] > 0).astype(int) != y)
    eF = np.mean(((XA[:, 0] + XB[:, 0]) / 2 > 0).astype(int) != y)
    assert abs(eA - eF) < 0.03, f"Bei rho=0.97 sollte Fusion kaum helfen (eA={eA:.3f}, eF={eF:.3f})"
    # und die Fehler fallen zusammen: P(beide falsch) ~ eps einzeln
    wrongA = (XA[:, 0] > 0).astype(int) != y
    wrongB = (XB[:, 0] > 0).astype(int) != y
    assert np.mean(wrongA & wrongB) > 0.7 * eA


def test_missing_modality_late_graceful():
    # Late Fusion mit fehlender Modalitaet = einfach die verbleibende Posterior
    XA, XB, y = make_complementary(n_per_class=600, sigma=1.0, seed=2)
    mA = fit_modality(XA, y)
    pA = mA.predict_proba(XA)
    accA = accuracy(y, predict_from_proba(pA))
    # sollte ~0.5 sein (A allein), nicht abstuerzen
    assert 0.40 < accA < 0.60


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} Tests bestanden.")
