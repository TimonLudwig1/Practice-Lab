"""Tests fuer die Base-Rate-Rechnung.  Aufruf:  python test_base_rate.py"""
import numpy as np

from base_rate import (ppv_bei_basisrate, benoetigte_fpr, alarme_pro_tag,
                       erwartete_kosten, bester_betriebspunkt)


# ----------------------- PPV / Bayes -----------------------
def test_axelsson_beispiel():
    # Das Zahlenbeispiel aus dem Skript (3.1): TPR .99, FPR .001, pi 1e-4 -> ~9 %
    p = ppv_bei_basisrate(0.99, 0.001, 1e-4)
    assert abs(p - 0.0901) < 0.001, p


def test_ppv_perfekter_detektor_ohne_fehlalarme():
    # FPR = 0 -> jeder Alarm ist echt, egal wie klein die Basisrate
    assert ppv_bei_basisrate(0.9, 0.0, 1e-9) == 1.0


def test_ppv_faellt_monoton_mit_kleinerer_basisrate():
    pis = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    werte = [ppv_bei_basisrate(0.99, 0.01, p) for p in pis]
    assert all(werte[i] > werte[i + 1] for i in range(len(werte) - 1)), werte


def test_ppv_bei_basisrate_eins():
    # pi = 1: alles ist ein Angriff -> jeder Alarm ist zwangslaeufig echt
    assert abs(ppv_bei_basisrate(0.5, 0.5, 1.0) - 1.0) < 1e-12


def test_ppv_akzeptiert_arrays():
    out = ppv_bei_basisrate(0.99, 0.001, np.array([1e-4, 1e-2]))
    assert out.shape == (2,) and out[1] > out[0]


def test_ppv_nutzloser_detektor():
    # TPR == FPR (Muenzwurf) -> PPV == Basisrate (kein Informationsgewinn)
    for pi in (0.01, 0.2, 0.5):
        assert abs(ppv_bei_basisrate(0.5, 0.5, pi) - pi) < 1e-12


# ----------------------- Umkehrung -----------------------
def test_benoetigte_fpr_ist_inverse_von_ppv():
    tpr, pi, ziel = 0.99, 1e-4, 0.5
    f = benoetigte_fpr(tpr, pi, ziel)
    assert abs(ppv_bei_basisrate(tpr, f, pi) - ziel) < 1e-9


def test_benoetigte_fpr_strenger_bei_kleinerer_basisrate():
    f_gross = benoetigte_fpr(0.99, 1e-2, 0.5)
    f_klein = benoetigte_fpr(0.99, 1e-4, 0.5)
    assert f_klein < f_gross


def test_benoetigte_fpr_lehnt_unsinn_ab():
    for ziel in (0.0, 1.0, 1.5, -0.2):
        try:
            benoetigte_fpr(0.99, 1e-4, ziel)
            assert False, f"ziel_ppv={ziel} haette abgelehnt werden muessen"
        except ValueError:
            pass


# ----------------------- Absolute Zahlen -----------------------
def test_alarme_pro_tag():
    echt, falsch = alarme_pro_tag(1e7, tpr=0.99, fpr=0.001, pi=1e-4)
    assert abs(echt - 990.0) < 1.0                 # 0.99 * 1e-4 * 1e7
    assert abs(falsch - 9999.0) < 1.0              # 0.001 * 0.9999 * 1e7
    assert falsch > 10 * echt                      # der Kern des Problems


def test_alarme_konsistent_mit_ppv():
    # PPV muss sich auch aus den absoluten Zahlen ergeben
    tpr, fpr, pi = 0.9, 0.02, 1e-3
    echt, falsch = alarme_pro_tag(1e6, tpr, fpr, pi)
    assert abs(echt / (echt + falsch) - ppv_bei_basisrate(tpr, fpr, pi)) < 1e-9


# ----------------------- Kosten -----------------------
def test_erwartete_kosten_summiert_beide_fehlerarten():
    # 1e6 Flows, pi=1e-3 -> 1000 Angriffe; TPR=0.9 -> 100 verpasst
    # FPR=0.01 -> ~9990 Fehlalarme
    k = erwartete_kosten(1e6, tpr=0.9, fpr=0.01, pi=1e-3,
                         kosten_fehlalarm=1.0, kosten_verpasst=100.0)
    erwartet = 0.01 * 0.999 * 1e6 * 1.0 + 0.1 * 1e-3 * 1e6 * 100.0
    assert abs(k - erwartet) < 1e-6


def test_bester_betriebspunkt_waehlt_minimum():
    # Kuenstliche ROC mit drei Punkten; 1e6 Flows, pi=1e-3 -> 1000 Angriffe.
    #   Punkt 0 (nie Alarm)    : 1000 verpasst * 100        = 100_000
    #   Punkt 1 (Kompromiss)   : ~999 Fehlal.*1 + 100 verp.*100 =  10_999  <- Minimum
    #   Punkt 2 (Dauer-Alarm)  : ~499_500 Fehlalarme * 1     = 499_500
    fpr = np.array([0.0, 0.001, 0.5])
    tpr = np.array([0.0, 0.90, 1.0])
    thr = np.array([1.0, 0.5, 0.0])
    i, s, k = bester_betriebspunkt(fpr, tpr, thr, 1e6, 1e-3,
                                   kosten_fehlalarm=1.0, kosten_verpasst=100.0)
    assert i == 1 and s == 0.5, (i, s, k)
    assert k < erwartete_kosten(1e6, 0.0, 0.0, 1e-3, 1.0, 100.0)   # besser als nie alarmieren
    assert k < erwartete_kosten(1e6, 1.0, 0.5, 1e-3, 1.0, 100.0)   # besser als dauernd alarmieren


def test_kostenoptimum_verschiebt_sich_mit_den_kosten():
    # Wird ein verpasster Angriff sehr teuer, lohnt sich der aggressivere Punkt.
    fpr = np.array([0.0, 0.001, 0.5])
    tpr = np.array([0.0, 0.90, 1.0])
    thr = np.array([1.0, 0.5, 0.0])
    i_billig, _, _ = bester_betriebspunkt(fpr, tpr, thr, 1e6, 1e-3, 1.0, 100.0)
    i_teuer, _, _ = bester_betriebspunkt(fpr, tpr, thr, 1e6, 1e-3, 1.0, 10_000.0)
    assert i_billig == 1 and i_teuer == 2, (i_billig, i_teuer)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Starte {len(tests)} Tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("Alle Tests bestanden.")
