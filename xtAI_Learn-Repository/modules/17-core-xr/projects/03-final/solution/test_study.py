"""Tests fuer Datengenerator und Statistik-Werkzeuge.  Aufruf:  python test_study.py"""
import numpy as np
from scipy import stats

from generate_study import (generate_study, generate_naiv_ohne_counterbalancing, WAHRHEIT)
from stats_tools import (cohen_dz, rank_biserial, paired_comparison, bonferroni,
                         holm_bonferroni, order_effect)

DF = generate_study(n_participants=24, seed=3)


def _paare(df, spalte):
    a = df[df.condition == "3DoF"].sort_values("participant")[spalte].values
    b = df[df.condition == "6DoF"].sort_values("participant")[spalte].values
    return a, b


# ------------------------------ Generator ------------------------------
def test_generator_reproduzierbar():
    assert generate_study(seed=3).equals(generate_study(seed=3))
    assert not generate_study(seed=3).equals(generate_study(seed=4))


def test_within_subject_struktur():
    assert DF.participant.nunique() == 24
    assert len(DF) == 48                                 # 2 Zeilen je Proband
    assert set(DF.condition) == {"3DoF", "6DoF"}
    for _, g in DF.groupby("participant"):
        assert set(g.condition) == {"3DoF", "6DoF"}      # jeder hat beide


def test_counterbalancing_ausgeglichen():
    ersten = DF[DF.position == 1].condition.value_counts()
    assert ersten["3DoF"] == ersten["6DoF"] == 12        # 12 beginnen je mit einer Bedingung


def test_werte_in_gueltigen_bereichen():
    assert DF.presence.between(1, 7).all()
    assert DF.comfort.between(1, 7).all()
    assert (DF.sickness >= 0).all()
    assert (DF.time >= 5).all()


# ------------------------------ Effektstaerken ------------------------------
def test_cohen_dz_von_hand():
    # diffs [1,2,3] -> mean 2, sd 1 -> dz = 2
    assert np.isclose(cohen_dz([2, 4, 6], [1, 2, 3]), 2.0)


def test_cohen_dz_vorzeichen():
    assert cohen_dz([5, 6, 7], [1, 1, 1]) > 0            # x > y (nicht-degeneriert)
    assert cohen_dz([1, 1, 1], [5, 6, 7]) < 0


def test_rank_biserial_extreme():
    assert np.isclose(rank_biserial([2, 3, 4], [1, 1, 1]), 1.0)    # alle Diffs positiv
    assert np.isclose(rank_biserial([1, 1, 1], [2, 3, 4]), -1.0)   # alle negativ


def test_rank_biserial_symmetrisch_nahe_null():
    x = np.array([1, 2, 3, 4]); y = np.array([2, 1, 4, 3])         # 2 hoch, 2 runter, gleiche Betraege
    assert abs(rank_biserial(x, y)) < 1e-9


def test_rank_biserial_robust_gegen_ausreisser():
    # ein extremer Ausreisser aendert die Raenge kaum -> r bleibt nahe 1
    x = np.array([2, 3, 4, 1000.0]); y = np.array([1, 1, 1, 1.0])
    assert rank_biserial(x, y) == 1.0


# ------------------------------ Korrekturen ------------------------------
def test_bonferroni_schwelle_und_flags():
    schwelle, sig = bonferroni([0.01, 0.03, 0.20, 0.04], alpha=0.05)
    assert np.isclose(schwelle, 0.0125)                  # 0.05 / 4
    assert sig == [True, False, False, False]            # nur 0.01 < 0.0125


def test_holm_ist_maechtiger_als_bonferroni():
    # p=0.03 ueberlebt Holm (0.05/3 = 0.0167? nein 0.03>0.0167 -> faellt). Nimm klareres Beispiel:
    pvals = [0.001, 0.013, 0.5]
    _, bonf = bonferroni(pvals, 0.05)                    # Schwelle 0.0167
    holm = holm_bonferroni(pvals, 0.05)
    # 0.013: Bonferroni 0.013<0.0167 sig; Holm k=1 -> 0.05/2=0.025, 0.013<0.025 sig. Beide sig.
    assert bonf[0] and bonf[1] and not bonf[2]
    assert holm[0] and holm[1] and not holm[2]


def test_holm_stoppt_beim_ersten_nicht_signifikanten():
    # Holm arbeitet auf SORTIERTEN p-Werten: beide 0.001 werden signifikant, BEVOR die 0.5
    # ueberhaupt an die Reihe kommt und den Stopp ausloest.
    holm = holm_bonferroni([0.001, 0.5, 0.001], 0.05)
    assert holm[0] is True and holm[1] is False and holm[2] is True
    # Ein echter Stopp: der zweitkleinste ist schon zu gross -> alles danach n.s.
    holm2 = holm_bonferroni([0.001, 0.30, 0.40], 0.05)
    assert holm2[0] is True and holm2[1] is False and holm2[2] is False


# ------------------------------ Die Auswertung findet die Wahrheit ------------------------------
def test_paired_comparison_stimmt_mit_scipy():
    a, b = _paare(DF, "presence")
    r = paired_comparison(a, b)
    assert np.isclose(r["wilcoxon_p"], stats.wilcoxon(a, b).pvalue)
    assert np.isclose(r["t_p"], stats.ttest_rel(a, b).pvalue)


def test_analyse_findet_die_eingebauten_effekte():
    # 6 DoF: mehr presence, weniger sickness, weniger Zeit - alle signifikant (Wilcoxon)
    a, b = _paare(DF, "presence"); assert b.mean() > a.mean() and stats.wilcoxon(a, b).pvalue < 0.01
    a, b = _paare(DF, "sickness"); assert b.mean() < a.mean() and stats.wilcoxon(a, b).pvalue < 0.01
    a, b = _paare(DF, "time");     assert b.mean() < a.mean() and stats.wilcoxon(a, b).pvalue < 0.01


def test_comfort_ist_der_wacklige_befund():
    # kleiner Effekt: roh signifikant, aber nach Bonferroni nicht
    a, b = _paare(DF, "comfort")
    p = stats.wilcoxon(a, b).pvalue
    assert 0.0125 < p < 0.05                              # zwischen roh-alpha und Bonferroni-Schwelle
    assert abs(cohen_dz(a, b)) < 0.7                      # deutlich kleiner als die grossen Effekte


def test_order_effekt_carryover_sickness():
    oe = order_effect(DF, "sickness")
    assert oe["mean_zweite"] > oe["mean_erste"]          # zweite Sitzung kraenker (Carryover)


def test_ohne_counterbalancing_wird_effekt_maskiert():
    # DER Kernbefund: naive Analyse unterschaetzt den wahren Sickness-Effekt (12) deutlich
    dfn = generate_naiv_ohne_counterbalancing(seed=3)
    a, b = _paare(dfn, "sickness")
    gemessen = a.mean() - b.mean()
    wahr = -WAHRHEIT["sickness_effekt"]                   # 12
    assert gemessen < 0.7 * wahr                          # deutlich unterschaetzt
    assert gemessen > 0                                   # Richtung stimmt noch


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Starte {len(tests)} Tests ...")
    for name, tf in tests:
        tf(); print(f"  {name} ... OK")
    print("Alle Tests bestanden.")
