"""Statistik-Werkzeuge fuer die Nutzerstudien-Auswertung — from scratch, wo sinnvoll.

`pingouin`/`statsmodels` fehlen in dieser Umgebung. Das ist ein Gluecksfall: die Effektstaerken
und Korrekturen sind ein paar Zeilen, und wer sie selbst schreibt, versteht sie. Die Testkerne
(Wilcoxon, paired t) kommen aus scipy.stats.

Alle Funktionen fuer GEPAARTE (within-subject) Daten: x, y sind zwei Messungen DERSELBEN
Probanden (z. B. presence unter 3 DoF vs. unter 6 DoF).
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def cohen_dz(x, y) -> float:
    """Effektstaerke fuer gepaarte Daten (parametrisch): mittlere Differenz / SD der Differenz.

    Faustregel (Cohen): |dz| ~ 0.2 klein, 0.5 mittel, 0.8 gross.
    Vorzeichen sagt die Richtung (positiv: x > y im Mittel).
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    return float(d.mean() / d.std(ddof=1))


def rank_biserial(x, y) -> float:
    """Effektstaerke fuer den Wilcoxon-Test (nichtparametrisch): matched-pairs rank-biserial r.

        r = (W+ - W-) / (W+ + W-)
    W+ = Summe der Raenge der positiven Differenzen, W- die der negativen. r in [-1, 1].
    Interpretation wie eine Korrelation; robust gegen Ausreisser (nutzt nur Raenge).
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]                                # Nulldifferenzen werden verworfen
    if len(d) == 0:
        return 0.0
    raenge = stats.rankdata(np.abs(d))
    w_plus = raenge[d > 0].sum()
    w_minus = raenge[d < 0].sum()
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total) if total > 0 else 0.0


def paired_comparison(x, y) -> dict:
    """Vollstaendiger Vergleich zweier gepaarter Messungen.

    Rueckgabe: dict mit Mittelwerten, paired-t-p, Wilcoxon-p, cohen_dz, rank_biserial.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    return {
        "mean_x": float(x.mean()), "mean_y": float(y.mean()),
        "t_p": float(stats.ttest_rel(x, y).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(x, y).pvalue),
        "cohen_dz": cohen_dz(x, y),
        "rank_biserial": rank_biserial(x, y),
    }


def bonferroni(pvalues, alpha: float = 0.05):
    """Bonferroni-Korrektur fuer m Tests. Rueckgabe: (korrigiertes_alpha, Liste signifikant?).

    Idee (Skript 5.2): wer m Tests bei alpha macht, hat eine Familien-Fehlerrate ~ m*alpha.
    Bonferroni testet stattdessen jeden bei alpha/m -> Familienrate <= alpha. Sehr konservativ.
    """
    m = len(pvalues)
    schwelle = alpha / m
    return schwelle, [p < schwelle for p in pvalues]


def holm_bonferroni(pvalues, alpha: float = 0.05):
    """Holm-Bonferroni (schrittweise, weniger konservativ als Bonferroni, gleiche Garantie).

    p-Werte aufsteigend sortieren; den k-kleinsten gegen alpha/(m-k) testen; beim ersten
    Nicht-Signifikanten STOPPEN (alle folgenden ebenfalls n.s.).
    Rueckgabe: dict {index -> signifikant?} in Original-Reihenfolge.
    """
    p = list(pvalues)
    m = len(p)
    ordnung = sorted(range(m), key=lambda i: p[i])
    signifikant = {}
    noch_signifikant = True
    for k, idx in enumerate(ordnung):
        schwelle = alpha / (m - k)
        if noch_signifikant and p[idx] < schwelle:
            signifikant[idx] = True
        else:
            noch_signifikant = False
            signifikant[idx] = False
    return signifikant


def order_effect(df, outcome: str) -> dict:
    """Prueft auf einen Reihenfolge-Effekt (Carryover/Lernen): erste vs. zweite Sitzung.

    Erwartet Spalten 'position' (1/2) und `outcome`. Vergleich BETWEEN (verschiedene Zeilen),
    daher Mann-Whitney-U statt Wilcoxon.
    """
    erste = df[df["position"] == 1][outcome].values
    zweite = df[df["position"] == 2][outcome].values
    u = stats.mannwhitneyu(erste, zweite)
    return {"mean_erste": float(erste.mean()), "mean_zweite": float(zweite.mean()),
            "p": float(u.pvalue)}
