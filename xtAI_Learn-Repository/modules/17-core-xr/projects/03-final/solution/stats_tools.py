"""Statistical tools for the user study analysis — from scratch, where that makes sense.

`pingouin`/`statsmodels` are missing in this environment. That is a stroke of luck: the effect
sizes and corrections are a few lines, and whoever writes them themselves understands them. The
test cores (Wilcoxon, paired t) come from scipy.stats.

All functions are for PAIRED (within-subject) data: x, y are two measurements of THE SAME
participants (e.g. presence under 3 DoF vs. under 6 DoF).
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def cohen_dz(x, y) -> float:
    """The effect size for paired data (parametric): mean difference / SD of the difference.

    Rule of thumb (Cohen): |dz| ~ 0.2 small, 0.5 medium, 0.8 large.
    The sign gives the direction (positive: x > y on average).
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    return float(d.mean() / d.std(ddof=1))


def rank_biserial(x, y) -> float:
    """The effect size for the Wilcoxon test (non-parametric): the matched-pairs rank-biserial r.

        r = (W+ - W-) / (W+ + W-)
    W+ = the sum of the ranks of the positive differences, W- that of the negative ones.
    r lies in [-1, 1]. Interpreted like a correlation; robust against outliers (it uses only
    ranks).
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]                                # zero differences are discarded
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total) if total > 0 else 0.0


def paired_comparison(x, y) -> dict:
    """A complete comparison of two paired measurements.

    Returns: a dict with the means, the paired-t p, the Wilcoxon p, cohen_dz, rank_biserial.
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
    """The Bonferroni correction for m tests. Returns: (corrected_alpha, a list of significant?).

    The idea (script 5.2): whoever runs m tests at alpha has a family-wise error rate ~ m*alpha.
    Bonferroni instead tests each at alpha/m -> the family rate is <= alpha. Very conservative.
    """
    m = len(pvalues)
    threshold = alpha / m
    return threshold, [p < threshold for p in pvalues]


def holm_bonferroni(pvalues, alpha: float = 0.05):
    """Holm-Bonferroni (step-wise, less conservative than Bonferroni, the same guarantee).

    Sort the p-values ascending; test the k-th smallest against alpha/(m-k); STOP at the first
    non-significant one (all following ones are non-significant too).
    Returns: a dict {index -> significant?} in the original order.
    """
    p = list(pvalues)
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    significant = {}
    still_significant = True
    for k, idx in enumerate(order):
        threshold = alpha / (m - k)
        if still_significant and p[idx] < threshold:
            significant[idx] = True
        else:
            still_significant = False
            significant[idx] = False
    return significant


def order_effect(df, outcome: str) -> dict:
    """Checks for an order effect (carryover/learning): the first vs. the second session.

    Expects the columns 'position' (1/2) and `outcome`. The comparison is BETWEEN (different
    rows), hence Mann-Whitney U instead of Wilcoxon.
    """
    first = df[df["position"] == 1][outcome].values
    second = df[df["position"] == 2][outcome].values
    u = stats.mannwhitneyu(first, second)
    return {"mean_first": float(first.mean()), "mean_second": float(second.mean()),
            "p": float(u.pvalue)}
