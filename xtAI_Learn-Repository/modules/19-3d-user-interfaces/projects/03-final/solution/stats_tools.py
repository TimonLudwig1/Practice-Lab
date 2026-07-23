"""Statistical building blocks (from scratch, since statsmodels/pingouin are missing).
P03-final module 19.
We use Wilcoxon from scipy.stats; the effect size and the multiplicity correction ourselves.
"""
import numpy as np


def rank_biserial_from_diffs(diffs):
    """The rank-biserial correlation as the effect size for the Wilcoxon signed-rank test.
    r = (W+ - W-) / (W+ + W-), computed from the rank sums of the positive/negative
    differences."""
    diffs = np.asarray(diffs, float)
    nz = diffs[diffs != 0]
    if nz.size == 0:
        return 0.0
    ranks = _rankdata(np.abs(nz))
    Wp = ranks[nz > 0].sum()
    Wm = ranks[nz < 0].sum()
    return (Wp - Wm) / (Wp + Wm)


def _rankdata(a):
    """Average ranks (ties -> the mid-rank), like scipy.stats.rankdata."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, float)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def holm_bonferroni(pvals):
    """Holm-Bonferroni corrected p-values (in the order of the input)."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, float)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        prev = max(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj
