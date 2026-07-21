"""BLEU (Papineni et al. 2002) von Hand — Skript-Abschnitt 6.1."""
import math
from collections import Counter


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    r"""Korpus-BLEU über tokenisierte Hypothesen/Referenzen (Listen von Token-Listen).

    BLEU = BP · exp(Σ_n (1/N) log p_n), mit geclippter n-Gramm-Präzision p_n und
    Brevity Penalty BP = min(1, exp(1 - r/c)). Rückgabe in [0, 100].
    """
    p_num = [0] * max_n
    p_den = [0] * max_n
    c_len = r_len = 0
    for hyp, ref in zip(hyps, refs):
        c_len += len(hyp)
        r_len += len(ref)
        for n in range(1, max_n + 1):
            h_ng = _ngrams(hyp, n)
            r_ng = _ngrams(ref, n)
            p_num[n - 1] += sum(min(c, r_ng[g]) for g, c in h_ng.items())   # clipped
            p_den[n - 1] += max(sum(h_ng.values()), 1)
    # +1-Glättung, falls eine höhere Ordnung 0 Treffer hat (sonst log(0))
    if min(p_num) == 0:
        precisions = [(p_num[i] + 1) / (p_den[i] + 1) for i in range(max_n)]
    else:
        precisions = [p_num[i] / p_den[i] for i in range(max_n)]
    geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if c_len > r_len else math.exp(1 - r_len / max(c_len, 1))
    return 100 * bp * geo
