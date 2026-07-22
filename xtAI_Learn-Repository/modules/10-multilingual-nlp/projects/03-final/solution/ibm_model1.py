r"""IBM Model 1 — statistical word translation via EM (Brown et al. 1993).

The classical foundation of machine translation and the conceptual predecessor of
**attention** ("soft alignment", script 4.2): from purely *parallel* sentences — without any
dictionary — the EM algorithm learns a translation table $t(e\mid f)$ and thereby word
**alignments**. Purely count-based: fast, CPU-only, no neural training.

Model (for a sentence pair with source sentence $f_1^m$ and target sentence $e_1^l$):
$$P(e\mid f) = \frac{\epsilon}{(m+1)^l}\prod_{j=1}^{l}\sum_{i=0}^{m} t(e_j\mid f_i),$$
where $f_0=\text{NULL}$. All alignments are equally probable (the Model 1 assumption);
only $t(e\mid f)$ is learned. EM converges here to a **global** optimum
(the likelihood is convex in $t$).
"""
from collections import defaultdict

from data import tokenize

NULL = "<null>"


def train(pairs, n_iter=5, min_count=1, verbose=True):
    r"""EM for IBM Model 1. pairs: list of (german, english).

    Learns t[e][f] = p(english word e | german word f). Returns: dict-of-dict.
    """
    # Tokenize the sentences; extend the source (DE) by NULL.
    corpus = []
    for de, en in pairs:
        f = [NULL] + tokenize(de)
        e = tokenize(en)
        if e and len(f) > 1:
            corpus.append((e, f))

    # Initialize t(e|f) uniformly (only over pairs that actually co-occur).
    t = defaultdict(lambda: defaultdict(float))
    co = defaultdict(set)
    for e, f in corpus:
        for ew in e:
            for fw in f:
                co[ew].add(fw)
    for ew, fs in co.items():
        init = 1.0 / len(fs)
        for fw in fs:
            t[ew][fw] = init

    for it in range(1, n_iter + 1):
        count = defaultdict(lambda: defaultdict(float))   # count[e][f]
        total = defaultdict(float)                        # total[f]
        loglik = 0.0
        for e, f in corpus:
            for ew in e:
                # Normalization s_total(e) = sum_f t(e|f)
                s_total = sum(t[ew][fw] for fw in f)
                loglik += _safe_log(s_total)
                for fw in f:
                    c = t[ew][fw] / s_total               # expected count (E-step)
                    count[ew][fw] += c
                    total[fw] += c
        # M-step: t(e|f) = count(e,f) / total(f)
        for ew in count:
            for fw in count[ew]:
                t[ew][fw] = count[ew][fw] / total[fw]
        if verbose:
            print(f"  EM iteration {it}: log likelihood {loglik:,.0f}")
    return t


def _safe_log(x):
    from math import log
    return log(x) if x > 0 else 0.0


def align(t, de_sent, en_sent):
    r"""Viterbi alignment (Model 1): each target word $e_j$ to $\arg\max_i t(e_j\mid f_i)$.

    Returns: list of (english word, aligned german word or NULL).
    """
    f = [NULL] + tokenize(de_sent)
    e = tokenize(en_sent)
    alignment = []
    for ew in e:
        best = max(f, key=lambda fw: t.get(ew, {}).get(fw, 0.0))
        alignment.append((ew, best))
    return alignment


def translate(t, de_sent):
    r"""Rough translation DE->EN: map each german word to its most probable english
    counterpart $\arg\max_e t(e\mid f)$ (word-for-word, without reordering or a language
    model — the well-known weakness of Model 1 alone)."""
    # invert: for each german word f the e with the largest t(e|f)
    best_e_for_f = {}
    # build the inversion only for the f occurring in the sentence
    fs = tokenize(de_sent)
    # collect candidates: all e that have f as a key
    cache = getattr(translate, "_cache", None)
    if cache is None or cache[0] is not t:
        inv = defaultdict(list)
        for ew, row in t.items():
            for fw, p in row.items():
                inv[fw].append((p, ew))
        translate._cache = (t, inv)
    inv = translate._cache[1]
    out = []
    for fw in fs:
        cands = inv.get(fw)
        if cands:
            out.append(max(cands)[1])
    return " ".join(out)
