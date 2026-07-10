r"""IBM Model 1 — statistische Wort-Übersetzung via EM (Brown et al. 1993).

Das klassische Fundament der maschinellen Übersetzung und der konzeptuelle Vorläufer der
**Attention** ("weiches Alignment", Skript 4.2): Aus rein *parallelen* Sätzen — ganz ohne
Wörterbuch — lernt der EM-Algorithmus eine Übersetzungstabelle $t(e\mid f)$ und damit
Wort-**Alignments**. Rein zählbasiert: schnell, CPU-only, kein neuronales Training.

Modell (für ein Satzpaar mit Quellsatz $f_1^m$ und Zielsatz $e_1^l$):
$$P(e\mid f) = \frac{\epsilon}{(m+1)^l}\prod_{j=1}^{l}\sum_{i=0}^{m} t(e_j\mid f_i),$$
wobei $f_0=\text{NULL}$. Alle Alignments sind gleich wahrscheinlich (Model-1-Annahme);
gelernt wird nur $t(e\mid f)$. EM konvergiert hier gegen ein **globales** Optimum
(die Likelihood ist konvex in $t$).
"""
from collections import defaultdict

from data import tokenize

NULL = "<null>"


def train(pairs, n_iter=5, min_count=1, verbose=True):
    r"""EM für IBM Model 1. pairs: Liste von (deutsch, englisch).

    Lernt t[e][f] = p(englisches Wort e | deutsches Wort f). Rückgabe: dict-of-dict.
    """
    # Sätze tokenisieren; Quelle (DE) um NULL erweitern.
    corpus = []
    for de, en in pairs:
        f = [NULL] + tokenize(de)
        e = tokenize(en)
        if e and len(f) > 1:
            corpus.append((e, f))

    # t(e|f) uniform initialisieren (nur über tatsächlich kookkurrierende Paare).
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
                # Normierung s_total(e) = Σ_f t(e|f)
                s_total = sum(t[ew][fw] for fw in f)
                loglik += _safe_log(s_total)
                for fw in f:
                    c = t[ew][fw] / s_total               # erwartete Anzahl (E-Schritt)
                    count[ew][fw] += c
                    total[fw] += c
        # M-Schritt: t(e|f) = count(e,f) / total(f)
        for ew in count:
            for fw in count[ew]:
                t[ew][fw] = count[ew][fw] / total[fw]
        if verbose:
            print(f"  EM-Iteration {it}: log-Likelihood {loglik:,.0f}")
    return t


def _safe_log(x):
    from math import log
    return log(x) if x > 0 else 0.0


def align(t, de_sent, en_sent):
    r"""Viterbi-Alignment (Model 1): jedes Zielwort $e_j$ auf $\arg\max_i t(e_j\mid f_i)$.

    Rückgabe: Liste von (englisches Wort, ausgerichtetes deutsches Wort oder NULL).
    """
    f = [NULL] + tokenize(de_sent)
    e = tokenize(en_sent)
    alignment = []
    for ew in e:
        best = max(f, key=lambda fw: t.get(ew, {}).get(fw, 0.0))
        alignment.append((ew, best))
    return alignment


def translate(t, de_sent):
    r"""Grob-Übersetzung DE->EN: jedes deutsche Wort auf sein wahrscheinlichstes
    englisches Pendant $\arg\max_e t(e\mid f)$ abbilden (Wort-für-Wort, ohne Reordering
    oder Sprachmodell — die bekannte Schwäche von Model 1 allein)."""
    # invertiere: pro deutschem Wort f das e mit größtem t(e|f)
    best_e_for_f = {}
    # baue Umkehrung nur für die im Satz vorkommenden f
    fs = tokenize(de_sent)
    # Sammle Kandidaten: alle e, die f als Key haben
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
