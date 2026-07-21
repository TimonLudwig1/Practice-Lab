"""Cross-linguales Embedding-Alignment — der Kern des Projekts.

Enthält die geschlossene **orthogonale Procrustes**-Lösung und die Retrieval-Maße
**Nearest-Neighbour** und **CSLS** (gegen *Hubness*). Siehe Skript-Abschnitt 3.
"""
import numpy as np


def orthogonal_procrustes(X, Y):
    r"""Löse min_W ||X W - Y||_F  unter  W^T W = I.

    X, Y: (n, d) — gestapelte Quell-/Ziel-Embeddings der Anker-Paare (Zeilen = Wörter).
    Geschlossene Lösung: M = X^T Y = U Σ V^T  ⟹  W = U V^T   (bildet Quelle → Ziel ab).
    Rückgabe: W (d, d) orthogonal.
    """
    M = X.T @ Y                       # (d, d) = Quelle^T · Ziel
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def _topk_mean(sims, k):
    """Mittel der k größten Werte je Zeile (für die CSLS-Nachbarschaftsdichte)."""
    k = min(k, sims.shape[1])
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def nearest_neighbor(query_vecs, tgt_emb):
    """Für jede (bereits projizierte, normierte) Query den ähnlichsten Ziel-Index.

    query_vecs: (q, d), tgt_emb: (Vt, d) — beide längen-normiert ⇒ Skalarprodukt = Kosinus.
    Rückgabe: (q,) Indizes der nächsten Ziel-Wörter.
    """
    sims = query_vecs @ tgt_emb.T          # (q, Vt)
    return np.argmax(sims, axis=1)


def csls(query_vecs, tgt_emb, src_pool, k=10):
    r"""CSLS-Retrieval gegen Hubness (Conneau et al. 2018).

    CSLS(z, y) = 2 cos(z, y) − r_T(z) − r_S(y), mit
      r_T(z) = mittlere Kosinus-Ähnlichkeit von z zu seinen k nächsten Ziel-Nachbarn,
      r_S(y) = mittlere Kosinus-Ähnlichkeit von y zu seinen k nächsten (projizierten)
               Quell-Nachbarn aus dem Pool.

    query_vecs: (q, d) projizierte Queries;  tgt_emb: (Vt, d);
    src_pool:   (Ns, d) projizierte Quell-Embeddings (für r_S).
    Rückgabe: (q,) Indizes der besten Ziel-Wörter nach CSLS.
    """
    sim_qt = query_vecs @ tgt_emb.T        # (q, Vt) cos(query, target)
    r_T = _topk_mean(sim_qt, k)            # (q,)  Dichte um jede Query im Zielraum
    sim_ts = tgt_emb @ src_pool.T          # (Vt, Ns) cos(target, source-pool)
    r_S = _topk_mean(sim_ts, k)            # (Vt,) Dichte um jedes Ziel im Quellraum
    scores = 2 * sim_qt - r_T[:, None] - r_S[None, :]
    return np.argmax(scores, axis=1)


def precision_at_1(pred_indices, gold_words, tgt_vocab):
    """Anteil der Queries, deren top-1-Vorhersage exakt das Gold-Wort ist."""
    ok = sum(tgt_vocab[p] == g for p, g in zip(pred_indices, gold_words))
    return ok / len(gold_words)
