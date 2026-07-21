"""Cross-linguales Embedding-Alignment — der Kern des Projekts.

Implementiere die geschlossene **orthogonale Procrustes**-Lösung und das **CSLS**-Retrieval.
`nearest_neighbor`, `_topk_mean` und `precision_at_1` sind vorgegeben. Siehe Skript-
Abschnitt 3, und prüfe mit `python test_align.py`.
"""
import numpy as np


def orthogonal_procrustes(X, Y):
    r"""Löse min_W ||X W - Y||_F  unter  W^T W = I.

    X, Y: (n, d) — gestapelte Quell-/Ziel-Embeddings der Anker-Paare (Zeilen = Wörter).

    TODO (geschlossene Lösung, Skript 3.2):
      1. M = X^T Y            (Quelle^T · Ziel — die RICHTUNG ist wichtig!)
      2. U, _, Vt = np.linalg.svd(M)
      3. return U @ Vt        (orthogonal, bildet Quelle -> Ziel ab: X W ≈ Y)
    """
    raise NotImplementedError("Aufgabe 1: orthogonal_procrustes implementieren")


def _topk_mean(sims, k):
    """Mittel der k größten Werte je Zeile (für die CSLS-Nachbarschaftsdichte). Vorgegeben."""
    k = min(k, sims.shape[1])
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def nearest_neighbor(query_vecs, tgt_emb):
    """Für jede (projizierte, normierte) Query den ähnlichsten Ziel-Index. Vorgegeben."""
    sims = query_vecs @ tgt_emb.T          # (q, Vt) = Kosinus (Vektoren sind normiert)
    return np.argmax(sims, axis=1)


def csls(query_vecs, tgt_emb, src_pool, k=10):
    r"""CSLS-Retrieval gegen Hubness (Conneau et al. 2018):

        CSLS(z, y) = 2 cos(z, y) − r_T(z) − r_S(y)

    query_vecs: (q, d) projizierte Queries; tgt_emb: (Vt, d); src_pool: (Ns, d)
    projizierte Quell-Embeddings (für r_S). Rückgabe: (q,) beste Ziel-Indizes.

    TODO:
      1. sim_qt = query_vecs @ tgt_emb.T                  # (q, Vt) cos(query, target)
      2. r_T = _topk_mean(sim_qt, k)                      # (q,)  Dichte um jede Query
      3. sim_ts = tgt_emb @ src_pool.T                    # (Vt, Ns)
         r_S = _topk_mean(sim_ts, k)                      # (Vt,) Dichte um jedes Ziel
      4. scores = 2*sim_qt - r_T[:,None] - r_S[None,:]
      5. return np.argmax(scores, axis=1)
    """
    raise NotImplementedError("Aufgabe 2: csls implementieren")


def precision_at_1(pred_indices, gold_words, tgt_vocab):
    """Anteil der Queries, deren top-1-Vorhersage exakt das Gold-Wort ist. Vorgegeben."""
    ok = sum(tgt_vocab[p] == g for p, g in zip(pred_indices, gold_words))
    return ok / len(gold_words)
