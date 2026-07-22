"""Cross-lingual embedding alignment — the core of the project.

Contains the closed-form **orthogonal Procrustes** solution and the retrieval measures
**nearest neighbour** and **CSLS** (against *hubness*). See script section 3.
"""
import numpy as np


def orthogonal_procrustes(X, Y):
    r"""Solve min_W ||X W - Y||_F  subject to  W^T W = I.

    X, Y: (n, d) — stacked source/target embeddings of the anchor pairs (rows = words).
    Closed-form solution: M = X^T Y = U S V^T  =>  W = U V^T   (maps source -> target).
    Returns: W (d, d) orthogonal.
    """
    M = X.T @ Y                       # (d, d) = source^T . target
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def _topk_mean(sims, k):
    """Mean of the k largest values per row (for the CSLS neighbourhood density)."""
    k = min(k, sims.shape[1])
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def nearest_neighbor(query_vecs, tgt_emb):
    """For each (already projected, normalized) query the most similar target index.

    query_vecs: (q, d), tgt_emb: (Vt, d) — both length-normalized => dot product = cosine.
    Returns: (q,) indices of the nearest target words.
    """
    sims = query_vecs @ tgt_emb.T          # (q, Vt)
    return np.argmax(sims, axis=1)


def csls(query_vecs, tgt_emb, src_pool, k=10):
    r"""CSLS retrieval against hubness (Conneau et al. 2018).

    CSLS(z, y) = 2 cos(z, y) - r_T(z) - r_S(y), with
      r_T(z) = mean cosine similarity of z to its k nearest target neighbours,
      r_S(y) = mean cosine similarity of y to its k nearest (projected) source
               neighbours from the pool.

    query_vecs: (q, d) projected queries;  tgt_emb: (Vt, d);
    src_pool:   (Ns, d) projected source embeddings (for r_S).
    Returns: (q,) indices of the best target words by CSLS.
    """
    sim_qt = query_vecs @ tgt_emb.T        # (q, Vt) cos(query, target)
    r_T = _topk_mean(sim_qt, k)            # (q,)  density around each query in target space
    sim_ts = tgt_emb @ src_pool.T          # (Vt, Ns) cos(target, source pool)
    r_S = _topk_mean(sim_ts, k)            # (Vt,) density around each target in source space
    scores = 2 * sim_qt - r_T[:, None] - r_S[None, :]
    return np.argmax(scores, axis=1)


def precision_at_1(pred_indices, gold_words, tgt_vocab):
    """Share of queries whose top-1 prediction is exactly the gold word."""
    ok = sum(tgt_vocab[p] == g for p, g in zip(pred_indices, gold_words))
    return ok / len(gold_words)
