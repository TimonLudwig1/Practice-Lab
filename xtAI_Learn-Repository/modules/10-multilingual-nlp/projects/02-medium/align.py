"""Cross-lingual embedding alignment — the core of the project.

Implement the closed-form **orthogonal Procrustes** solution and the **CSLS** retrieval.
`nearest_neighbor`, `_topk_mean` and `precision_at_1` are given. See script section 3,
and check with `python test_align.py`.
"""
import numpy as np


def orthogonal_procrustes(X, Y):
    r"""Solve min_W ||X W - Y||_F  subject to  W^T W = I.

    X, Y: (n, d) — stacked source/target embeddings of the anchor pairs (rows = words).

    TODO (closed-form solution, script 3.2):
      1. M = X^T Y            (source^T . target — the DIRECTION matters!)
      2. U, _, Vt = np.linalg.svd(M)
      3. return U @ Vt        (orthogonal, maps source -> target: X W ~ Y)
    """
    raise NotImplementedError("Task 1: implement orthogonal_procrustes")


def _topk_mean(sims, k):
    """Mean of the k largest values per row (for the CSLS neighbourhood density). Given."""
    k = min(k, sims.shape[1])
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def nearest_neighbor(query_vecs, tgt_emb):
    """For each (projected, normalized) query the most similar target index. Given."""
    sims = query_vecs @ tgt_emb.T          # (q, Vt) = cosine (vectors are normalized)
    return np.argmax(sims, axis=1)


def csls(query_vecs, tgt_emb, src_pool, k=10):
    r"""CSLS retrieval against hubness (Conneau et al. 2018):

        CSLS(z, y) = 2 cos(z, y) - r_T(z) - r_S(y)

    query_vecs: (q, d) projected queries; tgt_emb: (Vt, d); src_pool: (Ns, d)
    projected source embeddings (for r_S). Returns: (q,) best target indices.

    TODO:
      1. sim_qt = query_vecs @ tgt_emb.T                  # (q, Vt) cos(query, target)
      2. r_T = _topk_mean(sim_qt, k)                      # (q,)  density around each query
      3. sim_ts = tgt_emb @ src_pool.T                    # (Vt, Ns)
         r_S = _topk_mean(sim_ts, k)                      # (Vt,) density around each target
      4. scores = 2*sim_qt - r_T[:,None] - r_S[None,:]
      5. return np.argmax(scores, axis=1)
    """
    raise NotImplementedError("Task 2: implement csls")


def precision_at_1(pred_indices, gold_words, tgt_vocab):
    """Share of queries whose top-1 prediction is exactly the gold word. Given."""
    ok = sum(tgt_vocab[p] == g for p, g in zip(pred_indices, gold_words))
    return ok / len(gold_words)
