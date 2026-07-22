"""Test suite for alignment & retrieval (fast, synthetic — no embedding building).

    python test_align.py

Checks the closed-form Procrustes solution and the retrieval measures on controlled data
where the correct answer is known.
"""
import numpy as np

from align import (orthogonal_procrustes, nearest_neighbor, csls, precision_at_1)

rng = np.random.RandomState(0)


def random_orthogonal(d):
    Q, _ = np.linalg.qr(rng.randn(d, d))
    return Q


def test_procrustes_recovers_rotation():
    # Y = X Q for a known rotation Q -> Procrustes must return Q.
    d, n = 20, 300
    X = rng.randn(n, d)
    Q = random_orthogonal(d)
    Y = X @ Q
    W = orthogonal_procrustes(X, Y)
    assert np.allclose(W, Q, atol=1e-6), "W should be the exact rotation Q"
    assert np.allclose(X @ W, Y, atol=1e-6), "X W should hit Y"
    print("  Procrustes recovers a known rotation ... OK")


def test_procrustes_orthogonal():
    d = 15
    X = rng.randn(120, d)
    Y = rng.randn(120, d)
    W = orthogonal_procrustes(X, Y)
    assert np.allclose(W @ W.T, np.eye(d), atol=1e-6), "W must be orthogonal"
    assert abs(abs(np.linalg.det(W)) - 1.0) < 1e-6
    print("  Procrustes solution is orthogonal ... OK")


def test_procrustes_direction():
    # Direction check: M = X^T Y (source^T target), not Y^T X.
    d, n = 10, 200
    X = rng.randn(n, d)
    Q = random_orthogonal(d)
    Y = X @ Q
    W = orthogonal_procrustes(X, Y)
    # Residual in the correct direction ~ 0, in the wrong direction large.
    err_correct = np.linalg.norm(X @ W - Y)
    err_wrong = np.linalg.norm(Y @ W - X)
    assert err_correct < 1e-6 < err_wrong, "W must map source->target (X W ~ Y)"
    print("  Alignment direction source->target ... OK")


def test_procrustes_denoises():
    # With noise: X W should be close to Y, markedly better than without alignment.
    d, n = 30, 500
    X = rng.randn(n, d)
    Q = random_orthogonal(d)
    Y = X @ Q + 0.1 * rng.randn(n, d)
    W = orthogonal_procrustes(X, Y)
    err_aligned = np.linalg.norm(X @ W - Y)
    err_raw = np.linalg.norm(X - Y)
    assert err_aligned < 0.5 * err_raw
    print("  Procrustes reduces the residual under noise ... OK")


def _two_spaces(d=12, V=60):
    """Target space = random unit vectors; source space = rotated back.
    Each source row i translates exactly to target row i."""
    Y = rng.randn(V, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    Q = random_orthogonal(d)
    X = Y @ Q.T                      # X @ Q = Y
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, Y, Q


def test_nn_retrieval_perfect_on_clean():
    X, Y, Q = _two_spaces()
    W = orthogonal_procrustes(X, Y)          # should be ~Q
    proj = X @ W
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    pred = nearest_neighbor(proj, Y)
    assert np.array_equal(pred, np.arange(len(Y))), "each query must hit its counterpart"
    print("  NN retrieval perfect on clean spaces ... OK")


def test_csls_matches_nn_on_clean():
    X, Y, Q = _two_spaces()
    W = orthogonal_procrustes(X, Y)
    proj = X @ W
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    pred = csls(proj, Y, proj, k=5)
    assert np.array_equal(pred, np.arange(len(Y)))
    print("  CSLS retrieval perfect on clean spaces ... OK")


def test_csls_fixes_hub():
    # Construct a hub: one target vector h lies close to MANY queries and is
    # wrongly the NN for a query whose true target is elsewhere. CSLS penalizes h.
    d = 8
    # two "real" pairs + one hub
    q0 = np.zeros(d); q0[0] = 1.0
    t0 = q0.copy()                          # correct target for q0 (cos 1.0)
    hub = np.ones(d) / np.sqrt(d)           # hub vector
    # further queries, all fairly close to the hub -> the hub becomes "popular"
    extra = hub + 0.05 * rng.randn(30, d)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    # tilt q0 towards the hub, so that NN(q0)=hub instead of t0
    q = 0.4 * q0 + 0.75 * hub
    q /= np.linalg.norm(q)
    queries = np.vstack([q, extra])         # (31, d); we evaluate query 0 only
    targets = np.vstack([t0, hub])          # index 0 = correct, 1 = hub
    # NN wrongly picks the hub for q0
    nn_pred = nearest_neighbor(queries[:1], targets)
    csls_pred = csls(queries[:1], targets, queries, k=10)
    assert nn_pred[0] == 1, "setup: NN should pick the hub here"
    assert csls_pred[0] == 0, "CSLS should correct the hub and pick t0"
    print("  CSLS corrects hubness ... OK")


def test_precision_at_1():
    vocab = ["a", "b", "c"]
    assert precision_at_1(np.array([0, 1, 2]), ["a", "b", "c"], vocab) == 1.0
    assert precision_at_1(np.array([0, 0, 2]), ["a", "b", "c"], vocab) == 2 / 3
    print("  precision_at_1 ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} tests ...")
    for t in tests:
        t()
    print("All tests passed.")
