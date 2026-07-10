"""Testsuite für Alignment & Retrieval (schnell, synthetisch — kein Embedding-Bau).

    python test_align.py

Prüft die geschlossene Procrustes-Lösung und die Retrieval-Maße auf kontrollierten
Daten, bei denen die richtige Antwort bekannt ist.
"""
import numpy as np

from align import (orthogonal_procrustes, nearest_neighbor, csls, precision_at_1)

rng = np.random.RandomState(0)


def random_orthogonal(d):
    Q, _ = np.linalg.qr(rng.randn(d, d))
    return Q


def test_procrustes_recovers_rotation():
    # Y = X Q fuer eine bekannte Rotation Q -> Procrustes muss Q zurueckgeben.
    d, n = 20, 300
    X = rng.randn(n, d)
    Q = random_orthogonal(d)
    Y = X @ Q
    W = orthogonal_procrustes(X, Y)
    assert np.allclose(W, Q, atol=1e-6), "W sollte die exakte Rotation Q sein"
    assert np.allclose(X @ W, Y, atol=1e-6), "X W sollte Y treffen"
    print("  Procrustes rekonstruiert bekannte Rotation ... OK")


def test_procrustes_orthogonal():
    d = 15
    X = rng.randn(120, d)
    Y = rng.randn(120, d)
    W = orthogonal_procrustes(X, Y)
    assert np.allclose(W @ W.T, np.eye(d), atol=1e-6), "W muss orthogonal sein"
    assert abs(abs(np.linalg.det(W)) - 1.0) < 1e-6
    print("  Procrustes-Lösung ist orthogonal ... OK")


def test_procrustes_direction():
    # Kontrolle der Richtung: M = X^T Y (Quelle^T Ziel), nicht Y^T X.
    d, n = 10, 200
    X = rng.randn(n, d)
    Q = random_orthogonal(d)
    Y = X @ Q
    W = orthogonal_procrustes(X, Y)
    # Residuum in korrekter Richtung ~ 0, in falscher Richtung gross.
    err_correct = np.linalg.norm(X @ W - Y)
    err_wrong = np.linalg.norm(Y @ W - X)
    assert err_correct < 1e-6 < err_wrong, "W muss Quelle->Ziel abbilden (X W ~ Y)"
    print("  Alignment-Richtung Quelle->Ziel ... OK")


def test_procrustes_denoises():
    # Mit Rauschen: X W soll nah an Y sein, deutlich besser als ohne Alignment.
    d, n = 30, 500
    X = rng.randn(n, d)
    Q = random_orthogonal(d)
    Y = X @ Q + 0.1 * rng.randn(n, d)
    W = orthogonal_procrustes(X, Y)
    err_aligned = np.linalg.norm(X @ W - Y)
    err_raw = np.linalg.norm(X - Y)
    assert err_aligned < 0.5 * err_raw
    print("  Procrustes reduziert Residuum bei Rauschen ... OK")


def _two_spaces(d=12, V=60):
    """Zielraum = zufaellige Einheitsvektoren; Quellraum = per Rotation zurueckgedreht.
    Jede Quell-Zeile i uebersetzt exakt zu Ziel-Zeile i."""
    Y = rng.randn(V, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    Q = random_orthogonal(d)
    X = Y @ Q.T                      # X @ Q = Y
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, Y, Q


def test_nn_retrieval_perfect_on_clean():
    X, Y, Q = _two_spaces()
    W = orthogonal_procrustes(X, Y)          # sollte ~Q sein
    proj = X @ W
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    pred = nearest_neighbor(proj, Y)
    assert np.array_equal(pred, np.arange(len(Y))), "jede Query muss ihr Pendant treffen"
    print("  NN-Retrieval perfekt auf sauberen Räumen ... OK")


def test_csls_matches_nn_on_clean():
    X, Y, Q = _two_spaces()
    W = orthogonal_procrustes(X, Y)
    proj = X @ W
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    pred = csls(proj, Y, proj, k=5)
    assert np.array_equal(pred, np.arange(len(Y)))
    print("  CSLS-Retrieval perfekt auf sauberen Räumen ... OK")


def test_csls_fixes_hub():
    # Konstruiere einen Hub: ein Ziel-Vektor h liegt nahe an VIELEN Queries und ist
    # faelschlich NN fuer eine Query, deren echtes Ziel woanders liegt. CSLS bestraft h.
    d = 8
    # zwei "echte" Paare + ein Hub
    q0 = np.zeros(d); q0[0] = 1.0
    t0 = q0.copy()                          # korrektes Ziel fuer q0 (cos 1.0)
    hub = np.ones(d) / np.sqrt(d)           # Hub-Vektor
    # weitere Queries, die alle recht nah am Hub liegen -> Hub wird "beliebt"
    extra = hub + 0.05 * rng.randn(30, d)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    # q0 Richtung Hub kippen, sodass NN(q0)=hub statt t0
    q = 0.4 * q0 + 0.75 * hub
    q /= np.linalg.norm(q)
    queries = np.vstack([q, extra])         # (31, d); nur Query 0 werten wir aus
    targets = np.vstack([t0, hub])          # index 0 = korrekt, 1 = Hub
    # NN waehlt faelschlich den Hub fuer q0
    nn_pred = nearest_neighbor(queries[:1], targets)
    csls_pred = csls(queries[:1], targets, queries, k=10)
    assert nn_pred[0] == 1, "Aufbau: NN soll hier den Hub waehlen"
    assert csls_pred[0] == 0, "CSLS soll den Hub korrigieren und t0 waehlen"
    print("  CSLS korrigiert Hubness ... OK")


def test_precision_at_1():
    vocab = ["a", "b", "c"]
    assert precision_at_1(np.array([0, 1, 2]), ["a", "b", "c"], vocab) == 1.0
    assert precision_at_1(np.array([0, 0, 2]), ["a", "b", "c"], vocab) == 2 / 3
    print("  precision_at_1 ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Starte {len(tests)} Tests ...")
    for t in tests:
        t()
    print("Alle Tests bestanden.")
