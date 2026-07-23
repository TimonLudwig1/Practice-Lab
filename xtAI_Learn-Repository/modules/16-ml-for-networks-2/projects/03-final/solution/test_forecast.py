"""Tests for the traffic simulation, the features and the forecasting.

Call:  python test_forecast.py      (~30 s)
"""
import numpy as np

from traffic_sim import (load_topology, backbone_subgraph, row_normalized_adjacency,
                         simulate_traffic)
from forecast import (seasonal_naive, mase, time_split, build_features, train_ridge,
                      WEEK, DAY)

G = backbone_subgraph(load_topology())
A = row_normalized_adjacency(G)
Y, E = simulate_traffic(G, weeks=4, decay=0.2, spread=0.75, seed=0)


# ---------------------- topology ----------------------
def test_topology_is_real_and_connected():
    import networkx as nx
    assert nx.is_connected(G)
    assert 50 <= G.number_of_nodes() <= 200
    assert G.number_of_edges() > G.number_of_nodes()      # not a tree


def test_adjacency_matrix_is_row_normalized():
    row_sums = A.sum(axis=1)
    assert np.allclose(row_sums[row_sums > 0], 1.0)
    assert np.all(np.diag(A) == 0)                        # no self-loop here (unlike the GCN!)


# ---------------------- simulation ----------------------
def test_traffic_shape_and_positive():
    assert Y.shape == (24 * 7 * 4, G.number_of_nodes())
    assert np.all(Y >= 1.0) and np.all(np.isfinite(Y))


def test_daily_pattern_exists():
    # The autocorrelation at lag 24 has to be high (period 24 h)
    y = Y[:, 0]
    r24 = np.corrcoef(y[24:], y[:-24])[0, 1]
    r12 = np.corrcoef(y[12:], y[:-12])[0, 1]
    assert r24 > 0.5, r24
    assert r24 > r12, (r24, r12)      # 24h more similar than 12h (the opposite phase)


def test_weekend_has_less_traffic():
    t = np.arange(Y.shape[0]); day = (t // 24) % 7
    weekday = Y[day < 5].mean()
    weekend = Y[day >= 5].mean()
    assert weekend < weekday, (weekend, weekday)


def test_large_as_carry_more_traffic():
    degree = np.array([d for _, d in G.degree()], float)
    mean_traffic = Y.mean(axis=0)
    assert np.corrcoef(degree, mean_traffic)[0, 1] > 0.3          # the gravity idea


def test_events_propagate_only_with_spread():
    # spread=0: an outbreak stays local -> the neighbours see nothing of it
    _, E_local = simulate_traffic(G, weeks=2, decay=0.9, spread=0.0, seed=1)
    _, E_wide = simulate_traffic(G, weeks=2, decay=0.0, spread=0.95, seed=1)
    # the share of nodes that ever see an event at all
    affected_local = (E_local.max(axis=0) > 1).mean()
    affected_wide = (E_wide.max(axis=0) > 1).mean()
    assert affected_wide > affected_local, (affected_wide, affected_local)


def test_simulation_is_reproducible():
    Y1, _ = simulate_traffic(G, weeks=1, seed=42)
    Y2, _ = simulate_traffic(G, weeks=1, seed=42)
    assert np.array_equal(Y1, Y2)


# ---------------------- split / features ----------------------
def test_time_split_has_no_future_in_the_training_set():
    # THE leakage check: every training time point lies BEFORE every test time point.
    train_idx, test_idx = time_split(Y.shape[0])
    assert max(train_idx) < min(test_idx)
    assert min(train_idx) > WEEK                           # enough history for all lags


def test_calendar_is_cyclically_encoded():
    # 11 p.m. and midnight have to be CLOSE together (raw they would be maximally far apart)
    def encode(h):
        return np.array([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)])
    d_23_0 = np.linalg.norm(encode(23) - encode(0))
    d_23_11 = np.linalg.norm(encode(23) - encode(11))
    assert d_23_0 < d_23_11, (d_23_0, d_23_11)


def test_graph_features_add_columns():
    idx = range(WEEK + 1, WEEK + 5)
    X_without, y1 = build_features(Y, A, idx, with_graph=False)
    X_with, y2 = build_features(Y, A, idx, with_graph=True)
    assert X_with.shape[1] == X_without.shape[1] + 3       # neighbour t-1, t-2, 2-hop
    assert np.array_equal(y1, y2)                          # the target stays the same
    assert X_without.shape[0] == len(idx) * G.number_of_nodes()


# ---------------------- metrics / models ----------------------
def test_mase_of_the_baseline_against_itself_is_one():
    _, test_idx = time_split(Y.shape[0])
    yn = seasonal_naive(Y, test_idx).ravel()
    yt = Y[list(test_idx)].ravel()
    assert np.isclose(mase(yt, yn, yn), 1.0)


def test_seasonal_naive_reaches_one_week_back():
    _, test_idx = time_split(Y.shape[0])
    yn = seasonal_naive(Y, test_idx)
    expected = Y[[k - WEEK for k in test_idx]]
    assert np.array_equal(yn, expected)


def test_ridge_beats_the_naive_baseline():
    train_idx, test_idx = time_split(Y.shape[0])
    yn = seasonal_naive(Y, test_idx).ravel()
    yt, yp = train_ridge(Y, A, train_idx, test_idx, with_graph=False)
    assert mase(yt, yp, yn) < 1.0                          # otherwise the model would be worthless


def test_graph_helps_when_traffic_propagates():
    # The core statement of the project: with a strong spread the graph has to help measurably.
    Ys, _ = simulate_traffic(G, weeks=4, decay=0.0, spread=0.95, seed=0)
    tr, te = time_split(Ys.shape[0])
    yn = seasonal_naive(Ys, te).ravel()
    m_without = mase(*train_ridge(Ys, A, tr, te, False), yn)
    m_with = mase(*train_ridge(Ys, A, tr, te, True), yn)
    assert m_with < m_without, (m_with, m_without)
    assert (m_without - m_with) / m_without > 0.05         # at least 5 % better


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("All tests passed.")
