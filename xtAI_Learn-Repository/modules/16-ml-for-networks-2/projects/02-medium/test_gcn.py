"""Tests for the normalized adjacency, the GCN and the link split.

Call:  python test_gcn.py      (~40 s: loads the graph and trains briefly)
"""
import numpy as np
import torch

from gcn import GCN, MLPWithoutStructure, normalized_adjacency, edge_score
from graph_data import LinkSplit

# A mini graph to compute by hand:  0 - 1 - 2   (a path)
MINI = [(0, 1), (1, 2)]
N_MINI = 3


# ---------------------- the normalized adjacency ----------------------
def test_a_hat_is_symmetric():
    A = normalized_adjacency(MINI, N_MINI).to_dense()
    assert torch.allclose(A, A.T), A


def test_a_hat_has_self_loops():
    # Without (A+I) the diagonal would be 0 -> the node would forget itself.
    A = normalized_adjacency(MINI, N_MINI).to_dense()
    assert torch.all(torch.diag(A) > 0), torch.diag(A)


def test_a_hat_values_recomputed_by_hand():
    # Path 0-1-2, with self-loops: d0=2, d1=3, d2=2
    # A_hat[u,v] = 1/(sqrt(d_u)*sqrt(d_v))
    A = normalized_adjacency(MINI, N_MINI).to_dense()
    d = np.array([2.0, 3.0, 2.0])
    assert np.isclose(A[0, 0].item(), 1/np.sqrt(d[0]*d[0]))     # self-loop 0
    assert np.isclose(A[0, 1].item(), 1/np.sqrt(d[0]*d[1]))     # edge 0-1
    assert np.isclose(A[1, 1].item(), 1/np.sqrt(d[1]*d[1]))     # self-loop 1
    assert A[0, 2].item() == 0.0                                # no edge 0-2


def test_a_hat_eigenvalues_in_minus_one_to_one():
    # This is the entire reason for the symmetric normalization: stable depth.
    A = normalized_adjacency(MINI, N_MINI).to_dense()
    ev = torch.linalg.eigvalsh(A)
    assert torch.all(ev <= 1.0 + 1e-6) and torch.all(ev >= -1.0 - 1e-6), ev


def test_a_hat_is_sparse_and_the_right_size():
    A = normalized_adjacency(MINI, N_MINI)
    assert A.is_sparse
    assert tuple(A.shape) == (N_MINI, N_MINI)
    # 2 edges * 2 directions + 3 self-loops = 7
    assert A._nnz() == 7, A._nnz()


def test_a_hat_damps_hubs():
    # A star: node 0 is a hub with 5 leaves. The message of the hub to a leaf
    # has to be weaker than that of a leaf to itself.
    star = [(0, i) for i in range(1, 6)]
    A = normalized_adjacency(star, 6).to_dense()
    assert A[1, 0] < A[1, 1], "the hub message should be damped"


def test_isolated_node_produces_no_nan():
    # Node 3 without edges (only a self-loop) must not lead to a division by zero
    A = normalized_adjacency(MINI, 4).to_dense()
    assert torch.all(torch.isfinite(A))


# ---------------------- GCN ----------------------
def test_gcn_output_shape():
    A = normalized_adjacency(MINI, N_MINI)
    Z = GCN(N_MINI, dim=8)(A)
    assert Z.shape == (N_MINI, 8)
    assert torch.all(torch.isfinite(Z))


def test_edge_score_is_symmetric():
    Z = torch.randn(5, 4)
    s1 = edge_score(Z, torch.tensor([[0, 1]]))
    s2 = edge_score(Z, torch.tensor([[1, 0]]))
    assert torch.allclose(s1, s2)     # the dot product is symmetric => undirected


def test_message_passing_mixes_neighbors():
    # The core of the GNN: after one A_hat multiplication the state of a node must have
    # changed BECAUSE its neighbours have flowed into it.
    A = normalized_adjacency(MINI, N_MINI).to_dense()
    H = torch.eye(N_MINI)                 # every node "knows" only itself
    H1 = A @ H
    assert H1[0, 1] > 0, "after 1 hop node 0 should know something about node 1"
    assert H1[0, 2] == 0, "node 2 is 2 hops away - still invisible after 1 hop"
    H2 = A @ H1
    assert H2[0, 2] > 0, "after 2 hops node 2 should be visible"


# ---------------------- split / leakage ----------------------
_SPLIT = LinkSplit(test_fraction=0.10, seed=0)


def test_test_edges_are_removed_from_the_training_graph():
    # THE leakage check (script 2.5, trap 1)
    for u, v in _SPLIT.test_pos[:200]:
        assert not _SPLIT.G_train.has_edge(_SPLIT.nodes[u], _SPLIT.nodes[v])
        assert _SPLIT.G.has_edge(_SPLIT.nodes[u], _SPLIT.nodes[v])   # present in the original


def test_split_is_complete():
    assert len(_SPLIT.train_pos) + len(_SPLIT.test_pos) == _SPLIT.G.number_of_edges()


def test_negatives_are_not_real_edges():
    for u, v in _SPLIT.negative_uniform(300, seed=1):
        assert not _SPLIT.G.has_edge(_SPLIT.nodes[u], _SPLIT.nodes[v])
        assert u != v


def test_degree_confound_really_exists():
    # The finding from project 01: uniformly drawn pairs are almost always leaf x leaf.
    uni = _SPLIT.negative_uniform(500, seed=2)
    matched = _SPLIT.negative_degree_matched(_SPLIT.test_pos[:500], seed=2)
    med_pos = np.median(_SPLIT.degree_product(_SPLIT.test_pos))
    med_uni = np.median(_SPLIT.degree_product(uni))
    med_matched = np.median(_SPLIT.degree_product(matched))
    assert med_uni < med_pos / 20, (med_pos, med_uni)          # >20x difference
    assert med_matched > med_pos / 5, (med_pos, med_matched)   # matched: comparable


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("All tests passed.")
