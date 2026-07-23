"""A graph convolutional network (Kipf & Welling 2017) — from scratch, without torch_geometric.

The whole magic is two lines of mathematics:

    A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}          # computed once in advance
    H^(k) = sigma( A_hat @ H^(k-1) @ W^(k) )               # one layer

Why exactly like that (script 2.3):
  * (A + I)   self-loops - otherwise the node drops out of its OWN update.
  * D^{-1/2} ... D^{-1/2}  symmetric normalization. Without it you are summing neighbours: a hub
    of degree 2312 would get activations ~1000x larger than a leaf of degree 2. Symmetric
    (instead of row-wise D^{-1}A), because that (a) leaves A_hat symmetric, (b) damps the message
    of a hub (hubs are uninformative - the same intuition as Adamic-Adar/IDF) and (c) keeps the
    eigenvalues in [-1, 1] => stable depth.

SPARSE is not optional here: a dense 10670x10670 matrix would be ~455 MB, but 99.96 % of it are
zeros. torch.sparse.mm solves that.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def normalized_adjacency(edges, n: int) -> torch.Tensor:
    """Computes A_hat = D^{-1/2}(A+I)D^{-1/2} as a sparse COO tensor.

    edges: a list of (u, v) index pairs (undirected, every edge listed ONCE).
    """
    rows = [u for u, _ in edges] + [v for _, v in edges] + list(range(n))
    cols = [v for _, v in edges] + [u for u, _ in edges] + list(range(n))
    r = torch.tensor(rows, dtype=torch.long)
    c = torch.tensor(cols, dtype=torch.long)
    values = torch.ones(len(rows))

    degree = torch.zeros(n).scatter_add_(0, r, values)        # degree incl. the self-loop
    d_inv_sqrt = degree.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0                 # catch isolated nodes

    norm_values = d_inv_sqrt[r] * d_inv_sqrt[c]               # 1/sqrt(d_u) * 1/sqrt(d_v)
    A = torch.sparse_coo_tensor(torch.stack([r, c]), norm_values, (n, n))
    return A.coalesce()


class GCN(nn.Module):
    """A 2-layer GCN. Produces one embedding per node.

    The graph has no node features (it is pure topology!). That is why we learn the input
    representation ourselves: an embedding table as H^(0). This makes the model - as GCN
    generally is - **transductive** (script 2.3): for a new node there would be no row in the
    table. That is exactly the problem GraphSAGE solves.

    Why only 2 layers? Over-smoothing (script 2.4) - and in a small-world graph (mean path ~3.7)
    two hops already see a large part of the network anyway.
    """

    def __init__(self, n_nodes: int, dim: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(n_nodes, dim)         # H^(0), learned
        nn.init.normal_(self.emb.weight, std=0.1)
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)

    def forward(self, A_hat: torch.Tensor) -> torch.Tensor:
        H = torch.sparse.mm(A_hat, self.emb.weight)   # layer 1: 1 hop
        H = torch.relu(self.W1(H))
        H = torch.sparse.mm(A_hat, H)                 # layer 2: 2 hops
        return self.W2(H)


class MLPWithoutStructure(nn.Module):
    """The control group: identical to the GCN, but WITHOUT message passing (no A_hat).

    If this model is just as good, the topology contributes nothing - and the whole GNN would be
    superfluous effort. That is exactly what a control group is for.
    """

    def __init__(self, n_nodes: int, dim: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(n_nodes, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)

    def forward(self, A_hat=None):
        H = torch.relu(self.W1(self.emb.weight))
        return self.W2(H)


def edge_score(Z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """The score of an edge = the dot product of the two node embeddings.

    A high score <=> the two nodes 'fit together' <=> an edge is likely.
    """
    return (Z[pairs[:, 0]] * Z[pairs[:, 1]]).sum(dim=1)
