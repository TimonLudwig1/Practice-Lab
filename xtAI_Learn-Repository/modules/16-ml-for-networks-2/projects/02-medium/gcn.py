"""A graph convolutional network (Kipf & Welling 2017) — from scratch, without torch_geometric.

>>> YOUR TASK <<<  Fill in the four functions/methods marked with TODO.

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

Reference solution: solution/gcn.py — try it yourself first!
"""
from __future__ import annotations
import torch
import torch.nn as nn


def normalized_adjacency(edges, n: int) -> torch.Tensor:
    """Compute A_hat = D^{-1/2}(A+I)D^{-1/2} as a SPARSE COO tensor.

    edges: a list of (u, v) index pairs (undirected, every edge listed ONCE).

    Blueprint:
      1. Build the index lists. Because the graph is undirected, every edge needs BOTH
         directions: rows = [u...] + [v...] + [0..n-1]   (the last piece = the self-loops!)
         cols = [v...] + [u...] + [0..n-1]
      2. degree = torch.zeros(n).scatter_add_(0, rows, values)   # degree incl. the self-loop
      3. d_inv_sqrt = degree.pow(-0.5);  catch Inf (isolated nodes!) -> 0
      4. norm_values[i] = d_inv_sqrt[rows[i]] * d_inv_sqrt[cols[i]]
      5. torch.sparse_coo_tensor(torch.stack([rows, cols]), norm_values, (n, n)).coalesce()

    Self-check (the tests check exactly this):
      - A_hat is symmetric
      - the diagonal is > 0 (self-loops!)
      - for the path 0-1-2 it holds that A_hat[0,1] = 1/sqrt(2*3)
      - all eigenvalues lie in [-1, 1]
    """
    # TODO
    raise NotImplementedError


class GCN(nn.Module):
    """A 2-layer GCN. Produces one embedding per node.

    The graph has NO node features (pure topology). That is why we learn the input
    representation ourselves: an nn.Embedding table as H^(0). This makes the model - as GCN
    generally is - **transductive** (script 2.3).

    Why only 2 layers? Over-smoothing (script 2.4).
    """

    def __init__(self, n_nodes: int, dim: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(n_nodes, dim)         # H^(0), learned
        nn.init.normal_(self.emb.weight, std=0.1)
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)

    def forward(self, A_hat: torch.Tensor) -> torch.Tensor:
        """Two GCN layers.

        Layer 1:  H = relu( W1( A_hat @ emb ) )     # 1 hop
        Layer 2:  Z = W2( A_hat @ H )               # 2 hops
        Hint: torch.sparse.mm(A_hat, X) for sparse @ dense.
        """
        # TODO
        raise NotImplementedError


class MLPWithoutStructure(nn.Module):
    """The control group: identical to the GCN, but WITHOUT message passing (A_hat is ignored).

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
        """Like GCN.forward, but WITHOUT any A_hat multiplication."""
        # TODO
        raise NotImplementedError


def edge_score(Z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """The score of an edge = the dot product of the two node embeddings.

    Z:     (n, dim) node embeddings
    pairs: (k, 2) long tensor of node pairs
    Returns: (k,) scores. High <=> the two fit together <=> an edge is likely.

    Hint: (Z[pairs[:,0]] * Z[pairs[:,1]]).sum(dim=1)   - the row-wise dot product.
    """
    # TODO
    raise NotImplementedError
