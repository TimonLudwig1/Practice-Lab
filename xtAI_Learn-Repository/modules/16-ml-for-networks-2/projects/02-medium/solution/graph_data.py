"""Load the graph, build the link prediction split, draw negative examples — infrastructure (given).

Dataset: the real AS peering topology (SNAP oregon1_010331, as in project 01).

The most important part here are the TWO kinds of negative examples:
  - uniform        : random node pairs (the standard in the literature)
  - degree-matched : pairs with the same degree distribution as the real edges
Project 01 has shown why that is decisive: uniformly drawn pairs are almost always leaf x leaf in
a scale-free graph (median degree product 4 vs. 500 for real edges). Every degree-based quantity
then separates them trivially - without knowing anything about structure.
"""
from __future__ import annotations
import gzip
import os
import random
import urllib.request

import networkx as nx
import numpy as np

URL = "https://snap.stanford.edu/data/oregon1_010331.txt.gz"
PATH = "datasets/oregon1_010331.txt.gz"


def load_graph(path: str = PATH) -> nx.Graph:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})  # without UA: 403
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    with gzip.open(path, "rt") as f:
        lines = [l for l in f if not l.startswith("#")]
    return nx.parse_edgelist(lines, nodetype=int)


class LinkSplit:
    """Holds the graph, the index mapping and the train/test split of the edges.

    IMPORTANT (script 2.5, trap 1): `train_pos` contains the graph WITHOUT the test edges.
    Only on that may features/embeddings/the adjacency be computed - otherwise leakage.
    """

    def __init__(self, test_fraction: float = 0.10, seed: int = 0):
        self.seed = seed
        self.G = load_graph()
        self.nodes = sorted(self.G.nodes())
        self.idx = {k: i for i, k in enumerate(self.nodes)}
        self.n = len(self.nodes)

        edges = [(self.idx[u], self.idx[v]) for u, v in self.G.edges()]
        random.Random(seed).shuffle(edges)
        n_test = int(test_fraction * len(edges))
        self.test_pos = edges[:n_test]
        self.train_pos = edges[n_test:]

        # the graph WITHOUT the test edges - the basis for everything that is learned
        self.G_train = self.G.copy()
        self.G_train.remove_edges_from([(self.nodes[u], self.nodes[v])
                                        for u, v in self.test_pos])
        self._degree = {i: self.G.degree(self.nodes[i]) for i in range(self.n)}

    def _is_edge(self, u: int, v: int) -> bool:
        return self.G.has_edge(self.nodes[u], self.nodes[v])

    def negative_uniform(self, k: int, seed=None):
        """Random non-edges (the usual, but misleading, standard)."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        out = []
        while len(out) < k:
            u, v = rng.integers(0, self.n, 2)
            if u != v and not self._is_edge(int(u), int(v)):
                out.append((int(u), int(v)))
        return out

    def negative_degree_matched(self, positive, seed=None):
        """Non-edges with (as far as possible) the same degrees as the corresponding real edge.

        This 'switches off' the degree as a feature - what remains is genuine structure.
        """
        rnd = random.Random(self.seed if seed is None else seed)
        bins = {}
        for i in range(self.n):
            bins.setdefault(self._degree[i], []).append(i)
        degree_values = np.array(sorted(bins))

        def draw(d):
            g = int(degree_values[np.argmin(np.abs(degree_values - d))])
            return rnd.choice(bins[g])

        out = []
        for (u0, v0) in positive:
            for _ in range(40):                    # several attempts per edge
                u, v = draw(self._degree[u0]), draw(self._degree[v0])
                if u != v and not self._is_edge(u, v):
                    out.append((u, v))
                    break
        return out

    def degree_product(self, pairs):
        return np.array([self._degree[u] * self._degree[v] for u, v in pairs])

    def overview(self) -> str:
        return (f"Graph: {self.n:,} nodes, {self.G.number_of_edges():,} edges\n"
                f"Split: {len(self.train_pos):,} training edges / "
                f"{len(self.test_pos):,} test edges (removed from G_train)")
