"""Traffic simulation on the REAL AS topology.

WHY SYNTHETIC? The topology is real (SNAP oregon1), the traffic is simulated. The reason: public
traffic matrices of real backbones (Abilene, GEANT via SNDlib) could not be loaded without
hurdles at build time (Zenodo mirror: HTTP 403). Instead of taking a poor substitute, we simulate
traffic with the properties that real network traffic demonstrably has — and we disclose the
generator, so that every assumption is visible and changeable.

Modelled properties (script 3.1):
  * base load ~ log(degree)        (the gravity idea: large AS carry more traffic)
  * a daily pattern (period 24 h)  + a node-individual phase
  * a weekly pattern (weekend ~ -30 %)
  * AR(1) noise                    (traffic is correlated, not white noise)
  * PROPAGATING EVENTS             (flash crowd / congestion): an outbreak at ONE node
    spreads across the edges with a delay.

The last element is the actual point of the project: ONLY if traffic spreads spatially can the
topology provide any additional information at all. `decay` and `spread` dose exactly that — and
thereby make the central question measurable:
**How strongly does traffic have to propagate for a graph model to be worth it?**
"""
from __future__ import annotations
import gzip
import os
import urllib.request

import networkx as nx
import numpy as np

URL = "https://snap.stanford.edu/data/oregon1_010331.txt.gz"
PATH = "datasets/oregon1_010331.txt.gz"


def load_topology(path: str = PATH) -> nx.Graph:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})  # without UA: 403
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    with gzip.open(path, "rt") as f:
        lines = [l for l in f if not l.startswith("#")]
    return nx.parse_edgelist(lines, nodetype=int)


def backbone_subgraph(G: nx.Graph, max_nodes: int = 200) -> nx.Graph:
    """A connected subgraph around the largest hubs.

    Why a subgraph? A real backbone has ~10-100 PoPs; simulating 10,670 AS hourly would be
    neither realistic nor necessary. We keep the real structure (hubs + their neighbourhood),
    only smaller.
    """
    hubs = [n for n, _ in sorted(G.degree(), key=lambda t: -t[1])[:12]]
    selected = set(hubs)
    for h in hubs:
        selected.update(list(G.neighbors(h))[:30])
    S = G.subgraph(list(selected)[:max_nodes]).copy()
    return S.subgraph(max(nx.connected_components(S), key=len)).copy()


def row_normalized_adjacency(G: nx.Graph) -> np.ndarray:
    """Row-normalized adjacency: (A_norm @ y)[i] = the mean of y over the neighbours of i."""
    A = nx.to_numpy_array(G)
    return A / np.maximum(A.sum(axis=1, keepdims=True), 1.0)


def simulate_traffic(G: nx.Graph, weeks: int = 4, decay: float = 0.2, spread: float = 0.75,
                     n_events: int = 300, seed: int = 0):
    """Produces a traffic time series Y of shape (T, n) — T hours x n nodes.

    decay  : how strongly an event persists at the node ITSELF        (0 = not at all)
    spread : how strongly it carries over to the NEIGHBOURS           (high = strong propagation)

    Rule of thumb: the smaller `decay` and the larger `spread`, the more information sits in the
    topology (and the less in the node's own past).

    Returns: (Y, E) with E = the pure event component (for diagnosis).
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    T = 24 * 7 * weeks
    A_norm = row_normalized_adjacency(G)

    degree = np.array([d for _, d in G.degree()], dtype=float)
    base_load = 50 + 20 * np.log1p(degree)               # the gravity idea
    hour = np.arange(T) % 24
    day = (np.arange(T) // 24) % 7
    weekly_pattern = np.where(day >= 5, 0.7, 1.0)        # the weekend is weaker
    phase = rng.uniform(0, 2 * np.pi, n)                 # every node slightly shifted

    # --- propagating events ---
    outbreak = np.zeros((T, n))
    for _ in range(n_events):
        outbreak[rng.integers(0, T - 6), rng.integers(0, n)] += rng.uniform(300, 800)
    E = np.zeros((T, n))
    for k in range(1, T):
        E[k] = decay * E[k - 1] + spread * (A_norm @ E[k - 1]) + outbreak[k]

    # --- total traffic ---
    Y = np.zeros((T, n))
    noise = np.zeros(n)
    for k in range(T):
        noise = 0.8 * noise + rng.normal(0, 3, n)        # AR(1)
        season = 1 + 0.6 * np.sin(2 * np.pi * hour[k] / 24 - np.pi / 2 + 0.3 * phase)
        Y[k] = np.maximum(base_load * season * weekly_pattern[k] + noise + E[k], 1.0)
    return Y, E


def summary_stats(G: nx.Graph, Y: np.ndarray, E: np.ndarray) -> str:
    node_list = list(G.nodes())
    idx = {k: i for i, k in enumerate(node_list)}
    corr = np.mean([np.corrcoef(Y[:, idx[u]], Y[:, idx[v]])[0, 1]
                    for u, v in list(G.edges())[:200]])
    return (f"Topology: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (real)\n"
            f"Traffic : {Y.shape[0]} hours x {Y.shape[1]} nodes (simulated)\n"
            f"mean correlation of neighbouring nodes: {corr:.3f}\n"
            f"share of propagating events in the traffic: {100*E.sum()/Y.sum():.1f} %")
