"""Graph laden, Link-Prediction-Split bauen, Negativbeispiele ziehen — Infrastruktur (gegeben).

Datensatz: echte AS-Peering-Topologie (SNAP oregon1_010331, wie Projekt 01).

Der wichtigste Teil hier sind die ZWEI Arten von Negativbeispielen:
  - uniform         : zufaellige Knotenpaare (der Literatur-Standard)
  - grad-gematcht   : Paare mit derselben Gradverteilung wie die echten Kanten
Projekt 01 hat gezeigt, warum das entscheidend ist: uniform gezogene Paare sind in einem
scale-free-Graphen fast immer Blatt x Blatt (Median-Gradprodukt 4 vs. 500 bei echten Kanten).
Jede gradbasierte Groesse trennt sie dann trivial - ohne irgendetwas ueber Struktur zu wissen.
"""
from __future__ import annotations
import gzip
import os
import random
import urllib.request

import networkx as nx
import numpy as np

URL = "https://snap.stanford.edu/data/oregon1_010331.txt.gz"
PFAD = "daten/oregon1_010331.txt.gz"


def lade_graph(pfad: str = PFAD) -> nx.Graph:
    os.makedirs(os.path.dirname(pfad), exist_ok=True)
    if not os.path.exists(pfad):
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})  # ohne UA: 403
        with urllib.request.urlopen(req, timeout=60) as r, open(pfad, "wb") as f:
            f.write(r.read())
    with gzip.open(pfad, "rt") as f:
        zeilen = [z for z in f if not z.startswith("#")]
    return nx.parse_edgelist(zeilen, nodetype=int)


class LinkSplit:
    """Haelt den Graphen, die Index-Abbildung und den Trainings-/Test-Split der Kanten.

    WICHTIG (Skript 2.5, Falle 1): `train_pos` enthaelt den Graphen OHNE die Testkanten.
    Nur darauf duerfen Features/Embeddings/Adjazenz berechnet werden - sonst Leakage.
    """

    def __init__(self, test_anteil: float = 0.10, seed: int = 0):
        self.seed = seed
        self.G = lade_graph()
        self.knoten = sorted(self.G.nodes())
        self.idx = {k: i for i, k in enumerate(self.knoten)}
        self.n = len(self.knoten)

        kanten = [(self.idx[u], self.idx[v]) for u, v in self.G.edges()]
        random.Random(seed).shuffle(kanten)
        n_test = int(test_anteil * len(kanten))
        self.test_pos = kanten[:n_test]
        self.train_pos = kanten[n_test:]

        # Graph OHNE Testkanten - die Grundlage fuer alles Gelernte
        self.G_train = self.G.copy()
        self.G_train.remove_edges_from([(self.knoten[u], self.knoten[v])
                                        for u, v in self.test_pos])
        self._grad = {i: self.G.degree(self.knoten[i]) for i in range(self.n)}

    def _ist_kante(self, u: int, v: int) -> bool:
        return self.G.has_edge(self.knoten[u], self.knoten[v])

    def negative_uniform(self, k: int, seed=None):
        """Zufaellige Nicht-Kanten (der uebliche, aber irrefuehrende Standard)."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        out = []
        while len(out) < k:
            u, v = rng.integers(0, self.n, 2)
            if u != v and not self._ist_kante(int(u), int(v)):
                out.append((int(u), int(v)))
        return out

    def negative_grad_gematcht(self, positive, seed=None):
        """Nicht-Kanten mit (moeglichst) denselben Graden wie die jeweilige echte Kante.

        Damit ist der Grad als Merkmal 'ausgeschaltet' - was uebrig bleibt, ist echte Struktur.
        """
        rnd = random.Random(self.seed if seed is None else seed)
        bins = {}
        for i in range(self.n):
            bins.setdefault(self._grad[i], []).append(i)
        grad_werte = np.array(sorted(bins))

        def ziehe(d):
            g = int(grad_werte[np.argmin(np.abs(grad_werte - d))])
            return rnd.choice(bins[g])

        out = []
        for (u0, v0) in positive:
            for _ in range(40):                    # mehrere Versuche je Kante
                u, v = ziehe(self._grad[u0]), ziehe(self._grad[v0])
                if u != v and not self._ist_kante(u, v):
                    out.append((u, v))
                    break
        return out

    def gradprodukt(self, paare):
        return np.array([self._grad[u] * self._grad[v] for u, v in paare])

    def uebersicht(self) -> str:
        return (f"Graph: {self.n:,} Knoten, {self.G.number_of_edges():,} Kanten\n"
                f"Split: {len(self.train_pos):,} Trainingskanten / "
                f"{len(self.test_pos):,} Testkanten (aus G_train entfernt)")
