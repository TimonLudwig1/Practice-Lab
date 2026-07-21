"""Verkehrssimulation auf der ECHTEN AS-Topologie.

WARUM SYNTHETISCH? Die Topologie ist echt (SNAP oregon1), der Verkehr ist simuliert. Grund:
oeffentliche Verkehrsmatrizen echter Backbones (Abilene, GEANT via SNDlib) waren beim Bau nicht
ohne Huerden ladbar (Zenodo-Mirror: HTTP 403). Statt einen schlechten Ersatz zu nehmen,
simulieren wir Verkehr mit den Eigenschaften, die echter Netzverkehr nachweislich hat — und
legen den Generator offen, damit jede Annahme sichtbar und veraenderbar ist.

Modellierte Eigenschaften (Skript 3.1):
  * Grundlast ~ log(Grad)          (Gravity-Idee: grosse AS tragen mehr Verkehr)
  * Tagesgang (Periode 24 h)       + knotenindividuelle Phase
  * Wochengang (Wochenende ~ -30 %)
  * AR(1)-Rauschen                 (Verkehr ist korreliert, nicht weisses Rauschen)
  * PROPAGIERENDE EREIGNISSE       (Flash Crowd / Congestion): ein Ausbruch an EINEM Knoten
    breitet sich mit Verzoegerung ueber die Kanten aus.

Das letzte Element ist der eigentliche Punkt des Projekts: NUR wenn Verkehr sich raeumlich
ausbreitet, kann die Topologie ueberhaupt Zusatzinformation liefern. Ueber `decay` und `spread`
laesst sich genau das dosieren — und damit die zentrale Frage messen:
**Wie stark muss Verkehr propagieren, damit ein Graph-Modell sich lohnt?**
"""
from __future__ import annotations
import gzip
import os
import urllib.request

import networkx as nx
import numpy as np

URL = "https://snap.stanford.edu/data/oregon1_010331.txt.gz"
PFAD = "datasets/oregon1_010331.txt.gz"


def lade_topologie(pfad: str = PFAD) -> nx.Graph:
    os.makedirs(os.path.dirname(pfad), exist_ok=True)
    if not os.path.exists(pfad):
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})  # ohne UA: 403
        with urllib.request.urlopen(req, timeout=60) as r, open(pfad, "wb") as f:
            f.write(r.read())
    with gzip.open(pfad, "rt") as f:
        zeilen = [z for z in f if not z.startswith("#")]
    return nx.parse_edgelist(zeilen, nodetype=int)


def backbone_teilgraph(G: nx.Graph, max_knoten: int = 200) -> nx.Graph:
    """Zusammenhaengender Teilgraph rund um die groessten Hubs.

    Warum ein Teilgraph? Ein realer Backbone hat ~10-100 PoPs; 10 670 AS stuendlich zu
    simulieren waere weder realistisch noch noetig. Wir behalten die echte Struktur
    (Hubs + deren Nachbarschaft), nur kleiner.
    """
    hubs = [n for n, _ in sorted(G.degree(), key=lambda t: -t[1])[:12]]
    ausgewaehlt = set(hubs)
    for h in hubs:
        ausgewaehlt.update(list(G.neighbors(h))[:30])
    S = G.subgraph(list(ausgewaehlt)[:max_knoten]).copy()
    return S.subgraph(max(nx.connected_components(S), key=len)).copy()


def normalisierte_nachbarschaft(G: nx.Graph) -> np.ndarray:
    """Zeilennormierte Adjazenz: (A_norm @ y)[i] = Mittelwert von y ueber die Nachbarn von i."""
    A = nx.to_numpy_array(G)
    return A / np.maximum(A.sum(axis=1, keepdims=True), 1.0)


def simuliere_verkehr(G: nx.Graph, wochen: int = 4, decay: float = 0.2, spread: float = 0.75,
                      n_ereignisse: int = 300, seed: int = 0):
    """Erzeugt eine Verkehrs-Zeitreihe Y der Form (T, n) — T Stunden x n Knoten.

    decay  : wie stark ein Ereignis am Knoten SELBST nachwirkt   (0 = gar nicht)
    spread : wie stark es auf die NACHBARN uebergeht             (hoch = starke Propagation)

    Faustregel: je kleiner `decay` und je groesser `spread`, desto mehr Information steckt in
    der Topologie (und desto weniger in der eigenen Vergangenheit des Knotens).

    Rueckgabe: (Y, E) mit E = reiner Ereignisanteil (fuer Diagnose).
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    T = 24 * 7 * wochen
    A_norm = normalisierte_nachbarschaft(G)

    grad = np.array([d for _, d in G.degree()], dtype=float)
    grundlast = 50 + 20 * np.log1p(grad)                 # Gravity-Idee
    stunde = np.arange(T) % 24
    tag = (np.arange(T) // 24) % 7
    wochengang = np.where(tag >= 5, 0.7, 1.0)            # Wochenende schwaecher
    phase = rng.uniform(0, 2 * np.pi, n)                 # jeder Knoten leicht verschoben

    # --- propagierende Ereignisse ---
    ausbruch = np.zeros((T, n))
    for _ in range(n_ereignisse):
        ausbruch[rng.integers(0, T - 6), rng.integers(0, n)] += rng.uniform(300, 800)
    E = np.zeros((T, n))
    for k in range(1, T):
        E[k] = decay * E[k - 1] + spread * (A_norm @ E[k - 1]) + ausbruch[k]

    # --- Gesamtverkehr ---
    Y = np.zeros((T, n))
    rausch = np.zeros(n)
    for k in range(T):
        rausch = 0.8 * rausch + rng.normal(0, 3, n)      # AR(1)
        saison = 1 + 0.6 * np.sin(2 * np.pi * stunde[k] / 24 - np.pi / 2 + 0.3 * phase)
        Y[k] = np.maximum(grundlast * saison * wochengang[k] + rausch + E[k], 1.0)
    return Y, E


def kennzahlen(G: nx.Graph, Y: np.ndarray, E: np.ndarray) -> str:
    kn = list(G.nodes())
    idx = {k: i for i, k in enumerate(kn)}
    korr = np.mean([np.corrcoef(Y[:, idx[u]], Y[:, idx[v]])[0, 1]
                    for u, v in list(G.edges())[:200]])
    return (f"Topologie: {G.number_of_nodes()} Knoten, {G.number_of_edges()} Kanten (echt)\n"
            f"Verkehr  : {Y.shape[0]} Stunden x {Y.shape[1]} Knoten (simuliert)\n"
            f"mittlere Korrelation benachbarter Knoten: {korr:.3f}\n"
            f"Anteil propagierender Ereignisse am Verkehr: {100*E.sum()/Y.sum():.1f} %")
