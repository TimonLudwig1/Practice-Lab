"""Ein Graph Convolutional Network (Kipf & Welling 2017) — from scratch, ohne torch_geometric.

Der ganze Zauber sind zwei Zeilen Mathematik:

    A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}          # einmal vorab berechnen
    H^(k) = sigma( A_hat @ H^(k-1) @ W^(k) )               # eine Schicht

Warum genau so (Skript 2.3):
  * (A + I)   Self-Loops - sonst faellt der Knoten aus seiner EIGENEN Aktualisierung heraus.
  * D^{-1/2} ... D^{-1/2}  symmetrische Normierung. Ohne sie summiert man Nachbarn: ein Hub mit
    Grad 2312 bekaeme Aktivierungen ~1000x groesser als ein Blatt mit Grad 2. Symmetrisch
    (statt zeilenweise D^{-1}A), weil das (a) A_hat symmetrisch laesst, (b) die Botschaft eines
    Hubs daempft (Hubs sind uninformativ - dieselbe Intuition wie Adamic-Adar/IDF) und (c) die
    Eigenwerte in [-1, 1] haelt => stabile Tiefe.

SPARSE ist hier nicht optional: eine dichte 10670x10670-Matrix waere ~455 MB, aber 99.96 %
davon sind Null. torch.sparse.mm loest das.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def normalisierte_adjazenz(kanten, n: int) -> torch.Tensor:
    """Berechnet A_hat = D^{-1/2}(A+I)D^{-1/2} als sparse COO-Tensor.

    kanten: Liste von (u, v)-Indexpaaren (ungerichtet, jede Kante EINMAL).
    """
    zeilen = [u for u, _ in kanten] + [v for _, v in kanten] + list(range(n))
    spalten = [v for _, v in kanten] + [u for u, _ in kanten] + list(range(n))
    z = torch.tensor(zeilen, dtype=torch.long)
    s = torch.tensor(spalten, dtype=torch.long)
    werte = torch.ones(len(zeilen))

    grad = torch.zeros(n).scatter_add_(0, z, werte)          # Grad inkl. Self-Loop
    d_inv_sqrt = grad.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0                 # isolierte Knoten abfangen

    norm_werte = d_inv_sqrt[z] * d_inv_sqrt[s]                # 1/sqrt(d_u) * 1/sqrt(d_v)
    A = torch.sparse_coo_tensor(torch.stack([z, s]), norm_werte, (n, n))
    return A.coalesce()


class GCN(nn.Module):
    """2-schichtiges GCN. Erzeugt fuer jeden Knoten ein Embedding.

    Der Graph hat keine Knoten-Features (eine reine Topologie!). Deshalb lernen wir die
    Eingangs-Repraesentation selbst: eine Embedding-Tabelle als H^(0). Damit ist das Modell
    - wie GCN generell - **transduktiv** (Skript 2.3): fuer einen neuen Knoten gaebe es keine
    Zeile in der Tabelle. Genau dieses Problem loest GraphSAGE.

    Warum nur 2 Schichten? Over-Smoothing (Skript 2.4) - und im Small-World-Graphen
    (mittlerer Pfad ~3.7) sieht man mit 2 Hops ohnehin schon einen grossen Teil des Netzes.
    """

    def __init__(self, n_knoten: int, dim: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(n_knoten, dim)        # H^(0), gelernt
        nn.init.normal_(self.emb.weight, std=0.1)
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)

    def forward(self, A_hat: torch.Tensor) -> torch.Tensor:
        H = torch.sparse.mm(A_hat, self.emb.weight)   # Schicht 1: 1 Hop
        H = torch.relu(self.W1(H))
        H = torch.sparse.mm(A_hat, H)                 # Schicht 2: 2 Hops
        return self.W2(H)


class MLPOhneStruktur(nn.Module):
    """Kontrollgruppe: identisch zum GCN, aber OHNE Message Passing (kein A_hat).

    Wenn dieses Modell genauso gut ist, bringt die Topologie nichts - und das ganze GNN
    waere ueberfluessiger Aufwand. Genau dafuer ist eine Kontrollgruppe da.
    """

    def __init__(self, n_knoten: int, dim: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(n_knoten, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)

    def forward(self, A_hat=None):
        H = torch.relu(self.W1(self.emb.weight))
        return self.W2(H)


def kanten_score(Z: torch.Tensor, paare: torch.Tensor) -> torch.Tensor:
    """Score einer Kante = Skalarprodukt der beiden Knoten-Embeddings.

    Hoher Score <=> die beiden Knoten 'passen zusammen' <=> Kante wahrscheinlich.
    """
    return (Z[paare[:, 0]] * Z[paare[:, 1]]).sum(dim=1)
