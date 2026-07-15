"""Ein Graph Convolutional Network (Kipf & Welling 2017) — from scratch, ohne torch_geometric.

>>> DEINE AUFGABE <<<  Vier TODOs. Der ganze Zauber sind zwei Zeilen Mathematik:

    A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}          # einmal vorab berechnen
    H^(k) = sigma( A_hat @ H^(k-1) @ W^(k) )               # eine Schicht

Warum genau so (Skript 2.3 — verstehe das, bevor du tippst):
  * (A + I)   Self-Loops - sonst faellt der Knoten aus seiner EIGENEN Aktualisierung heraus.
  * D^{-1/2} ... D^{-1/2}  symmetrische Normierung. Ohne sie summiert man Nachbarn: ein Hub mit
    Grad 2312 bekaeme Aktivierungen ~1000x groesser als ein Blatt mit Grad 2. Symmetrisch
    (statt zeilenweise D^{-1}A), weil das (a) A_hat symmetrisch laesst, (b) die Botschaft eines
    Hubs daempft (Hubs sind uninformativ - dieselbe Intuition wie Adamic-Adar/IDF) und (c) die
    Eigenwerte in [-1, 1] haelt => stabile Tiefe.

SPARSE ist hier nicht optional: eine dichte 10670x10670-Matrix waere ~455 MB, aber 99.96 %
davon sind Null. Nutze torch.sparse_coo_tensor / torch.sparse.mm.

Musterloesung: loesung/gcn.py — erst selbst versuchen!
"""
from __future__ import annotations
import torch
import torch.nn as nn


def normalisierte_adjazenz(kanten, n: int) -> torch.Tensor:
    """Berechne A_hat = D^{-1/2}(A+I)D^{-1/2} als SPARSE COO-Tensor.

    kanten: Liste von (u, v)-Indexpaaren (ungerichtet, jede Kante EINMAL aufgefuehrt).

    Bauplan:
      1. Index-Listen aufbauen. Weil der Graph ungerichtet ist, braucht jede Kante BEIDE
         Richtungen: zeilen = [u...] + [v...] + [0..n-1]   (das letzte Stueck = Self-Loops!)
         spalten = [v...] + [u...] + [0..n-1]
      2. grad = torch.zeros(n).scatter_add_(0, zeilen, werte)   # Grad inkl. Self-Loop
      3. d_inv_sqrt = grad.pow(-0.5);  Inf abfangen (isolierte Knoten!) -> 0
      4. norm_werte[i] = d_inv_sqrt[zeilen[i]] * d_inv_sqrt[spalten[i]]
      5. torch.sparse_coo_tensor(torch.stack([zeilen, spalten]), norm_werte, (n, n)).coalesce()

    Selbstcheck (die Tests pruefen genau das):
      - A_hat ist symmetrisch
      - die Diagonale ist > 0 (Self-Loops!)
      - fuer den Pfad 0-1-2 gilt A_hat[0,1] = 1/sqrt(2*3)
      - alle Eigenwerte liegen in [-1, 1]
    """
    # TODO
    raise NotImplementedError


class GCN(nn.Module):
    """2-schichtiges GCN. Erzeugt fuer jeden Knoten ein Embedding.

    Der Graph hat KEINE Knoten-Features (reine Topologie). Deshalb lernen wir die
    Eingangs-Repraesentation selbst: eine nn.Embedding-Tabelle als H^(0). Damit ist das Modell
    - wie GCN generell - **transduktiv** (Skript 2.3).

    Warum nur 2 Schichten? Over-Smoothing (Skript 2.4).
    """

    def __init__(self, n_knoten: int, dim: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(n_knoten, dim)        # H^(0), gelernt
        nn.init.normal_(self.emb.weight, std=0.1)
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)

    def forward(self, A_hat: torch.Tensor) -> torch.Tensor:
        """Zwei GCN-Schichten.

        Schicht 1:  H = relu( W1( A_hat @ emb ) )     # 1 Hop
        Schicht 2:  Z = W2( A_hat @ H )               # 2 Hops
        Tipp: torch.sparse.mm(A_hat, X) fuer sparse @ dense.
        """
        # TODO
        raise NotImplementedError


class MLPOhneStruktur(nn.Module):
    """Kontrollgruppe: identisch zum GCN, aber OHNE Message Passing (A_hat wird ignoriert).

    Wenn dieses Modell genauso gut ist, bringt die Topologie nichts - und das ganze GNN waere
    ueberfluessiger Aufwand. Genau dafuer ist eine Kontrollgruppe da.
    """

    def __init__(self, n_knoten: int, dim: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(n_knoten, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)

    def forward(self, A_hat=None):
        """Wie GCN.forward, aber OHNE jede A_hat-Multiplikation."""
        # TODO
        raise NotImplementedError


def kanten_score(Z: torch.Tensor, paare: torch.Tensor) -> torch.Tensor:
    """Score einer Kante = Skalarprodukt der beiden Knoten-Embeddings.

    Z:     (n, dim) Knoten-Embeddings
    paare: (k, 2) Long-Tensor von Knotenpaaren
    Rueckgabe: (k,) Scores. Hoch <=> die beiden passen zusammen <=> Kante wahrscheinlich.

    Tipp: (Z[paare[:,0]] * Z[paare[:,1]]).sum(dim=1)   - zeilenweises Skalarprodukt.
    """
    # TODO
    raise NotImplementedError
