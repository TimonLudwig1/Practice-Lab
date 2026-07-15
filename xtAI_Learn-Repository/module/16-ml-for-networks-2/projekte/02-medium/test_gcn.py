"""Tests fuer die normalisierte Adjazenz, das GCN und den Link-Split.

Aufruf:  python test_gcn.py      (~40 s: laedt den Graphen und trainiert kurz)
"""
import numpy as np
import torch

from gcn import GCN, MLPOhneStruktur, normalisierte_adjazenz, kanten_score
from graph_data import LinkSplit

# Mini-Graph zum Rechnen von Hand:  0 - 1 - 2   (Pfad)
MINI = [(0, 1), (1, 2)]
N_MINI = 3


# ---------------------- normalisierte Adjazenz ----------------------
def test_a_hat_ist_symmetrisch():
    A = normalisierte_adjazenz(MINI, N_MINI).to_dense()
    assert torch.allclose(A, A.T), A


def test_a_hat_hat_self_loops():
    # Ohne (A+I) waere die Diagonale 0 -> der Knoten vergaesse sich selbst.
    A = normalisierte_adjazenz(MINI, N_MINI).to_dense()
    assert torch.all(torch.diag(A) > 0), torch.diag(A)


def test_a_hat_werte_von_hand_nachgerechnet():
    # Pfad 0-1-2, mit Self-Loops: d0=2, d1=3, d2=2
    # A_hat[u,v] = 1/(sqrt(d_u)*sqrt(d_v))
    A = normalisierte_adjazenz(MINI, N_MINI).to_dense()
    d = np.array([2.0, 3.0, 2.0])
    assert np.isclose(A[0, 0].item(), 1/np.sqrt(d[0]*d[0]))     # Self-Loop 0
    assert np.isclose(A[0, 1].item(), 1/np.sqrt(d[0]*d[1]))     # Kante 0-1
    assert np.isclose(A[1, 1].item(), 1/np.sqrt(d[1]*d[1]))     # Self-Loop 1
    assert A[0, 2].item() == 0.0                                # keine Kante 0-2


def test_a_hat_eigenwerte_in_minus_eins_bis_eins():
    # Das ist der ganze Grund fuer die symmetrische Normierung: stabile Tiefe.
    A = normalisierte_adjazenz(MINI, N_MINI).to_dense()
    ew = torch.linalg.eigvalsh(A)
    assert torch.all(ew <= 1.0 + 1e-6) and torch.all(ew >= -1.0 - 1e-6), ew


def test_a_hat_ist_sparse_und_richtig_gross():
    A = normalisierte_adjazenz(MINI, N_MINI)
    assert A.is_sparse
    assert tuple(A.shape) == (N_MINI, N_MINI)
    # 2 Kanten * 2 Richtungen + 3 Self-Loops = 7
    assert A._nnz() == 7, A._nnz()


def test_a_hat_daempft_hubs():
    # Stern: Knoten 0 ist Hub mit 5 Blaettern. Die Botschaft des Hubs an ein Blatt
    # muss schwaecher sein als die eines Blattes an sich selbst.
    stern = [(0, i) for i in range(1, 6)]
    A = normalisierte_adjazenz(stern, 6).to_dense()
    assert A[1, 0] < A[1, 1], "Hub-Botschaft muesste gedaempft sein"


def test_isolierter_knoten_erzeugt_kein_nan():
    # Knoten 3 ohne Kanten (nur Self-Loop) darf nicht zu Division durch 0 fuehren
    A = normalisierte_adjazenz(MINI, 4).to_dense()
    assert torch.all(torch.isfinite(A))


# ---------------------- GCN ----------------------
def test_gcn_ausgabeform():
    A = normalisierte_adjazenz(MINI, N_MINI)
    Z = GCN(N_MINI, dim=8)(A)
    assert Z.shape == (N_MINI, 8)
    assert torch.all(torch.isfinite(Z))


def test_kanten_score_ist_symmetrisch():
    Z = torch.randn(5, 4)
    s1 = kanten_score(Z, torch.tensor([[0, 1]]))
    s2 = kanten_score(Z, torch.tensor([[1, 0]]))
    assert torch.allclose(s1, s2)     # Skalarprodukt ist symmetrisch => ungerichtet


def test_message_passing_mischt_nachbarn():
    # Kern des GNN: nach einer A_hat-Multiplikation muss sich der Zustand eines Knotens
    # geaendert haben, WEIL seine Nachbarn eingeflossen sind.
    A = normalisierte_adjazenz(MINI, N_MINI).to_dense()
    H = torch.eye(N_MINI)                 # jeder Knoten "kennt" nur sich selbst
    H1 = A @ H
    assert H1[0, 1] > 0, "Knoten 0 muesste nach 1 Hop etwas von Knoten 1 wissen"
    assert H1[0, 2] == 0, "Knoten 2 ist 2 Hops weg - nach 1 Hop noch unsichtbar"
    H2 = A @ H1
    assert H2[0, 2] > 0, "nach 2 Hops muesste Knoten 2 sichtbar sein"


# ---------------------- Split / Leakage ----------------------
_SPLIT = LinkSplit(test_anteil=0.10, seed=0)


def test_testkanten_sind_aus_dem_trainingsgraphen_entfernt():
    # DIE Leakage-Pruefung (Skript 2.5, Falle 1)
    for u, v in _SPLIT.test_pos[:200]:
        assert not _SPLIT.G_train.has_edge(_SPLIT.knoten[u], _SPLIT.knoten[v])
        assert _SPLIT.G.has_edge(_SPLIT.knoten[u], _SPLIT.knoten[v])   # im Original schon


def test_split_ist_vollstaendig():
    assert len(_SPLIT.train_pos) + len(_SPLIT.test_pos) == _SPLIT.G.number_of_edges()


def test_negative_sind_keine_echten_kanten():
    for u, v in _SPLIT.negative_uniform(300, seed=1):
        assert not _SPLIT.G.has_edge(_SPLIT.knoten[u], _SPLIT.knoten[v])
        assert u != v


def test_grad_confound_existiert_wirklich():
    # Der Befund aus Projekt 01: uniform gezogene Paare sind fast immer Blatt x Blatt.
    uni = _SPLIT.negative_uniform(500, seed=2)
    gem = _SPLIT.negative_grad_gematcht(_SPLIT.test_pos[:500], seed=2)
    med_pos = np.median(_SPLIT.gradprodukt(_SPLIT.test_pos))
    med_uni = np.median(_SPLIT.gradprodukt(uni))
    med_gem = np.median(_SPLIT.gradprodukt(gem))
    assert med_uni < med_pos / 20, (med_pos, med_uni)      # >20x Unterschied
    assert med_gem > med_pos / 5, (med_pos, med_gem)       # gematcht: vergleichbar


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Starte {len(tests)} Tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("Alle Tests bestanden.")
