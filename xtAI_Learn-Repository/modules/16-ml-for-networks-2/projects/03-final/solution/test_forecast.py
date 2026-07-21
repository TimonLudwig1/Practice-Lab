"""Tests fuer Verkehrssimulation, Features und Forecasting.

Aufruf:  python test_forecast.py      (~30 s)
"""
import numpy as np

from traffic_sim import (lade_topologie, backbone_teilgraph, normalisierte_nachbarschaft,
                         simuliere_verkehr)
from forecast import (saisonal_naiv, mase, zeit_split, baue_features, trainiere_ridge,
                      WOCHE, TAG)

G = backbone_teilgraph(lade_topologie())
A = normalisierte_nachbarschaft(G)
Y, E = simuliere_verkehr(G, wochen=4, decay=0.2, spread=0.75, seed=0)


# ---------------------- Topologie ----------------------
def test_topologie_ist_echt_und_zusammenhaengend():
    import networkx as nx
    assert nx.is_connected(G)
    assert 50 <= G.number_of_nodes() <= 200
    assert G.number_of_edges() > G.number_of_nodes()      # kein Baum


def test_nachbarschaftsmatrix_ist_zeilennormiert():
    zeilensummen = A.sum(axis=1)
    assert np.allclose(zeilensummen[zeilensummen > 0], 1.0)
    assert np.all(np.diag(A) == 0)                        # kein Self-Loop hier (anders als GCN!)


# ---------------------- Simulation ----------------------
def test_verkehr_form_und_positiv():
    assert Y.shape == (24 * 7 * 4, G.number_of_nodes())
    assert np.all(Y >= 1.0) and np.all(np.isfinite(Y))


def test_tagesgang_existiert():
    # Autokorrelation bei Lag 24 muss hoch sein (Periode 24 h)
    y = Y[:, 0]
    r24 = np.corrcoef(y[24:], y[:-24])[0, 1]
    r12 = np.corrcoef(y[12:], y[:-12])[0, 1]
    assert r24 > 0.5, r24
    assert r24 > r12, (r24, r12)      # 24h aehnlicher als 12h (Gegenphase)


def test_wochenende_hat_weniger_verkehr():
    t = np.arange(Y.shape[0]); tag = (t // 24) % 7
    werktag = Y[tag < 5].mean()
    wochenende = Y[tag >= 5].mean()
    assert wochenende < werktag, (wochenende, werktag)


def test_grosse_as_tragen_mehr_verkehr():
    grad = np.array([d for _, d in G.degree()], float)
    mittel = Y.mean(axis=0)
    assert np.corrcoef(grad, mittel)[0, 1] > 0.3          # Gravity-Idee


def test_ereignisse_propagieren_nur_bei_spread():
    # spread=0: ein Ausbruch bleibt lokal -> Nachbarn sehen nichts davon
    _, E_lokal = simuliere_verkehr(G, wochen=2, decay=0.9, spread=0.0, seed=1)
    _, E_breit = simuliere_verkehr(G, wochen=2, decay=0.0, spread=0.95, seed=1)
    # Anteil der Knoten, die ueberhaupt je ein Ereignis sehen
    betroffen_lokal = (E_lokal.max(axis=0) > 1).mean()
    betroffen_breit = (E_breit.max(axis=0) > 1).mean()
    assert betroffen_breit > betroffen_lokal, (betroffen_breit, betroffen_lokal)


def test_simulation_ist_reproduzierbar():
    Y1, _ = simuliere_verkehr(G, wochen=1, seed=42)
    Y2, _ = simuliere_verkehr(G, wochen=1, seed=42)
    assert np.array_equal(Y1, Y2)


# ---------------------- Split / Features ----------------------
def test_zeit_split_hat_keine_zukunft_im_training():
    # DIE Leakage-Pruefung: jeder Trainingszeitpunkt liegt VOR jedem Testzeitpunkt.
    train_idx, test_idx = zeit_split(Y.shape[0])
    assert max(train_idx) < min(test_idx)
    assert min(train_idx) > WOCHE                          # genug Historie fuer alle Lags


def test_kalender_ist_zyklisch_kodiert():
    # 23 Uhr und 0 Uhr muessen NAH beieinander liegen (roh waeren sie maximal weit entfernt)
    def kodiere(h):
        return np.array([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)])
    d_23_0 = np.linalg.norm(kodiere(23) - kodiere(0))
    d_23_11 = np.linalg.norm(kodiere(23) - kodiere(11))
    assert d_23_0 < d_23_11, (d_23_0, d_23_11)


def test_graph_features_fuegen_spalten_hinzu():
    idx = range(WOCHE + 1, WOCHE + 5)
    X_ohne, y1 = baue_features(Y, A, idx, mit_graph=False)
    X_mit, y2 = baue_features(Y, A, idx, mit_graph=True)
    assert X_mit.shape[1] == X_ohne.shape[1] + 3          # nachbar t-1, t-2, 2-hop
    assert np.array_equal(y1, y2)                          # Ziel bleibt gleich
    assert X_ohne.shape[0] == len(idx) * G.number_of_nodes()


# ---------------------- Metriken / Modelle ----------------------
def test_mase_der_baseline_gegen_sich_selbst_ist_eins():
    _, test_idx = zeit_split(Y.shape[0])
    yn = saisonal_naiv(Y, test_idx).ravel()
    yw = Y[list(test_idx)].ravel()
    assert np.isclose(mase(yw, yn, yn), 1.0)


def test_saisonal_naiv_greift_eine_woche_zurueck():
    _, test_idx = zeit_split(Y.shape[0])
    yn = saisonal_naiv(Y, test_idx)
    erwartet = Y[[k - WOCHE for k in test_idx]]
    assert np.array_equal(yn, erwartet)


def test_ridge_schlaegt_die_naive_baseline():
    train_idx, test_idx = zeit_split(Y.shape[0])
    yn = saisonal_naiv(Y, test_idx).ravel()
    yt, yp = trainiere_ridge(Y, A, train_idx, test_idx, mit_graph=False)
    assert mase(yt, yp, yn) < 1.0                          # sonst waere das Modell wertlos


def test_graph_hilft_wenn_verkehr_propagiert():
    # Kernaussage des Projekts: bei starker Ausbreitung muss der Graph messbar helfen.
    Ys, _ = simuliere_verkehr(G, wochen=4, decay=0.0, spread=0.95, seed=0)
    tr, te = zeit_split(Ys.shape[0])
    yn = saisonal_naiv(Ys, te).ravel()
    m_ohne = mase(*trainiere_ridge(Ys, A, tr, te, False), yn)
    m_mit = mase(*trainiere_ridge(Ys, A, tr, te, True), yn)
    assert m_mit < m_ohne, (m_mit, m_ohne)
    assert (m_ohne - m_mit) / m_ohne > 0.05                # mindestens 5 % besser


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Starte {len(tests)} Tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("Alle Tests bestanden.")
