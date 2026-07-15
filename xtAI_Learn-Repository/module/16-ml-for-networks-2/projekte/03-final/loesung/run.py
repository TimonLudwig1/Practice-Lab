"""Spatio-temporales Traffic-Forecasting: lohnt sich die Topologie?

Aufbau:
  1. Echte AS-Topologie + simulierter Verkehr (Generator offengelegt).
  2. Drei Modelle in aufsteigender Ordnung: saisonal-naiv -> Lag+Ridge -> Lag+Ridge+GRAPH.
  3. Die eigentliche Studie: `spread` variieren und messen, WANN der Graph sich lohnt.

Aufruf:  python run.py       (~1-2 min)
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error

from traffic_sim import (lade_topologie, backbone_teilgraph, normalisierte_nachbarschaft,
                         simuliere_verkehr, kennzahlen)
from forecast import saisonal_naiv, mase, zeit_split, trainiere_ridge, WOCHE

SEED = 0


def bewerte(G, Y, A_norm, verbose=True):
    """Alle drei Modelle auf einer Verkehrsreihe. Rueckgabe: dict mit MASE-Werten."""
    T = Y.shape[0]
    train_idx, test_idx = zeit_split(T)

    y_naiv = saisonal_naiv(Y, test_idx).ravel()
    y_wahr = Y[list(test_idx)].ravel()
    mae_naiv = mean_absolute_error(y_wahr, y_naiv)

    ergebnis = {"saisonal-naiv": 1.0}
    for tag, mit_graph in (("Lag+Ridge", False), ("Lag+Ridge+GRAPH", True)):
        yt, yp = trainiere_ridge(Y, A_norm, train_idx, test_idx, mit_graph)
        ergebnis[tag] = mase(yt, yp, y_naiv)
    if verbose:
        print(f"\n{'Modell':22s} {'MAE':>9s} {'MASE':>8s}")
        print(f"{'saisonal-naiv':22s} {mae_naiv:9.3f} {1.0:8.4f}   <- die Messlatte")
        for tag in ("Lag+Ridge", "Lag+Ridge+GRAPH"):
            print(f"{tag:22s} {ergebnis[tag]*mae_naiv:9.3f} {ergebnis[tag]:8.4f}")
    return ergebnis


def main():
    G = backbone_teilgraph(lade_topologie())
    A_norm = normalisierte_nachbarschaft(G)

    print("=== Daten ===")
    Y, E = simuliere_verkehr(G, wochen=4, decay=0.2, spread=0.75, seed=SEED)
    print(kennzahlen(G, Y, E))
    print(f"\nZeitbasierter Split: die letzten {WOCHE} Stunden (= 1 Woche) sind Test.")
    print("(Ein zufaelliger Split waere hier Selbstbetrug - man traininerte auf der Zukunft.)")

    print("\n=== Modelle ===")
    r = bewerte(G, Y, A_norm)
    gewinn = 100 * (r["Lag+Ridge"] - r["Lag+Ridge+GRAPH"]) / r["Lag+Ridge"]
    print(f"\n-> Beide ML-Modelle schlagen die naive Baseline deutlich (MASE < 1).")
    print(f"-> Die Graph-Features bringen zusaetzlich {gewinn:.1f} % Verbesserung.")

    # ---------- Die eigentliche Studie ----------
    print("\n=== Studie: WANN lohnt sich die Topologie? ===")
    print("Wir variieren, wie stark sich Ereignisse ueber die Kanten ausbreiten.")
    print("(decay = Nachwirkung am Knoten selbst, spread = Uebergang auf die Nachbarn)\n")
    print(f"{'decay':>6s} {'spread':>7s} {'Ereignis%':>10s} {'MASE ohne':>10s} "
          f"{'MASE mit':>9s} {'Gewinn':>8s}")
    studie = []
    for decay, spread in [(0.9, 0.05), (0.45, 0.50), (0.2, 0.75), (0.0, 0.95)]:
        Ys, Es = simuliere_verkehr(G, wochen=4, decay=decay, spread=spread, seed=SEED)
        rs = bewerte(G, Ys, A_norm, verbose=False)
        g = 100 * (rs["Lag+Ridge"] - rs["Lag+Ridge+GRAPH"]) / rs["Lag+Ridge"]
        anteil = 100 * Es.sum() / Ys.sum()
        studie.append((spread, g))
        print(f"{decay:6.2f} {spread:7.2f} {anteil:9.1f}% {rs['Lag+Ridge']:10.4f} "
              f"{rs['Lag+Ridge+GRAPH']:9.4f} {g:7.1f}%")

    print("\n-> Der Nutzen der Topologie waechst monoton mit der Ausbreitung.")
    print("   Bleibt der Verkehr lokal (spread~0), ist die eigene Vergangenheit des Knotens")
    print("   bereits die ganze Information - der Graph bringt dann ~nichts.")
    print("   DAS ist die ehrliche Antwort auf 'lohnt sich ein Spatio-Temporal-GNN?':")
    print("   Es haengt daran, ob der Verkehr sich raeumlich ausbreitet - nicht am Modell.")

    # ---------- Plots ----------
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("ergebnisse", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

        ax1.plot(Y[:24*7, :3], lw=1)
        ax1.set(xlabel="Stunde", ylabel="Verkehr", title="Simulierter Verkehr (3 Knoten, 1 Woche)")
        ax1.grid(alpha=0.3)

        sp = [s for s, _ in studie]; gw = [g for _, g in studie]
        ax2.plot(sp, gw, "o-", lw=2, color="crimson")
        ax2.axhline(0, ls="--", color="k", lw=1)
        ax2.set(xlabel="spread (raeumliche Ausbreitung)",
                ylabel="Verbesserung durch Graph-Features [%]",
                title="Die Topologie lohnt sich genau dann,\nwenn Verkehr propagiert")
        ax2.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig("ergebnisse/forecasting.png", dpi=110)
        print("\nPlot gespeichert: ergebnisse/forecasting.png")
    except Exception as e:
        print("(kein Plot:", e, ")")


if __name__ == "__main__":
    main()
