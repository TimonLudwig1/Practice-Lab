"""Das Experiment: Motion-to-Photon-Latenz und ihre Gegenmittel.

  1. Das Latenz-Budget aufschluesseln (warum 90 Hz eng ist).
  2. Wie stark haengt die Welt nach? (Fehler in Grad ueber Latenz)
  3. Prediction: der Sweet Spot (horizon = latency) und der Overshoot.
  4. Timewarp: warum es die Latenz "unsichtbar" macht.

Aufruf:  python run.py        (~2 s)
"""
from __future__ import annotations
import numpy as np

from head_motion import generate_head_yaw
from latency import (LatencyBudget, displayed_pose, predicted_pose, timewarped_pose,
                     angular_error)

FS = 1000


def main():
    t, truth, velocity = generate_head_yaw(duration_s=20.0, fs=FS, seed=0)
    print("=== Kopfbewegung (simuliert) ===")
    print(f"Dauer {t[-1]:.0f}s @ {FS} Hz | max Winkelgeschwindigkeit "
          f"{np.abs(velocity).max():.0f} deg/s (real bis ~500)")

    # ---------- 1) Latenz-Budget ----------
    print("\n=== 1) Motion-to-Photon-Budget (Skript 3.1) ===")
    budget = LatencyBudget(sensor=1.5, fusion=1.0, app=4.0, render=11.1, scanout=3.0, display=2.0)
    print(budget.report())
    print("-> Allein ein Frame bei 90 Hz = 11.1 ms frisst das halbe 20-ms-Budget.")

    # ---------- 2) reine Latenz ----------
    print("\n=== 2) Wie stark haengt die Welt nach? (ohne Gegenmittel) ===")
    print(f"{'Latenz':>8s} {'mittl. Fehler':>14s} {'max Fehler':>11s}")
    for L in (5, 11, 20, 50):
        e = angular_error(displayed_pose(truth, L, FS), truth)
        print(f"{L:6d}ms {e.mean():12.2f}deg {e.max():10.2f}deg")
    print("-> Bei 50 ms weicht das Bild im Mittel 5 Grad von der Wahrheit ab - deutlich spuerbar.")

    # ---------- 3) Prediction ----------
    print("\n=== 3) Prediction: extrapolieren statt hinterherhinken ===")
    L = 20
    e_ohne = angular_error(displayed_pose(truth, L, FS), truth).mean()
    print(f"Bei {L} ms Latenz, horizon variiert (ideal = Latenz):")
    print(f"{'horizon':>9s} {'mittl. Fehler':>14s} {'max Fehler':>11s} {'vs ohne':>9s}")
    for H in (0, 10, 20, 30, 40):
        e = angular_error(predicted_pose(truth, velocity, L, H, FS), truth)
        print(f"{H:7d}ms {e.mean():12.3f}deg {e.max():10.2f}deg {100*(1-e.mean()/e_ohne):+7.0f}%")
    print("-> U-Form: horizon = Latenz ist optimal (~-86%). ZU viel Vorhersage (40 ms) ist so")
    print("   schlecht wie gar keine - man extrapoliert an der Bewegung vorbei.")

    # Overshoot an Richtungswechseln
    beschl = np.abs(np.gradient(velocity, 1 / FS))
    wende = beschl > np.percentile(beschl, 90)
    e_pred = angular_error(predicted_pose(truth, velocity, L, L, FS), truth)
    print(f"\nOvershoot: Prediction-Fehler an Richtungswechseln {e_pred[wende].mean():.3f} deg "
          f"vs. bei glatter Bewegung {e_pred[~wende].mean():.3f} deg")
    print("-> Genau dort liegt Prediction daneben: sie extrapoliert in die ALTE Richtung weiter.")

    # ---------- 4) Timewarp ----------
    print("\n=== 4) Timewarp: das fertige Bild nachschieben ===")
    e_render = angular_error(displayed_pose(truth, 20, FS), truth).mean()
    e_warp = angular_error(timewarped_pose(truth, render_latency_ms=20, warp_latency_ms=2, fs=FS),
                           truth).mean()
    print(f"  Render-Latenz 20 ms, ohne Timewarp: {e_render:.2f} deg")
    print(f"  mit Timewarp (Warp holt Pose bei 2 ms): {e_warp:.2f} deg  "
          f"({100*(1-e_warp/e_render):.0f}% weniger)")
    print("-> Timewarp macht die Latenz nicht KLEINER, aber fuer die ORIENTIERUNG unsichtbar -")
    print("   und genau darauf reagiert das Vestibularsystem am empfindlichsten.")
    print("   (Bei TRANSLATION ginge das nicht: dann fehlt verdeckte Information -> Disokklusion.)")

    # ---------- Plots ----------
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.3))

        s = slice(2000, 4000)   # 2 s Ausschnitt
        ax1.plot(t[s], truth[s], "k", lw=2, label="wahr (jetzt)")
        ax1.plot(t[s], displayed_pose(truth, 50, FS)[s], lw=1.3, label="50 ms Latenz (haengt nach)")
        ax1.plot(t[s], predicted_pose(truth, velocity, 50, 50, FS)[s], lw=1.3,
                 label="+ Prediction")
        ax1.set(xlabel="Zeit [s]", ylabel="yaw [Grad]", title="Latenz laesst die Welt nachhaengen")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        hs = np.arange(0, 45, 2)
        errs = [angular_error(predicted_pose(truth, velocity, 20, h, FS), truth).mean() for h in hs]
        ax2.plot(hs, errs, "o-", lw=2)
        ax2.axvline(20, ls="--", color="crimson", label="horizon = Latenz")
        ax2.set(xlabel="Prediction-Horizont [ms]", ylabel="mittl. Fehler [Grad]",
                title="Prediction: U-Form um den Sweet Spot")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        Ls = np.arange(0, 55, 5)
        ax3.plot(Ls, [angular_error(displayed_pose(truth, L, FS), truth).mean() for L in Ls],
                 "o-", lw=2, label="ohne Gegenmittel")
        ax3.plot(Ls, [angular_error(predicted_pose(truth, velocity, L, L, FS), truth).mean()
                      for L in Ls], "s-", lw=2, label="mit Prediction")
        ax3.plot(Ls, [angular_error(timewarped_pose(truth, L, 2, FS), truth).mean()
                      for L in Ls], "^-", lw=2, label="mit Timewarp")
        ax3.set(xlabel="Render-Latenz [ms]", ylabel="mittl. Fehler [Grad]",
                title="Gegenmittel im Vergleich")
        ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

        plt.tight_layout(); plt.savefig("results/motion_to_photon.png", dpi=110)
        print("\nPlot gespeichert: results/motion_to_photon.png")
    except Exception as e:
        print("(kein Plot:", e, ")")


if __name__ == "__main__":
    main()
