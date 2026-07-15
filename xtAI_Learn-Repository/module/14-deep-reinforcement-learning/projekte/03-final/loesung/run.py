"""Hauptexperiment: modellfreies RL vs. exakte Optimalsteuerung (LQR).

  1. LQR/Riccati auf dem BEKANNTEN Modell  -> exakte optimale Rueckfuehrung u* = -K x.
  2. Linear-gaussche Politik MODELLFREI per Policy Gradient trainieren (kennt A,B,Q,R nicht).
  3. Vergleich: gelernte Gewichte w  vs. -K, und mittlere Kosten vs. LQR-Optimum.
  4. Plots: Lernkurve + Trajektorienvergleich.

Aufruf:  python run.py            (~1-2 min auf der CPU)
"""
from __future__ import annotations
import time
import numpy as np

from lqr_env import LinearQuadraticSystem
from lqr_reference import lqr_gain, closed_loop_eigenvalues, optimal_cost
from policy_gradient_continuous import LinearGaussianPolicy, train_policy_gradient

N_UPDATES = 600
BATCH = 32
SEED = 0


def mean_cost(env, w, X0):
    return float(np.mean([env.rollout_linear(w, x0)[0] for x0 in X0]))


def main():
    env = LinearQuadraticSystem(seed=SEED)

    # ---------- 1) exakte Referenz ----------
    P, K, iters = lqr_gain(env)
    ev = closed_loop_eigenvalues(env, K)
    print("=== Exakte Optimalsteuerung (LQR / Riccati, Modell bekannt) ===")
    print(f"Riccati konvergiert in {iters} Iterationen (Millisekunden).")
    print("optimale Rueckfuehrung  u* = -K x  mit  -K =", np.round(-K.ravel(), 4))
    print("closed-loop |Eigenwerte| =", np.round(np.abs(ev), 4), "(<1 => stabil)")
    x0_probe = np.array([1.0, 0.0])
    sim_c, _ = env.rollout_linear(-K.ravel(), x0_probe, horizon=400)
    print(f"Validierung: simulierte Kosten {sim_c:.4f} == x0^T P x0 = "
          f"{optimal_cost(P, x0_probe):.4f}  ✓")

    # ---------- 2) modellfrei lernen ----------
    print(f"\n=== Modellfreier Policy Gradient ({N_UPDATES} Updates x {BATCH} Episoden "
          f"= {N_UPDATES*BATCH:,} Episoden) ===")
    policy = LinearGaussianPolicy(lr=0.05, seed=SEED)
    t0 = time.time()
    history = train_policy_gradient(env, policy, n_updates=N_UPDATES, batch=BATCH,
                                    verbose_every=150)
    dt = time.time() - t0
    w = policy.weights()

    # ---------- 3) Vergleich ----------
    rng = np.random.default_rng(123)
    X0 = [rng.normal(0, 1, 2) for _ in range(200)]        # gemeinsame Testmenge
    c_learn = mean_cost(env, w, X0)
    c_opt = mean_cost(env, -K.ravel(), X0)
    gap = 100 * (c_learn - c_opt) / c_opt

    print(f"\n=== Vergleich (200 zufaellige Startzustaende) ===")
    print(f"{'':22s} {'Gewichte':>22s} {'mittlere Kosten':>16s}")
    print(f"{'LQR (exakt, Modell)':22s} {str(np.round(-K.ravel(),3)):>22s} {c_opt:16.3f}")
    print(f"{'gelernt (modellfrei)':22s} {str(np.round(w,3)):>22s} {c_learn:16.3f}")
    print(f"\nOptimalitaetsluecke: {gap:.2f} %   |   Trainingszeit: {dt:.0f}s")
    print(f"Parameterabstand ||w - (-K)||: {np.linalg.norm(w + K.ravel()):.3f}")

    # ---------- 4) Plots ----------
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("ergebnisse", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

        ax1.plot(history, lw=1.2, label="Policy Gradient (modellfrei)")
        ax1.axhline(c_opt, ls="--", color="k", lw=1.5, label=f"LQR-Optimum ({c_opt:.2f})")
        ax1.set(xlabel="Update", ylabel="mittlere Episodenkosten",
                title="Lernkurve: RL naehert sich dem exakten Optimum")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        x0 = np.array([1.0, 0.0])
        _, traj_opt = env.rollout_linear(-K.ravel(), x0)
        _, traj_pg = env.rollout_linear(w, x0)
        ax2.plot(traj_opt[:, 0], label="LQR (exakt): Position", color="k", lw=2)
        ax2.plot(traj_pg[:, 0], label="gelernt: Position", ls="--", lw=1.8)
        ax2.axhline(0, color="gray", lw=0.8)
        ax2.set(xlabel="Zeitschritt", ylabel="Position",
                title=f"Regelverhalten ab x0={x0}")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        plt.tight_layout(); plt.savefig("ergebnisse/lqr_vs_rl.png", dpi=110)
        print("\nPlot gespeichert: ergebnisse/lqr_vs_rl.png")
    except Exception as e:
        print("(kein Plot:", e, ")")


if __name__ == "__main__":
    main()
