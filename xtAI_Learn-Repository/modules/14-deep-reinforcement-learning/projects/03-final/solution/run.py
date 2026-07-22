"""Main experiment: model-free RL vs. exact optimal control (LQR).

  1. LQR/Riccati on the KNOWN model  -> the exact optimal feedback u* = -K x.
  2. Train a linear-Gaussian policy MODEL-FREE by policy gradient (does not know A,B,Q,R).
  3. Comparison: the learned weights w vs. -K, and the mean cost vs. the LQR optimum.
  4. Plots: learning curve + trajectory comparison.

Call:  python run.py            (~1-2 min on the CPU)
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

    # ---------- 1) exact reference ----------
    P, K, iters = lqr_gain(env)
    ev = closed_loop_eigenvalues(env, K)
    print("=== Exact optimal control (LQR / Riccati, model known) ===")
    print(f"Riccati converges in {iters} iterations (milliseconds).")
    print("optimal feedback  u* = -K x  with  -K =", np.round(-K.ravel(), 4))
    print("closed-loop |eigenvalues| =", np.round(np.abs(ev), 4), "(<1 => stable)")
    x0_probe = np.array([1.0, 0.0])
    sim_c, _ = env.rollout_linear(-K.ravel(), x0_probe, horizon=400)
    print(f"Validation: simulated cost {sim_c:.4f} == x0^T P x0 = "
          f"{optimal_cost(P, x0_probe):.4f}  OK")

    # ---------- 2) learn model-free ----------
    print(f"\n=== Model-free policy gradient ({N_UPDATES} updates x {BATCH} episodes "
          f"= {N_UPDATES*BATCH:,} episodes) ===")
    policy = LinearGaussianPolicy(lr=0.05, seed=SEED)
    t0 = time.time()
    history = train_policy_gradient(env, policy, n_updates=N_UPDATES, batch=BATCH,
                                    verbose_every=150)
    dt = time.time() - t0
    w = policy.weights()

    # ---------- 3) comparison ----------
    rng = np.random.default_rng(123)
    X0 = [rng.normal(0, 1, 2) for _ in range(200)]        # a shared test set
    c_learn = mean_cost(env, w, X0)
    c_opt = mean_cost(env, -K.ravel(), X0)
    gap = 100 * (c_learn - c_opt) / c_opt

    print(f"\n=== Comparison (200 random start states) ===")
    print(f"{'':22s} {'weights':>22s} {'mean cost':>16s}")
    print(f"{'LQR (exact, model)':22s} {str(np.round(-K.ravel(),3)):>22s} {c_opt:16.3f}")
    print(f"{'learned (model-free)':22s} {str(np.round(w,3)):>22s} {c_learn:16.3f}")
    print(f"\nOptimality gap: {gap:.2f} %   |   training time: {dt:.0f}s")
    print(f"Parameter distance ||w - (-K)||: {np.linalg.norm(w + K.ravel()):.3f}")

    # ---------- 4) plots ----------
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

        ax1.plot(history, lw=1.2, label="policy gradient (model-free)")
        ax1.axhline(c_opt, ls="--", color="k", lw=1.5, label=f"LQR optimum ({c_opt:.2f})")
        ax1.set(xlabel="update", ylabel="mean episode cost",
                title="Learning curve: RL approaches the exact optimum")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        x0 = np.array([1.0, 0.0])
        _, traj_opt = env.rollout_linear(-K.ravel(), x0)
        _, traj_pg = env.rollout_linear(w, x0)
        ax2.plot(traj_opt[:, 0], label="LQR (exact): position", color="k", lw=2)
        ax2.plot(traj_pg[:, 0], label="learned: position", ls="--", lw=1.8)
        ax2.axhline(0, color="gray", lw=0.8)
        ax2.set(xlabel="time step", ylabel="position",
                title=f"Control behavior from x0={x0}")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        plt.tight_layout(); plt.savefig("results/lqr_vs_rl.png", dpi=110)
        print("\nPlot saved: results/lqr_vs_rl.png")
    except Exception as e:
        print("(no plot:", e, ")")


if __name__ == "__main__":
    main()
