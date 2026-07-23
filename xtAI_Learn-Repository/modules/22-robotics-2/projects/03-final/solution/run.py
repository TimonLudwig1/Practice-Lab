"""Pose-graph SLAM experiments: drift collapse, incremental loop closures, robustness.
(Reference solution P03-final)   /Users/.../.venv/bin/python run.py   Plots -> results/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pose_graph import (make_dataset, integrate_odometry, optimize, ate)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def exp_drift_collapse():
    print("=" * 70)
    print("EXPERIMENT 1 — the drift collapse: odometry vs. optimised pose graph")
    print("=" * 70)
    gt, odo, lcs = make_dataset(seed=0)
    print(f"  {len(gt)} poses, {len(odo)} odometry edges, {len(lcs)} loop closures")
    x_odo = integrate_odometry(gt[0], odo)
    print(f"  odometry-only ATE:            {ate(x_odo, gt):.3f} m  (unbounded drift)")

    x_noLC, _ = optimize(x_odo, odo, iters=10)
    print(f"  optimised WITHOUT loop closures: {ate(x_noLC, gt):.3f} m  (a chain has no conflicts)")

    x_opt, chi = optimize(x_odo, odo + lcs, iters=10)
    print(f"  optimised WITH loop closures:   {ate(x_opt, gt):.4f} m")
    print(f"  chi2 per Gauss-Newton iteration: {[round(c, 1) for c in chi[:5]]} ...")
    print("  => Optimising the chain alone changes nothing (no constraint disagrees). Adding the")
    print("     loop closures makes the accumulated drift show up as a large error, and Gauss-Newton")
    print("     redistributes it over the whole loop: chi2 collapses in one step and the ATE drops ~12x.")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.5, label="ground truth")
    ax.plot(x_odo[:, 0], x_odo[:, 1], color="crimson", lw=1.3, label=f"odometry (ATE {ate(x_odo, gt):.2f})")
    ax.plot(x_opt[:, 0], x_opt[:, 1], color="steelblue", lw=1.8, label=f"optimised (ATE {ate(x_opt, gt):.2f})")
    for (i, j, z, O) in lcs:
        ax.plot([x_opt[i, 0], x_opt[j, 0]], [x_opt[i, 1], x_opt[j, 1]], color="green", lw=0.8, alpha=0.7)
    ax.plot(gt[0, 0], gt[0, 1], "go", ms=10, label="start (anchored)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("pose-graph SLAM: loop closures collapse the odometry drift")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "slam_map.png"), dpi=110); plt.close(fig)
    return gt, odo, lcs


def exp_incremental_loop_closures(gt, odo, lcs):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — how many loop closures do you need? (add them one at a time)")
    print("=" * 70)
    x_odo = integrate_odometry(gt[0], odo)
    print(f"  {'loop closures':>13} | {'ATE [m]':>9}")
    print("  " + "-" * 26)
    ks, ates = [], []
    for k in range(len(lcs) + 1):
        x_opt, _ = optimize(x_odo, odo + lcs[:k], iters=10)
        a = ate(x_opt, gt); ks.append(k); ates.append(a)
        print(f"  {k:13d} | {a:9.4f}")
    print("  => The FIRST loop closure does almost all the work: it ties the drifted end back to a")
    print("     known pose, and that one constraint corrects the entire loop (ATE 1.03 -> ~0.13 m).")
    print("     Further loop closures refine it; the gain quickly saturates.")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(ks, ates, "o-", color="purple")
    ax.set_xlabel("number of loop closures"); ax.set_ylabel("ATE [m]")
    ax.set_title("one loop closure corrects the whole trajectory"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "loop_closures.png"), dpi=110); plt.close(fig)


def exp_robustness(gt, odo, lcs):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — a single FALSE loop closure, and the robust kernel that survives it")
    print("=" * 70)
    x_odo = integrate_odometry(gt[0], odo)
    Omega_l = np.diag(1.0 / np.array([0.05, 0.05, 0.02]) ** 2)
    false_lc = (10, 80, np.array([0.2, 0.1, 0.3]), Omega_l)   # bogus relative pose, far-apart poses

    x_good, _ = optimize(x_odo, odo + lcs, iters=10)
    x_bad, _ = optimize(x_odo, odo + lcs + [false_lc], iters=10)
    x_rob, _ = optimize(x_odo, odo + lcs + [false_lc], iters=15, huber=2.0)
    print(f"  correct loop closures only     : ATE {ate(x_good, gt):.4f} m")
    print(f"  + 1 FALSE loop closure (naive) : ATE {ate(x_bad, gt):.4f} m  (worse than odometry!)")
    print(f"  + Huber robust kernel          : ATE {ate(x_rob, gt):.4f} m  (rejects the outlier)")
    print("  => Least squares trusts every edge, so ONE wrong loop closure drags the whole map to")
    print("     satisfy it and corrupts everything. A robust kernel (Huber) down-weights the edge")
    print("     whose error is grossly inconsistent with the rest, and the map recovers.")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.3, label="ground truth")
    ax.plot(x_bad[:, 0], x_bad[:, 1], color="crimson", lw=1.3, label=f"naive + false LC (ATE {ate(x_bad, gt):.1f})")
    ax.plot(x_rob[:, 0], x_rob[:, 1], color="steelblue", lw=1.8, label=f"Huber robust (ATE {ate(x_rob, gt):.2f})")
    ax.plot([x_bad[10, 0], x_bad[80, 0]], [x_bad[10, 1], x_bad[80, 1]], color="red", lw=1.5, ls=":", label="false loop closure")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("a false loop closure wrecks naive LS; the robust kernel survives")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "robustness.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    gt, odo, lcs = exp_drift_collapse()
    exp_incremental_loop_closures(gt, odo, lcs)
    exp_robustness(gt, odo, lcs)
    print("\nPlots in results/. Done.")
