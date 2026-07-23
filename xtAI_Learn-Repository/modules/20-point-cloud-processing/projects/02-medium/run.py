"""ICP experiments: convergence, init sensitivity (local minimum), partial overlap.
(Reference solution P02-medium)   /Users/.../.venv/bin/python run.py    Plots -> results/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from icp import (kabsch, apply_transform, icp, rotation_angle, make_shape, random_rigid)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def exp_convergence():
    print("=" * 66)
    print("EXPERIMENT 1 — ICP convergence (moderate misalignment, noise)")
    print("=" * 66)
    target = make_shape(1400, seed=0)
    R_gt, t_gt = random_rigid(angle_deg=35, seed=1, tmag=0.4)
    rng = np.random.default_rng(2)
    source = apply_transform(target, R_gt, t_gt) + rng.normal(0, 0.01, target.shape)

    R, t, hist = icp(source, target, max_iter=60)
    # ICP maps source->target; the true solution is the inverse of (R_gt,t_gt).
    # Quality: residual rotation error of R @ R_gt (should be ~ the identity)
    ang_err = rotation_angle(R @ R_gt)
    aligned = apply_transform(source, R, t)
    print(f"  start RMSE {hist[0]:.4f} -> final RMSE {hist[-1]:.5f} in {len(hist)} iterations")
    print(f"  residual rotation error: {ang_err:.3f} deg  (near 0 = rotated back correctly)")
    print(f"  monotonically decreasing? {all(hist[i] >= hist[i+1] - 1e-9 for i in range(len(hist)-1))}")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(hist, "o-"); ax.set_yscale("log")
    ax.set_xlabel("iteration"); ax.set_ylabel("RMSE (log)")
    ax.set_title("ICP converges monotonically"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "convergence.png"), dpi=110); plt.close(fig)
    return dict(rmse=hist[-1], ang_err=ang_err)


def exp_basin():
    print("\n" + "=" * 66)
    print("EXPERIMENT 2 — convergence basin: ICP is only LOCAL")
    print("=" * 66)
    target = make_shape(1200, seed=0)
    angles = np.arange(0, 181, 15)
    final_err, success = [], []
    for a in angles:
        R_gt, t_gt = random_rigid(angle_deg=a, seed=10 + a, tmag=0.3)
        rng = np.random.default_rng(100 + a)
        source = apply_transform(target, R_gt, t_gt) + rng.normal(0, 0.008, target.shape)
        R, t, hist = icp(source, target, max_iter=80)
        ang = rotation_angle(R @ R_gt)
        final_err.append(ang)
        success.append(ang < 5.0)
    print(f"  {'misalignment [deg]':>18} | {'residual err [deg]':>18} | successful?")
    print("  " + "-" * 55)
    for a, e, s in zip(angles, final_err, success):
        print(f"  {a:18d} | {e:18.2f} | {'yes' if s else 'NO (local minimum)'}")
    last_ok = max([a for a, s in zip(angles, success) if s], default=0)
    print(f"  => ICP converges correctly up to ~{last_ok} deg of misalignment, beyond that it locks")
    print("     into a WRONG minimum. Hence ICP needs a good initialisation (global registration).")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(angles, final_err, "o-", color="crimson")
    ax.axhline(5, ls="--", color="gray", label="success threshold 5 deg")
    ax.set_xlabel("initial misalignment [deg]"); ax.set_ylabel("final rotation error [deg]")
    ax.set_title("convergence basin of ICP"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "basin.png"), dpi=110); plt.close(fig)
    return dict(last_ok=last_ok)


def exp_partial_outliers():
    print("\n" + "=" * 66)
    print("EXPERIMENT 3 — partial overlap + outliers: trimmed ICP helps")
    print("=" * 66)
    target_full = make_shape(1600, seed=0)
    # the source sees only a PART (x>-0.2) + outliers
    R_gt, t_gt = random_rigid(angle_deg=25, seed=5, tmag=0.3)
    rng = np.random.default_rng(7)
    part = target_full[target_full[:, 0] > -0.2]
    source = apply_transform(part, R_gt, t_gt) + rng.normal(0, 0.01, part.shape)
    n_out = int(0.10 * len(source))
    outliers = rng.uniform(-1.5, 1.5, (n_out, 3))
    source_noisy = np.vstack([source, apply_transform(outliers, R_gt, t_gt)])

    R1, t1, h1 = icp(source_noisy, target_full, max_iter=80, trim_ratio=1.0)
    R2, t2, h2 = icp(source_noisy, target_full, max_iter=80, trim_ratio=0.8)
    e1, e2 = rotation_angle(R1 @ R_gt), rotation_angle(R2 @ R_gt)
    print(f"  vanilla ICP (all pairs)      : residual error {e1:6.2f} deg, RMSE {h1[-1]:.4f}")
    print(f"  trimmed ICP (best 80% pairs) : residual error {e2:6.2f} deg, RMSE {h2[-1]:.4f}")
    print("  => outliers + non-overlapping points drag vanilla ICP away; trimming")
    print("     discards the bad correspondences and stays robust.")
    return dict(e_vanilla=e1, e_trim=e2)


if __name__ == "__main__":
    exp_convergence()
    exp_basin()
    exp_partial_outliers()
    print("\nPlots in results/. Done.")
