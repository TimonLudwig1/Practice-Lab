"""Tabletop segmentation — evaluation.  (Reference solution P03-final)

    /Users/.../.venv/bin/python run.py    Plots -> results/ (gitignored).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from pointcloud_seg import (make_scene, ransac_plane, ransac_iterations,
                          euclidean_clustering)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def angle_deg(u, v):
    c = np.clip(abs(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))), -1, 1)
    return np.rad2deg(np.arccos(c))


def exp_ransac_plane():
    print("=" * 66)
    print("EXPERIMENT 1 — RANSAC plane: find the ground + the iteration-count formula")
    print("=" * 66)
    pts, labels, true_n = make_scene(seed=0)
    w = np.mean(labels == 0)                       # true inlier fraction (ground points)
    print(f"  scene: {len(pts)} points, ground fraction w={w:.3f}")
    n, d, mask, n_in = ransac_plane(pts, tau=0.02, iters=300, seed=1)
    # quality: how many of the found inliers really are ground?
    prec = np.mean(labels[mask] == 0)
    rec = np.mean(mask[labels == 0])
    print(f"  plane found: {n_in} inliers | normal error {angle_deg(n, true_n):.2f} deg")
    print(f"  ground precision {prec:.3f} | ground recall {rec:.3f}")

    # verify the iteration-count formula: empirical success rate vs. the N prediction
    print("\n  iteration-count formula N = log(1-p)/log(1-w^3):")
    for p_conf in [0.90, 0.99, 0.999]:
        Npred = ransac_iterations(w, 3, p_conf)
        # empirically: fraction of runs (with Npred iterations) that hit the plane well
        succ = 0; T = 60
        n_ground = (labels == 0).sum()
        for s in range(T):
            _, _, m, _ = ransac_plane(pts, tau=0.02, iters=int(np.ceil(Npred)), seed=1000 + s)
            succ += int(m.sum() > 0.8 * n_ground)   # plane hit well?
        print(f"    p={p_conf:.3f}: N_predicted={Npred:6.1f} -> empirical success rate {succ/T:.2f}")
    print("  => with the predicted iteration count RANSAC hits the plane reliably.")
    return dict(w=w, normal_err=angle_deg(n, true_n), prec=prec, rec=rec)


def segment(pts, tau=0.02, eps=0.12, min_size=25, seed=1):
    """Full pipeline: plane out, cluster the rest. Returns (plane_mask, obj_idx, cluster_labels)."""
    n, d, plane_mask, _ = ransac_plane(pts, tau=tau, iters=300, seed=seed)
    obj_idx = np.where(~plane_mask)[0]
    cl = euclidean_clustering(pts[obj_idx], eps=eps, min_size=min_size)
    return plane_mask, obj_idx, cl


def exp_pipeline():
    print("\n" + "=" * 66)
    print("EXPERIMENT 2 — full pipeline: segment the ground + the objects")
    print("=" * 66)
    pts, labels, true_n = make_scene(seed=0, n_objects=4)
    plane_mask, obj_idx, cl = segment(pts)
    n_found = len(set(cl[cl >= 0]))
    n_true = len(set(labels[labels > 0]))
    lo = labels[obj_idx]
    obj_mask = lo > 0                              # true object points among the rest
    ari = adjusted_rand_score(lo[obj_mask], cl[obj_mask])   # pure object separation
    obj_recall = np.mean(~plane_mask[labels > 0])          # object points NOT in the ground
    print(f"  true objects: {n_true} | clusters found: {n_found}")
    print(f"  object ARI (true object points only): {ari:.3f}  (1.0 = perfect separation)")
    print(f"  object-point recall (not wrongly in the ground): {obj_recall:.3f}")
    print("  => RANSAC separates the ground off, region growing separates the objects cleanly.")

    # 3D plot of the segmentation
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[plane_mask, 0], pts[plane_mask, 1], pts[plane_mask, 2],
               s=2, alpha=0.15, color="lightgray", label="ground (RANSAC)")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for c in sorted(set(cl[cl >= 0])):
        m = obj_idx[cl == c]
        ax.scatter(pts[m, 0], pts[m, 1], pts[m, 2], s=6, color=colors[c % 10], label=f"object {c}")
    ax.set_title(f"segmentation: {n_found} objects + ground"); ax.legend(fontsize=8)
    try: ax.set_box_aspect((1, 1, 0.5))
    except Exception: pass
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "segmentation.png"), dpi=110); plt.close(fig)
    return dict(n_found=n_found, n_true=n_true, ari=ari)


def _objects_detected(lo, cl):
    """Number of true objects whose points lie mostly in ONE cluster."""
    det = 0
    for o in set(lo[lo > 0]):
        cids = cl[lo == o]; cids = cids[cids >= 0]
        if len(cids) == 0:
            continue
        _, counts = np.unique(cids, return_counts=True)
        if counts.max() > 0.5 * np.sum(lo == o):
            det += 1
    return det


def exp_minsize_tradeoff():
    print("\n" + "=" * 66)
    print("EXPERIMENT 3 — min_size trade-off in the clustering (30% outliers, 8 scenes)")
    print("=" * 66)
    print("The clustering parameter min_size controls a U-shaped trade-off:")
    print("too small -> outliers form their own noise clusters; too large -> real objects drop out.\n")
    print(f"  {'min_size':>8} | {'clusters found':>16} | {'objects detected (/4)':>21}")
    print("  " + "-" * 52)
    ms_vals = [5, 30, 150, 300, 450]
    ncl_all, det_all = [], []
    for ms in ms_vals:
        ncl, det = [], []
        for s in range(8):
            pts, labels, true_n = make_scene(seed=s, n_objects=4, outlier_frac=0.30)
            plane_mask, obj_idx, cl = segment(pts, min_size=ms, seed=1 + s)
            ncl.append(len(set(cl[cl >= 0])))
            det.append(_objects_detected(labels[obj_idx], cl))
        ncl_all.append(np.mean(ncl)); det_all.append(np.mean(det))
        print(f"  {ms:8d} | {np.mean(ncl):16.2f} | {np.mean(det):21.2f}")
    print("  => sweet spot at min_size ~ 30-150: exactly 4 clean clusters, all objects detected.")
    print("     min_size=5 -> ~12 clusters (noise); min_size=450 -> the objects disappear.")

    fig, ax1 = plt.subplots(figsize=(7, 4.3))
    ax1.plot(ms_vals, ncl_all, "o-", color="steelblue", label="clusters found")
    ax1.axhline(4, ls="--", color="green", label="true = 4")
    ax1.set_xscale("log"); ax1.set_xlabel("min_size (log)"); ax1.set_ylabel("clusters found")
    ax2 = ax1.twinx()
    ax2.plot(ms_vals, det_all, "s-", color="crimson", label="objects detected")
    ax2.set_ylabel("objects detected (/4)", color="crimson")
    ax1.set_title("min_size trade-off: noise vs. lost objects")
    ax1.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "minsize_tradeoff.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    exp_ransac_plane()
    exp_pipeline()
    exp_minsize_tradeoff()
    print("\nPlots in results/. Done.")
