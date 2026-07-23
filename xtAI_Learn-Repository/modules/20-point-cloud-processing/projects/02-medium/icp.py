"""ICP registration from scratch: Kabsch SVD + iteration.

Module 20 — 3D Point Cloud Processing.

>>> YOUR TASK <<<
Implement the two core functions: `kabsch` and the ICP iteration loop in `icp` (the `# TODO`
spots). Everything else (apply_transform, the trimming scaffold, the data generator, run.py, the
tests) is given. The solution is in solution/.
"""
import numpy as np
from scipy.spatial import cKDTree


# ==========================================================================
# Kabsch / Procrustes — the closed-form SVD solution   >>> YOUR TURN <<<
# ==========================================================================
def kabsch(P, Q):
    """Optimal rigid transformation (R,t) minimising sum_i ||R p_i + t - q_i||^2.

    P, Q: (n,3) CORRESPONDING point sets. Return R (3,3) and t (3,).

    Steps (script ch. 7):
      1. Centroids p_bar, q_bar; centre both sets (Pc, Qc).
      2. Cross-covariance H = Pc^T @ Qc  (3x3).
      3. SVD: U, S, Vt = np.linalg.svd(H); V = Vt.T.
      4. Det correction d = sign(det(V @ U.T)); R = V @ diag(1,1,d) @ U.T
         (prevents a reflection with det=-1).
      5. t = q_bar - R @ p_bar.
    """
    # TODO: implement the 5 steps
    raise NotImplementedError


def apply_transform(pts, R, t):
    """Apply (R,t) to (n,3) points: R p + t (vectorised).   [given]"""
    return pts @ R.T + t


# ==========================================================================
# ICP   >>> YOUR TURN (the iteration loop) <<<
# ==========================================================================
def icp(source, target, max_iter=60, tol=1e-7, trim_ratio=1.0, init_R=None, init_t=None):
    """Point-to-point ICP: aligns source onto target. Returns (R_total, t_total, rmse_history).

    In every iteration:
      (1) nearest correspondence per current source point via tree.query(src),
      (2) optional trimming (only the trim_ratio nearest pairs, otherwise all),
      (3) dR, dt = kabsch(...) on the (trimmed) pairs,
      (4) src = apply_transform(src, dR, dt); accumulate the total transformation
          (R <- dR @ R, t <- dR @ t + dt),
      (5) log the RMSE of the correspondence distances; stop when the RMSE changes
          by less than tol.
    """
    tree = cKDTree(target)
    R = np.eye(3) if init_R is None else init_R.copy()
    t = np.zeros(3) if init_t is None else init_t.copy()
    src = apply_transform(source, R, t)
    history = []
    for _ in range(max_iter):
        dist, idx = tree.query(src)
        corr = target[idx]
        # trimming scaffold (given): pick the pairs to be used
        if trim_ratio < 1.0:
            keep = np.argsort(dist)[:max(3, int(trim_ratio * len(dist)))]
            use_src, use_corr, use_dist = src[keep], corr[keep], dist[keep]
        else:
            use_src, use_corr, use_dist = src, corr, dist
        # TODO: (3) Kabsch on (use_src, use_corr); (4) update src + accumulate R,t;
        #       (5) compute the rmse from use_dist, append to history, stop on convergence.
        raise NotImplementedError
    return R, t, history


def rotation_angle(R):
    """Rotation angle (degrees): arccos((tr(R)-1)/2).   [given]"""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(c))


# ==========================================================================
# Synthetic test shape (asymmetric -> the rotation is unique)   [given]
# ==========================================================================
def make_shape(n=1200, seed=0):
    """Points on an anisotropic ellipsoid + a Gaussian bump (breaks all symmetries)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = dirs * np.array([1.0, 0.6, 0.4])
    w = np.exp(-((pts - np.array([1.0, 0, 0])) ** 2).sum(1) / 0.15)
    return pts + w[:, None] * np.array([0.35, 0.0, 0.0])


def random_rigid(angle_deg, seed=0, tmag=0.5):
    """Random rigid transformation with the given rotation angle."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    th = np.deg2rad(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return R, rng.normal(size=3) * tmag
