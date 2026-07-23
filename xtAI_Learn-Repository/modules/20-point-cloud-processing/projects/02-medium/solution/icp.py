"""ICP registration from scratch: Kabsch SVD + iteration.  (Reference solution P02-medium)

Module 20 — 3D Point Cloud Processing.

Core: the closed-form SVD solution of the Procrustes problem (Kabsch, script ch. 7) and the
ICP iteration (nearest correspondence -> Kabsch -> apply, script ch. 6-8). Pure numpy/scipy.
"""
import numpy as np
from scipy.spatial import cKDTree


# ==========================================================================
# Kabsch / Procrustes — the closed-form SVD solution
# ==========================================================================
def kabsch(P, Q):
    """Optimal rigid transformation (R,t) minimising sum_i ||R p_i + t - q_i||^2.

    P, Q: (n,3) CORRESPONDING point sets. Returns R (3,3, det=+1) and t (3,).

    Derivation (script ch. 7): centre -> H = sum (p_i-p_bar)(q_i-q_bar)^T; SVD H=U S V^T;
    R = V diag(1,1,det(V U^T)) U^T; t = q_bar - R p_bar. The det correction prevents reflections.
    """
    p_bar = P.mean(0)
    q_bar = Q.mean(0)
    Pc = P - p_bar
    Qc = Q - q_bar
    H = Pc.T @ Qc                          # (3,3) cross-covariance
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T))    # +/-1, prevents a reflection
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ U.T
    t = q_bar - R @ p_bar
    return R, t


def apply_transform(pts, R, t):
    """Apply (R,t) to (n,3) points: R p + t (vectorised)."""
    return pts @ R.T + t


# ==========================================================================
# ICP
# ==========================================================================
def icp(source, target, max_iter=60, tol=1e-7, trim_ratio=1.0, init_R=None, init_t=None):
    """Point-to-point ICP: aligns source onto target.

    Iterates: (1) nearest correspondence per source point (kd-tree),
              (2) optional trimming (keep only the trim_ratio nearest pairs),
              (3) solve Kabsch, (4) apply. Stops on convergence.

    Returns (R_total, t_total, rmse_history). R_total/t_total map the ORIGINAL source.
    """
    tree = cKDTree(target)
    R = np.eye(3) if init_R is None else init_R.copy()
    t = np.zeros(3) if init_t is None else init_t.copy()
    src = apply_transform(source, R, t)
    history = []
    for _ in range(max_iter):
        dist, idx = tree.query(src)
        corr = target[idx]
        if trim_ratio < 1.0:                       # trimmed ICP: only the best pairs
            keep = np.argsort(dist)[:max(3, int(trim_ratio * len(dist)))]
            dR, dt = kabsch(src[keep], corr[keep])
            rmse = np.sqrt(np.mean(dist[keep] ** 2))
        else:
            dR, dt = kabsch(src, corr)
            rmse = np.sqrt(np.mean(dist ** 2))
        src = apply_transform(src, dR, dt)
        R = dR @ R                                 # accumulate the total transformation
        t = dR @ t + dt
        history.append(rmse)
        if len(history) > 1 and abs(history[-2] - history[-1]) < tol:
            break
    return R, t, history


def rotation_angle(R):
    """Rotation angle (degrees) of a rotation matrix: arccos((tr(R)-1)/2)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(c))


# ==========================================================================
# Synthetic test shape (asymmetric -> the rotation is uniquely determined)
# ==========================================================================
def make_shape(n=1200, seed=0):
    """Points on an anisotropic ellipsoid + a Gaussian bump (breaks all symmetries)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    axes = np.array([1.0, 0.6, 0.4])                # different axes -> not spherically symmetric
    pts = dirs * axes
    # bump on the +x side: pushes points near (1,0,0) outwards
    w = np.exp(-((pts - np.array([1.0, 0, 0])) ** 2).sum(1) / 0.15)
    pts = pts + (w[:, None]) * np.array([0.35, 0.0, 0.0])
    return pts


def random_rigid(angle_deg, seed=0, tmag=0.5):
    """Random rigid transformation with the given rotation angle."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    th = np.deg2rad(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    t = rng.normal(size=3) * tmag
    return R, t
