"""Pose-graph SLAM in 2D (SE(2)): Gauss-Newton on a sparse information matrix.
(Reference solution P03-final)   Module 22 — Robotics 2.

Nodes = robot poses (x, y, theta). Edges = relative-pose constraints (z, Omega): odometry between
consecutive poses, loop closures between revisited ones. We minimise
    sum_ij  e_ij^T Omega_ij e_ij,    e_ij = t2v( Z_ij^-1 X_i^-1 X_j )
by Gauss-Newton: H dx = -b with H = sum J^T Omega J (SPARSE), anchoring pose 0. A single loop
closure collapses the accumulated odometry drift. Everything from scratch (numpy + scipy.sparse).
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ==========================================================================
# SE(2) utilities
# ==========================================================================
def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def v2t(x):
    """Pose vector (x, y, theta) -> 3x3 homogeneous transform."""
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([[c, -s, x[0]],
                     [s,  c, x[1]],
                     [0,  0, 1.0]])


def t2v(T):
    """Homogeneous transform -> pose vector (x, y, theta)."""
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


# ==========================================================================
# Edge error and its analytic Jacobians (Grisetti et al. tutorial)
# ==========================================================================
def edge_error(xi, xj, z):
    """e_ij = t2v( Z^-1 Xi^-1 Xj ), angle wrapped. Zero when xi,xj match the measurement z."""
    e = t2v(np.linalg.inv(v2t(z)) @ np.linalg.inv(v2t(xi)) @ v2t(xj))
    e[2] = wrap(e[2])
    return e


def edge_jacobians(xi, xj, z):
    """Analytic Jacobians A = de/dxi, B = de/dxj (each 3x3)."""
    c, s = np.cos(xi[2]), np.sin(xi[2])
    RiT = np.array([[c, s], [-s, c]])            # R_i^T
    dRiT = np.array([[-s, c], [-c, -s]])         # d R_i^T / d theta_i
    cz, sz = np.cos(z[2]), np.sin(z[2])
    RijT = np.array([[cz, sz], [-sz, cz]])       # R_z^T
    A = np.zeros((3, 3)); B = np.zeros((3, 3))
    A[:2, :2] = -RijT @ RiT
    A[:2, 2] = RijT @ dRiT @ (xj[:2] - xi[:2])
    A[2, 2] = -1.0
    B[:2, :2] = RijT @ RiT
    B[2, 2] = 1.0
    return A, B


# ==========================================================================
# The optimiser: Gauss-Newton on the sparse pose graph
# ==========================================================================
def optimize(x0, edges, iters=10, huber=None, anchor_weight=1e6):
    """Gauss-Newton pose-graph optimisation.

    x0     : (N,3) initial pose estimate (e.g. from odometry).
    edges  : list of (i, j, z, Omega) constraints.
    huber  : if set, robust Huber down-weighting of edges whose Mahalanobis error exceeds it.
    Returns (x_opt (N,3), chi2_history).
    """
    x = np.array(x0, float).reshape(-1, 3)
    N = len(x)
    chi_hist = []
    for _ in range(iters):
        H = lil_matrix((3 * N, 3 * N))
        b = np.zeros(3 * N)
        chi = 0.0
        for (i, j, z, Omega) in edges:
            e = edge_error(x[i], x[j], z)
            chi += e @ Omega @ e
            Ow = Omega
            if huber is not None:                # robust kernel: cap the influence of outliers
                d = np.sqrt(e @ Omega @ e)
                if d > huber:
                    Ow = Omega * (huber / d)
            A, B = edge_jacobians(x[i], x[j], z)
            ii, jj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
            H[ii, ii] += A.T @ Ow @ A
            H[ii, jj] += A.T @ Ow @ B
            H[jj, ii] += B.T @ Ow @ A
            H[jj, jj] += B.T @ Ow @ B
            b[ii] += A.T @ Ow @ e
            b[jj] += B.T @ Ow @ e
        H[0:3, 0:3] += np.eye(3) * anchor_weight   # anchor pose 0 (remove the gauge freedom)
        dx = spsolve(H.tocsr(), -b)
        x = x + dx.reshape(-1, 3)
        x[:, 2] = wrap(x[:, 2])
        chi_hist.append(float(chi))
    return x, chi_hist


# ==========================================================================
# Synthetic dataset: a robot drives a closed rectangular loop
# ==========================================================================
def make_dataset(seed=0, side_steps=30, step=0.3,
                 odo_sigma=(0.02, 0.02, 0.01), lc_sigma=(0.05, 0.05, 0.02), lc_radius=0.4):
    """Ground-truth loop + noisy odometry edges + loop-closure edges.

    Returns (gt, odo_edges, loop_closures) where gt is (T,3), and each edge is (i, j, z, Omega).
    Loop closures connect non-consecutive poses that pass within lc_radius of each other.
    """
    rng = np.random.default_rng(seed)
    gt = [np.array([0.0, 0.0, 0.0])]
    for _side in range(4):
        for _ in range(side_steps):
            gt.append(t2v(v2t(gt[-1]) @ v2t([step, 0, 0])))       # drive forward
        gt.append(t2v(v2t(gt[-1]) @ v2t([0, 0, np.pi / 2])))      # turn 90 degrees
    gt = np.array(gt)
    T = len(gt)
    Omega_o = np.diag(1.0 / np.array(odo_sigma) ** 2)
    Omega_l = np.diag(1.0 / np.array(lc_sigma) ** 2)

    odo_edges = []
    for i in range(T - 1):
        z_true = t2v(np.linalg.inv(v2t(gt[i])) @ v2t(gt[i + 1]))
        z = z_true + rng.normal(0, odo_sigma); z[2] = wrap(z[2])
        odo_edges.append((i, i + 1, z, Omega_o))

    loop_closures = []
    for i in range(T):
        for j in range(i + side_steps, T):       # only non-adjacent revisits
            if np.linalg.norm(gt[i][:2] - gt[j][:2]) < lc_radius:
                z_true = t2v(np.linalg.inv(v2t(gt[i])) @ v2t(gt[j]))
                z = z_true + rng.normal(0, lc_sigma); z[2] = wrap(z[2])
                loop_closures.append((i, j, z, Omega_l))
    return gt, odo_edges, loop_closures


def integrate_odometry(gt0, odo_edges):
    """Chain the odometry edges into a (drifting) trajectory estimate."""
    x = [np.array(gt0, float)]
    for (i, j, z, _) in odo_edges:
        if j == i + 1:
            x.append(t2v(v2t(x[-1]) @ v2t(z)))
    return np.array(x)


def ate(x, gt):
    """Absolute trajectory error (RMSE of positions); pose 0 is anchored, so no alignment needed."""
    return float(np.sqrt(np.mean(np.sum((x[:, :2] - gt[:, :2]) ** 2, axis=1))))
