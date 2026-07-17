"""ICP-Registrierung from scratch: Kabsch-SVD + Iteration.  (Referenzloesung P02-medium)

Modul 20 — 3D Point Cloud Processing.

Kern: die geschlossene SVD-Loesung des Procrustes-Problems (Kabsch, Skript Kap. 7) und die
ICP-Iteration (naechste Korrespondenz -> Kabsch -> anwenden, Skript Kap. 6-8). Reines numpy/scipy.
"""
import numpy as np
from scipy.spatial import cKDTree


# ==========================================================================
# Kabsch / Procrustes — die geschlossene SVD-Loesung
# ==========================================================================
def kabsch(P, Q):
    """Optimale Rigid-Transformation (R,t), die sum_i ||R p_i + t - q_i||^2 minimiert.

    P, Q: (n,3) KORRESPONDIERENDE Punktmengen. Gibt R (3,3, det=+1) und t (3,) zurueck.

    Herleitung (Skript Kap. 7): zentrieren -> H = sum (p_i-p_bar)(q_i-q_bar)^T; SVD H=U S V^T;
    R = V diag(1,1,det(V U^T)) U^T; t = q_bar - R p_bar. Die Det-Korrektur verhindert Spiegelungen.
    """
    p_bar = P.mean(0)
    q_bar = Q.mean(0)
    Pc = P - p_bar
    Qc = Q - q_bar
    H = Pc.T @ Qc                          # (3,3) Kreuz-Kovarianz
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T))    # +/-1, verhindert Spiegelung
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ U.T
    t = q_bar - R @ p_bar
    return R, t


def apply_transform(pts, R, t):
    """Wende (R,t) auf (n,3)-Punkte an: R p + t (vektorisiert)."""
    return pts @ R.T + t


# ==========================================================================
# ICP
# ==========================================================================
def icp(source, target, max_iter=60, tol=1e-7, trim_ratio=1.0, init_R=None, init_t=None):
    """Point-to-point ICP: richtet source auf target aus.

    Iteriert: (1) naechste Korrespondenz je Quellpunkt (kd-Baum),
              (2) optional Trimming (nur die trim_ratio-naechsten Paare behalten),
              (3) Kabsch loesen, (4) anwenden. Abbruch bei Konvergenz.

    Gibt (R_total, t_total, rmse_history) zurueck. R_total/t_total bilden das ORIGINAL-source ab.
    """
    tree = cKDTree(target)
    R = np.eye(3) if init_R is None else init_R.copy()
    t = np.zeros(3) if init_t is None else init_t.copy()
    src = apply_transform(source, R, t)
    history = []
    for _ in range(max_iter):
        dist, idx = tree.query(src)
        corr = target[idx]
        if trim_ratio < 1.0:                       # Trimmed ICP: nur die besten Paare
            keep = np.argsort(dist)[:max(3, int(trim_ratio * len(dist)))]
            dR, dt = kabsch(src[keep], corr[keep])
            rmse = np.sqrt(np.mean(dist[keep] ** 2))
        else:
            dR, dt = kabsch(src, corr)
            rmse = np.sqrt(np.mean(dist ** 2))
        src = apply_transform(src, dR, dt)
        R = dR @ R                                 # akkumuliere Gesamt-Transformation
        t = dR @ t + dt
        history.append(rmse)
        if len(history) > 1 and abs(history[-2] - history[-1]) < tol:
            break
    return R, t, history


def rotation_angle(R):
    """Rotationswinkel (Grad) einer Rotationsmatrix: arccos((tr(R)-1)/2)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(c))


# ==========================================================================
# Synthetische Testform (asymmetrisch -> Rotation eindeutig bestimmbar)
# ==========================================================================
def make_shape(n=1200, seed=0):
    """Punkte auf einem anisotropen Ellipsoid + Gauss-Beule (bricht alle Symmetrien)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    axes = np.array([1.0, 0.6, 0.4])                # verschiedene Achsen -> nicht kugelsymmetrisch
    pts = dirs * axes
    # Beule auf der +x-Seite: verschiebt Punkte nahe (1,0,0) nach aussen
    w = np.exp(-((pts - np.array([1.0, 0, 0])) ** 2).sum(1) / 0.15)
    pts = pts + (w[:, None]) * np.array([0.35, 0.0, 0.0])
    return pts


def random_rigid(angle_deg, seed=0, tmag=0.5):
    """Zufaellige Rigid-Transformation mit gegebenem Rotationswinkel."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    th = np.deg2rad(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    t = rng.normal(size=3) * tmag
    return R, t
