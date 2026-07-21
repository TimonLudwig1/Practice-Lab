"""ICP-Registrierung from scratch: Kabsch-SVD + Iteration.

Modul 20 — 3D Point Cloud Processing.

>>> DEINE AUFGABE <<<
Implementiere die zwei Kernfunktionen `kabsch` und die ICP-Iterationsschleife in `icp` (die
`# TODO`-Stellen). Alles andere (apply_transform, Trimming-Geruest, Datengenerator, run.py, Tests)
ist vorgegeben. Loesung in solution/.
"""
import numpy as np
from scipy.spatial import cKDTree


# ==========================================================================
# Kabsch / Procrustes — die geschlossene SVD-Loesung   >>> DU BIST DRAN <<<
# ==========================================================================
def kabsch(P, Q):
    """Optimale Rigid-Transformation (R,t), die sum_i ||R p_i + t - q_i||^2 minimiert.

    P, Q: (n,3) KORRESPONDIERENDE Punktmengen. Gib R (3,3) und t (3,) zurueck.

    Schritte (Skript Kap. 7):
      1. Schwerpunkte p_bar, q_bar; beide Mengen zentrieren (Pc, Qc).
      2. Kreuz-Kovarianz H = Pc^T @ Qc  (3x3).
      3. SVD: U, S, Vt = np.linalg.svd(H); V = Vt.T.
      4. Det-Korrektur d = sign(det(V @ U.T)); R = V @ diag(1,1,d) @ U.T
         (verhindert Spiegelung mit det=-1).
      5. t = q_bar - R @ p_bar.
    """
    # TODO: implementiere die 5 Schritte
    raise NotImplementedError


def apply_transform(pts, R, t):
    """Wende (R,t) auf (n,3)-Punkte an: R p + t (vektorisiert).   [vorgegeben]"""
    return pts @ R.T + t


# ==========================================================================
# ICP   >>> DU BIST DRAN (die Iterationsschleife) <<<
# ==========================================================================
def icp(source, target, max_iter=60, tol=1e-7, trim_ratio=1.0, init_R=None, init_t=None):
    """Point-to-point ICP: richtet source auf target aus. Gibt (R_total, t_total, rmse_history).

    In jeder Iteration:
      (1) naechste Korrespondenz je aktuellem Quellpunkt via tree.query(src),
      (2) optional Trimming (nur die trim_ratio-naechsten Paare, sonst alle),
      (3) dR, dt = kabsch(...) auf den (getrimmten) Paaren,
      (4) src = apply_transform(src, dR, dt); Gesamt-Transformation akkumulieren
          (R <- dR @ R, t <- dR @ t + dt),
      (5) RMSE der Korrespondenzdistanzen protokollieren; Abbruch, wenn sich der RMSE
          um weniger als tol aendert.
    """
    tree = cKDTree(target)
    R = np.eye(3) if init_R is None else init_R.copy()
    t = np.zeros(3) if init_t is None else init_t.copy()
    src = apply_transform(source, R, t)
    history = []
    for _ in range(max_iter):
        dist, idx = tree.query(src)
        corr = target[idx]
        # Trimming-Geruest (vorgegeben): waehle die zu nutzenden Paare
        if trim_ratio < 1.0:
            keep = np.argsort(dist)[:max(3, int(trim_ratio * len(dist)))]
            use_src, use_corr, use_dist = src[keep], corr[keep], dist[keep]
        else:
            use_src, use_corr, use_dist = src, corr, dist
        # TODO: (3) Kabsch auf (use_src, use_corr); (4) src updaten + R,t akkumulieren;
        #       (5) rmse aus use_dist berechnen, an history anhaengen, Abbruch bei Konvergenz.
        raise NotImplementedError
    return R, t, history


def rotation_angle(R):
    """Rotationswinkel (Grad): arccos((tr(R)-1)/2).   [vorgegeben]"""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(c))


# ==========================================================================
# Synthetische Testform (asymmetrisch -> Rotation eindeutig)   [vorgegeben]
# ==========================================================================
def make_shape(n=1200, seed=0):
    """Punkte auf anisotropem Ellipsoid + Gauss-Beule (bricht alle Symmetrien)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = dirs * np.array([1.0, 0.6, 0.4])
    w = np.exp(-((pts - np.array([1.0, 0, 0])) ** 2).sum(1) / 0.15)
    return pts + w[:, None] * np.array([0.35, 0.0, 0.0])


def random_rigid(angle_deg, seed=0, tmag=0.5):
    """Zufaellige Rigid-Transformation mit gegebenem Rotationswinkel."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    th = np.deg2rad(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return R, rng.normal(size=3) * tmag
