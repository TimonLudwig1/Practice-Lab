"""Tabletop-Segmentierung: RANSAC-Ebene + Euclidean Clustering.  (Referenzloesung P03-final)

Modul 20 — 3D Point Cloud Processing.

Pipeline (Skript Kap. 9-10): eine Szene aus (leicht geneigtem) Boden + Objekten + Ausreissern.
  1. RANSAC findet die dominante Ebene (Boden) -> Inlier abtrennen.
  2. Die Rest-Punkte per Euclidean Clustering (Region Growing, kd-Baum) in Objekte trennen.
  3. Auswertung gegen ground-truth-Labels (Boden-Recall/Precision, Objekt-ARI, Objektzahl).
Alles from scratch (numpy/scipy); nur die Metrik ARI kommt aus sklearn. CPU-Sekunden.
"""
import numpy as np
from scipy.spatial import cKDTree


# ==========================================================================
# Szenen-Generator (offengelegt, reproduzierbar)
# ==========================================================================
def make_scene(seed=0, n_ground=3000, n_objects=4, tilt_deg=8.0,
               ground_noise=0.008, outlier_frac=0.05):
    """Boden (leicht geneigte Ebene) + n_objects Objekte (Kugeln/Quader) + Ausreisser.

    Rueckgabe: pts (N,3), labels (N,) mit 0=Boden, 1..k=Objekte, -1=Ausreisser; plus
    plane_normal (die wahre Bodennormale).
    """
    rng = np.random.default_rng(seed)
    # --- Boden: Ebene z=0, um eine zufaellige Achse leicht gekippt ---
    gx = rng.uniform(-2, 2, n_ground)
    gy = rng.uniform(-2, 2, n_ground)
    gz = rng.normal(0, ground_noise, n_ground)
    ground = np.column_stack([gx, gy, gz])
    axis = rng.normal(size=3); axis[2] = 0; axis /= np.linalg.norm(axis)
    th = np.deg2rad(tilt_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    Rt = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    ground = ground @ Rt.T
    plane_normal = Rt @ np.array([0.0, 0.0, 1.0])
    pts_list = [ground]; lab_list = [np.zeros(n_ground, int)]

    # --- Objekte auf dem Boden (mit Mindestabstand, damit sie nicht verschmelzen) ---
    centers = []
    for k in range(1, n_objects + 1):
        for _try in range(100):                # rejection sampling: Zentren >= 0.9 auseinander
            cx, cy = rng.uniform(-1.5, 1.5, 2)
            if all((cx - ux) ** 2 + (cy - uy) ** 2 > 0.9 ** 2 for ux, uy in centers):
                break
        centers.append((cx, cy))
        kind = rng.integers(2)
        n_obj = rng.integers(250, 450)
        if kind == 0:                          # Kugel
            r = rng.uniform(0.18, 0.32)
            d = rng.normal(size=(n_obj, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
            obj = d * r + np.array([cx, cy, r])
        else:                                  # Quader-Oberflaeche
            s = rng.uniform(0.2, 0.4, 3)
            obj = rng.uniform(-1, 1, (n_obj, 3)) * s
            ax = rng.integers(0, 3, n_obj)     # jeden Punkt auf eine Wuerfelseite druecken
            obj[np.arange(n_obj), ax] = np.sign(obj[np.arange(n_obj), ax]) * s[ax]
            obj = obj + np.array([cx, cy, s[2]])
        obj = obj @ Rt.T                       # mit dem Boden mitkippen
        obj = obj + rng.normal(0, 0.006, obj.shape)
        pts_list.append(obj); lab_list.append(np.full(len(obj), k, int))

    pts = np.vstack(pts_list); labels = np.concatenate(lab_list)
    # --- Ausreisser im Volumen ---
    n_out = int(outlier_frac * len(pts))
    lo, hi = pts.min(0), pts.max(0)
    outliers = rng.uniform(lo, hi, (n_out, 3))
    pts = np.vstack([pts, outliers]); labels = np.concatenate([labels, np.full(n_out, -1, int)])
    perm = rng.permutation(len(pts))
    return pts[perm], labels[perm], plane_normal


# ==========================================================================
# RANSAC-Ebenenschaetzung
# ==========================================================================
def fit_plane_3pts(p1, p2, p3):
    """Ebene aus 3 Punkten: Normale (Einheit) + Offset d mit n . x = d."""
    n = np.cross(p2 - p1, p3 - p1)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return None, None
    n = n / nn
    return n, n @ p1


def ransac_plane(pts, tau=0.02, iters=200, seed=0):
    """RANSAC-Ebene: gib (normal, d, inlier_mask, n_inliers) des besten Modells.
    Verfeinert die Ebene am Ende per PCA ueber alle Inlier."""
    rng = np.random.default_rng(seed)
    N = len(pts)
    best_mask = None; best_count = -1; best_n = None; best_d = None
    for _ in range(iters):
        i, j, k = rng.choice(N, 3, replace=False)
        n, d = fit_plane_3pts(pts[i], pts[j], pts[k])
        if n is None:
            continue
        dist = np.abs(pts @ n - d)
        mask = dist < tau
        c = int(mask.sum())
        if c > best_count:
            best_count, best_mask, best_n, best_d = c, mask, n, d
    # PCA-Verfeinerung auf den Inliern
    inl = pts[best_mask]
    c0 = inl.mean(0)
    C = (inl - c0).T @ (inl - c0)
    evals, evecs = np.linalg.eigh(C)
    n_ref = evecs[:, 0]                          # kleinster Eigenwert = Normale
    d_ref = n_ref @ c0
    mask_ref = np.abs(pts @ n_ref - d_ref) < tau
    return n_ref, d_ref, mask_ref, int(mask_ref.sum())


def ransac_iterations(w, s, p):
    """Theoretische Iterationszahl N = log(1-p) / log(1 - w^s)."""
    return np.log(1 - p) / np.log(1 - w ** s)


# ==========================================================================
# Euclidean Clustering (Region Growing per kd-Baum)
# ==========================================================================
def euclidean_clustering(pts, eps=0.06, min_size=30):
    """Region Growing: Punkte, deren eps-Nachbarschaften sich beruehren, bilden ein Cluster.
    Gibt labels (n,) mit 0,1,... fuer Cluster und -1 fuer Rauschen (Cluster < min_size)."""
    tree = cKDTree(pts)
    n = len(pts)
    labels = np.full(n, -1, int)
    visited = np.zeros(n, bool)
    cid = 0
    for start in range(n):
        if visited[start]:
            continue
        # BFS ueber eps-Nachbarschaften
        queue = [start]; visited[start] = True; comp = []
        while queue:
            i = queue.pop()
            comp.append(i)
            for j in tree.query_ball_point(pts[i], eps):
                if not visited[j]:
                    visited[j] = True; queue.append(j)
        if len(comp) >= min_size:
            labels[comp] = cid; cid += 1
    return labels
