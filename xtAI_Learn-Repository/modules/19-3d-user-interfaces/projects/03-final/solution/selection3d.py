"""A comparative 3D selection study: ray-casting vs cone vs bubble under clutter.
(Reference solution P03-final)  Module 19 — 3D User Interfaces.

The scene: a target object at distance L plus N distractors angularly surrounding it (a variable
density). The pointer (a ray from the origin) has angular motor noise sigma_theta. Three
selection techniques:

  RAY-CASTING : select the object INTERSECTED by the ray with the smallest t (the nearest hit);
                nothing hit -> a mis-selection. Precise, but for small/distant targets the noisy
                ray misses, and distractors in front occlude.
  CONE        : among all objects in the cone (half angle alpha) the one with the smallest angle
                to the axis. It makes small/distant targets easier, but picks the wrong one in a
                crowd.
  BUBBLE      : the object with the smallest angle to the SURFACE (angle_to_center - angular_radius);
                it always captures the angularly nearest one -> best when sparse, over-selects
                when dense.

Everything is purely geometric, CPU milliseconds. The generator is disclosed and reproducible
with a seed.
"""
import numpy as np


# ==========================================================================
# Geometry helpers
# ==========================================================================
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def angle_between(u, v):
    """The angle (rad) between two vectors."""
    c = np.dot(_unit(u), _unit(v))
    return np.arccos(np.clip(c, -1.0, 1.0))


def ray_sphere_t(o, d, c, R):
    """The distance t>0 to the front intersection of ray(o,d)/sphere(c,R), else None. (as in P01)"""
    m = o - c
    b = m @ d
    cc = m @ m - R * R
    disc = b * b - cc
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    t = -b - sq
    if t < 0:
        t = -b + sq
    return t if t >= 0 else None


# ==========================================================================
# The scene generator
# ==========================================================================
def make_trial(rng, L_target=4.0, n_distractors=6, spread_deg=6.0, r=0.12,
               occlude_frac=0.4):
    """Produces a scene. Object 0 is the target (distance L_target); the distractors lie
    angularly around the target (up to spread_deg), some of them NEARER (occlusion).

    Returns: pos (n,3), radii (n,), target_idx=0, target_dir (3,).
    """
    # the target direction: forwards (+x) with a slight random tilt
    base = _unit([1.0, rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)])
    target_pos = L_target * base
    pos = [target_pos]
    radii = [r]
    # two orthonormal tangent directions to base
    tmp = np.array([0.0, 1.0, 0.0]) if abs(base[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = _unit(np.cross(base, tmp))
    e2 = _unit(np.cross(base, e1))
    for _ in range(n_distractors):
        ang = np.deg2rad(rng.uniform(0.3, spread_deg))     # the angular proximity to the target
        phi = rng.uniform(0, 2 * np.pi)
        offset_dir = _unit(base + np.tan(ang) * (np.cos(phi) * e1 + np.sin(phi) * e2))
        # the distance: with probability occlude_frac nearer (occluding), else around L_target
        if rng.random() < occlude_frac:
            L = L_target * rng.uniform(0.4, 0.9)
        else:
            L = L_target * rng.uniform(0.9, 1.4)
        pos.append(L * offset_dir)
        radii.append(r)
    return np.array(pos), np.array(radii), 0, base


def noisy_ray_dir(target_dir, sigma_theta, rng):
    """The noisy pointing direction: the intended direction + a 2D Gaussian angular error."""
    base = _unit(target_dir)
    tmp = np.array([0.0, 1.0, 0.0]) if abs(base[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = _unit(np.cross(base, tmp))
    e2 = _unit(np.cross(base, e1))
    ex, ey = rng.normal(0, sigma_theta, 2)
    return _unit(base + np.tan(ex) * e1 + np.tan(ey) * e2)


# ==========================================================================
# The selection techniques  (o = the origin, here (0,0,0))
# ==========================================================================
def select_raycast(pos, radii, o, d):
    best_i, best_t = None, np.inf
    for i in range(len(pos)):
        t = ray_sphere_t(o, d, pos[i], radii[i])
        if t is not None and t < best_t:
            best_i, best_t = i, t
    return best_i


def select_cone(pos, radii, o, d, alpha_deg=8.0):
    alpha = np.deg2rad(alpha_deg)
    best_i, best_ang = None, np.inf
    for i in range(len(pos)):
        ang = angle_between(d, pos[i] - o)
        if ang < alpha and ang < best_ang:
            best_i, best_ang = i, ang
    return best_i


def select_bubble(pos, radii, o, d):
    best_i, best_gap = None, np.inf
    for i in range(len(pos)):
        dist = np.linalg.norm(pos[i] - o)
        ang = angle_between(d, pos[i] - o)
        ang_radius = np.arctan2(radii[i], dist)
        gap = ang - ang_radius            # the angle to the surface (can be negative = the ray hits)
        if gap < best_gap:
            best_i, best_gap = i, gap
    return best_i


TECHNIQUES = {"raycast": select_raycast, "cone": select_cone, "bubble": select_bubble}


def run_technique(name, pos, radii, o, d):
    if name == "cone":
        return select_cone(pos, radii, o, d)
    return TECHNIQUES[name](pos, radii, o, d)


# ==========================================================================
# ISO 9241-9 throughput  (isolated targets, reciprocal tapping)
# ==========================================================================
def throughput(amplitude_deg, hit_offsets_rad, movement_times):
    """The effective throughput (bits/s) after ISO 9241-9.

    hit_offsets_rad: the angular deviations of the clicks from the target centre (1D, along the
    movement axis).
    W_e = 4.133 * std(hit_offsets); D_e = amplitude; ID_e = log2(D_e/W_e + 1); TP = ID_e / mean(MT).
    """
    A = np.deg2rad(amplitude_deg)
    W_e = 4.133 * np.std(hit_offsets_rad)
    ID_e = np.log2(A / W_e + 1.0)
    return ID_e / np.mean(movement_times), ID_e, np.rad2deg(W_e)
