"""Pointing precision & reach: the angular Fitts' law and Go-Go.  (Reference solution P02-medium)

Module 19 — 3D User Interfaces.

We model two selection techniques MECHANISTICALLY (not via an assumed Fitts curve):

  RAY-CASTING: a target (radius r) at distance L subtends the angular radius
    theta_r ~ arctan(r/L)  -> it shrinks with L. The pointing has a fixed ANGULAR noise
    sigma_theta. A hit if the 2D angular error < theta_r. The hit rate falls with L.

  GO-GO (Poupyrev 1996): the virtual hand follows the real one non-linearly
    r_v = r_r                    for r_r < D
    r_v = r_r + k (r_r - D)^2    for r_r >= D
  The reach grows quadratically -> distant targets become reachable. BUT: the C/D gain
  g = dr_v/dr_r = 1 + 2k(r_r-D) also amplifies the HAND noise beyond D
    sigma_v = g * sigma_r  -> precision falls in the extended range. That is the actual
  Go-Go trade-off (reach against precision), worked out exactly here.
"""
import numpy as np


# ==========================================================================
# The angular model (ray-casting)
# ==========================================================================
def angular_radius(r, L):
    """The angular radius (rad) of a target with radius r at distance L. arctan(r/L)."""
    return np.arctan2(r, L)


def p_hit_raycasting(r, L, sigma_theta):
    """The hit probability for ray-casting.

    The 2D angular error (azimuth, elevation) is N(0, sigma_theta^2) per axis; its magnitude
    is Rayleigh distributed. A hit if the magnitude < theta_r = angular_radius(r, L):

        P(hit) = 1 - exp( - theta_r^2 / (2 sigma_theta^2) ).

    theta_r ~ r/L shrinks with the distance -> P(hit) falls.
    """
    theta_r = angular_radius(r, L)
    return 1.0 - np.exp(-(theta_r**2) / (2.0 * sigma_theta**2))


# ==========================================================================
# Go-Go
# ==========================================================================
def go_go(r_r, D, k):
    """The non-linear Go-Go mapping from the real to the virtual hand distance."""
    r_r = np.asarray(r_r, float)
    return np.where(r_r < D, r_r, r_r + k * (r_r - D)**2)


def go_go_gain(r_r, D, k):
    """The C/D gain g = dr_v/dr_r. 1 up close, 1+2k(r_r-D) beyond D."""
    r_r = np.asarray(r_r, float)
    return np.where(r_r < D, 1.0, 1.0 + 2.0 * k * (r_r - D))


def go_go_inverse(L, D, k):
    """Which real hand distance r_r produces the virtual reach L? (solves r_v=L)

    For L < D:  r_r = L.
    For L >= D: k r_r^2 + (1-2kD) r_r + (kD^2 - L) = 0, the physical (larger) root.
    """
    L = np.asarray(L, float)
    if k <= 0:
        return L.copy()
    a = k
    b = 1.0 - 2.0 * k * D
    c = k * D * D - L
    disc = np.maximum(b * b - 4 * a * c, 0.0)
    r_far = (-b + np.sqrt(disc)) / (2 * a)
    return np.where(L < D, L, r_far)


def p_hit_gogo(r, L, D, k, sigma_r, arm_length):
    """The hit probability for Go-Go/virtual hand for a target of radius r at distance L.

    - A reach of L requires the real hand distance r_r = go_go_inverse(L). If r_r > arm_length,
      the target is UNREACHABLE -> P(hit)=0 (that is the virtual hand ceiling).
    - Otherwise: the fingertip noise is sigma_v = g(r_r)*sigma_r (amplified beyond D).
      A hit if the 2D position error < r:  P(hit)=1-exp(-r^2/(2 sigma_v^2)).
    """
    r_r = float(go_go_inverse(L, D, k))
    if r_r > arm_length:
        return 0.0
    g = float(go_go_gain(r_r, D, k))
    sigma_v = g * sigma_r
    return 1.0 - np.exp(-(r**2) / (2.0 * sigma_v**2))


# ==========================================================================
# The angular Fitts' law (the characterization)
# ==========================================================================
def angular_id(theta_D, theta_W):
    """The index of difficulty (bits): log2(theta_D/theta_W + 1)."""
    return np.log2(theta_D / theta_W + 1.0)


def simulate_fitts_times(theta_D, theta_W, a, b, motor_sigma, n_reps, rng):
    """Simulates movement times for reciprocal tapping: MT = a + b*ID + noise.
    Returns (IDs, MTs) for all repetitions (for the regression)."""
    ids, mts = [], []
    for tD, tW in zip(theta_D, theta_W):
        ID = angular_id(tD, tW)
        for _ in range(n_reps):
            mt = a + b * ID + rng.normal(0, motor_sigma)
            ids.append(ID); mts.append(max(mt, 1e-3))
    return np.array(ids), np.array(mts)


def fit_fitts(ids, mts):
    """A linear regression MT = a + b*ID. Returns (a, b, R^2)."""
    b, a = np.polyfit(ids, mts, 1)  # polyfit: the highest degree first -> slope, intercept
    pred = a + b * ids
    ss_res = np.sum((mts - pred)**2)
    ss_tot = np.sum((mts - mts.mean())**2)
    r2 = 1.0 - ss_res / ss_tot
    return a, b, r2
