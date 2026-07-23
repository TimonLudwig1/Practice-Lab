"""Pointing precision & reach: the angular Fitts' law and Go-Go.

Module 19 — 3D User Interfaces.

>>> YOUR TASK <<<
Implement the three core functions marked with `# TODO` (go_go, p_hit_raycasting,
p_hit_gogo). The rest (the gain, the inverse, the Fitts fit, run.py, the tests) is given.
The solution is in solution/.

Two selection techniques, modelled MECHANISTICALLY:

  RAY-CASTING: a target (radius r) at distance L subtends the angular radius
    theta_r ~ arctan(r/L), which shrinks with L. The pointing has a fixed ANGULAR noise
    sigma_theta. A hit if the 2D angular error < theta_r.

  GO-GO: the virtual hand follows the real one non-linearly; the reach grows quadratically, but
    the C/D gain g=dr_v/dr_r also amplifies the noise beyond D (sigma_v=g*sigma_r).
"""
import numpy as np


# ==========================================================================
# The angular model (ray-casting)
# ==========================================================================
def angular_radius(r, L):
    """The angular radius (rad) of a target of radius r at distance L. arctan(r/L).   [given]"""
    return np.arctan2(r, L)


def p_hit_raycasting(r, L, sigma_theta):
    """The hit probability for ray-casting.   >>> THIS IS YOUR PART <<<

    The 2D angular error is N(0, sigma_theta^2) per axis; its magnitude is Rayleigh distributed.
    A hit if the magnitude < theta_r = angular_radius(r, L). For the Rayleigh distribution:

        P(hit) = 1 - exp( - theta_r^2 / (2 * sigma_theta^2) ).
    """
    # TODO: compute theta_r and return the Rayleigh hit probability
    raise NotImplementedError


# ==========================================================================
# Go-Go
# ==========================================================================
def go_go(r_r, D, k):
    """The non-linear Go-Go mapping real -> virtual hand distance.   >>> THIS IS YOUR PART <<<

        r_v = r_r                    for r_r < D
        r_v = r_r + k (r_r - D)^2    for r_r >= D

    A hint: np.where(...) works element-wise (r_r can be a scalar or an array).
    """
    r_r = np.asarray(r_r, float)
    # TODO: implement the piecewise Go-Go function
    raise NotImplementedError


def go_go_gain(r_r, D, k):
    """The C/D gain g = dr_v/dr_r. 1 up close, 1+2k(r_r-D) beyond D.   [given]"""
    r_r = np.asarray(r_r, float)
    return np.where(r_r < D, 1.0, 1.0 + 2.0 * k * (r_r - D))


def go_go_inverse(L, D, k):
    """The real hand distance r_r that produces the virtual reach L (solves r_v=L).   [given]

    L<D: r_r=L.  L>=D: k r_r^2 + (1-2kD) r_r + (kD^2 - L)=0, the physical (larger) root.
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
    """The hit probability for Go-Go/the virtual hand for a target of radius r at distance L.
    >>> THIS IS YOUR PART <<<

    The steps:
      1. r_r = go_go_inverse(L, D, k) — the real hand distance for this reach.
      2. If r_r > arm_length: the target is UNREACHABLE -> return 0.0.
      3. Otherwise the fingertip noise is sigma_v = go_go_gain(r_r,D,k) * sigma_r,
         a hit if the 2D position error < r:  P(hit) = 1 - exp(-r^2 / (2 sigma_v^2)).
    """
    # TODO: implement the 3 steps
    raise NotImplementedError


# ==========================================================================
# The angular Fitts' law (the characterization)   [all given]
# ==========================================================================
def angular_id(theta_D, theta_W):
    """The index of difficulty (bits): log2(theta_D/theta_W + 1)."""
    return np.log2(theta_D / theta_W + 1.0)


def simulate_fitts_times(theta_D, theta_W, a, b, motor_sigma, n_reps, rng):
    """Simulates reciprocal tapping times MT = a + b*ID + noise. Returns (IDs, MTs)."""
    ids, mts = [], []
    for tD, tW in zip(theta_D, theta_W):
        ID = angular_id(tD, tW)
        for _ in range(n_reps):
            mt = a + b * ID + rng.normal(0, motor_sigma)
            ids.append(ID); mts.append(max(mt, 1e-3))
    return np.array(ids), np.array(mts)


def fit_fitts(ids, mts):
    """A linear regression MT = a + b*ID. Returns (a, b, R^2)."""
    b, a = np.polyfit(ids, mts, 1)
    pred = a + b * ids
    ss_res = np.sum((mts - pred)**2)
    ss_tot = np.sum((mts - mts.mean())**2)
    return a, b, 1.0 - ss_res / ss_tot
