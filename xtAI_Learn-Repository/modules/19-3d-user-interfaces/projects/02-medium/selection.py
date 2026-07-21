"""Zeige-Praezision & Reichweite: angulares Fitts' Law und Go-Go.

Modul 19 — 3D User Interfaces.

>>> DEINE AUFGABE <<<
Implementiere die drei mit `# TODO` markierten Kernfunktionen (go_go, p_hit_raycasting,
p_hit_gogo). Der Rest (Gain, Inverse, Fitts-Fit, run.py, Tests) ist vorgegeben. Loesung in
solution/.

Zwei Selektionstechniken, MECHANISTISCH modelliert:

  RAY-CASTING: Ziel (Radius r) in Distanz L subtendiert angularen Radius theta_r ~ arctan(r/L),
    der mit L schrumpft. Zeigen hat festes ANGULARES Rauschen sigma_theta.
    Treffer, wenn 2D-Winkelfehler < theta_r.

  GO-GO: virtuelle Hand folgt real nichtlinear; Reichweite waechst quadratisch, aber die
    C/D-Verstaerkung g=dr_v/dr_r verstaerkt jenseits von D auch das Rauschen (sigma_v=g*sigma_r).
"""
import numpy as np


# ==========================================================================
# Angulares Modell (Ray-Casting)
# ==========================================================================
def angular_radius(r, L):
    """Angularer Radius (rad) eines Ziels Radius r in Distanz L. arctan(r/L).   [vorgegeben]"""
    return np.arctan2(r, L)


def p_hit_raycasting(r, L, sigma_theta):
    """Trefferwahrscheinlichkeit beim Ray-Casting.   >>> DU BIST DRAN <<<

    Der 2D-Winkelfehler ist je Achse N(0, sigma_theta^2); sein Betrag ist Rayleigh-verteilt.
    Treffer, wenn Betrag < theta_r = angular_radius(r, L). Fuer die Rayleigh-Verteilung gilt:

        P(hit) = 1 - exp( - theta_r^2 / (2 * sigma_theta^2) ).
    """
    # TODO: berechne theta_r und gib die Rayleigh-Trefferwahrscheinlichkeit zurueck
    raise NotImplementedError


# ==========================================================================
# Go-Go
# ==========================================================================
def go_go(r_r, D, k):
    """Nichtlineare Go-Go-Abbildung reale -> virtuelle Handdistanz.   >>> DU BIST DRAN <<<

        r_v = r_r                    fuer r_r < D
        r_v = r_r + k (r_r - D)^2    fuer r_r >= D

    Tipp: np.where(...) arbeitet elementweise (r_r kann Skalar oder Array sein).
    """
    r_r = np.asarray(r_r, float)
    # TODO: implementiere die stueckweise Go-Go-Funktion
    raise NotImplementedError


def go_go_gain(r_r, D, k):
    """C/D-Verstaerkung g = dr_v/dr_r. 1 im Nahbereich, 1+2k(r_r-D) jenseits D.   [vorgegeben]"""
    r_r = np.asarray(r_r, float)
    return np.where(r_r < D, 1.0, 1.0 + 2.0 * k * (r_r - D))


def go_go_inverse(L, D, k):
    """Reale Handdistanz r_r, die virtuelle Reichweite L erzeugt (loest r_v=L).   [vorgegeben]

    L<D: r_r=L.  L>=D: k r_r^2 + (1-2kD) r_r + (kD^2 - L)=0, physik. (groessere) Wurzel.
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
    """Trefferwahrscheinlichkeit fuer Go-Go/Virtual-Hand bei Ziel-Radius r in Distanz L.
    >>> DU BIST DRAN <<<

    Schritte:
      1. r_r = go_go_inverse(L, D, k) — reale Handdistanz fuer diese Reichweite.
      2. Ist r_r > arm_length: Ziel UNERREICHBAR -> return 0.0.
      3. Sonst Fingerspitzen-Rauschen sigma_v = go_go_gain(r_r,D,k) * sigma_r,
         Treffer wenn 2D-Positionsfehler < r:  P(hit) = 1 - exp(-r^2 / (2 sigma_v^2)).
    """
    # TODO: implementiere die 3 Schritte
    raise NotImplementedError


# ==========================================================================
# Angulares Fitts' Law (Charakterisierung)   [alles vorgegeben]
# ==========================================================================
def angular_id(theta_D, theta_W):
    """Index of Difficulty (bits): log2(theta_D/theta_W + 1)."""
    return np.log2(theta_D / theta_W + 1.0)


def simulate_fitts_times(theta_D, theta_W, a, b, motor_sigma, n_reps, rng):
    """Simuliert reciprocal-tapping-Zeiten MT = a + b*ID + Rauschen. Gibt (IDs, MTs)."""
    ids, mts = [], []
    for tD, tW in zip(theta_D, theta_W):
        ID = angular_id(tD, tW)
        for _ in range(n_reps):
            mt = a + b * ID + rng.normal(0, motor_sigma)
            ids.append(ID); mts.append(max(mt, 1e-3))
    return np.array(ids), np.array(mts)


def fit_fitts(ids, mts):
    """Lineare Regression MT = a + b*ID. Gibt (a, b, R^2)."""
    b, a = np.polyfit(ids, mts, 1)
    pred = a + b * ids
    ss_res = np.sum((mts - pred)**2)
    ss_tot = np.sum((mts - mts.mean())**2)
    return a, b, 1.0 - ss_res / ss_tot
