"""Zeige-Praezision & Reichweite: angulares Fitts' Law und Go-Go.  (Referenzloesung P02-medium)

Modul 19 — 3D User Interfaces.

Wir modellieren zwei Selektionstechniken MECHANISTISCH (nicht per angenommener Fitts-Kurve):

  RAY-CASTING: Ein Ziel (Radius r) in Distanz L subtendiert den angularen Radius
    theta_r ~ arctan(r/L)  -> schrumpft mit L. Das Zeigen hat festes ANGULARES Rauschen
    sigma_theta. Getroffen, wenn der 2D-Winkelfehler < theta_r. Trefferrate faellt mit L.

  GO-GO (Poupyrev 1996): Die virtuelle Hand folgt der realen nichtlinear
    r_v = r_r                    fuer r_r < D
    r_v = r_r + k (r_r - D)^2    fuer r_r >= D
  Reichweite waechst quadratisch -> ferne Ziele erreichbar. ABER: die C/D-Verstaerkung
  g = dr_v/dr_r = 1 + 2k(r_r-D) verstaerkt jenseits von D auch das HAND-Rauschen
    sigma_v = g * sigma_r  -> Praezision faellt im gestreckten Bereich. Das ist der
  eigentliche Go-Go-Trade-off (Reichweite gegen Praezision), hier exakt herausgearbeitet.
"""
import numpy as np


# ==========================================================================
# Angulares Modell (Ray-Casting)
# ==========================================================================
def angular_radius(r, L):
    """Angularer Radius (rad) eines Ziels mit Radius r in Distanz L. arctan(r/L)."""
    return np.arctan2(r, L)


def p_hit_raycasting(r, L, sigma_theta):
    """Trefferwahrscheinlichkeit beim Ray-Casting.

    Der 2D-Winkelfehler (Azimut, Elevation) ist je Achse N(0, sigma_theta^2); sein Betrag
    ist Rayleigh-verteilt. Treffer, wenn Betrag < theta_r = angular_radius(r, L):

        P(hit) = 1 - exp( - theta_r^2 / (2 sigma_theta^2) ).

    theta_r ~ r/L schrumpft mit der Distanz -> P(hit) faellt.
    """
    theta_r = angular_radius(r, L)
    return 1.0 - np.exp(-(theta_r**2) / (2.0 * sigma_theta**2))


# ==========================================================================
# Go-Go
# ==========================================================================
def go_go(r_r, D, k):
    """Nichtlineare Go-Go-Abbildung reale -> virtuelle Handdistanz."""
    r_r = np.asarray(r_r, float)
    return np.where(r_r < D, r_r, r_r + k * (r_r - D)**2)


def go_go_gain(r_r, D, k):
    """C/D-Verstaerkung g = dr_v/dr_r. 1 im Nahbereich, 1+2k(r_r-D) jenseits von D."""
    r_r = np.asarray(r_r, float)
    return np.where(r_r < D, 1.0, 1.0 + 2.0 * k * (r_r - D))


def go_go_inverse(L, D, k):
    """Welche reale Handdistanz r_r erzeugt die virtuelle Reichweite L? (loest r_v=L)

    Fuer L < D:  r_r = L.
    Fuer L >= D: k r_r^2 + (1-2kD) r_r + (kD^2 - L) = 0, die physik. (groessere) Wurzel.
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

    - Reichweite L erfordert reale Handdistanz r_r = go_go_inverse(L). Ist r_r > arm_length,
      ist das Ziel UNERREICHBAR -> P(hit)=0 (das ist der Virtual-Hand-Deckel).
    - Sonst: Fingerspitzen-Rauschen sigma_v = g(r_r)*sigma_r (verstaerkt jenseits D).
      Treffer, wenn 2D-Positionsfehler < r:  P(hit)=1-exp(-r^2/(2 sigma_v^2)).
    """
    r_r = float(go_go_inverse(L, D, k))
    if r_r > arm_length:
        return 0.0
    g = float(go_go_gain(r_r, D, k))
    sigma_v = g * sigma_r
    return 1.0 - np.exp(-(r**2) / (2.0 * sigma_v**2))


# ==========================================================================
# Angulares Fitts' Law (Charakterisierung)
# ==========================================================================
def angular_id(theta_D, theta_W):
    """Index of Difficulty (bits): log2(theta_D/theta_W + 1)."""
    return np.log2(theta_D / theta_W + 1.0)


def simulate_fitts_times(theta_D, theta_W, a, b, motor_sigma, n_reps, rng):
    """Simuliert Bewegungszeiten fuer reciprocal tapping: MT = a + b*ID + Rauschen.
    Gibt (IDs, MTs) fuer alle Wiederholungen zurueck (zur Regression)."""
    ids, mts = [], []
    for tD, tW in zip(theta_D, theta_W):
        ID = angular_id(tD, tW)
        for _ in range(n_reps):
            mt = a + b * ID + rng.normal(0, motor_sigma)
            ids.append(ID); mts.append(max(mt, 1e-3))
    return np.array(ids), np.array(mts)


def fit_fitts(ids, mts):
    """Lineare Regression MT = a + b*ID. Gibt (a, b, R^2)."""
    b, a = np.polyfit(ids, mts, 1)  # polyfit: hoechster Grad zuerst -> steigung, achsenabschnitt
    pred = a + b * ids
    ss_res = np.sum((mts - pred)**2)
    ss_tot = np.sum((mts - mts.mean())**2)
    r2 = 1.0 - ss_res / ss_tot
    return a, b, r2
