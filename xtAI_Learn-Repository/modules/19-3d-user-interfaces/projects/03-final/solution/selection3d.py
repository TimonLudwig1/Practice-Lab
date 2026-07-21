"""Vergleichende 3D-Selektionsstudie: Ray-Casting vs Cone vs Bubble unter Clutter.
(Referenzloesung P03-final)  Modul 19 — 3D User Interfaces.

Szene: ein Zielobjekt in Distanz L plus N Distraktoren, die es angular umgeben (variable
Dichte). Der Zeiger (Strahl vom Ursprung) hat angulares Motor-Rauschen sigma_theta. Drei
Selektionstechniken:

  RAY-CASTING : selektiere das vom Strahl GESCHNITTENE Objekt mit kleinstem t (naechster Treffer);
                nichts getroffen -> Fehlselektion. Praezise, aber bei kleinen/fernen Zielen
                verfehlt der verrauschte Strahl, und vordere Distraktoren verdecken.
  CONE        : unter allen Objekten im Kegel (Halbwinkel alpha) das mit kleinstem Winkel zur
                Achse. Erleichtert kleine/ferne Ziele, waehlt aber im Gedraenge den falschen.
  BUBBLE      : das Objekt mit kleinstem Winkel zur OBERFLAECHE (angle_to_center - angular_radius);
                faengt immer das angular naechste ein -> top bei duenn, ueber-selektiert bei dicht.

Alles rein geometrisch, CPU-Millisekunden. Generator offengelegt und mit Seed reproduzierbar.
"""
import numpy as np


# ==========================================================================
# Geometrie-Helfer
# ==========================================================================
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def angle_between(u, v):
    """Winkel (rad) zwischen zwei Vektoren."""
    c = np.dot(_unit(u), _unit(v))
    return np.arccos(np.clip(c, -1.0, 1.0))


def ray_sphere_t(o, d, c, R):
    """Distanz t>0 zum vorderen Schnittpunkt Strahl(o,d)/Kugel(c,R), sonst None. (wie P01)"""
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
# Szenen-Generator
# ==========================================================================
def make_trial(rng, L_target=4.0, n_distractors=6, spread_deg=6.0, r=0.12,
               occlude_frac=0.4):
    """Erzeugt eine Szene. Objekt 0 ist das Ziel (Distanz L_target); die Distraktoren
    liegen angular um das Ziel (bis spread_deg), teils NAeHER (Verdeckung).

    Rueckgabe: pos (n,3), radii (n,), target_idx=0, target_dir (3,).
    """
    # Ziel-Richtung: vorwaerts (+x) mit leichter zufaelliger Neigung
    base = _unit([1.0, rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)])
    target_pos = L_target * base
    pos = [target_pos]
    radii = [r]
    # zwei orthonormale Tangentenrichtungen zu base
    tmp = np.array([0.0, 1.0, 0.0]) if abs(base[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = _unit(np.cross(base, tmp))
    e2 = _unit(np.cross(base, e1))
    for _ in range(n_distractors):
        ang = np.deg2rad(rng.uniform(0.3, spread_deg))     # angulare Naehe zum Ziel
        phi = rng.uniform(0, 2 * np.pi)
        offset_dir = _unit(base + np.tan(ang) * (np.cos(phi) * e1 + np.sin(phi) * e2))
        # Distanz: mit Wahrsch. occlude_frac naeher (verdeckend), sonst um L_target herum
        if rng.random() < occlude_frac:
            L = L_target * rng.uniform(0.4, 0.9)
        else:
            L = L_target * rng.uniform(0.9, 1.4)
        pos.append(L * offset_dir)
        radii.append(r)
    return np.array(pos), np.array(radii), 0, base


def noisy_ray_dir(target_dir, sigma_theta, rng):
    """Verrauschte Zeigerichtung: intendierte Richtung + 2D-Gauss-Winkelfehler."""
    base = _unit(target_dir)
    tmp = np.array([0.0, 1.0, 0.0]) if abs(base[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = _unit(np.cross(base, tmp))
    e2 = _unit(np.cross(base, e1))
    ex, ey = rng.normal(0, sigma_theta, 2)
    return _unit(base + np.tan(ex) * e1 + np.tan(ey) * e2)


# ==========================================================================
# Selektionstechniken  (o = Ursprung, hier (0,0,0))
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
        gap = ang - ang_radius            # Winkel zur Oberflaeche (kann negativ = Strahl trifft)
        if gap < best_gap:
            best_i, best_gap = i, gap
    return best_i


TECHNIQUES = {"raycast": select_raycast, "cone": select_cone, "bubble": select_bubble}


def run_technique(name, pos, radii, o, d):
    if name == "cone":
        return select_cone(pos, radii, o, d)
    return TECHNIQUES[name](pos, radii, o, d)


# ==========================================================================
# ISO 9241-9 Throughput  (isolierte Ziele, reciprocal tapping)
# ==========================================================================
def throughput(amplitude_deg, hit_offsets_rad, movement_times):
    """Effektiver Throughput (bits/s) nach ISO 9241-9.

    hit_offsets_rad: angulare Abweichungen der Klicks vom Zielzentrum (1D, entlang Bewegungsachse).
    W_e = 4.133 * std(hit_offsets); D_e = amplitude; ID_e = log2(D_e/W_e + 1); TP = ID_e / mean(MT).
    """
    A = np.deg2rad(amplitude_deg)
    W_e = 4.133 * np.std(hit_offsets_rad)
    ID_e = np.log2(A / W_e + 1.0)
    return ID_e / np.mean(movement_times), ID_e, np.rad2deg(W_e)
