"""Realistische Kopfbewegung erzeugen — Infrastruktur, vollstaendig vorgegeben.

Wir modellieren die yaw-Drehung eines Kopfes (Blick nach links/rechts) als Summe mehrerer
Sinus-Komponenten plus gelegentliche schnelle "Blick-Rucke" (sakkadenartige Kopfwenden). Das
erzeugt die zwei Dinge, auf die es fuer die Latenz-Analyse ankommt: hohe Winkelgeschwindigkeiten
(real bis ~500 deg/s) UND scharfe Richtungswechsel (dort schlaegt Prediction fehl).

Der Einfachheit halber ist die Orientierung hier ein Skalar (yaw in Grad). Die gesamte Analyse
gilt genauso fuer volle 3D-Orientierungen — dann interpoliert/extrapoliert man Quaternionen per
SLERP statt Zahlen per Linear (siehe Projekt 01). Die Physik der Latenz bleibt identisch.
"""
from __future__ import annotations
import numpy as np


def generate_head_yaw(duration_s: float = 20.0, fs: int = 1000, seed: int = 0):
    """Erzeugt eine yaw-Trajektorie.

    Rueckgabe: (t, angle_deg, velocity_deg_s) — jeweils Arrays der Laenge duration_s*fs.
    fs = Abtastrate in Hz (1000 = 1 ms Aufloesung, fein genug fuer Latenzen von wenigen ms).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # glatte Grundbewegung: mehrere ueberlagerte Schwingungen
    angle = (30 * np.sin(2 * np.pi * 0.5 * t)
             + 15 * np.sin(2 * np.pi * 1.1 * t + 1.0)
             + 8 * np.sin(2 * np.pi * 2.3 * t + 0.5))

    # ein paar schnelle Kopfwenden (halber Cosinus-Puls, ~150 ms)
    for _ in range(6):
        t0 = rng.uniform(1, duration_s - 1)
        amp = rng.uniform(-25, 25)
        dur = 0.15
        maske = (t >= t0) & (t < t0 + dur)
        angle[maske] += amp * 0.5 * (1 - np.cos(2 * np.pi * (t[maske] - t0) / dur))

    velocity = np.gradient(angle, 1 / fs)      # deg/s
    return t, angle, velocity
