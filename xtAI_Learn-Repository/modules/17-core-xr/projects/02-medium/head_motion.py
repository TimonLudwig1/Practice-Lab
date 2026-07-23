"""Produce realistic head motion — infrastructure, fully given.

We model the yaw rotation of a head (looking left/right) as a sum of several sine components plus
occasional fast "gaze jerks" (saccade-like head turns). That produces the two things that matter
for the latency analysis: high angular velocities (up to ~500 deg/s in reality) AND sharp changes
of direction (which is where prediction fails).

For simplicity the orientation here is a scalar (yaw in degrees). The entire analysis holds just
as well for full 3D orientations — you then interpolate/extrapolate quaternions via SLERP instead
of numbers via linear extrapolation (see project 01). The physics of latency stays identical.
"""
from __future__ import annotations
import numpy as np


def generate_head_yaw(duration_s: float = 20.0, fs: int = 1000, seed: int = 0):
    """Produces a yaw trajectory.

    Returns: (t, angle_deg, velocity_deg_s) — each an array of length duration_s*fs.
    fs = the sampling rate in Hz (1000 = 1 ms resolution, fine enough for latencies of a few ms).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # smooth base motion: several superimposed oscillations
    angle = (30 * np.sin(2 * np.pi * 0.5 * t)
             + 15 * np.sin(2 * np.pi * 1.1 * t + 1.0)
             + 8 * np.sin(2 * np.pi * 2.3 * t + 0.5))

    # a few fast head turns (a half cosine pulse, ~150 ms)
    for _ in range(6):
        t0 = rng.uniform(1, duration_s - 1)
        amp = rng.uniform(-25, 25)
        dur = 0.15
        mask = (t >= t0) & (t < t0 + dur)
        angle[mask] += amp * 0.5 * (1 - np.cos(2 * np.pi * (t[mask] - t0) / dur))

    velocity = np.gradient(angle, 1 / fs)      # deg/s
    return t, angle, velocity
