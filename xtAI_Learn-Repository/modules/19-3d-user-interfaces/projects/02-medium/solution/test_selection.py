"""Testsuite P02-medium. pytest fehlt -> __main__-Runner.
    /Users/.../.venv/bin/python test_selection.py
"""
import numpy as np
from selection import (angular_radius, p_hit_raycasting, go_go, go_go_gain,
                      go_go_inverse, p_hit_gogo, angular_id,
                      simulate_fitts_times, fit_fitts)


def test_angular_radius_shrinks_with_distance():
    r = 0.1
    assert angular_radius(r, 1) > angular_radius(r, 10)
    # kleiner Winkel: arctan(r/L) ~ r/L
    assert abs(angular_radius(0.01, 10.0) - 0.001) < 1e-4


def test_p_hit_raycasting_falls_with_distance():
    s = np.deg2rad(1.0)
    p_near = p_hit_raycasting(0.1, 1.0, s)
    p_far = p_hit_raycasting(0.1, 16.0, s)
    assert p_near > 0.9 and p_far < 0.2
    assert 0.0 <= p_far < p_near <= 1.0


def test_go_go_continuity_and_amplification():
    D, k = 0.6, 0.2
    # stetig an D
    left = float(go_go(D - 1e-6, D, k)); right = float(go_go(D + 1e-6, D, k))
    assert abs(left - right) < 1e-4
    # C1: Ableitung an D ist 1 (Gain-Sprungfrei)
    assert abs(float(go_go_gain(D - 1e-6, D, k)) - 1.0) < 1e-6
    assert abs(float(go_go_gain(D + 1e-9, D, k)) - 1.0) < 1e-3
    # jenseits von D verstaerkt (Gain > 1, r_v > r_r)
    assert float(go_go(1.0, D, k)) > 1.0
    assert float(go_go_gain(1.0, D, k)) > 1.0


def test_go_go_inverse_roundtrip():
    D, k = 0.6, 0.2
    for L in [0.3, 0.7, 1.5, 3.0]:
        r_r = float(go_go_inverse(L, D, k))
        assert abs(float(go_go(r_r, D, k)) - L) < 1e-6


def test_virtual_hand_cannot_reach_far():
    # k=0 -> reine Virtual Hand: jenseits der Armlaenge unerreichbar
    assert p_hit_gogo(0.08, 1.5, D=0.6, k=0.0, sigma_r=0.01, arm_length=0.7) == 0.0
    # nah dagegen gut treffbar
    assert p_hit_gogo(0.08, 0.4, D=0.6, k=0.0, sigma_r=0.01, arm_length=0.7) > 0.9


def test_gogo_extends_reach_but_loses_precision():
    D, k, arm, sr, r = 0.45, 60.0, 0.7, 0.005, 0.08
    # Go-Go erreicht ferne Ziele (P>0), Virtual Hand nicht
    assert p_hit_gogo(r, 2.0, D, k, sr, arm) > 0.0
    assert p_hit_gogo(r, 2.0, D, 0.0, sr, arm) == 0.0
    # Praezision faellt mit Streckung: fernes Ziel schlechter als nahes (beide via Go-Go)
    assert p_hit_gogo(r, 3.0, D, k, sr, arm) < p_hit_gogo(r, 0.6, D, k, sr, arm)
    # jenseits der max. Reichweite (voll gestreckt) unerreichbar
    assert p_hit_gogo(r, 5.0, D, k, sr, arm) == 0.0


def test_angular_id_and_fitts_fit():
    # ID steigt, wenn theta_W schrumpft
    assert angular_id(0.5, 0.05) > angular_id(0.5, 0.2)
    # Fitts-Fit gewinnt die Parameter zurueck
    rng = np.random.default_rng(1)
    tD = np.full(5, np.deg2rad(30)); tW = np.array([2*angular_radius(0.1, L) for L in [1,2,4,8,16.]])
    ids, mts = simulate_fitts_times(tD, tW, a=0.15, b=0.20, motor_sigma=0.04, n_reps=60, rng=rng)
    a, b, r2 = fit_fitts(ids, mts)
    assert abs(a - 0.15) < 0.05 and abs(b - 0.20) < 0.05 and r2 > 0.6


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} Tests bestanden.")
