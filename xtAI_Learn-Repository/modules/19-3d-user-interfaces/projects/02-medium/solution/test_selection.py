"""Test suite P02-medium. pytest is missing -> a __main__ runner.
    /Users/.../.venv/bin/python test_selection.py
"""
import numpy as np
from selection import (angular_radius, p_hit_raycasting, go_go, go_go_gain,
                      go_go_inverse, p_hit_gogo, angular_id,
                      simulate_fitts_times, fit_fitts)


def test_angular_radius_shrinks_with_distance():
    r = 0.1
    assert angular_radius(r, 1) > angular_radius(r, 10)
    # a small angle: arctan(r/L) ~ r/L
    assert abs(angular_radius(0.01, 10.0) - 0.001) < 1e-4


def test_p_hit_raycasting_falls_with_distance():
    s = np.deg2rad(1.0)
    p_near = p_hit_raycasting(0.1, 1.0, s)
    p_far = p_hit_raycasting(0.1, 16.0, s)
    assert p_near > 0.9 and p_far < 0.2
    assert 0.0 <= p_far < p_near <= 1.0


def test_go_go_continuity_and_amplification():
    D, k = 0.6, 0.2
    # continuous at D
    left = float(go_go(D - 1e-6, D, k)); right = float(go_go(D + 1e-6, D, k))
    assert abs(left - right) < 1e-4
    # C1: the derivative at D is 1 (no jump in the gain)
    assert abs(float(go_go_gain(D - 1e-6, D, k)) - 1.0) < 1e-6
    assert abs(float(go_go_gain(D + 1e-9, D, k)) - 1.0) < 1e-3
    # beyond D it amplifies (gain > 1, r_v > r_r)
    assert float(go_go(1.0, D, k)) > 1.0
    assert float(go_go_gain(1.0, D, k)) > 1.0


def test_go_go_inverse_roundtrip():
    D, k = 0.6, 0.2
    for L in [0.3, 0.7, 1.5, 3.0]:
        r_r = float(go_go_inverse(L, D, k))
        assert abs(float(go_go(r_r, D, k)) - L) < 1e-6


def test_virtual_hand_cannot_reach_far():
    # k=0 -> the pure virtual hand: unreachable beyond the arm length
    assert p_hit_gogo(0.08, 1.5, D=0.6, k=0.0, sigma_r=0.01, arm_length=0.7) == 0.0
    # up close, by contrast, easy to hit
    assert p_hit_gogo(0.08, 0.4, D=0.6, k=0.0, sigma_r=0.01, arm_length=0.7) > 0.9


def test_gogo_extends_reach_but_loses_precision():
    D, k, arm, sr, r = 0.45, 60.0, 0.7, 0.005, 0.08
    # Go-Go reaches distant targets (P>0), the virtual hand does not
    assert p_hit_gogo(r, 2.0, D, k, sr, arm) > 0.0
    assert p_hit_gogo(r, 2.0, D, 0.0, sr, arm) == 0.0
    # precision falls with extension: a distant target is worse than a near one (both via Go-Go)
    assert p_hit_gogo(r, 3.0, D, k, sr, arm) < p_hit_gogo(r, 0.6, D, k, sr, arm)
    # beyond the max reach (fully extended) it is unreachable
    assert p_hit_gogo(r, 5.0, D, k, sr, arm) == 0.0


def test_angular_id_and_fitts_fit():
    # ID rises when theta_W shrinks
    assert angular_id(0.5, 0.05) > angular_id(0.5, 0.2)
    # the Fitts fit recovers the parameters
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
    print(f"\n{passed}/{len(tests)} tests passed.")
