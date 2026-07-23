"""Tests for the motion-to-photon chain.  Call:  python test_latency.py"""
import numpy as np

from head_motion import generate_head_yaw
from latency import (LatencyBudget, ms_to_steps, displayed_pose, predicted_pose,
                     timewarped_pose, angular_error)

FS = 1000
t, TRUTH, VEL = generate_head_yaw(duration_s=20.0, fs=FS, seed=0)


# ----------------------------- basic building blocks -----------------------------
def test_ms_to_steps():
    assert ms_to_steps(20, 1000) == 20
    assert ms_to_steps(11.1, 1000) == 11
    assert ms_to_steps(0, 1000) == 0


def test_budget_sums_and_judges():
    b = LatencyBudget(sensor=1.5, fusion=1.0, app=4.0, render=11.1, scanout=3.0, display=2.0)
    assert np.isclose(b.total_ms, 22.6)
    assert "TOO HIGH" in b.report()
    b2 = LatencyBudget(sensor=1, render=8, scanout=2)
    assert b2.total_ms == 11 and "OK" in b2.report()


# ----------------------------- pure latency -----------------------------
def test_zero_latency_is_error_free():
    d = displayed_pose(TRUTH, 0, FS)
    assert np.array_equal(d, TRUTH)
    assert angular_error(d, TRUTH).max() == 0.0


def test_latency_shifts_by_exactly_the_right_time():
    # at 20 ms the display shows the pose from 20 samples ago (after the settling time)
    d = displayed_pose(TRUTH, 20, FS)
    assert np.allclose(d[100:], TRUTH[80:-20])


def test_more_latency_more_error():
    errors = [angular_error(displayed_pose(TRUTH, L, FS), TRUTH).mean() for L in (5, 11, 20, 50)]
    assert errors == sorted(errors)          # monotonically increasing
    assert errors[0] > 0


# ----------------------------- prediction -----------------------------
def test_prediction_horizon_zero_is_pure_latency():
    p = predicted_pose(TRUTH, VEL, latency_ms=20, horizon_ms=0, fs=FS)
    assert np.array_equal(p, displayed_pose(TRUTH, 20, FS))


def test_prediction_sweet_spot_is_horizon_equals_latency():
    # at a constant angular velocity horizon = latency is exact; here almost exact
    e = {H: angular_error(predicted_pose(TRUTH, VEL, 20, H, FS), TRUTH).mean()
         for H in (0, 10, 20, 30, 40)}
    best = min(e, key=e.get)
    assert best == 20, e
    assert e[20] < 0.2 * e[0]                 # clearly better than without prediction


def test_prediction_is_u_shaped():
    # too much prediction is worse again -> a U shape
    e = {H: angular_error(predicted_pose(TRUTH, VEL, 20, H, FS), TRUTH).mean()
         for H in (20, 30, 40)}
    assert e[20] < e[30] < e[40]


def test_prediction_overshoots_at_changes_of_direction():
    # prediction is worst where the movement turns (high acceleration)
    accel = np.abs(np.gradient(VEL, 1 / FS))
    turns = accel > np.percentile(accel, 90)
    e = angular_error(predicted_pose(TRUTH, VEL, 20, 20, FS), TRUTH)
    assert e[turns].mean() > 2 * e[~turns].mean(), (e[turns].mean(), e[~turns].mean())


def test_prediction_is_exact_at_constant_velocity():
    # an artificial case: a uniform rotation -> linear extrapolation is EXACT
    n = 2000
    truth = np.arange(n) * 0.1          # 0.1 degrees per sample = a constant velocity
    vel = np.gradient(truth, 1 / FS)
    p = predicted_pose(truth, vel, latency_ms=20, horizon_ms=20, fs=FS)
    assert angular_error(p, truth)[100:].max() < 1e-6


# ----------------------------- timewarp -----------------------------
def test_timewarp_reduces_to_the_warp_latency():
    # orientational timewarp: the effective latency = the warp latency, independent of the
    # render latency
    tw = timewarped_pose(TRUTH, render_latency_ms=50, warp_latency_ms=2, fs=FS)
    assert np.array_equal(tw, displayed_pose(TRUTH, 2, FS))


def test_timewarp_is_independent_of_the_render_latency():
    a = timewarped_pose(TRUTH, render_latency_ms=20, warp_latency_ms=2, fs=FS)
    b = timewarped_pose(TRUTH, render_latency_ms=50, warp_latency_ms=2, fs=FS)
    assert np.array_equal(a, b)         # for a pure rotation only the warp latency counts


def test_timewarp_beats_pure_latency():
    e_without = angular_error(displayed_pose(TRUTH, 20, FS), TRUTH).mean()
    e_warp = angular_error(timewarped_pose(TRUTH, 20, 2, FS), TRUTH).mean()
    assert e_warp < 0.2 * e_without


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests ...")
    for name, tf in tests:
        tf(); print(f"  {name} ... OK")
    print("All tests passed.")
