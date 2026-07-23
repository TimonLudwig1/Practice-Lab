"""Test suite P02-medium (control). pytest is missing -> __main__ runner.
    /Users/.../.venv/bin/python test_control.py
"""
import numpy as np
from control import (ArmModel, TRUE, reference, pd_gravity_control, computed_torque_control,
                     simulate, rms_second_half)

KP = 100.0 * np.eye(2)
KD = 20.0 * np.eye(2)


def test_gravity_compensation_holds_still():
    # at rest with e=0, dqd=0: both controllers output exactly gravity -> the arm stays put
    q = np.array([0.4, -0.7])
    for ctrl in (pd_gravity_control, computed_torque_control):
        tau = ctrl(TRUE, q, np.zeros(2), q, np.zeros(2), np.zeros(2), KP, KD)
        assert np.allclose(tau, TRUE.gravity(q), atol=1e-9)


def test_computed_torque_cancels_to_linear_error():
    # feed the computed torque into the true dynamics: ddq should equal ddqd - Kd edot - Kp e
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-2, 2, 2); dq = rng.normal(size=2)
        qd = rng.uniform(-2, 2, 2); dqd = rng.normal(size=2); ddqd = rng.normal(size=2)
        tau = computed_torque_control(TRUE, q, dq, qd, dqd, ddqd, KP, KD)
        ddq = TRUE.forward_dynamics(q, dq, tau)
        e, edot = q - qd, dq - dqd
        assert np.allclose(ddq, ddqd - KD @ edot - KP @ e, atol=1e-9)


def test_regulation_both_reach_setpoint():
    qd = np.array([0.6, -0.8])
    const_ref = lambda t: (qd, np.zeros(2), np.zeros(2))
    for ctrl in (pd_gravity_control, computed_torque_control):
        _, _, _, err = simulate(ctrl, [0, 0], [0, 0], KP, KD, ref_fn=const_ref, T=3.0)
        assert err[-1] < 1e-3, "both controllers must reach a constant setpoint"


def test_computed_torque_beats_pd_on_tracking():
    ref_w = lambda t: reference(t, 3.0)
    _, _, _, ep = simulate(pd_gravity_control, [0, 0], [0, 0], KP, KD, ref_fn=ref_w)
    _, _, _, ec = simulate(computed_torque_control, [0, 0], [0, 0], KP, KD, ref_fn=ref_w)
    rp, rc = rms_second_half(ep), rms_second_half(ec)
    assert rc < rp / 10.0, "computed torque should track far better than PD+gravity"
    assert rc < 1e-2


def test_error_decay_matches_linear_prediction():
    # critically damped: e(t) = e0 (1 + w0 t) exp(-w0 t), w0 = sqrt(Kp)
    e0 = np.array([0.5, -0.5]); qd = np.array([0.3, 0.2]); w0 = 10.0
    const_ref = lambda t: (qd, np.zeros(2), np.zeros(2))
    ts, Q, _, _ = simulate(computed_torque_control, qd + e0, [0, 0], KP, KD,
                           ref_fn=const_ref, T=1.5, dt=1e-4)
    analytic = np.outer((1 + w0 * ts) * np.exp(-w0 * ts), e0)
    assert np.abs((Q - qd) - analytic).max() < 5e-3


def test_model_error_degrades_tracking():
    ref_w = lambda t: reference(t, 2.0)
    _, _, _, e_exact = simulate(computed_torque_control, [0, 0], [0, 0], KP, KD,
                                ref_fn=ref_w, ctrl_model=TRUE)
    wrong = ArmModel(m1=1.5, m2=1.5)
    _, _, _, e_wrong = simulate(computed_torque_control, [0, 0], [0, 0], KP, KD,
                                ref_fn=ref_w, ctrl_model=wrong, plant=TRUE)
    assert rms_second_half(e_wrong) > rms_second_half(e_exact) * 5


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
