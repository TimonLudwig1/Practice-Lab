"""Test suite P03-final (MPC). pytest is missing -> __main__ runner.
    /Users/.../.venv/bin/python test_mpc.py
"""
import numpy as np
from mpc import (double_integrator, lqr, prediction_matrices, condense, MPC, simulate, lqr_cost)

A, B, Q, R = double_integrator(dt=0.1)
X0 = np.array([2.0, 0.0])


def test_prediction_matrices():
    # X = Sx x0 + Su U must reproduce a hand-rolled forward simulation
    N = 6
    Sx, Su = prediction_matrices(A, B, N)
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=2); U = rng.normal(size=N)
    X = Sx @ x0 + Su @ U
    x = x0.copy()
    for k in range(N):
        x = A @ x + B.flatten() * U[k]
        assert np.allclose(X[2 * k:2 * k + 2], x, atol=1e-12)


def test_hessian_positive_definite():
    H, F, _, _ = condense(A, B, Q, R, Q, N=10)
    assert np.allclose(H, H.T)
    assert np.linalg.eigvalsh(H).min() > 0, "the QP Hessian must be positive definite"


def test_unconstrained_mpc_equals_lqr():
    K, _ = lqr(A, B, Q, R)
    for N in [1, 5, 20]:
        mpc = MPC(A, B, Q, R, N=N, terminal="lqr")     # LQR terminal cost => exact for any N
        u0 = mpc.control(X0).item()
        assert abs(u0 - (-K @ X0).item()) < 1e-8, f"MPC must equal LQR (N={N})"


def test_stage_terminal_converges_to_lqr():
    K, _ = lqr(A, B, Q, R)
    u_lqr = (-K @ X0).item()
    e_short = abs(MPC(A, B, Q, R, N=1, terminal="stage").control(X0).item() - u_lqr)
    e_long = abs(MPC(A, B, Q, R, N=60, terminal="stage").control(X0).item() - u_lqr)
    assert e_long < e_short / 100, "a longer horizon must approach the LQR action"


def test_input_constraint_respected():
    u_max = 0.5
    mpc = MPC(A, B, Q, R, N=25, u_bound=u_max)
    _, us = simulate(mpc.control, A, B, X0, T=60)
    assert np.abs(us).max() <= u_max + 1e-6, "MPC must keep the input within its bound"


def test_state_constraint_respected_where_lqr_violates():
    v_max = 0.6
    K, _ = lqr(A, B, Q, R)
    xs_l, _ = simulate(lambda x: -K @ x, A, B, X0, T=60)
    mpc = MPC(A, B, Q, R, N=25, v_bound=v_max)
    xs_m, _ = simulate(mpc.control, A, B, X0, T=60)
    assert np.abs(xs_l[:, 1]).max() > v_max + 0.1, "LQR should violate the speed limit"
    assert np.abs(xs_m[:, 1]).max() <= v_max + 1e-3, "MPC must respect the speed limit"


def test_constrained_mpc_beats_clipped_lqr_cost():
    u_max = 0.5
    K, _ = lqr(A, B, Q, R)
    xs_l, us_l = simulate(lambda x: -K @ x, A, B, X0, T=80, clip=u_max)
    mpc = MPC(A, B, Q, R, N=25, u_bound=u_max)
    xs_m, us_m = simulate(mpc.control, A, B, X0, T=80)
    assert lqr_cost(xs_m, us_m, Q, R) <= lqr_cost(xs_l, us_l, Q, R) + 1e-6, \
        "constrained MPC should not be worse than naively clipped LQR"


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
