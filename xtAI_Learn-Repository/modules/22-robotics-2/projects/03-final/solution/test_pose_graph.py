"""Test suite P03-final (pose-graph SLAM). pytest is missing -> __main__ runner.
    /Users/.../.venv/bin/python test_pose_graph.py
"""
import numpy as np
from pose_graph import (wrap, v2t, t2v, edge_error, edge_jacobians,
                       optimize, make_dataset, integrate_odometry, ate)


def test_se2_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = np.array([rng.normal(), rng.normal(), wrap(rng.uniform(-4, 4))])
        assert np.allclose(t2v(v2t(x)), x, atol=1e-12)
    # composition: relative pose of x_j in x_i's frame, then re-composed, returns x_j
    xi = np.array([1.0, 2.0, 0.5]); xj = np.array([3.0, -1.0, -0.8])
    rel = t2v(np.linalg.inv(v2t(xi)) @ v2t(xj))
    assert np.allclose(t2v(v2t(xi) @ v2t(rel)), xj, atol=1e-12)


def test_edge_error_zero_when_consistent():
    xi = np.array([1.0, 0.5, 0.3])
    xj = np.array([2.0, 1.0, -0.4])
    z = t2v(np.linalg.inv(v2t(xi)) @ v2t(xj))     # the exact relative pose
    assert np.allclose(edge_error(xi, xj, z), 0.0, atol=1e-12)


def test_jacobians_match_numeric():
    rng = np.random.default_rng(1)
    for _ in range(30):
        xi = rng.normal(size=3); xj = rng.normal(size=3); z = rng.normal(size=3)
        A, B = edge_jacobians(xi, xj, z)
        eps = 1e-6; An = np.zeros((3, 3)); Bn = np.zeros((3, 3))
        for k in range(3):
            d = np.zeros(3); d[k] = eps
            An[:, k] = (edge_error(xi + d, xj, z) - edge_error(xi - d, xj, z)) / (2 * eps)
            Bn[:, k] = (edge_error(xi, xj + d, z) - edge_error(xi, xj - d, z)) / (2 * eps)
        assert np.abs(A - An).max() < 1e-5
        assert np.abs(B - Bn).max() < 1e-5


def test_optimize_reduces_chi2_monotonically():
    gt, odo, lcs = make_dataset(seed=0)
    x_odo = integrate_odometry(gt[0], odo)
    _, chi = optimize(x_odo, odo + lcs, iters=8)
    assert chi[-1] < chi[0], "chi2 must decrease"
    assert all(chi[i] >= chi[i + 1] - 1e-6 for i in range(len(chi) - 1)), "monotone decrease"


def test_loop_closure_collapses_drift():
    gt, odo, lcs = make_dataset(seed=0)
    x_odo = integrate_odometry(gt[0], odo)
    a_odo = ate(x_odo, gt)
    x_noLC, _ = optimize(x_odo, odo, iters=10)
    x_opt, _ = optimize(x_odo, odo + lcs, iters=10)
    assert abs(ate(x_noLC, gt) - a_odo) < 1e-2, "optimising a pure chain must not change the ATE"
    assert ate(x_opt, gt) < a_odo / 5, "loop closures must collapse the drift"


def test_single_loop_closure_already_helps():
    gt, odo, lcs = make_dataset(seed=0)
    x_odo = integrate_odometry(gt[0], odo)
    x_one, _ = optimize(x_odo, odo + lcs[:1], iters=10)
    assert ate(x_one, gt) < ate(x_odo, gt) / 3, "the first loop closure already corrects most drift"


def test_huber_survives_false_loop_closure():
    gt, odo, lcs = make_dataset(seed=0)
    x_odo = integrate_odometry(gt[0], odo)
    Omega_l = np.diag(1.0 / np.array([0.05, 0.05, 0.02]) ** 2)
    false_lc = (10, 80, np.array([0.2, 0.1, 0.3]), Omega_l)
    x_bad, _ = optimize(x_odo, odo + lcs + [false_lc], iters=10)
    x_rob, _ = optimize(x_odo, odo + lcs + [false_lc], iters=15, huber=2.0)
    assert ate(x_rob, gt) < ate(x_bad, gt) / 3, "the robust kernel must beat naive LS on a false LC"


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
