"""Test suite P03-final (navigation). pytest is missing -> __main__ runner.
    /Users/.../.venv/bin/python test_navigation.py
"""
import numpy as np
from navigation import (World, default_world, LANDMARKS, rrt, shortcut, densify,
                       path_length, unicycle_step, measure_ranges, ParticleFilter,
                       pure_pursuit, _wrap)

START = np.array([0.5, 0.5])
GOAL = np.array([9.0, 9.0])


def test_world_collision():
    w = World([(5.0, 5.0, 1.0)])
    assert w.in_collision([5.0, 5.0])
    assert not w.in_collision([0.5, 0.5])
    assert w.in_collision([-1.0, 5.0])            # outside the bounds
    # a segment straight through the obstacle is blocked, around it is free
    assert not w.segment_free([2.0, 5.0], [8.0, 5.0])
    assert w.segment_free([2.0, 0.5], [8.0, 0.5])
    # the margin tightens the test
    assert w.segment_free([5.0, 3.5], [5.0, 3.9])
    assert not w.segment_free([5.0, 3.5], [5.0, 3.9], margin=0.6)


def test_rrt_finds_collision_free_path():
    w = default_world()
    path, nodes, parents = rrt(START, GOAL, w, rng=np.random.default_rng(0), margin=0.3)
    assert path is not None, "RRT should find a path in this world"
    assert np.allclose(path[0], START) and np.allclose(path[-1], GOAL)
    for a, b in zip(path[:-1], path[1:]):
        assert w.segment_free(a, b, margin=0.3), "every edge must be collision-free"


def test_shortcut_shortens_and_stays_free():
    w = default_world()
    rng = np.random.default_rng(1)
    path, _, _ = rrt(START, GOAL, w, rng=rng, margin=0.3)
    sm = shortcut(path, w, rng=rng, margin=0.3)
    assert path_length(sm) <= path_length(path) + 1e-9
    assert path_length(sm) < path_length(path)      # should become strictly shorter
    for a, b in zip(sm[:-1], sm[1:]):
        assert w.segment_free(a, b, margin=0.3)


def test_densify_keeps_geometry():
    P = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    D = densify(P, ds=0.1)
    assert abs(path_length(D) - path_length(P)) < 1e-9      # the length is preserved
    assert len(D) > len(P) * 10
    assert np.allclose(D[0], P[0]) and np.allclose(D[-1], P[-1])


def test_unicycle_and_wrap():
    # straight ahead: only x changes
    s = unicycle_step(np.array([0.0, 0.0, 0.0]), 1.0, 0.0, 0.5)
    assert np.allclose(s, [0.5, 0.0, 0.0])
    # pure rotation
    s = unicycle_step(np.array([0.0, 0.0, 0.0]), 0.0, np.pi, 1.0)
    assert abs(abs(s[2]) - np.pi) < 1e-9 and np.allclose(s[:2], [0.0, 0.0])
    # _wrap maps onto a 2pi interval around 0; +pi and -pi are THE SAME angle,
    # hence test on the absolute value.
    assert abs(abs(_wrap(3 * np.pi)) - np.pi) < 1e-9
    assert abs(abs(_wrap(-3 * np.pi)) - np.pi) < 1e-9
    assert abs(_wrap(0.5) - 0.5) < 1e-12
    assert abs(_wrap(2 * np.pi + 0.3) - 0.3) < 1e-12
    assert -np.pi - 1e-12 <= _wrap(123.456) <= np.pi + 1e-12


def test_measure_ranges():
    z = measure_ranges([0.5, 0.5, 0.0], sigma_r=0.0)
    assert np.allclose(z, np.linalg.norm(LANDMARKS - np.array([0.5, 0.5]), axis=1))


def test_particle_filter_converges_to_true_pose():
    rng = np.random.default_rng(0)
    true = np.array([3.0, 4.0, 0.5])
    # the filter starts with an offset; without motion, measurements only -> must lock on
    pf = ParticleFilter(800, true + np.array([0.8, -0.7, 0.4]), rng=rng)
    for _ in range(25):
        pf.predict(0.0, 0.0, 0.1, (0.05, 0.05))
        pf.update(measure_ranges(true, sigma_r=0.25, rng=rng), sigma_r=0.25)
        pf.resample_if_needed()
    assert np.linalg.norm(pf.estimate()[:2] - true[:2]) < 0.35


def test_particle_filter_weights_and_neff():
    rng = np.random.default_rng(2)
    pf = ParticleFilter(100, np.array([1.0, 1.0, 0.0]), rng=rng)
    assert abs(pf.weights.sum() - 1.0) < 1e-12
    assert abs(pf.n_eff() - 100) < 1e-6            # equal weights -> N_eff = M
    pf.update(measure_ranges([1.0, 1.0, 0.0], sigma_r=0.25), sigma_r=0.25)
    assert abs(pf.weights.sum() - 1.0) < 1e-9
    assert pf.n_eff() <= 100 + 1e-9                 # unequal weights -> N_eff drops


def test_pure_pursuit_steers_toward_path_and_clamps():
    path = densify(np.array([[0.0, 0.0], [5.0, 0.0]]), ds=0.1)
    # robot looks up (+90 deg), path goes right -> must turn right (omega<0)
    v, om = pure_pursuit(path, np.array([0.0, 0.0, np.pi / 2]))
    assert om < 0 and abs(om) <= 1.5 + 1e-9
    # exactly on course -> barely any turn, full speed
    v2, om2 = pure_pursuit(path, np.array([0.0, 0.0, 0.0]))
    assert abs(om2) < 1e-6 and v2 > 0.7
    # saturation: an extreme heading error -> |omega| = omega_max
    _, om3 = pure_pursuit(path, np.array([0.0, 0.0, np.pi]))
    assert abs(abs(om3) - 1.5) < 1e-9


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
