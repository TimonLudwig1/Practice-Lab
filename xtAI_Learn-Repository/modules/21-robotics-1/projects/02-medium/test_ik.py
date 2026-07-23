"""Test suite P02-medium (IK). pytest is missing -> __main__ runner.
    /Users/.../.venv/bin/python test_ik.py
"""
import numpy as np
from ik import fk, joints, jacobian, analytic_ik_2link, numeric_ik, nullspace_step

L2 = [1.0, 1.0]
L3 = [0.9, 0.7, 0.5]


def test_fk_matches_closed_form():
    q = np.array([0.7, 1.1])
    closed = np.array([np.cos(q[0]) + np.cos(q[0]+q[1]),
                       np.sin(q[0]) + np.sin(q[0]+q[1])])
    assert np.allclose(fk(q, L2), closed, atol=1e-12)
    # stretched -> (2,0)
    assert np.allclose(fk([0.0, 0.0], L2), [2.0, 0.0], atol=1e-12)


def test_jacobian_matches_numeric():
    for L, q in [(L2, np.array([0.7, 1.1])), (L3, np.array([0.4, 0.6, -0.3]))]:
        Ja = jacobian(q, L)
        h = 1e-6
        cols = []
        for k in range(len(q)):
            e = np.zeros(len(q)); e[k] = h
            cols.append((fk(q+e, L) - fk(q-e, L)) / (2*h))
        assert np.allclose(Ja, np.column_stack(cols), atol=1e-6)


def test_jacobian_determinant_2link():
    q = np.array([0.3, 1.234])
    assert abs(np.linalg.det(jacobian(q, L2)) - L2[0]*L2[1]*np.sin(q[1])) < 1e-12
    # singularity at q2 = 0 and pi
    for q2 in [0.0, np.pi]:
        assert abs(np.linalg.det(jacobian([0.4, q2], L2))) < 1e-12


def test_analytic_ik_reaches_target():
    for tgt in [(1.0, 1.0), (1.5, 0.3), (-0.5, 1.2), (0.2, -1.6)]:
        sols = analytic_ik_2link(tgt, L2)
        assert len(sols) >= 1, f"{tgt} should be reachable"
        for s in sols:
            assert np.allclose(fk(s, L2), tgt, atol=1e-9)


def test_analytic_ik_two_solutions_and_unreachable():
    # generic target -> exactly two solutions with opposite q2
    sols = analytic_ik_2link((1.0, 1.0), L2)
    assert len(sols) == 2
    assert np.sign(sols[0][1]) == -np.sign(sols[1][1])
    # workspace boundary (stretched) -> only one solution
    assert len(analytic_ik_2link((0.0, 2.0), L2)) == 1
    # outside -> none
    assert analytic_ik_2link((2.5, 0.0), L2) == []
    assert analytic_ik_2link((0.0, 0.0), [1.2, 0.5]) == []   # inside the hole


def test_numeric_ik_converges():
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(50):
        r = rng.uniform(0.4, 1.7); th = rng.uniform(-np.pi, np.pi)
        tgt = np.array([r*np.cos(th), r*np.sin(th)])
        q, hist, conv, _ = numeric_ik(tgt, L2, rng.uniform(-np.pi, np.pi, 2),
                                      method="dls", lam=0.05)
        if conv:
            ok += 1
            assert np.linalg.norm(tgt - fk(q, L2)) < 1e-5
    assert ok >= 40, f"DLS should converge most of the time, was {ok}/50"


def test_pinv_explodes_dls_bounded_near_singularity():
    tgt = np.array([0.6, 0.9])
    _, _, _, ms_p = numeric_ik(tgt, L2, np.array([0.2, 1e-4]), method="pinv", max_iter=100)
    _, _, _, ms_d = numeric_ik(tgt, L2, np.array([0.2, 1e-4]), method="dls", lam=0.1,
                               max_iter=100)
    assert ms_p > 1000, f"pinv should explode near the singularity (was {ms_p:.1f})"
    assert ms_d < 50, f"DLS should stay bounded (was {ms_d:.1f})"


def test_nullspace_does_not_move_endeffector():
    q = np.array([0.4, 0.6, -0.3])
    x0 = fk(q, L3)
    rng = np.random.default_rng(2)
    for _ in range(50):
        dq = nullspace_step(q, L3, rng.normal(size=3))
        n = np.linalg.norm(dq)
        if n < 1e-12:
            continue
        dq = 0.01 * dq / n
        assert np.linalg.norm(fk(q + dq, L3) - x0) < 1e-4
    # non-redundant arm (n=m=2): the null space is trivial
    assert np.linalg.norm(nullspace_step(np.array([0.3, 0.8]), L2, np.array([1.0, 1.0]))) < 1e-9


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
