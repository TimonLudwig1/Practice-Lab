"""Testsuite P02-medium (ICP). pytest fehlt -> __main__-Runner.
    /Users/.../.venv/bin/python test_icp.py
"""
import numpy as np
from icp import (kabsch, apply_transform, icp, rotation_angle, make_shape, random_rigid)


def test_kabsch_recovers_known_transform():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(50, 3))
    R_gt, t_gt = random_rigid(angle_deg=40, seed=1)
    Q = apply_transform(P, R_gt, t_gt)
    R, t = kabsch(P, Q)
    assert np.allclose(R, R_gt, atol=1e-9)
    assert np.allclose(t, t_gt, atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9        # echte Rotation, keine Spiegelung


def test_kabsch_is_rotation_not_reflection():
    # Auch bei entarteten/verrauschten Daten muss det(R)=+1 sein
    rng = np.random.default_rng(3)
    P = rng.normal(size=(30, 3)); P[:, 2] *= 0.001    # fast planar
    Q = P + rng.normal(0, 0.5, P.shape)
    R, t = kabsch(P, Q)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_apply_transform_shape():
    pts = np.random.default_rng(1).normal(size=(10, 3))
    R, t = random_rigid(30, seed=2)
    out = apply_transform(pts, R, t)
    assert out.shape == (10, 3)
    # inverse anwenden gibt Original zurueck
    back = apply_transform(out - t, R.T, np.zeros(3))
    assert np.allclose(back, pts, atol=1e-9)


def test_icp_converges_clean():
    target = make_shape(1000, seed=0)
    R_gt, t_gt = random_rigid(angle_deg=30, seed=1, tmag=0.3)
    source = apply_transform(target, R_gt, t_gt)
    R, t, hist = icp(source, target, max_iter=60)
    assert rotation_angle(R @ R_gt) < 2.0, "ICP sollte die Rotation zurueckdrehen"
    assert hist[-1] < 1e-3
    # monoton fallend
    assert all(hist[i] >= hist[i + 1] - 1e-9 for i in range(len(hist) - 1))


def test_icp_basin_small_ok_large_fails():
    # Kleine Fehlstellung konvergiert, sehr grosse (150 deg) rastet im lokalen Minimum ein
    target = make_shape(1000, seed=0)
    # klein: 20 deg -> Erfolg
    Rs_gt, ts_gt = random_rigid(angle_deg=20, seed=6, tmag=0.2)
    R_small, _, _ = icp(apply_transform(target, Rs_gt, ts_gt), target, max_iter=60)
    assert rotation_angle(R_small @ Rs_gt) < 3.0
    # gross: 150 deg -> erwartetes Scheitern (Rest-Fehler bleibt gross)
    Rl_gt, tl_gt = random_rigid(angle_deg=150, seed=4, tmag=0.3)
    R_large, _, _ = icp(apply_transform(target, Rl_gt, tl_gt), target, max_iter=80)
    assert rotation_angle(R_large @ Rl_gt) > 10.0


def test_trimmed_icp_beats_vanilla_with_outliers():
    target = make_shape(1400, seed=0)
    R_gt, t_gt = random_rigid(angle_deg=25, seed=5, tmag=0.3)
    rng = np.random.default_rng(7)
    src = apply_transform(target, R_gt, t_gt) + rng.normal(0, 0.01, target.shape)
    out = apply_transform(rng.uniform(-1.5, 1.5, (200, 3)), R_gt, t_gt)
    src_noisy = np.vstack([src, out])
    Rv, _, _ = icp(src_noisy, target, max_iter=80, trim_ratio=1.0)
    Rt, _, _ = icp(src_noisy, target, max_iter=80, trim_ratio=0.8)
    ev, et = rotation_angle(Rv @ R_gt), rotation_angle(Rt @ R_gt)
    assert et <= ev + 1e-6
    assert et < 5.0


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
