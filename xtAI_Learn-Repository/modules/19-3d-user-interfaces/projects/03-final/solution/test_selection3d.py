"""Test suite P03-final. pytest is missing -> a __main__ runner.
    /Users/.../.venv/bin/python test_selection3d.py
"""
import numpy as np
from selection3d import (angle_between, ray_sphere_t, make_trial, noisy_ray_dir,
                        run_technique, select_raycast, select_cone, select_bubble)
from stats_tools import rank_biserial_from_diffs, holm_bonferroni, _rankdata

ORIGIN = np.zeros(3)


def _acc(L, nd, spread, occ, sig_deg, n=800, seed=0, r=0.12):
    rng = np.random.default_rng(seed); sig = np.deg2rad(sig_deg)
    hit = {t: 0 for t in ["raycast", "cone", "bubble"]}
    for _ in range(n):
        pos, radii, tgt, base = make_trial(rng, L_target=L, n_distractors=nd,
                                           spread_deg=spread, r=r, occlude_frac=occ)
        d = noisy_ray_dir(base, sig, rng)
        for t in hit:
            hit[t] += (run_technique(t, pos, radii, ORIGIN, d) == tgt)
    return {t: hit[t] / n for t in hit}


def test_geometry_helpers():
    assert abs(angle_between([1, 0, 0], [0, 1, 0]) - np.pi / 2) < 1e-9
    assert abs(angle_between([1, 0, 0], [1, 0, 0])) < 1e-9
    # a ray along +x hits the sphere around (5,0,0), R=1 at t=4
    assert abs(ray_sphere_t(ORIGIN, np.array([1., 0, 0]), np.array([5., 0, 0]), 1.0) - 4.0) < 1e-9
    assert ray_sphere_t(ORIGIN, np.array([0., 1, 0]), np.array([5., 0, 0]), 1.0) is None


def test_trial_structure():
    rng = np.random.default_rng(0)
    pos, radii, tgt, base = make_trial(rng, L_target=4, n_distractors=5)
    assert tgt == 0 and pos.shape == (6, 3) and radii.shape == (6,)
    assert abs(np.linalg.norm(pos[0]) - 4.0) < 1e-6      # the target at distance L
    assert abs(np.linalg.norm(base) - 1.0) < 1e-9        # the direction is normalized


def test_perfect_aim_selects_target():
    # without noise and distractors every technique has to hit the target
    rng = np.random.default_rng(1)
    pos, radii, tgt, base = make_trial(rng, L_target=3, n_distractors=0)
    for t in ["raycast", "cone", "bubble"]:
        assert run_technique(t, pos, radii, ORIGIN, base) == tgt


def test_isolated_all_high():
    acc = _acc(L=1.5, nd=1, spread=20, occ=0.0, sig_deg=1.5)
    for t, v in acc.items():
        assert v > 0.9, f"{t} only {v:.3f} on an isolated target"


def test_raycasting_falls_with_distance():
    near = _acc(L=2, nd=2, spread=16, occ=0.15, sig_deg=1.5)["raycast"]
    far = _acc(L=16, nd=2, spread=16, occ=0.15, sig_deg=1.5)["raycast"]
    assert near > far + 0.3, f"ray-casting should fall with distance ({near:.3f}->{far:.3f})"


def test_volume_beats_raycast_far():
    acc = _acc(L=8, nd=3, spread=14, occ=0.2, sig_deg=1.5)
    assert acc["cone"] > acc["raycast"] + 0.3
    assert acc["bubble"] > acc["raycast"] + 0.3


def test_bubble_overselects_in_dense():
    acc = _acc(L=2, nd=6, spread=6, occ=0.15, sig_deg=1.0)
    assert acc["cone"] > acc["bubble"], f"cone {acc['cone']:.3f} should beat bubble {acc['bubble']:.3f}"


def test_stats_rank_biserial_and_holm():
    # all differences positive -> rank-biserial = +1
    assert abs(rank_biserial_from_diffs([0.1, 0.2, 0.05, 0.3]) - 1.0) < 1e-9
    assert abs(rank_biserial_from_diffs([-0.1, -0.2, -0.05]) + 1.0) < 1e-9
    # Holm: the smallest p * m
    adj = holm_bonferroni([0.01, 0.04, 0.03])
    assert abs(adj[0] - 0.03) < 1e-9 and np.all((adj >= 0) & (adj <= 1))
    # _rankdata with ties
    assert np.allclose(_rankdata([10, 20, 20, 40]), [1, 2.5, 2.5, 4])


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
