"""Testsuite P03-final (Segmentierung). pytest fehlt -> __main__-Runner.
    /Users/.../.venv/bin/python test_pointcloud_seg.py
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from pointcloud_seg import (make_scene, fit_plane_3pts, ransac_plane,
                          ransac_iterations, euclidean_clustering)


def _angle(u, v):
    c = np.clip(abs(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))), -1, 1)
    return np.rad2deg(np.arccos(c))


def test_fit_plane_3pts():
    # drei Punkte in der xy-Ebene -> Normale +-z, d=0
    n, d = fit_plane_3pts(np.array([0, 0, 0.]), np.array([1, 0, 0.]), np.array([0, 1, 0.]))
    assert abs(abs(n[2]) - 1.0) < 1e-9 and abs(d) < 1e-9
    # kollineare Punkte -> None
    n2, d2 = fit_plane_3pts(np.array([0, 0, 0.]), np.array([1, 1, 1.]), np.array([2, 2, 2.]))
    assert n2 is None


def test_ransac_iterations_formula():
    # w=0.5, s=3, p=0.99 -> N ~ 34-35
    N = ransac_iterations(0.5, 3, 0.99)
    assert 33 < N < 36
    # sinkt w, steigt N stark
    assert ransac_iterations(0.3, 3, 0.99) > ransac_iterations(0.5, 3, 0.99)


def test_ransac_finds_ground():
    pts, labels, true_n = make_scene(seed=0)
    n, d, mask, n_in = ransac_plane(pts, tau=0.02, iters=300, seed=1)
    # gefundene Normale ~ wahre Bodennormale
    assert _angle(n, true_n) < 2.0
    # hoher Boden-Recall und -Precision
    assert np.mean(mask[labels == 0]) > 0.9
    assert np.mean(labels[mask] == 0) > 0.9


def test_euclidean_clustering_separates_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0, 0], 0.05, (100, 3))
    b = rng.normal([2, 0, 0], 0.05, (100, 3))
    cl = euclidean_clustering(np.vstack([a, b]), eps=0.2, min_size=20)
    assert len(set(cl[cl >= 0])) == 2
    # innerhalb eines Blobs gleiches Label
    assert len(set(cl[:100])) == 1 and len(set(cl[100:])) == 1


def test_full_pipeline_ari():
    pts, labels, true_n = make_scene(seed=0, n_objects=4)
    _, _, plane_mask, _ = ransac_plane(pts, tau=0.02, iters=300, seed=1)
    obj_idx = np.where(~plane_mask)[0]
    cl = euclidean_clustering(pts[obj_idx], eps=0.12, min_size=25)
    lo = labels[obj_idx]; m = lo > 0
    assert adjusted_rand_score(lo[m], cl[m]) > 0.9
    assert len(set(cl[cl >= 0])) == 4


def test_minsize_controls_noise_clusters():
    pts, labels, true_n = make_scene(seed=1, n_objects=4, outlier_frac=0.3)
    _, _, plane_mask, _ = ransac_plane(pts, tau=0.02, iters=300, seed=1)
    obj = pts[~plane_mask]
    n_small = len(set(euclidean_clustering(obj, eps=0.12, min_size=5)) - {-1})
    n_big = len(set(euclidean_clustering(obj, eps=0.12, min_size=30)) - {-1})
    assert n_small > n_big    # kleines min_size -> mehr (Rausch-)Cluster


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
