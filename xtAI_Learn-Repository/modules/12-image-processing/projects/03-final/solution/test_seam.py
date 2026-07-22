"""Test suite (fast).

    python test_seam.py
"""
import numpy as np

from seam_carving import energy, cumulative_energy, find_seam, remove_seam, carve


def test_cumulative_dp_known():
    e = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float)
    M = cumulative_energy(e)
    expected = np.array([[1, 2, 3], [5, 6, 8], [12, 13, 15]], float)
    assert np.array_equal(M, expected), M
    assert M[-1].min() == 12.0
    print("  DP cumulative energy (known example) ... OK")


def test_seam_is_connected():
    rng = np.random.RandomState(0)
    M = cumulative_energy(rng.rand(30, 25))
    seam = find_seam(M)
    assert len(seam) == 30
    assert np.all(np.abs(np.diff(seam)) <= 1), "the seam must be connected (|delta|<=1)"
    assert seam.min() >= 0 and seam.max() < 25
    print("  Seam is connected & inside the image ... OK")


def test_remove_seam_shrinks_width():
    img = np.zeros((10, 8, 3), np.uint8)
    seam = np.arange(10) % 8
    out = remove_seam(img, seam)
    assert out.shape == (10, 7, 3)
    print("  remove_seam shrinks the width by 1 ... OK")


def test_carve_reduces_width():
    img = np.random.RandomState(1).rand(20, 30, 3) * 255
    out = carve(img, 5)
    assert out.shape == (20, 25, 3)
    print("  carve removes N seams ... OK")


def test_carve_removes_low_energy_region():
    # an image with a clearly low-energy (constant) column region in the middle surrounded
    # by a high-frequency border: seam carving must first remove the smooth middle.
    rng = np.random.RandomState(2)
    img = rng.rand(40, 60, 3) * 255                 # structure everywhere
    img[:, 25:35, :] = 128.0                        # a smooth region (low energy)
    e_before = energy(img)[:, 25:35].mean()
    out = carve(img, 8)
    # after removing the smooth region the remaining mean energy is higher
    assert out.shape[1] == 52
    assert energy(out).mean() > energy(img).mean()  # the "more important" material remained
    print("  carve removes low-energy regions first ... OK")


def test_energy_shape():
    assert energy(np.random.rand(12, 15)).shape == (12, 15)
    assert energy(np.random.rand(12, 15, 3)).shape == (12, 15)
    print("  Energy shape (gray & color) ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} tests ...")
    for t in tests:
        t()
    print("All tests passed.")
