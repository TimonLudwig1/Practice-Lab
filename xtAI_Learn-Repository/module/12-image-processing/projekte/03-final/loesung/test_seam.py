"""Testsuite (schnell).

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
    print("  DP kumulierte Energie (bekanntes Beispiel) ... OK")


def test_seam_is_connected():
    rng = np.random.RandomState(0)
    M = cumulative_energy(rng.rand(30, 25))
    seam = find_seam(M)
    assert len(seam) == 30
    assert np.all(np.abs(np.diff(seam)) <= 1), "Naht muss zusammenhängend sein (|Δ|<=1)"
    assert seam.min() >= 0 and seam.max() < 25
    print("  Naht ist zusammenhängend & im Bild ... OK")


def test_remove_seam_shrinks_width():
    img = np.zeros((10, 8, 3), np.uint8)
    seam = np.arange(10) % 8
    out = remove_seam(img, seam)
    assert out.shape == (10, 7, 3)
    print("  remove_seam verkleinert Breite um 1 ... OK")


def test_carve_reduces_width():
    img = np.random.RandomState(1).rand(20, 30, 3) * 255
    out = carve(img, 5)
    assert out.shape == (20, 25, 3)
    print("  carve entfernt N Nähte ... OK")


def test_carve_removes_low_energy_region():
    # Bild mit einer klar energiearmen (konstanten) Spalten-Region in der Mitte umgeben
    # von hochfrequentem Rand: Seam Carving muss zuerst die glatte Mitte abtragen.
    rng = np.random.RandomState(2)
    img = rng.rand(40, 60, 3) * 255                 # überall Struktur
    img[:, 25:35, :] = 128.0                        # glatte Region (niedrige Energie)
    e_before = energy(img)[:, 25:35].mean()
    out = carve(img, 8)
    # nach dem Abtragen der glatten Region ist die verbleibende mittlere Energie höher
    assert out.shape[1] == 52
    assert energy(out).mean() > energy(img).mean()  # es blieb das „wichtigere" Material
    print("  carve trägt zuerst energiearme Regionen ab ... OK")


def test_energy_shape():
    assert energy(np.random.rand(12, 15)).shape == (12, 15)
    assert energy(np.random.rand(12, 15, 3)).shape == (12, 15)
    print("  Energie-Form (grau & Farbe) ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Starte {len(tests)} Tests ...")
    for t in tests:
        t()
    print("Alle Tests bestanden.")
