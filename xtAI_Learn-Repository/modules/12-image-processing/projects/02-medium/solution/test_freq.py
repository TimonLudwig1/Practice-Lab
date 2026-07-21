"""Testsuite (schnell — kleine Bilder).

    python test_freq.py
"""
import numpy as np

from imaging import circular_convolve, kernel_to_mask, add_salt_pepper
from frequency import gaussian_lowpass_mask, apply_frequency_filter, spectrum
from denoise import psnr, denoise_gaussian, denoise_median

rng = np.random.RandomState(0)


def test_convolution_theorem():
    # zirkuläre Faltung im Ortsraum == Multiplikation mit der Kernel-FFT im Frequenzraum
    img = rng.rand(32, 40) * 255
    K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float); K /= K.sum()
    spatial = circular_convolve(img, K)
    freq = apply_frequency_filter(img, kernel_to_mask(K, img.shape))
    assert np.abs(spatial - freq).max() < 1e-8, np.abs(spatial - freq).max()
    print("  Faltungssatz (zirkulär) ... OK")


def test_lowpass_smooths():
    img = rng.rand(48, 48) * 255
    mask = gaussian_lowpass_mask(img.shape, 0.05)
    low = apply_frequency_filter(img, mask)
    assert low.std() < img.std(), "Tiefpass muss die Varianz senken (glätten)"
    # DC-Wert (Mittel) bleibt erhalten
    assert abs(low.mean() - img.mean()) < 1e-6
    print("  Tiefpass glättet & erhält den Mittelwert ... OK")


def test_highpass_removes_dc():
    img = rng.rand(48, 48) * 255
    mask = gaussian_lowpass_mask(img.shape, 0.05)
    high = apply_frequency_filter(img, 1 - mask)     # Hochpass
    assert abs(high.mean()) < 1e-6, "Hochpass muss den DC-Anteil (Mittel) entfernen"
    print("  Hochpass entfernt den DC-Anteil ... OK")


def test_lowpass_mask_props():
    mask = gaussian_lowpass_mask((30, 30), 0.05)
    assert abs(mask[0, 0] - 1.0) < 1e-9, "Maske muss bei DC (Index 0,0) = 1 sein"
    assert mask.min() >= 0 and mask.max() <= 1.0
    print("  Tiefpass-Maske in [0,1], DC=1 ... OK")


def test_psnr():
    x = rng.rand(20, 20) * 255
    assert psnr(x, x) == float("inf")
    noisy = x + rng.normal(0, 10, x.shape)
    assert 20 < psnr(x, noisy) < 40
    print("  PSNR ... OK")


def test_median_beats_gaussian_on_saltpepper():
    # strukturiertes (glattes) Bild — nur dann zeigt der Median seinen Vorteil:
    # er entfernt Ausreißer und erhält die glatte Struktur, der Gauß verschmiert sie.
    img = np.tile(np.linspace(30, 220, 64), (64, 1))       # glatter Verlauf
    sp = add_salt_pepper(img, 0.06, seed=1)
    assert psnr(img, denoise_median(sp, 3)) > psnr(img, denoise_gaussian(sp, 1.2))
    print("  Median schlägt Gauß bei Salz&Pfeffer ... OK")


def test_spectrum_shape():
    s = spectrum(rng.rand(16, 24) * 255)
    assert s.shape == (16, 24) and np.all(np.isfinite(s))
    print("  Spektrum-Form ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Starte {len(tests)} Tests ...")
    for t in tests:
        t()
    print("Alle Tests bestanden.")
