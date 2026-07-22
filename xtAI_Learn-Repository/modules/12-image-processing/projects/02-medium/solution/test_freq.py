"""Test suite (fast — small images).

    python test_freq.py
"""
import numpy as np

from imaging import circular_convolve, kernel_to_mask, add_salt_pepper
from frequency import gaussian_lowpass_mask, apply_frequency_filter, spectrum
from denoise import psnr, denoise_gaussian, denoise_median

rng = np.random.RandomState(0)


def test_convolution_theorem():
    # circular convolution in the spatial domain == multiplication with the kernel FFT in the frequency domain
    img = rng.rand(32, 40) * 255
    K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float); K /= K.sum()
    spatial = circular_convolve(img, K)
    freq = apply_frequency_filter(img, kernel_to_mask(K, img.shape))
    assert np.abs(spatial - freq).max() < 1e-8, np.abs(spatial - freq).max()
    print("  Convolution theorem (circular) ... OK")


def test_lowpass_smooths():
    img = rng.rand(48, 48) * 255
    mask = gaussian_lowpass_mask(img.shape, 0.05)
    low = apply_frequency_filter(img, mask)
    assert low.std() < img.std(), "the low-pass must lower the variance (smooth)"
    # the DC value (mean) is preserved
    assert abs(low.mean() - img.mean()) < 1e-6
    print("  Low-pass smooths & preserves the mean ... OK")


def test_highpass_removes_dc():
    img = rng.rand(48, 48) * 255
    mask = gaussian_lowpass_mask(img.shape, 0.05)
    high = apply_frequency_filter(img, 1 - mask)     # high-pass
    assert abs(high.mean()) < 1e-6, "the high-pass must remove the DC part (mean)"
    print("  High-pass removes the DC part ... OK")


def test_lowpass_mask_props():
    mask = gaussian_lowpass_mask((30, 30), 0.05)
    assert abs(mask[0, 0] - 1.0) < 1e-9, "the mask must be = 1 at DC (index 0,0)"
    assert mask.min() >= 0 and mask.max() <= 1.0
    print("  Low-pass mask in [0,1], DC=1 ... OK")


def test_psnr():
    x = rng.rand(20, 20) * 255
    assert psnr(x, x) == float("inf")
    noisy = x + rng.normal(0, 10, x.shape)
    assert 20 < psnr(x, noisy) < 40
    print("  PSNR ... OK")


def test_median_beats_gaussian_on_saltpepper():
    # a structured (smooth) image — only then does the median show its advantage:
    # it removes outliers and preserves the smooth structure, the Gaussian smears it.
    img = np.tile(np.linspace(30, 220, 64), (64, 1))       # a smooth gradient
    sp = add_salt_pepper(img, 0.06, seed=1)
    assert psnr(img, denoise_median(sp, 3)) > psnr(img, denoise_gaussian(sp, 1.2))
    print("  Median beats Gaussian on salt&pepper ... OK")


def test_spectrum_shape():
    s = spectrum(rng.rand(16, 24) * 255)
    assert s.shape == (16, 24) and np.all(np.isfinite(s))
    print("  Spectrum shape ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} tests ...")
    for t in tests:
        t()
    print("All tests passed.")
