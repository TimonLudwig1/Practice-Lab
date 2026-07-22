"""Frequency domain: spectrum, low-/high-pass, filter application (script section 3).

Implement `gaussian_lowpass_mask` and `apply_frequency_filter`. `spectrum` (visualization)
is given. Check with `python test_freq.py`.
"""
import numpy as np


def spectrum(img):
    """Log-magnitude spectrum, centered (DC in the middle) — given."""
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(F))


def gaussian_lowpass_mask(shape, cutoff):
    r"""Gaussian low-pass mask in the frequency domain (DC at the origin, matching `np.fft.fft2`).

    TODO:
      - H, W = shape;  u = np.fft.fftfreq(H)[:, None];  v = np.fft.fftfreq(W)[None, :]
        (fftfreq puts the DC component at index 0 — exactly like fft2);
      - r2 = u**2 + v**2;
      - return np.exp(-r2 / (2 * cutoff**2))   # mask in [0,1], 1 at DC
    """
    raise NotImplementedError("Task 1: implement gaussian_lowpass_mask")


def apply_frequency_filter(img, mask):
    """Filter an image with a frequency mask.

    TODO:
      - F = np.fft.fft2(img)
      - return np.real(np.fft.ifft2(F * mask))
    """
    raise NotImplementedError("Task 2: implement apply_frequency_filter")
