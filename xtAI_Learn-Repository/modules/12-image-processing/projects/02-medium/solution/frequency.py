"""Frequency domain: spectrum, low-/high-pass, filter application (script section 3)."""
import numpy as np


def spectrum(img):
    """Log-magnitude spectrum, centered (DC in the middle) — for visualization."""
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(F))


def gaussian_lowpass_mask(shape, cutoff):
    """Gaussian low-pass mask in the frequency domain (DC at the origin, matching `np.fft.fft2`).

    cutoff = the standard deviation in normalized frequencies (0..0.5). Returns: a mask in
    [0,1], 1 at the lowest frequency (DC).
    """
    H, W = shape
    u = np.fft.fftfreq(H)[:, None]      # DC at index 0
    v = np.fft.fftfreq(W)[None, :]
    r2 = u**2 + v**2
    return np.exp(-r2 / (2 * cutoff**2))


def apply_frequency_filter(img, mask):
    """Filter an image with a frequency mask: F = fft2(img); real(ifft2(F*mask))."""
    F = np.fft.fft2(img)
    return np.real(np.fft.ifft2(F * mask))
