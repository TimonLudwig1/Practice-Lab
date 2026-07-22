"""Denoising & metric (script section 2)."""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def psnr(clean, test):
    """Peak signal-to-noise ratio in dB (higher = closer to the clean image)."""
    mse = np.mean((clean.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def denoise_gaussian(img, sigma=1.2):
    """Linear Gaussian filter — good against Gaussian noise (blurs edges)."""
    return gaussian_filter(img, sigma)


def denoise_median(img, size=3):
    """Nonlinear median filter — good against salt&pepper (edge-preserving)."""
    return median_filter(img, size=size)
