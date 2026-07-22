"""Denoising & metric (script section 2).

Implement `psnr`. The two filters (Gaussian/median via SciPy) are given.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def psnr(clean, test):
    r"""Peak signal-to-noise ratio in dB (higher = closer to the clean image).

    TODO:
      - mse = the mean of the squared difference (as float64);
      - if mse == 0: return float("inf");
      - else: return 10 * log10(255**2 / mse).
    """
    raise NotImplementedError("Task 3: implement psnr")


def denoise_gaussian(img, sigma=1.2):
    """Linear Gaussian filter — good against Gaussian noise (blurs edges). Given."""
    return gaussian_filter(img, sigma)


def denoise_median(img, size=3):
    """Nonlinear median filter — good against salt&pepper (edge-preserving). Given."""
    return median_filter(img, size=size)
