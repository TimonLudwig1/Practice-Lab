"""Entrauschen & Metrik (Skript-Abschnitt 2)."""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def psnr(clean, test):
    """Peak Signal-to-Noise Ratio in dB (höher = näher am sauberen Bild)."""
    mse = np.mean((clean.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def denoise_gaussian(img, sigma=1.2):
    """Linearer Gauß-Filter — gut gegen gaußsches Rauschen (verwischt Kanten)."""
    return gaussian_filter(img, sigma)


def denoise_median(img, size=3):
    """Nichtlinearer Median-Filter — gut gegen Salz&Pfeffer (kantenerhaltend)."""
    return median_filter(img, size=size)
