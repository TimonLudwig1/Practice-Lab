"""Entrauschen & Metrik (Skript-Abschnitt 2).

Implementiere `psnr`. Die beiden Filter (Gauß/Median via SciPy) sind vorgegeben.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def psnr(clean, test):
    r"""Peak Signal-to-Noise Ratio in dB (höher = näher am sauberen Bild).

    TODO:
      - mse = Mittel der quadratischen Differenz (als float64);
      - falls mse == 0: return float("inf");
      - sonst: return 10 * log10(255**2 / mse).
    """
    raise NotImplementedError("Aufgabe 3: psnr implementieren")


def denoise_gaussian(img, sigma=1.2):
    """Linearer Gauß-Filter — gut gegen gaußsches Rauschen (verwischt Kanten). Vorgegeben."""
    return gaussian_filter(img, sigma)


def denoise_median(img, size=3):
    """Nichtlinearer Median-Filter — gut gegen Salz&Pfeffer (kantenerhaltend). Vorgegeben."""
    return median_filter(img, size=size)
