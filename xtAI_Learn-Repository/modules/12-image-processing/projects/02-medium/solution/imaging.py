"""Bild laden, Rauschen erzeugen, Referenz-Faltung (vorgegeben)."""
import numpy as np
import matplotlib.cbook as cbook
from PIL import Image


def load_gray():
    """Graustufen-Beispielbild (Grace Hopper) als float64-Array in [0,255]."""
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        return np.asarray(Image.open(f).convert("L"), dtype=np.float64)


def add_gaussian_noise(img, sigma=25.0, seed=0):
    rng = np.random.RandomState(seed)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255)


def add_salt_pepper(img, p=0.05, seed=0):
    rng = np.random.RandomState(seed)
    out = img.copy()
    m = rng.rand(*img.shape)
    out[m < p] = 0
    out[m > 1 - p] = 255
    return out


def circular_convolve(img, kernel):
    """*Zirkuläre* 2D-Faltung im Ortsraum (wrap-around) — Referenz für den Faltungssatz.

    Genau für diese Randbedingung gilt der Satz exakt (Multiplikation im Frequenzraum).
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    out = np.zeros_like(img, dtype=np.float64)
    for u in range(kh):
        for v in range(kw):
            out += kernel[u, v] * np.roll(img, (u - pad_h, v - pad_w), axis=(0, 1))
    return out


def kernel_to_mask(kernel, shape):
    """Bettet einen kleinen Kernel in Bildgröße ein (Zentrum im Ursprung) und gibt seine
    FFT zurück — der Frequenz-„Filter", der der zirkulären Faltung mit `kernel` entspricht."""
    kh, kw = kernel.shape
    Kp = np.zeros(shape, dtype=np.float64)
    Kp[:kh, :kw] = kernel
    Kp = np.roll(Kp, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(Kp)
