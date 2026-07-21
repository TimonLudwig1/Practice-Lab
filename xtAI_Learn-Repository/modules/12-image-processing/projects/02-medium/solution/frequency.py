"""Frequenzraum: Spektrum, Tief-/Hochpass, Filteranwendung (Skript-Abschnitt 3)."""
import numpy as np


def spectrum(img):
    """Log-Magnituden-Spektrum, zentriert (DC in der Mitte) — für die Visualisierung."""
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(F))


def gaussian_lowpass_mask(shape, cutoff):
    """Gaußsche Tiefpass-Maske im Frequenzraum (DC im Ursprung, passend zu `np.fft.fft2`).

    cutoff = Standardabweichung in normierten Frequenzen (0..0.5). Rückgabe: Maske in [0,1],
    1 bei der niedrigsten Frequenz (DC).
    """
    H, W = shape
    u = np.fft.fftfreq(H)[:, None]      # DC bei Index 0
    v = np.fft.fftfreq(W)[None, :]
    r2 = u**2 + v**2
    return np.exp(-r2 / (2 * cutoff**2))


def apply_frequency_filter(img, mask):
    """Bild mit einer Frequenz-Maske filtern: F = fft2(img); real(ifft2(F*mask))."""
    F = np.fft.fft2(img)
    return np.real(np.fft.ifft2(F * mask))
