"""Frequenzraum: Spektrum, Tief-/Hochpass, Filteranwendung (Skript-Abschnitt 3).

Implementiere `gaussian_lowpass_mask` und `apply_frequency_filter`. `spectrum` (Visualisierung)
ist vorgegeben. Prüfe mit `python test_freq.py`.
"""
import numpy as np


def spectrum(img):
    """Log-Magnituden-Spektrum, zentriert (DC in der Mitte) — vorgegeben."""
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(F))


def gaussian_lowpass_mask(shape, cutoff):
    r"""Gaußsche Tiefpass-Maske im Frequenzraum (DC im Ursprung, passend zu `np.fft.fft2`).

    TODO:
      - H, W = shape;  u = np.fft.fftfreq(H)[:, None];  v = np.fft.fftfreq(W)[None, :]
        (fftfreq legt die DC-Komponente auf Index 0 — genau wie fft2);
      - r2 = u**2 + v**2;
      - return np.exp(-r2 / (2 * cutoff**2))   # Maske in [0,1], 1 bei DC
    """
    raise NotImplementedError("Aufgabe 1: gaussian_lowpass_mask implementieren")


def apply_frequency_filter(img, mask):
    """Bild mit einer Frequenz-Maske filtern.

    TODO:
      - F = np.fft.fft2(img)
      - return np.real(np.fft.ifft2(F * mask))
    """
    raise NotImplementedError("Aufgabe 2: apply_frequency_filter implementieren")
