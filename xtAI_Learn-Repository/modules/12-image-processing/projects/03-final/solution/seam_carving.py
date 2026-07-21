"""Musterlösung — Seam Carving (content-aware Image Resizing).

Verkleinert ein Bild in der Breite, ohne wichtige Inhalte zu verzerren: es entfernt
iterativ **energiearme senkrechte Nähte** (zusammenhängende Pixelpfade geringster
Gradienten-Energie). Die optimale Naht findet **dynamische Programmierung** (Skript 6;
DP wie in Modul 06/07). Reines NumPy, CPU, Sekunden.
"""
import numpy as np


def energy(img):
    """Energie-Karte = summierter Gradientenbetrag über die Farbkanäle (L1-Gradient).

    img: (H, W) oder (H, W, C), float. Rückgabe: (H, W) mit „Wichtigkeit" pro Pixel.
    """
    if img.ndim == 2:
        gy, gx = np.gradient(img)
        return np.abs(gx) + np.abs(gy)
    e = np.zeros(img.shape[:2], dtype=np.float64)
    for c in range(img.shape[2]):
        gy, gx = np.gradient(img[:, :, c].astype(np.float64))
        e += np.abs(gx) + np.abs(gy)
    return e


def cumulative_energy(e):
    r"""Kumulierte Minimalenergie per DP (von oben nach unten):
    M(i,j) = e(i,j) + min(M(i-1,j-1), M(i-1,j), M(i-1,j+1)).
    Ränder mit +inf abgesichert. Rückgabe: (H, W)-Matrix M.
    """
    H, W = e.shape
    M = e.astype(np.float64).copy()
    for i in range(1, H):
        left = np.concatenate(([np.inf], M[i - 1, :-1]))
        up = M[i - 1]
        right = np.concatenate((M[i - 1, 1:], [np.inf]))
        M[i] += np.minimum(np.minimum(left, up), right)
    return M


def find_seam(M):
    """Backtracking der billigsten Naht: Start am Minimum der letzten Zeile, nach oben je
    zum kleinsten der (bis zu) drei oberen Nachbarn. Rückgabe: (H,) Spaltenindizes."""
    H, W = M.shape
    seam = np.empty(H, dtype=int)
    seam[-1] = int(np.argmin(M[-1]))
    for i in range(H - 2, -1, -1):
        j = seam[i + 1]
        lo, hi = max(0, j - 1), min(W, j + 2)
        seam[i] = lo + int(np.argmin(M[i, lo:hi]))
    return seam


def remove_seam(img, seam):
    """Entfernt die Naht (ein Pixel pro Zeile) → Breite um 1 kleiner."""
    H, W = img.shape[:2]
    if img.ndim == 3:
        out = np.empty((H, W - 1, img.shape[2]), dtype=img.dtype)
        for i in range(H):
            out[i] = np.delete(img[i], seam[i], axis=0)
    else:
        out = np.empty((H, W - 1), dtype=img.dtype)
        for i in range(H):
            out[i] = np.delete(img[i], seam[i])
    return out


def carve(img, num_seams, record=False):
    """Entfernt `num_seams` senkrechte Nähte nacheinander (Energie wird jeweils neu
    berechnet). record=True gibt zusätzlich die erste Naht + Energie zurück (für Viz)."""
    first = None
    for k in range(num_seams):
        e = energy(img)
        M = cumulative_energy(e)
        seam = find_seam(M)
        if record and k == 0:
            first = (e.copy(), seam.copy())
        img = remove_seam(img, seam)
    return (img, first) if record else img
