"""Reference solution — seam carving (content-aware image resizing).

Shrinks an image in width without distorting important content: it iteratively removes
**low-energy vertical seams** (connected pixel paths of least gradient energy). The optimal
seam is found by **dynamic programming** (script 6; DP as in modules 06/07). Pure NumPy, CPU,
seconds.
"""
import numpy as np


def energy(img):
    """Energy map = the summed gradient magnitude over the color channels (L1 gradient).

    img: (H, W) or (H, W, C), float. Returns: (H, W) with the "importance" per pixel.
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
    r"""Cumulative minimum energy via DP (top to bottom):
    M(i,j) = e(i,j) + min(M(i-1,j-1), M(i-1,j), M(i-1,j+1)).
    Borders guarded with +inf. Returns: the (H, W) matrix M.
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
    """Backtracking of the cheapest seam: start at the minimum of the last row, upwards to the
    smallest of the (up to) three upper neighbours. Returns: (H,) column indices."""
    H, W = M.shape
    seam = np.empty(H, dtype=int)
    seam[-1] = int(np.argmin(M[-1]))
    for i in range(H - 2, -1, -1):
        j = seam[i + 1]
        lo, hi = max(0, j - 1), min(W, j + 2)
        seam[i] = lo + int(np.argmin(M[i, lo:hi]))
    return seam


def remove_seam(img, seam):
    """Removes the seam (one pixel per row) → the width is smaller by 1."""
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
    """Removes `num_seams` vertical seams one after another (the energy is recomputed each
    time). record=True additionally returns the first seam + energy (for visualization)."""
    first = None
    for k in range(num_seams):
        e = energy(img)
        M = cumulative_energy(e)
        seam = find_seam(M)
        if record and k == 0:
            first = (e.copy(), seam.copy())
        img = remove_seam(img, seam)
    return (img, first) if record else img
