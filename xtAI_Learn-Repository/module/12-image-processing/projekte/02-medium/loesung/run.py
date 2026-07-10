"""Frequenzraum & Entrauschen — Demonstration (vorgegeben).

    python run.py            # zeigt Spektrum, Tief-/Hochpass, Faltungssatz, Entrauschen
    python run.py --save     # zusätzlich Abbildungen als PNG in ergebnisse/ speichern

Alles NumPy/SciPy, CPU, Sekunden.
"""
import argparse
import os
import numpy as np

from imaging import (load_gray, add_gaussian_noise, add_salt_pepper,
                     circular_convolve, kernel_to_mask)
from frequency import spectrum, gaussian_lowpass_mask, apply_frequency_filter
from denoise import psnr, denoise_gaussian, denoise_median


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    img = load_gray()
    print(f"Bild {img.shape}")

    # --- Faltungssatz: zirkuläre Faltung == Multiplikation im Frequenzraum ---
    K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float); K /= K.sum()
    spatial = circular_convolve(img, K)
    freq = apply_frequency_filter(img, kernel_to_mask(K, img.shape))
    print(f"Faltungssatz: max|spatial - freq| = {np.abs(spatial - freq).max():.2e}  (≈0 erwartet)")

    # --- Tief-/Hochpass ---
    lp = gaussian_lowpass_mask(img.shape, 0.05)
    low = apply_frequency_filter(img, lp)
    high = apply_frequency_filter(img, 1 - lp)
    print(f"Tiefpass std {low.std():.1f} (glatt) | Hochpass std {high.std():.1f}, mean {high.mean():.2f} (Kanten)")

    # --- Entrauschen: Gauß vs. Median gegen zwei Rauscharten ---
    print("\nEntrauschen (PSNR in dB, höher = besser):")
    for name, noisy in [("Gaußsches Rauschen", add_gaussian_noise(img, 25)),
                        ("Salz&Pfeffer", add_salt_pepper(img, 0.05))]:
        g = denoise_gaussian(noisy, 1.2)
        m = denoise_median(noisy, 3)
        print(f"  {name:20} | verrauscht {psnr(img, noisy):5.1f} | "
              f"Gauß {psnr(img, g):5.1f} | Median {psnr(img, m):5.1f}")
    print("\n-> Gauß und Median helfen ähnlich bei gaußschem Rauschen; bei Salz&Pfeffer")
    print("   gewinnt der Median deutlich (robust gegen Ausreißer, kantenerhaltend).")

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("ergebnisse", exist_ok=True)
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        for a, im, t in zip(ax, [img, spectrum(img), low, high],
                            ["Original", "Spektrum (log)", "Tiefpass", "Hochpass"]):
            a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
        plt.tight_layout(); plt.savefig("ergebnisse/frequenz.png", dpi=90)
        print("\nAbbildung gespeichert: ergebnisse/frequenz.png")


if __name__ == "__main__":
    main()
