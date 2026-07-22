"""Frequency domain & denoising — demonstration (given).

    python run.py            # shows spectrum, low-/high-pass, convolution theorem, denoising
    python run.py --save     # additionally save figures as PNG into results/

All NumPy/SciPy, CPU, seconds.
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
    print(f"Image {img.shape}")

    # --- convolution theorem: circular convolution == multiplication in the frequency domain ---
    K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float); K /= K.sum()
    spatial = circular_convolve(img, K)
    freq = apply_frequency_filter(img, kernel_to_mask(K, img.shape))
    print(f"Convolution theorem: max|spatial - freq| = {np.abs(spatial - freq).max():.2e}  (≈0 expected)")

    # --- low-/high-pass ---
    lp = gaussian_lowpass_mask(img.shape, 0.05)
    low = apply_frequency_filter(img, lp)
    high = apply_frequency_filter(img, 1 - lp)
    print(f"Low-pass std {low.std():.1f} (smooth) | high-pass std {high.std():.1f}, mean {high.mean():.2f} (edges)")

    # --- denoising: Gaussian vs. median against two noise types ---
    print("\nDenoising (PSNR in dB, higher = better):")
    for name, noisy in [("Gaussian noise", add_gaussian_noise(img, 25)),
                        ("Salt&pepper", add_salt_pepper(img, 0.05))]:
        g = denoise_gaussian(noisy, 1.2)
        m = denoise_median(noisy, 3)
        print(f"  {name:20} | noisy {psnr(img, noisy):5.1f} | "
              f"Gaussian {psnr(img, g):5.1f} | median {psnr(img, m):5.1f}")
    print("\n-> Gaussian and median help similarly with Gaussian noise; on salt&pepper")
    print("   the median wins clearly (robust against outliers, edge-preserving).")

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        for a, im, t in zip(ax, [img, spectrum(img), low, high],
                            ["Original", "Spectrum (log)", "Low-pass", "High-pass"]):
            a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
        plt.tight_layout(); plt.savefig("results/frequency.png", dpi=90)
        print("\nFigure saved: results/frequency.png")


if __name__ == "__main__":
    main()
