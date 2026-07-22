"""Demonstrate seam carving (given in the solution).

    python run.py                 # removes 150 seams, saves a comparison in results/
    python run.py --seams 200

Shows the original, the energy map, the first seam and the content-aware shrunk image.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from PIL import Image

from seam_carving import energy, carve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seams", type=int, default=150)
    args = ap.parse_args()

    with cbook.get_sample_data("grace_hopper.jpg") as f:
        img = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64)
    print(f"Original {img.shape[1]}x{img.shape[0]}, removing {args.seams} seams ...")

    out, (e0, seam0) = carve(img.copy(), args.seams, record=True)
    print(f"Result {out.shape[1]}x{out.shape[0]}  "
          f"(width {img.shape[1]} -> {out.shape[1]})")

    # draw the first seam onto the original
    overlay = img.copy()
    for i, j in enumerate(seam0):
        overlay[i, j] = [255, 0, 0]

    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(16, 5))
    for a, im, t, cmap in [
        (ax[0], img.astype(np.uint8), "Original", None),
        (ax[1], e0, "Energy (gradient)", "inferno"),
        (ax[2], overlay.astype(np.uint8), "first seam (red)", None),
        (ax[3], out.astype(np.uint8), f"content-aware -{args.seams}px", None),
    ]:
        a.imshow(im, cmap=cmap); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig("results/seam_carving.png", dpi=90)
    Image.fromarray(out.astype(np.uint8)).save("results/carved.png")
    print("Saved: results/seam_carving.png, results/carved.png")
    print("Observation: the main subject stays undistorted; the low-energy background")
    print("(flag/area) gets narrower — that is the content-aware effect.")


if __name__ == "__main__":
    main()
