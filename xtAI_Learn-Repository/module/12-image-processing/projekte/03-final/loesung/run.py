"""Seam Carving demonstrieren (vorgegeben in der Lösung).

    python run.py                 # trägt 150 Nähte ab, speichert Vergleich in ergebnisse/
    python run.py --seams 200

Zeigt Original, Energie-Karte, die erste Naht und das content-aware verkleinerte Bild.
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
    print(f"Original {img.shape[1]}x{img.shape[0]}, entferne {args.seams} Nähte ...")

    out, (e0, seam0) = carve(img.copy(), args.seams, record=True)
    print(f"Ergebnis {out.shape[1]}x{out.shape[0]}  "
          f"(Breite {img.shape[1]} -> {out.shape[1]})")

    # erste Naht auf dem Original einzeichnen
    overlay = img.copy()
    for i, j in enumerate(seam0):
        overlay[i, j] = [255, 0, 0]

    os.makedirs("ergebnisse", exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(16, 5))
    for a, im, t, cmap in [
        (ax[0], img.astype(np.uint8), "Original", None),
        (ax[1], e0, "Energie (Gradient)", "inferno"),
        (ax[2], overlay.astype(np.uint8), "erste Naht (rot)", None),
        (ax[3], out.astype(np.uint8), f"content-aware −{args.seams}px", None),
    ]:
        a.imshow(im, cmap=cmap); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig("ergebnisse/seam_carving.png", dpi=90)
    Image.fromarray(out.astype(np.uint8)).save("ergebnisse/carved.png")
    print("Gespeichert: ergebnisse/seam_carving.png, ergebnisse/carved.png")
    print("Beobachtung: das Hauptmotiv bleibt unverzerrt; der energiearme Hintergrund")
    print("(Flagge/Fläche) wird schmaler — das ist der content-aware-Effekt.")


if __name__ == "__main__":
    main()
