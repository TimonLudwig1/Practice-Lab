"""EuroSAT laden + Vorverarbeitung (vorgegeben).

EuroSAT: 27 000 Sentinel-2-Satellitenbilder (64x64 RGB), 10 Landnutzungsklassen
(AnnualCrop, Forest, River, Residential, ...). Eine ganz andere Domäne als ImageNet —
ideal, um zu zeigen, dass vortrainierte Merkmale *trotzdem* übertragbar sind.
"""
import os
import numpy as np
from torchvision import datasets

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def load_eurosat(n_train=3000, n_test=1500, seed=0):
    """Lädt EuroSAT (Download beim ersten Mal, ~90 MB, schnell) und gibt reproduzierbare
    Train/Test-Teilmengen als (PIL-Bild, Label)-Listen + Klassennamen zurück."""
    ds = datasets.EuroSAT(root=DATA_DIR, download=True)
    idx = np.random.RandomState(seed).permutation(len(ds))
    tr = [ds[int(i)] for i in idx[:n_train]]
    te = [ds[int(i)] for i in idx[n_train:n_train + n_test]]
    return tr, te, ds.classes
