"""EuroSAT-Daten für die drei Wege (vorgegeben).

Stellt zwei Sichten auf dieselben Bilder bereit:
  - `load_tensors`   : 64x64-Tensoren (nativ) für das From-Scratch-CNN;
  - `load_pil`       : PIL-Bilder + Labels für Transfer (Vorverarbeitung via Backbone).
Reproduzierbare Train/Test-Teilmengen über einen festen Seed.
"""
import os
import numpy as np
import torch
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), "daten")
_CLASSES = None


def _indices(n_total, n_train, n_test, seed):
    idx = np.random.RandomState(seed).permutation(n_total)
    return idx[:n_train], idx[n_train:n_train + n_test]


def load_tensors(n_train=4000, n_test=2000, seed=0):
    """64x64-Tensoren (C,H,W) in [0,1] + Labels, für das From-Scratch-CNN."""
    global _CLASSES
    ds = datasets.EuroSAT(root=DATA_DIR, download=True, transform=transforms.ToTensor())
    _CLASSES = ds.classes
    tr, te = _indices(len(ds), n_train, n_test, seed)
    Xtr = torch.stack([ds[int(i)][0] for i in tr]); ytr = torch.tensor([ds[int(i)][1] for i in tr])
    Xte = torch.stack([ds[int(i)][0] for i in te]); yte = torch.tensor([ds[int(i)][1] for i in te])
    return (Xtr, ytr), (Xte, yte), ds.classes


def load_pil(n_train=4000, n_test=2000, seed=0):
    """PIL-Bilder + Labels (für Transfer: das Backbone bringt seine eigene Vorverarbeitung)."""
    ds = datasets.EuroSAT(root=DATA_DIR, download=True)
    tr, te = _indices(len(ds), n_train, n_test, seed)
    train = [(ds[int(i)][0], ds[int(i)][1]) for i in tr]
    test = [(ds[int(i)][0], ds[int(i)][1]) for i in te]
    return train, test, ds.classes
