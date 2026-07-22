"""EuroSAT data for the three ways (given).

Provides two views of the same images:
  - `load_tensors`   : 64x64 tensors (native) for the from-scratch CNN;
  - `load_pil`       : PIL images + labels for transfer (preprocessing via the backbone).
Reproducible train/test subsets via a fixed seed.
"""
import os
import numpy as np
import torch
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")
_CLASSES = None


def _indices(n_total, n_train, n_test, seed):
    idx = np.random.RandomState(seed).permutation(n_total)
    return idx[:n_train], idx[n_train:n_train + n_test]


def load_tensors(n_train=4000, n_test=2000, seed=0):
    """64x64 tensors (C,H,W) in [0,1] + labels, for the from-scratch CNN."""
    global _CLASSES
    ds = datasets.EuroSAT(root=DATA_DIR, download=True, transform=transforms.ToTensor())
    _CLASSES = ds.classes
    tr, te = _indices(len(ds), n_train, n_test, seed)
    Xtr = torch.stack([ds[int(i)][0] for i in tr]); ytr = torch.tensor([ds[int(i)][1] for i in tr])
    Xte = torch.stack([ds[int(i)][0] for i in te]); yte = torch.tensor([ds[int(i)][1] for i in te])
    return (Xtr, ytr), (Xte, yte), ds.classes


def load_pil(n_train=4000, n_test=2000, seed=0):
    """PIL images + labels (for transfer: the backbone brings its own preprocessing)."""
    ds = datasets.EuroSAT(root=DATA_DIR, download=True)
    tr, te = _indices(len(ds), n_train, n_test, seed)
    train = [(ds[int(i)][0], ds[int(i)][1]) for i in tr]
    test = [(ds[int(i)][0], ds[int(i)][1]) for i in te]
    return train, test, ds.classes
