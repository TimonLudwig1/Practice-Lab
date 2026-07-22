"""Load EuroSAT + preprocessing (given).

EuroSAT: 27,000 Sentinel-2 satellite images (64x64 RGB), 10 land use classes (AnnualCrop,
Forest, River, Residential, ...). A completely different domain from ImageNet — ideal for
showing that pretrained features are transferable *nonetheless*.
"""
import os
import numpy as np
from torchvision import datasets

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def load_eurosat(n_train=3000, n_test=1500, seed=0):
    """Loads EuroSAT (downloaded the first time, ~90 MB, fast) and returns reproducible
    train/test subsets as (PIL image, label) lists + class names."""
    ds = datasets.EuroSAT(root=DATA_DIR, download=True)
    idx = np.random.RandomState(seed).permutation(len(ds))
    tr = [ds[int(i)] for i in idx[:n_train]]
    te = [ds[int(i)] for i in idx[n_train:n_train + n_test]]
    return tr, te, ds.classes
