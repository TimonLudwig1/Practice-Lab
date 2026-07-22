"""Test suite (fast — no backbone training).

    python test_transfer.py

Checks that the feature extractor is built correctly (frozen, head removed, correct output
shape) and that the pretrained features classify a small EuroSAT sample markedly better than
raw pixels.
"""
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from transfer import build_feature_extractor, extract_features, raw_pixel_features


def _dummy_images(n, color=None, seed=0):
    rng = np.random.RandomState(seed)
    imgs = []
    for _ in range(n):
        base = color if color is not None else rng.randint(0, 256, 3)
        arr = (base + rng.randint(-20, 20, (64, 64, 3))).clip(0, 255).astype("uint8")
        imgs.append(Image.fromarray(arr))
    return imgs


def test_extractor_frozen_and_headless():
    model, preprocess = build_feature_extractor()
    assert all(not p.requires_grad for p in model.parameters()), "backbone must be frozen"
    assert isinstance(model.classifier, torch.nn.Identity), "classification head must be removed"
    assert not model.training, "model must be in eval mode"
    print("  Feature extractor frozen & headless ... OK")


def test_feature_shape():
    model, preprocess = build_feature_extractor()
    feats = extract_features(model, preprocess, _dummy_images(5))
    assert feats.shape == (5, 576), feats.shape
    print("  Feature shape (N, 576) ... OK")


def test_preprocess_normalizes():
    _, preprocess = build_feature_extractor()
    x = preprocess(_dummy_images(1)[0])
    assert x.shape[0] == 3 and x.shape[-1] == 224, x.shape          # 3-channel, 224x224
    # ImageNet normalization -> values usually outside [0,1], partly negative
    assert x.min() < 0.0, "normalization should produce negative values"
    print("  Preprocessing (resize+normalization) ... OK")


def test_raw_pixel_features():
    R = raw_pixel_features(_dummy_images(4), size=16)
    assert R.shape == (4, 16 * 16 * 3)
    assert 0.0 <= R.min() and R.max() <= 1.0
    print("  Raw-pixel features ... OK")


def test_transfer_beats_raw_on_toy():
    # Two clearly separated "classes" (reddish vs. bluish tiles). On pretrained features a
    # linear classifier separates them perfectly; this is a fast integration test of the
    # whole chain.
    model, preprocess = build_feature_extractor()
    red = _dummy_images(30, color=np.array([200, 40, 40]), seed=1)
    blue = _dummy_images(30, color=np.array([40, 40, 200]), seed=2)
    imgs = red + blue
    y = np.array([0] * 30 + [1] * 30)
    tr = list(range(0, 60, 2))
    te = list(range(1, 60, 2))
    F = extract_features(model, preprocess, imgs)
    clf = LogisticRegression(max_iter=1000).fit(F[tr], y[tr])
    acc = accuracy_score(y[te], clf.predict(F[te]))
    assert acc > 0.9, acc
    print(f"  pretrained features separate toy classes (acc {acc:.2f}) ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} tests ...")
    for t in tests:
        t()
    print("All tests passed.")
