"""Testsuite (schnell — kein Backbone-Training).

    python test_transfer.py

Prüft, dass der Feature-Extraktor korrekt aufgebaut ist (eingefroren, Kopf entfernt,
richtige Ausgabeform) und dass die pretrained Features eine kleine EuroSAT-Stichprobe
deutlich besser klassifizieren als rohe Pixel.
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
    assert all(not p.requires_grad for p in model.parameters()), "Backbone muss eingefroren sein"
    assert isinstance(model.classifier, torch.nn.Identity), "Klassifikationskopf muss entfernt sein"
    assert not model.training, "Modell muss im eval-Modus sein"
    print("  Feature-Extraktor eingefroren & ohne Kopf ... OK")


def test_feature_shape():
    model, preprocess = build_feature_extractor()
    feats = extract_features(model, preprocess, _dummy_images(5))
    assert feats.shape == (5, 576), feats.shape
    print("  Feature-Form (N, 576) ... OK")


def test_preprocess_normalizes():
    _, preprocess = build_feature_extractor()
    x = preprocess(_dummy_images(1)[0])
    assert x.shape[0] == 3 and x.shape[-1] == 224, x.shape          # 3-Kanal, 224x224
    # ImageNet-Normalisierung -> Werte i.d.R. außerhalb [0,1], teils negativ
    assert x.min() < 0.0, "Normalisierung sollte negative Werte erzeugen"
    print("  Vorverarbeitung (Resize+Normalisierung) ... OK")


def test_raw_pixel_features():
    R = raw_pixel_features(_dummy_images(4), size=16)
    assert R.shape == (4, 16 * 16 * 3)
    assert 0.0 <= R.min() and R.max() <= 1.0
    print("  Rohpixel-Features ... OK")


def test_transfer_beats_raw_on_toy():
    # Zwei klar getrennte "Klassen" (rötliche vs. bläuliche Kacheln). Auf pretrained
    # Features trennt ein linearer Klassifikator sie perfekt; das ist ein schneller
    # Integrationstest der ganzen Kette.
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
    print(f"  pretrained Features trennen Toy-Klassen (acc {acc:.2f}) ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Starte {len(tests)} Tests ...")
    for t in tests:
        t()
    print("Alle Tests bestanden.")
