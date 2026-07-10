"""Testsuite (schnell — kein echtes Training).

    python test_cv.py

Prüft die Bausteine der drei Wege: From-Scratch-CNN (Form), Feature-Extraktor (eingefroren,
Form) und das Fine-Tuning-Setup (nur Kopf + letzter Block trainierbar).
"""
import torch
from PIL import Image
import numpy as np

from model import SmallCNN
from transfer import (build_feature_extractor, extract_features,
                      build_finetune_model, FT_PREPROCESS)


def _imgs(n):
    return [Image.fromarray(np.random.RandomState(i).randint(0, 255, (64, 64, 3), dtype="uint8"))
            for i in range(n)]


def test_smallcnn_forward():
    m = SmallCNN(n_classes=10)
    out = m(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 10), out.shape
    print("  SmallCNN-Forward (N,10) ... OK")


def test_smallcnn_learns_step():
    # Ein Gradientenschritt senkt den Loss auf einem festen Mini-Batch.
    torch.manual_seed(0)
    m = SmallCNN(10)
    x = torch.randn(16, 3, 64, 64); y = torch.randint(0, 10, (16,))
    opt = torch.optim.Adam(m.parameters(), 1e-3); crit = torch.nn.CrossEntropyLoss()
    l0 = crit(m(x), y).item()
    for _ in range(5):
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    assert crit(m(x), y).item() < l0
    print("  SmallCNN lernt (Loss sinkt) ... OK")


def test_feature_extractor_frozen():
    fe, preprocess = build_feature_extractor()
    assert all(not p.requires_grad for p in fe.parameters())
    assert not fe.training
    feats = extract_features(fe, preprocess, _imgs(5))
    assert feats.shape == (5, 576), feats.shape
    print("  Feature-Extraktor eingefroren & Form (N,576) ... OK")


def test_finetune_only_head_and_last_block_trainable():
    m = build_finetune_model(n_classes=10)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable, "es muss trainierbare Parameter geben"
    # nur Kopf (classifier) oder letzter Feature-Block (features.12) trainierbar
    assert all(n.startswith("classifier") or n.startswith("features.12") for n in trainable), trainable
    # der Rest ist eingefroren
    frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
    assert any(n.startswith("features.0") for n in frozen)
    print("  Fine-Tune: nur Kopf + letzter Block trainierbar ... OK")


def test_finetune_model_forward():
    m = build_finetune_model(n_classes=10)
    x = torch.stack([FT_PREPROCESS(im) for im in _imgs(3)])
    assert x.shape[-1] == 96
    with torch.no_grad():
        out = m(x)
    assert out.shape == (3, 10)
    print("  Fine-Tune-Modell-Forward (N,10) @96 ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Starte {len(tests)} Tests ...")
    for t in tests:
        t()
    print("Alle Tests bestanden.")
