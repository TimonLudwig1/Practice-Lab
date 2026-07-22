"""Test suite (fast — no real training).

    python test_cv.py

Checks the building blocks of the three ways: from-scratch CNN (shape), feature extractor
(frozen, shape) and the fine-tuning setup (only head + last block trainable).
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
    print("  SmallCNN forward (N,10) ... OK")


def test_smallcnn_learns_step():
    # One gradient step lowers the loss on a fixed mini-batch.
    torch.manual_seed(0)
    m = SmallCNN(10)
    x = torch.randn(16, 3, 64, 64); y = torch.randint(0, 10, (16,))
    opt = torch.optim.Adam(m.parameters(), 1e-3); crit = torch.nn.CrossEntropyLoss()
    l0 = crit(m(x), y).item()
    for _ in range(5):
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    assert crit(m(x), y).item() < l0
    print("  SmallCNN learns (loss falls) ... OK")


def test_feature_extractor_frozen():
    fe, preprocess = build_feature_extractor()
    assert all(not p.requires_grad for p in fe.parameters())
    assert not fe.training
    feats = extract_features(fe, preprocess, _imgs(5))
    assert feats.shape == (5, 576), feats.shape
    print("  Feature extractor frozen & shape (N,576) ... OK")


def test_finetune_only_head_and_last_block_trainable():
    m = build_finetune_model(n_classes=10)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable, "there must be trainable parameters"
    # only the head (classifier) or the last feature block (features.12) trainable
    assert all(n.startswith("classifier") or n.startswith("features.12") for n in trainable), trainable
    # the rest is frozen
    frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
    assert any(n.startswith("features.0") for n in frozen)
    print("  Fine-tune: only head + last block trainable ... OK")


def test_finetune_model_forward():
    m = build_finetune_model(n_classes=10)
    x = torch.stack([FT_PREPROCESS(im) for im in _imgs(3)])
    assert x.shape[-1] == 96
    with torch.no_grad():
        out = m(x)
    assert out.shape == (3, 10)
    print("  Fine-tune model forward (N,10) @96 ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} tests ...")
    for t in tests:
        t()
    print("All tests passed.")
