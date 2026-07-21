"""Weg (a): das From-Scratch-CNN trainieren (der „ohne pretrained"-Weg)."""
import torch
import torch.nn as nn

from model import SmallCNN


def train_from_scratch(train_data, test_data, epochs=10, lr=3e-4, batch_size=64, log=print):
    """train_data/test_data: (X, y) mit X (N,3,64,64). Mit einfacher Flip-Augmentation
    (für Satellitenbilder sind horizontale UND vertikale Flips gültig)."""
    Xtr, ytr = train_data
    Xte, yte = test_data
    model = SmallCNN(n_classes=int(ytr.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    N = len(Xtr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        for j in range(0, N, batch_size):
            b = perm[j:j + batch_size]
            xb = Xtr[b]
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, [3])            # horizontal
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, [2])            # vertikal
            opt.zero_grad(); crit(model(xb), ytr[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        log(f"  Scratch Epoche {ep}: Test-Acc {acc:.3f}")
    return model, acc
