"""Das kleine From-Scratch-CNN (der „ohne pretrained"-Weg)."""
import torch.nn as nn


class SmallCNN(nn.Module):
    """Ein kompaktes CNN nach dem Muster aus dem Skript (Abschnitt 2.2):
    [Conv -> BN -> ReLU -> Pool] x3 -> Global Average Pooling -> FC.
    Klein genug, um auf 64x64-EuroSAT-Bildern in ~1 Minute auf der CPU zu trainieren.
    """

    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                                          # Global Avg Pool
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))
