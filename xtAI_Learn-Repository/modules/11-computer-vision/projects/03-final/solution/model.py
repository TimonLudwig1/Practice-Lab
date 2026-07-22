"""The small from-scratch CNN (the "without pretrained" way)."""
import torch.nn as nn


class SmallCNN(nn.Module):
    """A compact CNN following the pattern from the script (section 2.2):
    [Conv -> BN -> ReLU -> Pool] x3 -> global average pooling -> FC.
    Small enough to train on 64x64 EuroSAT images in ~1 minute on the CPU.
    """

    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                                          # global avg pool
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))
