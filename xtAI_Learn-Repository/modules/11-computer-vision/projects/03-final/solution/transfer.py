"""Die beiden Transfer-Wege: Feature-Extraktion (Modus B) und Fine-Tuning (Modus C).

Backbone: MobileNetV3-Small (ImageNet-vortrainiert). Alles CPU-tauglich; Fine-Tuning wird
bewusst klein gehalten (nur letzter Block + Kopf, reduzierte Auflösung, wenige Epochen).
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1

# reduzierte Vorverarbeitung fürs Fine-Tuning (96px statt 224px -> viel billiger)
FT_PREPROCESS = transforms.Compose([
    transforms.Resize(96), transforms.CenterCrop(96), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---- Modus B: Feature-Extraktion (eingefrorenes Backbone) ------------------
def build_feature_extractor():
    model = mobilenet_v3_small(weights=WEIGHTS)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier = nn.Identity()
    model.eval()
    return model, WEIGHTS.transforms()


def extract_features(model, preprocess, pil_images, batch_size=128):
    feats = []
    for i in range(0, len(pil_images), batch_size):
        batch = torch.stack([preprocess(im) for im in pil_images[i:i + batch_size]])
        with torch.no_grad():
            feats.append(model(batch).numpy())
    return np.concatenate(feats, axis=0)


# ---- Modus C: Fine-Tuning (Kopf + letzter Block, klein) --------------------
def build_finetune_model(n_classes=10):
    """Pretrained Backbone mit neuem Kopf; nur der letzte Feature-Block und der Kopf
    sind trainierbar (alles andere eingefroren -> billig)."""
    model = mobilenet_v3_small(weights=WEIGHTS)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.features[-1].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model


def finetune(model, train_pairs, test_pairs, epochs=5, lr=5e-4, batch_size=32, log=print):
    """Kurzes Fine-Tuning bei reduzierter Auflösung. train/test: Listen von (PIL, label)."""
    Xtr = torch.stack([FT_PREPROCESS(im) for im, _ in train_pairs])
    ytr = torch.tensor([y for _, y in train_pairs])
    Xte = torch.stack([FT_PREPROCESS(im) for im, _ in test_pairs])
    yte = torch.tensor([y for _, y in test_pairs])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for j in range(0, len(Xtr), batch_size):
            b = perm[j:j + batch_size]
            opt.zero_grad(); crit(model(Xtr[b]), ytr[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        log(f"  Fine-Tune Epoche {ep}: Test-Acc {acc:.3f}")
    return acc
