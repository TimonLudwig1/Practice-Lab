"""Transfer Learning per Feature-Extraktion — der Kern des Projekts.

Ein **vortrainiertes** Backbone (MobileNetV3-Small, auf ImageNet trainiert) als
**eingefrorener Merkmalsextraktor** nutzen: Bilder -> Feature-Vektoren, auf denen dann ein
kleiner Klassifikator trainiert wird. Siehe Skript-Abschnitt 3.2 (Modus B).
"""
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


def build_feature_extractor():
    """Lädt das vortrainierte Backbone, **friert es ein** und entfernt den
    Klassifikationskopf, sodass der Forward-Pass **Feature-Vektoren** liefert.

    Rückgabe: (model im eval-Modus, preprocess-Transform).
    """
    model = mobilenet_v3_small(weights=WEIGHTS)
    for p in model.parameters():
        p.requires_grad = False              # einfrieren: kein Training des Backbones
    model.classifier = torch.nn.Identity()   # ImageNet-Kopf entfernen -> 576-dim Features
    model.eval()
    return model, WEIGHTS.transforms()       # exakt die Vorverarbeitung des Vortrainings


def extract_features(model, preprocess, pil_images, batch_size=128):
    """Schickt die Bilder (vorverarbeitet) durch das eingefrorene Netz und gibt die
    Feature-Matrix (N, 576) als NumPy-Array zurück."""
    feats = []
    for i in range(0, len(pil_images), batch_size):
        batch = torch.stack([preprocess(img) for img in pil_images[i:i + batch_size]])
        with torch.no_grad():
            feats.append(model(batch).numpy())
    return np.concatenate(feats, axis=0)


def raw_pixel_features(pil_images, size=16):
    """Baseline: Bilder auf size×size verkleinern und zu Vektoren plätten (rohe Pixel)."""
    out = []
    for img in pil_images:
        arr = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0
        out.append(arr.flatten())
    return np.stack(out)
