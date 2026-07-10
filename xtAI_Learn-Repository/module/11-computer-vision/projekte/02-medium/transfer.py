"""Transfer Learning per Feature-Extraktion — der Kern des Projekts.

Nutze ein **vortrainiertes** Backbone (MobileNetV3-Small, ImageNet) als **eingefrorenen
Merkmalsextraktor**: Bilder -> Feature-Vektoren, auf denen dann ein kleiner Klassifikator
trainiert wird. Siehe Skript-Abschnitt 3.2 (Modus B). Prüfe mit `python test_transfer.py`.
"""
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


def build_feature_extractor():
    """Lädt das vortrainierte Backbone, **friert es ein** und entfernt den
    Klassifikationskopf, sodass der Forward-Pass **Feature-Vektoren** liefert.

    TODO (Skript 3.2, Modus B):
      1. model = mobilenet_v3_small(weights=WEIGHTS)
      2. alle Parameter einfrieren:  for p in model.parameters(): p.requires_grad = False
      3. Kopf entfernen:  model.classifier = torch.nn.Identity()   (-> 576-dim Features)
      4. model.eval()  (Inferenzmodus)
      5. return model, WEIGHTS.transforms()   # exakt die Vorverarbeitung des Vortrainings
    """
    raise NotImplementedError("Aufgabe 1: build_feature_extractor implementieren")


def extract_features(model, preprocess, pil_images, batch_size=128):
    """Schickt die Bilder (vorverarbeitet) durch das eingefrorene Netz und gibt die
    Feature-Matrix (N, 576) als NumPy-Array zurück.

    TODO:
      - in Batches der Größe batch_size über pil_images gehen;
      - je Batch: torch.stack([preprocess(img) for img in batch]) -> (B,3,224,224);
      - unter `with torch.no_grad():` durch model schicken, .numpy() sammeln;
      - alle Batches mit np.concatenate(..., axis=0) zusammenfügen und zurückgeben.
    """
    raise NotImplementedError("Aufgabe 2: extract_features implementieren")


def raw_pixel_features(pil_images, size=16):
    """Baseline (vorgegeben): Bilder auf size×size verkleinern und zu Vektoren plätten."""
    out = []
    for img in pil_images:
        arr = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0
        out.append(arr.flatten())
    return np.stack(out)
