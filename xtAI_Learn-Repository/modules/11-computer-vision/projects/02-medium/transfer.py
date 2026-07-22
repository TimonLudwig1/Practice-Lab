"""Transfer learning via feature extraction — the core of the project.

Use a **pretrained** backbone (MobileNetV3-Small, ImageNet) as a **frozen feature
extractor**: images -> feature vectors, on which a small classifier is then trained. See
script section 3.2 (mode B). Check with `python test_transfer.py`.
"""
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


def build_feature_extractor():
    """Loads the pretrained backbone, **freezes it** and removes the classification head, so
    that the forward pass yields **feature vectors**.

    TODO (script 3.2, mode B):
      1. model = mobilenet_v3_small(weights=WEIGHTS)
      2. freeze all parameters:  for p in model.parameters(): p.requires_grad = False
      3. remove the head:  model.classifier = torch.nn.Identity()   (-> 576-dim features)
      4. model.eval()  (inference mode)
      5. return model, WEIGHTS.transforms()   # exactly the preprocessing of the pretraining
    """
    raise NotImplementedError("Task 1: implement build_feature_extractor")


def extract_features(model, preprocess, pil_images, batch_size=128):
    """Sends the images (preprocessed) through the frozen network and returns the feature
    matrix (N, 576) as a NumPy array.

    TODO:
      - go over pil_images in batches of size batch_size;
      - per batch: torch.stack([preprocess(img) for img in batch]) -> (B,3,224,224);
      - send it through model under `with torch.no_grad():`, collect .numpy();
      - join all batches with np.concatenate(..., axis=0) and return.
    """
    raise NotImplementedError("Task 2: implement extract_features")


def raw_pixel_features(pil_images, size=16):
    """Baseline (given): shrink the images to size×size and flatten them into vectors."""
    out = []
    for img in pil_images:
        arr = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0
        out.append(arr.flatten())
    return np.stack(out)
