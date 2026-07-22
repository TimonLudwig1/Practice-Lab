"""Transfer learning via feature extraction — the core of the project.

Use a **pretrained** backbone (MobileNetV3-Small, trained on ImageNet) as a **frozen feature
extractor**: images -> feature vectors, on which a small classifier is then trained. See
script section 3.2 (mode B).
"""
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


def build_feature_extractor():
    """Loads the pretrained backbone, **freezes it** and removes the classification head, so
    that the forward pass yields **feature vectors**.

    Returns: (model in eval mode, preprocess transform).
    """
    model = mobilenet_v3_small(weights=WEIGHTS)
    for p in model.parameters():
        p.requires_grad = False              # freeze: do not train the backbone
    model.classifier = torch.nn.Identity()   # remove the ImageNet head -> 576-dim features
    model.eval()
    return model, WEIGHTS.transforms()       # exactly the preprocessing of the pretraining


def extract_features(model, preprocess, pil_images, batch_size=128):
    """Sends the images (preprocessed) through the frozen network and returns the feature
    matrix (N, 576) as a NumPy array."""
    feats = []
    for i in range(0, len(pil_images), batch_size):
        batch = torch.stack([preprocess(img) for img in pil_images[i:i + batch_size]])
        with torch.no_grad():
            feats.append(model(batch).numpy())
    return np.concatenate(feats, axis=0)


def raw_pixel_features(pil_images, size=16):
    """Baseline: shrink the images to size×size and flatten them into vectors (raw pixels)."""
    out = []
    for img in pil_images:
        arr = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0
        out.append(arr.flatten())
    return np.stack(out)
