"""Full pipeline: pretrained feature extraction vs. raw-pixel baseline on EuroSAT.

    python run.py            # uses the feature cache in datasets/ (first run ~40 s extraction)
    python run.py --rebuild  # re-extract the features

Shows that a *frozen* ImageNet backbone massively raises the accuracy on satellite images
(~0.94 vs. ~0.41) — without training the backbone at all. Everything CPU.
"""
import argparse
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from data import load_eurosat, DATA_DIR
from transfer import build_feature_extractor, extract_features, raw_pixel_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    tr, te, classes = load_eurosat()
    ytr = np.array([y for _, y in tr])
    yte = np.array([y for _, y in te])
    print(f"EuroSAT: train {len(tr)}, test {len(te)}, {len(classes)} classes")

    cache = os.path.join(DATA_DIR, "feat_cache.npz")
    if not args.rebuild and os.path.exists(cache):
        d = np.load(cache)
        Ftr, Fte = d["Ftr"], d["Fte"]
    else:
        print("Extracting pretrained features (once, ~40 s) ...")
        model, preprocess = build_feature_extractor()
        Ftr = extract_features(model, preprocess, [img for img, _ in tr])
        Fte = extract_features(model, preprocess, [img for img, _ in te])
        np.savez(cache, Ftr=Ftr, Fte=Fte)

    Rtr = raw_pixel_features([img for img, _ in tr])
    Rte = raw_pixel_features([img for img, _ in te])

    raw_clf = LogisticRegression(max_iter=500).fit(Rtr, ytr)
    feat_clf = LogisticRegression(max_iter=1000).fit(Ftr, ytr)
    acc_raw = accuracy_score(yte, raw_clf.predict(Rte))
    acc_feat = accuracy_score(yte, feat_clf.predict(Fte))

    print(f"\nRaw-pixel baseline (16x16 + LogReg):   test accuracy {acc_raw:.3f}")
    print(f"Pretrained features (+ LogReg):        test accuracy {acc_feat:.3f}")
    print(f"  -> transfer learning wins by {acc_feat - acc_raw:+.3f}")

    # short error analysis
    pred = feat_clf.predict(Fte)
    cm = confusion_matrix(yte, pred)
    worst = np.argsort(np.diag(cm) / cm.sum(1))[:3]
    print("\nWeakest classes (feature model):")
    for c in worst:
        print(f"  {classes[c]:22} recall {cm[c, c] / cm[c].sum():.2f}")


if __name__ == "__main__":
    main()
