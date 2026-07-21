"""Volle Pipeline: pretrained Feature-Extraktion vs. Rohpixel-Baseline auf EuroSAT.

    python run.py            # nutzt Feature-Cache in datasets/ (erster Lauf ~40 s Extraktion)
    python run.py --rebuild  # Features neu extrahieren

Zeigt, dass ein *eingefrorenes* ImageNet-Backbone die Genauigkeit auf Satellitenbildern
massiv hebt (~0.94 vs. ~0.41) — ganz ohne das Backbone zu trainieren. Alles CPU.
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
    print(f"EuroSAT: Train {len(tr)}, Test {len(te)}, {len(classes)} Klassen")

    cache = os.path.join(DATA_DIR, "feat_cache.npz")
    if not args.rebuild and os.path.exists(cache):
        d = np.load(cache)
        Ftr, Fte = d["Ftr"], d["Fte"]
    else:
        print("Extrahiere pretrained Features (einmalig ~40 s) ...")
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

    print(f"\nRohpixel-Baseline (16x16 + LogReg):   Test-Accuracy {acc_raw:.3f}")
    print(f"Pretrained-Features (+ LogReg):       Test-Accuracy {acc_feat:.3f}")
    print(f"  -> Transfer Learning gewinnt um {acc_feat - acc_raw:+.3f}")

    # kurze Fehleranalyse
    pred = feat_clf.predict(Fte)
    cm = confusion_matrix(yte, pred)
    worst = np.argsort(np.diag(cm) / cm.sum(1))[:3]
    print("\nSchwächste Klassen (Feature-Modell):")
    for c in worst:
        print(f"  {classes[c]:22} Recall {cm[c, c] / cm[c].sum():.2f}")


if __name__ == "__main__":
    main()
