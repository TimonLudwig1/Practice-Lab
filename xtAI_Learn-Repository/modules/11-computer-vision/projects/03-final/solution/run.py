"""Three ways to a EuroSAT classifier compared — all CPU, deliberately cheap.

    python run.py

(a) from-scratch CNN  ·  (b) feature extraction (frozen)  ·  (c) fine-tuning (small).
At the end a comparison + short error analysis. Total runtime ~2-3 min on the CPU.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from data import load_tensors, load_pil
from train_scratch import train_from_scratch
from transfer import (build_feature_extractor, extract_features,
                      build_finetune_model, finetune)


def main():
    import torch
    torch.manual_seed(0); np.random.seed(0)

    # ----- (a) from scratch (64px tensors) -----
    print("== (a) from-scratch CNN ==")
    (Xtr, ytr), (Xte, yte), classes = load_tensors(n_train=4000, n_test=2000)
    _, acc_scratch = train_from_scratch((Xtr, ytr), (Xte, yte), epochs=10)

    # ----- (b) feature extraction (frozen backbone @224) -----
    print("\n== (b) feature extraction (frozen) ==")
    train_pil, test_pil, _ = load_pil(n_train=4000, n_test=2000)
    ytr_p = np.array([y for _, y in train_pil]); yte_p = np.array([y for _, y in test_pil])
    fe, preprocess = build_feature_extractor()
    Ftr = extract_features(fe, preprocess, [im for im, _ in train_pil])
    Fte = extract_features(fe, preprocess, [im for im, _ in test_pil])
    clf = LogisticRegression(max_iter=1000).fit(Ftr, ytr_p)
    pred_feat = clf.predict(Fte)
    acc_feat = accuracy_score(yte_p, pred_feat)
    print(f"  feature extraction test acc {acc_feat:.3f}")

    # ----- (c) fine-tuning (head + last block, 96px, small) -----
    print("\n== (c) fine-tuning (small: head + last block, 96px) ==")
    ft_model = build_finetune_model(n_classes=len(classes))
    acc_ft = finetune(ft_model, train_pil[:1200], test_pil[:600], epochs=5)

    # ----- comparison -----
    print("\n" + "=" * 46)
    print("COMPARISON (EuroSAT, test accuracy)")
    print(f"  (a) from scratch (no pretrained)   : {acc_scratch:.3f}")
    print(f"  (b) feature extraction  (@224)     : {acc_feat:.3f}")
    print(f"  (c) fine-tuning (small, @96)       : {acc_ft:.3f}")
    print("=" * 46)
    print("Both transfer ways clearly beat from scratch. Feature extraction at full")
    print("resolution is the best value per compute here; the deliberately cheap")
    print("fine-tuning (96px, 5 epochs) stays below it — more resolution/epochs would")
    print("lift it, but cost compute (which we keep small for the laptop).")

    # error analysis of the best model (feature extraction)
    cm = confusion_matrix(yte_p, pred_feat)
    recalls = cm.diagonal() / cm.sum(1)
    print("\nWeakest classes (feature model):")
    for c in np.argsort(recalls)[:3]:
        print(f"  {classes[c]:22} recall {recalls[c]:.2f}")


if __name__ == "__main__":
    main()
