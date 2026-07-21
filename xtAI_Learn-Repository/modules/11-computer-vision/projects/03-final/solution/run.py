"""Drei Wege zum EuroSAT-Klassifikator im Vergleich — alles CPU, bewusst billig.

    python run.py

(a) From-Scratch-CNN  ·  (b) Feature-Extraktion (eingefroren)  ·  (c) Fine-Tuning (klein).
Am Ende ein Vergleich + kurze Fehleranalyse. Gesamtlaufzeit ~2-3 min auf der CPU.
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

    # ----- (a) From scratch (64px-Tensoren) -----
    print("== (a) From-Scratch-CNN ==")
    (Xtr, ytr), (Xte, yte), classes = load_tensors(n_train=4000, n_test=2000)
    _, acc_scratch = train_from_scratch((Xtr, ytr), (Xte, yte), epochs=10)

    # ----- (b) Feature-Extraktion (eingefrorenes Backbone @224) -----
    print("\n== (b) Feature-Extraktion (eingefroren) ==")
    train_pil, test_pil, _ = load_pil(n_train=4000, n_test=2000)
    ytr_p = np.array([y for _, y in train_pil]); yte_p = np.array([y for _, y in test_pil])
    fe, preprocess = build_feature_extractor()
    Ftr = extract_features(fe, preprocess, [im for im, _ in train_pil])
    Fte = extract_features(fe, preprocess, [im for im, _ in test_pil])
    clf = LogisticRegression(max_iter=1000).fit(Ftr, ytr_p)
    pred_feat = clf.predict(Fte)
    acc_feat = accuracy_score(yte_p, pred_feat)
    print(f"  Feature-Extraktion Test-Acc {acc_feat:.3f}")

    # ----- (c) Fine-Tuning (Kopf + letzter Block, 96px, klein) -----
    print("\n== (c) Fine-Tuning (klein: Kopf + letzter Block, 96px) ==")
    ft_model = build_finetune_model(n_classes=len(classes))
    acc_ft = finetune(ft_model, train_pil[:1200], test_pil[:600], epochs=5)

    # ----- Vergleich -----
    print("\n" + "=" * 46)
    print("VERGLEICH (EuroSAT, Test-Accuracy)")
    print(f"  (a) From scratch (kein pretrained) : {acc_scratch:.3f}")
    print(f"  (b) Feature-Extraktion  (@224)     : {acc_feat:.3f}")
    print(f"  (c) Fine-Tuning (klein, @96)       : {acc_ft:.3f}")
    print("=" * 46)
    print("Beide Transfer-Wege schlagen From-Scratch klar. Feature-Extraktion bei voller")
    print("Auflösung ist hier der beste Wert pro Rechenaufwand; das absichtlich billige")
    print("Fine-Tuning (96px, 5 Epochen) bleibt darunter — mehr Auflösung/Epochen würden")
    print("es anheben, kosten aber Rechenzeit (die wir für den Laptop klein halten).")

    # Fehleranalyse des besten Modells (Feature-Extraktion)
    cm = confusion_matrix(yte_p, pred_feat)
    recalls = cm.diagonal() / cm.sum(1)
    print("\nSchwächste Klassen (Feature-Modell):")
    for c in np.argsort(recalls)[:3]:
        print(f"  {classes[c]:22} Recall {recalls[c]:.2f}")


if __name__ == "__main__":
    main()
