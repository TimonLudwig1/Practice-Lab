# Project 02 (medium): CNN image classification with PyTorch — an ablation study

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

## Goal

Build a convolutional neural network in PyTorch and check three claims of the script **experimentally**, cleanly as an ablation (never change more than one factor at a time):

1. **Architectural bias:** CNN (about 66k parameters) vs. MLP (about 235k parameters) — does structure beat raw capacity?
2. **Regularization:** dropout + weight decay (AdamW) + data augmentation — what happens to the train/test gap in the short run (3 epochs) versus the long run (10 epochs)?
3. **Optimizer:** Adam vs. SGD+momentum at an identical budget.

Plus error diagnostics: the confusion matrix and the most confident misclassifications.

**Why this format (Jupyter notebook):** the ablation lives on learning-curve plots right next to the training code; a notebook documents the experiment and the finding in one place.

**Why real data (Fashion-MNIST):** 70,000 real Zalando product images, 10 classes — more demanding than MNIST (shirt/t-shirt/pullover are hard even for humans), but small enough for a CPU or an Apple GPU. `torchvision` downloads it automatically (about 30 MB into `datasets/`, excluded from the repository via `.gitignore`). For the ablation we use 20,000 training images (the constant `SUBSET` — set it to 60,000 for peak performance).

## Prior knowledge

- Project 01 (you know what `loss.backward()` does internally)
- Script 2.1 (optimizers), 2.3 (regularization), 2.4 (BatchNorm), 2.5 (CNNs)

## Setup

```bash
source .venv/bin/activate
jupyter lab modules/05-machine-learning-2/projects/02-medium/image_classification_cnn.ipynb
```

It runs on an Apple GPU (`mps`), on CUDA or on a CPU — detected automatically. Reference runtime on an M-series Mac: about 5–10 minutes for the whole notebook.

## Tasks (step by step)

1. **Look at the data** (given): the loading pipeline with normalization and a second, augmented loader (RandomCrop + flip).
2. **TODO 1 — build the CNN:** two `[Conv3×3→BN→ReLU]×2→MaxPool` blocks (32, 64 channels) → global average pooling → dropout → linear. An assert cell checks the parameter count (**expected: 66,026** — work it out by hand beforehand, the formula is in script 2.5).
3. **TODO 2 — repair `evaluate`:** one decisive line is missing. Without it all test metrics are wrong — find it and explain which two mechanisms are affected.
4. **TODO 3 — complete the ablation configurations** (Adam / AdamW+wd / SGD+momentum, the augmented loader for configuration C).
5. **Run the ablation and interpret it** — compare against the expectations stated in the notebook text.
6. **Error diagnostics:** the confusion matrix; which pairs of classes does the network confuse, and why?
7. **The long-run experiment:** 10 epochs plain vs. regularized — is the overfitting gap visible?

## What should work in the end

- The parameter-count assert passes (66,026).
- The ablation runs through. Reference values after 3 epochs on the 20k subset: A (MLP) 0.853, B (CNN) 0.847, C (CNN + reg/aug) 0.775, D (CNN + SGD momentum) 0.829.
- **Note the finding, it is deliberately not the textbook one:** at this short budget the CNN does *not* yet beat the MLP. Only over 10 epochs does the regularized CNN pull ahead (0.861 vs. 0.848 for the plain CNN) — and it does so with 3.5 times fewer parameters. The inductive bias buys a better result per parameter, not faster progress per epoch; the GAP head and BatchNorm converge more slowly than a dense MLP on 28×28 greyscale.
- Using your own curves, you can explain why the regularized variant is behind after 3 epochs but has the smaller train/test gap and the better accuracy after 10.

## Reference solution

[`solution/solution.ipynb`](solution/solution.ipynb) — fully executed, with all curves, tables and interpretation texts.

---
---

# Projekt 02 (medium): CNN-Bildklassifikation mit PyTorch — Ablationsstudie (deutsche Fassung)

## Ziel

Ein Convolutional Neural Network in PyTorch bauen und drei Skript-Behauptungen **experimentell prüfen**, sauber als Ablation (immer nur eine Stellgröße ändern):

1. **Architektur-Bias:** CNN (≈66k Parameter) vs. MLP (≈235k Parameter) — schlägt Struktur rohe Kapazität?
2. **Regularisierung:** Dropout + Weight Decay (AdamW) + Datenaugmentierung — was passiert kurzfristig (3 Epochen) vs. langfristig (10 Epochen) mit der Train/Test-Schere?
3. **Optimierer:** Adam vs. SGD+Momentum bei identischem Budget.

Dazu Fehlerdiagnose: Confusion Matrix und die konfidentesten Fehlklassifikationen.

**Warum dieses Format (Jupyter Notebook):** Die Ablation lebt von Lernkurven-Plots direkt neben dem Trainingscode; ein Notebook dokumentiert Experiment und Befund in einem.

**Warum echte Daten (Fashion-MNIST):** 70 000 echte Zalando-Produktbilder, 10 Klassen — anspruchsvoller als MNIST (Shirt/T-Shirt/Pullover sind auch für Menschen schwer), aber klein genug für CPU/Apple-GPU. `torchvision` lädt automatisch (~30 MB nach `datasets/`, per `.gitignore` vom Repo ausgeschlossen). Für die Ablation nutzen wir 20 000 Trainingsbilder (Konstante `SUBSET` — auf 60 000 stellen für Bestleistung).

## Vorwissen

- Projekt 01 (du weißt, was `loss.backward()` intern tut)
- Skript 2.1 (Optimierer), 2.3 (Regularisierung), 2.4 (BatchNorm), 2.5 (CNNs)

## Setup

```bash
source .venv/bin/activate
jupyter lab modules/05-machine-learning-2/projects/02-medium/image_classification_cnn.ipynb
```

Läuft auf Apple-GPU (`mps`), CUDA oder CPU — wird automatisch erkannt. Referenz-Laufzeit auf einem M-Series-Mac: ca. 5–10 Minuten für das ganze Notebook.

## Aufgabenstellung (Schritt für Schritt)

1. **Daten ansehen** (fertig vorgegeben): Lade-Pipeline mit Normalisierung und einem zweiten, augmentierten Loader (RandomCrop + Flip).
2. **TODO 1 — CNN bauen:** zwei `[Conv3×3→BN→ReLU]×2→MaxPool`-Blöcke (32, 64 Kanäle) → Global Average Pooling → Dropout → Linear. Eine Assert-Zelle prüft die Parameterzahl (**erwartet: 66 026** — rechne sie vorher von Hand nach, Formel im Skript 2.5).
3. **TODO 2 — `evaluate` reparieren:** Eine entscheidende Zeile fehlt. Ohne sie sind alle Testmetriken falsch — finde sie und erkläre, welche zwei Mechanismen betroffen sind.
4. **TODO 3 — Ablations-Konfigurationen** vervollständigen (Adam / AdamW+wd / SGD+Momentum, Augmentierungs-Loader für Konfiguration C).
5. **Ablation ausführen und interpretieren** — vergleiche mit den Erwartungen im Notebook-Text.
6. **Fehlerdiagnose:** Confusion Matrix; welche Klassenpaare verwechselt das Netz und warum?
7. **Langzeit-Experiment:** 10 Epochen pur vs. regularisiert — sichtbare Overfitting-Schere?

## Was am Ende funktionieren soll

- Parameterzahl-Assert besteht (66 026).
- Die Ablation läuft durch. Referenzwerte nach 3 Epochen auf dem 20k-Subset: A (MLP) 0,853, B (CNN) 0,847, C (CNN + Reg/Aug) 0,775, D (CNN + SGD-Momentum) 0,829.
- **Achte auf den Befund, er ist bewusst nicht der aus dem Lehrbuch:** Bei diesem kurzen Budget schlägt das CNN das MLP noch *nicht*. Erst über 10 Epochen zieht das regularisierte CNN vorbei (0,861 gegenüber 0,848 für das pure CNN) — und zwar mit 3,5× weniger Parametern. Der induktive Bias kauft das bessere Ergebnis pro Parameter, nicht den schnelleren Fortschritt pro Epoche; GAP-Kopf und BatchNorm konvergieren auf 28×28-Graustufen langsamer als ein dichtes MLP.
- Du kannst anhand deiner Kurven erklären: warum die regularisierte Variante nach 3 Epochen zurückliegt, nach 10 aber die kleinere Train/Test-Schere und die bessere Accuracy hat.

## Musterlösung

[`solution/solution.ipynb`](solution/solution.ipynb) — vollständig ausgeführt, mit allen Kurven, Tabellen und Interpretationstexten.
