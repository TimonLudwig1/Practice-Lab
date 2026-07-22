# Project 02 (medium) — Transfer learning as a feature extractor (EuroSAT)

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 11 — Computer Vision** · Format: **Python project** (several modules + test suite)

## Why this format?

The core is a **practical skill**: correctly using a pretrained model as a feature extractor
(load, freeze, remove the head, preprocess correctly, pull features). As a **codebase** you
cleanly separate these building blocks and the **test suite** checks that you really froze
the backbone and tapped into it correctly.

## Goal

You use a **frozen** ImageNet backbone (MobileNetV3-Small) as a **feature extractor** on
**EuroSAT** (satellite images — a *completely different* domain from ImageNet) and train a
small classifier on top. Core points (script 3.2, mode B):

- **load** a pretrained model, **freeze** it (`requires_grad = False`) and **replace** the
  classification head with `Identity`;
- preprocess images with **`weights.transforms()`** exactly as in pretraining (incl. ImageNet
  normalization) and extract **feature vectors** (576-dim);
- **the aha moment:** a linear classifier on these features reaches **~0.94**, while the same
  method on **raw pixels** achieves only **~0.41** — *without* ever training the backbone.
  Transferable features clearly beat raw pixels.

## Prior knowledge

- **Script** section 3 (what a pretrained model is, the three usage modes, normalization,
  from-scratch vs. transfer).
- PyTorch/torchvision basics; scikit-learn (`LogisticRegression`).

## Project structure

```
02-medium/
  transfer.py        # build the feature extractor + pull features   <- YOU (task 1 + 2)
  data.py            # load EuroSAT + subsets                          (given)
  run.py             # pipeline: features vs. raw pixels + error analysis (given)
  test_transfer.py   # test suite (5 tests, fast)                     (given)
  solution/          # complete, tested reference solution
```

## Assignment

Little is given — the two practical cores are yours (`# TODO` in `transfer.py`):

1. **`build_feature_extractor`**: load pretrained MobileNetV3-Small, **freeze** it,
   `classifier = Identity`, `eval()`, and return `WEIGHTS.transforms()`.
2. **`extract_features`**: preprocess the images batch-wise and send them through the network
   under `torch.no_grad()`; return the feature matrix `(N, 576)`.

`raw_pixel_features` (the baseline), loading and classification are given.

**How to proceed:**

```bash
source ../../../../.venv/bin/activate
python test_transfer.py     # red -> fill in the tasks -> all 5 tests green
python run.py               # real EuroSAT: feature vs. raw-pixel accuracy
```

`run.py` downloads EuroSAT on the first run (~90 MB, fast) and caches the extracted features
in `datasets/` (rebuild with `--rebuild`).

## What should work in the end

- `python test_transfer.py` → **all 5 tests green**: backbone frozen & headless, feature
  shape `(N, 576)`, correct preprocessing (normalization produces negative values), the
  raw-pixel baseline, and an integration test that pretrained features separate toy classes.
- `python run.py` → **raw pixels ~0.41 vs. pretrained features ~0.94** on EuroSAT, plus a
  short error analysis of the weakest classes.

## Reference solution

Complete in [`solution/`](solution/) (all tests green, ~0.94). Try it yourself first — the
root `transfer.py` raises `NotImplementedError` until you fill in the TODOs.

---

# Projekt 02 (medium) — Transfer Learning als Feature-Extraktor (EuroSAT) (deutsche Fassung)

**Modul 11 — Computer Vision** · Format: **Python-Projekt** (mehrere Module + Testsuite)

## Warum dieses Format?

Der Kern ist eine **Praxis-Fähigkeit**: ein vortrainiertes Modell korrekt als
Merkmalsextraktor verwenden (laden, einfrieren, Kopf entfernen, richtig vorverarbeiten,
Features ziehen). Als **Codebasis** trennst du diese Bausteine sauber und die **Testsuite**
prüft, dass du das Backbone wirklich eingefroren und richtig angezapft hast.

## Ziel

Du nutzt ein **eingefrorenes** ImageNet-Backbone (MobileNetV3-Small) als
**Feature-Extraktor** auf **EuroSAT** (Satellitenbildern — einer *ganz anderen* Domäne als
ImageNet) und trainierst darauf einen kleinen Klassifikator. Kernpunkte (Skript 3.2, Modus B):

- ein pretrained Modell **laden**, **einfrieren** (`requires_grad = False`) und den
  Klassifikationskopf durch `Identity` **ersetzen**;
- Bilder mit **`weights.transforms()`** exakt so vorverarbeiten wie im Vortraining
  (inkl. ImageNet-Normalisierung) und **Feature-Vektoren** (576-dim) extrahieren;
- **der Aha-Moment:** ein linearer Klassifikator auf diesen Features erreicht **~0.94**,
  während dieselbe Methode auf **rohen Pixeln** nur **~0.41** schafft — *ohne* das Backbone
  je zu trainieren. Übertragbare Merkmale schlagen rohe Pixel deutlich.

## Vorwissen

- **Skript** Abschnitt 3 (was ein pretrained Modell ist, die drei Nutzungsarten,
  Normalisierung, from-scratch vs. Transfer).
- PyTorch/torchvision-Grundlagen; scikit-learn (`LogisticRegression`).

## Projektstruktur

```
02-medium/
  transfer.py        # Feature-Extraktor bauen + Features ziehen   <- DU (Aufgabe 1 + 2)
  data.py            # EuroSAT laden + Teilmengen                    (vorgegeben)
  run.py             # Pipeline: Features vs. Rohpixel + Fehleranalyse (vorgegeben)
  test_transfer.py   # Testsuite (5 Tests, schnell)                  (vorgegeben)
  solution/           # vollständige, getestete Musterlösung
```

## Aufgabenstellung

Wenig Vorgabe — die beiden Praxis-Kerne sind deine (`# TODO` in `transfer.py`):

1. **`build_feature_extractor`**: pretrained MobileNetV3-Small laden, **einfrieren**,
   `classifier = Identity`, `eval()`, und `WEIGHTS.transforms()` zurückgeben.
2. **`extract_features`**: die Bilder batchweise vorverarbeiten und unter `torch.no_grad()`
   durchs Netz schicken; Feature-Matrix `(N, 576)` zurückgeben.

`raw_pixel_features` (Baseline), das Laden und die Klassifikation sind vorgegeben.

**Vorgehen:**

```bash
source ../../../../.venv/bin/activate
python test_transfer.py     # rot -> Aufgaben füllen -> alle 5 Tests grün
python run.py               # echtes EuroSAT: Feature- vs. Rohpixel-Genauigkeit
```

`run.py` lädt EuroSAT beim ersten Lauf (~90 MB, schnell) und cached die extrahierten
Features in `datasets/` (Neuaufbau mit `--rebuild`).

## Was am Ende funktionieren soll

- `python test_transfer.py` → **alle 5 Tests grün**: Backbone eingefroren & ohne Kopf,
  Feature-Form `(N, 576)`, korrekte Vorverarbeitung (Normalisierung erzeugt negative Werte),
  Rohpixel-Baseline, und ein Integrationstest, dass pretrained Features Toy-Klassen trennen.
- `python run.py` → **Rohpixel ~0.41 vs. Pretrained-Features ~0.94** auf EuroSAT, plus eine
  kurze Fehleranalyse der schwächsten Klassen.

## Musterlösung

Vollständig in [`solution/`](solution/) (alle Tests grün, ~0.94). Erst selbst versuchen — die
Root-`transfer.py` wirft `NotImplementedError`, bis du die TODOs füllst.
