# Project 03 (final) — Image classifier for a new domain: three ways compared

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 11 — Computer Vision** · Format: **Python project** (free implementation, *no* code given)

## Why this scope?

The final project consolidates the **central practical decision** of the module: how do I
build a classifier for **my own** image domain? On real **EuroSAT satellite images** (10 land
use classes, a *completely different* domain from ImageNet) you compare the three ways from
script sections 2.3 & 3.2 **directly against each other**:

- **(a) "without pretrained"** — train a CNN **from scratch**;
- **(b) feature extraction** — a **frozen** pretrained backbone + your own classifier;
- **(c) fine-tuning** — **keep training** a pretrained backbone (partially).

This lets you experience quantitatively *why* transfer learning is the standard — and where
its limits/costs lie. **Everything is deliberately CPU-friendly** (small models, small
subsets, few epochs) and runs in **~2–3 minutes** without overheating the laptop.

**No code given.** This README is the specification. A complete, tested reference solution is
in [`solution/`](solution/) — **build it yourself first.**

## Goal

Build a EuroSAT classifier in **three** ways, **evaluate** it cleanly and **explain** the
difference.

## Prior knowledge

- **Script** section 2 (CNN building blocks, training from scratch) and **3** (pretrained
  models, the three usage modes, normalization).
- **Project 02** — you already built feature extraction (mode B) there; here it reappears in
  comparison with the other ways.
- PyTorch (`nn.Conv2d`, training loop), torchvision (`models`, `datasets.EuroSAT`),
  scikit-learn (`LogisticRegression`).

## Dataset

**EuroSAT** — 27,000 Sentinel-2 satellite images (64×64 RGB), 10 classes (AnnualCrop, Forest,
River, Residential, …). Automatic download via torchvision (~90 MB, fast). A **different
domain** from ImageNet — hence an honest transfer test.

## Specification of the three ways

**(a) From-scratch CNN** — the "without pretrained" way:
- a small CNN following script 2.2: `[Conv → BN → ReLU → Pool] × 3 → GlobalAvgPool → FC`;
- on the **native 64×64** images; simple **flip augmentation** (horizontal *and* vertical —
  valid for satellite images); Adam, ~10 epochs. **Keep it small!**

**(b) Feature extraction** (mode B) — **freeze** the backbone:
- load pretrained **MobileNetV3-Small**, freeze the parameters, replace the head with
  `Identity`; preprocess images with `weights.transforms()` (full 224 px, correct
  normalization) and pull **feature vectors**;
- train a **linear classifier** (LogReg) on top.

**(c) Fine-tuning** (mode C) — **partially** keep training the pretrained weights:
- set a new head for 10 classes; unfreeze **only** the last feature block **+** the head
  (the rest frozen → cheap), train with a **small learning rate**;
- **deliberately small**: reduced resolution (e.g. 96 px), a small subset, few epochs.

## Milestones

1. **Data**: load EuroSAT (a tensor view for (a); a PIL view for (b)/(c)); train/test split.
2. **(a)** define + train the from-scratch CNN; measure the test accuracy.
3. **(b)** build the feature extractor, pull features, train the classifier.
4. **(c)** briefly train the fine-tune model (head + last block).
5. **Comparison + error analysis** (confusion matrix, weakest classes) and **analysis** (see
   below).

## What should work in the end

Reference orders of magnitude (fixed seed, CPU, ~2–3 min):

| Way | Test accuracy |
|---|---|
| (a) From scratch (no pretrained) | **~0.76** |
| (b) Feature extraction (@224, frozen) | **~0.94** |
| (c) Fine-tuning (small, @96) | **~0.75** |

Both transfer ways are **competitive or better** than from scratch, and **feature extraction
at full resolution clearly dominates** — at minimal compute.

## Analysis (written, `ANALYSIS.md`)

Show/explain:

1. **Why does feature extraction beat the from-scratch CNN**, even though the backbone was
   **never** trained on satellite images? (Reference: transferable features, script 3.1.)
2. **Why does the *cheap* fine-tuning lie *below* feature extraction here**, even though
   fine-tuning is "more powerful"? Which two deliberate savings hold it back (resolution,
   epochs/data amount), and what would you change to lift it above (b) — and at what price?
3. **Data budget:** from scratch reaches ~0.76 — would it catch up with the transfer ways
   given **10×** more images and compute? Why (not)? (Reference: script 3.3.)
4. **Normalization:** what happens if you feed the pretrained models *without* the ImageNet
   normalization (`weights.transforms()`)? Why?

## Assessment criteria (master's level)

- **All three ways correctly** implemented — in particular: the backbone in (b) really frozen
  and with correct preprocessing; in (c) only the intended parts unfrozen.
- Clean, reproducible **evaluation** + error analysis.
- **Analysis** that places from-scratch vs. transfer quantitatively *and* conceptually and
  honestly discusses the compute/accuracy trade-off.
- Stayed **CPU-friendly** (no excessive training).

## Setup

```bash
source ../../../../.venv/bin/activate
# build your own implementation, then e.g.:
#   python run.py         # trains & compares all three ways (~2-3 min)
#   python test_cv.py     # test suite (fast)
```

Requires `torch`, `torchvision`, `scikit-learn`, `numpy`. EuroSAT is downloaded into
`datasets/` on the first run.

## Reference solution

Complete in [`solution/`](solution/): `data.py`, `model.py`, `train_scratch.py`,
`transfer.py`, `run.py`, `test_cv.py` (5 tests green). Reference: (a) ~0.76, (b) ~0.94,
(c) ~0.75. **Build it yourself first.**

---

# Projekt 03 (final) — Bildklassifikator für eine neue Domäne: drei Wege im Vergleich (deutsche Fassung)

**Modul 11 — Computer Vision** · Format: **Python-Projekt** (freie Umsetzung, *keine* Code-Vorgabe)

## Warum dieser Zuschnitt?

Das Abschlussprojekt konsolidiert die **zentrale praktische Entscheidung** des Moduls: Wie
baue ich einen Klassifikator für **meine eigene** Bilddomäne? Du vergleichst — auf echten
**EuroSAT-Satellitenbildern** (10 Landnutzungsklassen, eine *ganz andere* Domäne als
ImageNet) — die drei Wege aus Skript-Abschnitt 2.3 & 3.2 **direkt gegeneinander**:

- **(a) „ohne pretrained"** — ein CNN **von Grund auf** trainieren;
- **(b) Feature-Extraktion** — ein **eingefrorenes** pretrained Backbone + eigener Klassifikator;
- **(c) Fine-Tuning** — pretrained Backbone (teilweise) **weitertrainieren**.

So erlebst du quantitativ, *warum* Transfer Learning der Standard ist — und wo seine
Grenzen/Kosten liegen. **Alles ist bewusst CPU-freundlich** (kleine Modelle, kleine
Subsets, wenige Epochen) und läuft in **~2–3 Minuten** ohne den Laptop zu überhitzen.

**Keine Code-Vorgabe.** Diese README ist die Spezifikation. Eine vollständige, getestete
Musterlösung liegt in [`solution/`](solution/) — **erst selbst bauen.**

## Ziel

Einen EuroSAT-Klassifikator auf **drei** Arten bauen, sauber **evaluieren** und den
Unterschied **erklären**.

## Vorwissen

- **Skript** Abschnitt 2 (CNN-Bausteine, Training from scratch) und **3** (pretrained
  Modelle, die drei Nutzungsarten, Normalisierung).
- **Projekt 02** — Feature-Extraktion (Modus B) hast du dort schon gebaut; hier kommt es
  im Vergleich mit den anderen Wegen wieder vor.
- PyTorch (`nn.Conv2d`, Trainingsschleife), torchvision (`models`, `datasets.EuroSAT`),
  scikit-learn (`LogisticRegression`).

## Datensatz

**EuroSAT** — 27 000 Sentinel-2-Satellitenbilder (64×64 RGB), 10 Klassen (AnnualCrop,
Forest, River, Residential, …). Automatischer Download via torchvision (~90 MB, schnell).
Eine **andere Domäne** als ImageNet — deshalb ein ehrlicher Transfer-Test.

## Spezifikation der drei Wege

**(a) From-Scratch-CNN** — der „ohne pretrained"-Weg:
- ein kleines CNN nach Skript 2.2: `[Conv → BN → ReLU → Pool] × 3 → GlobalAvgPool → FC`;
- auf den **nativen 64×64**-Bildern; einfache **Flip-Augmentation** (horizontal *und*
  vertikal — bei Satellitenbildern gültig); Adam, ~10 Epochen. **Klein halten!**

**(b) Feature-Extraktion** (Modus B) — Backbone **einfrieren**:
- pretrained **MobileNetV3-Small** laden, Parameter einfrieren, Kopf durch `Identity`
  ersetzen; Bilder mit `weights.transforms()` vorverarbeiten (volle 224 px, korrekte
  Normalisierung) und **Feature-Vektoren** ziehen;
- darauf einen **linearen Klassifikator** (LogReg) trainieren.

**(c) Fine-Tuning** (Modus C) — pretrained Gewichte **teilweise** weitertrainieren:
- neuen Kopf für 10 Klassen setzen; **nur** den letzten Feature-Block **+** den Kopf
  auftauen (Rest eingefroren → billig), mit **kleiner Lernrate** trainieren;
- **bewusst klein**: reduzierte Auflösung (z. B. 96 px), kleines Subset, wenige Epochen.

## Milestones

1. **Daten**: EuroSAT laden (Tensor-Sicht für (a); PIL-Sicht für (b)/(c)); Train/Test-Split.
2. **(a)** From-Scratch-CNN definieren + trainieren; Test-Accuracy messen.
3. **(b)** Feature-Extraktor bauen, Features ziehen, Klassifikator trainieren.
4. **(c)** Fine-Tune-Modell (Kopf + letzter Block) kurz trainieren.
5. **Vergleich + Fehleranalyse** (Konfusionsmatrix, schwächste Klassen) und **Analyse** (s. u.).

## Was am Ende funktionieren soll

Referenz-Größenordnungen (fester Seed, CPU, ~2–3 min):

| Weg | Test-Accuracy |
|---|---|
| (a) From scratch (kein pretrained) | **~0.76** |
| (b) Feature-Extraktion (@224, eingefroren) | **~0.94** |
| (c) Fine-Tuning (klein, @96) | **~0.75** |

Beide Transfer-Wege sind **konkurrenzfähig oder besser** als From-Scratch, und die
**Feature-Extraktion bei voller Auflösung dominiert klar** — bei minimalem Rechenaufwand.

## Analyse (schriftlich, `ANALYSIS.md`)

Belege/erkläre:

1. **Warum schlägt Feature-Extraktion das From-Scratch-CNN**, obwohl das Backbone **nie**
   auf Satellitenbildern trainiert wurde? (Bezug: übertragbare Merkmale, Skript 3.1.)
2. **Warum liegt das *billige* Fine-Tuning hier *unter* der Feature-Extraktion**, obwohl
   Fine-Tuning „mächtiger" ist? Welche zwei bewussten Sparmaßnahmen bremsen es (Auflösung,
   Epochen/Datenmenge), und was würdest du ändern, um es über (b) zu heben — und zu welchem
   Preis?
3. **Datenbudget:** From-Scratch erreicht ~0.76 — würde es mit **10×** mehr Bildern und
   Rechenzeit die Transfer-Wege einholen? Warum (nicht)? (Bezug: Skript 3.3.)
4. **Normalisierung:** Was passiert, wenn du die pretrained Modelle *ohne* die
   ImageNet-Normalisierung (`weights.transforms()`) fütterst? Warum?

## Bewertungsmaßstab (Master-Niveau)

- **Alle drei Wege korrekt** umgesetzt — insbesondere: Backbone in (b) wirklich eingefroren
  und mit korrekter Vorverarbeitung; in (c) nur die vorgesehenen Teile auftauen.
- Saubere, reproduzierbare **Evaluation** + Fehleranalyse.
- **Analyse**, die From-Scratch vs. Transfer quantitativ *und* konzeptuell einordnet und
  den Rechenaufwand-/Genauigkeits-Kompromiss ehrlich diskutiert.
- **CPU-freundlich** geblieben (keine überzogenen Trainings).

## Setup

```bash
source ../../../../.venv/bin/activate
# eigene Umsetzung bauen, dann z. B.:
#   python run.py         # trainiert & vergleicht alle drei Wege (~2-3 min)
#   python test_cv.py     # Testsuite (schnell)
```

Benötigt `torch`, `torchvision`, `scikit-learn`, `numpy`. EuroSAT lädt beim ersten Lauf
nach `datasets/`.

## Musterlösung

Vollständig in [`solution/`](solution/): `data.py`, `model.py`, `train_scratch.py`,
`transfer.py`, `run.py`, `test_cv.py` (5 Tests grün). Referenz: (a) ~0.76, (b) ~0.94,
(c) ~0.75. **Erst selbst bauen.**
