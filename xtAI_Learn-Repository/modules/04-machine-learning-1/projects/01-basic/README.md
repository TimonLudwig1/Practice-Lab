# Project 01 (basic) — kNN by hand and the bias-variance tradeoff made tangible

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format:** Jupyter notebook (`knn_from_scratch.ipynb`).
**Why this format?** The two core lessons (scaling, the bias-variance arc) are visual insights — accuracy curves and decision boundaries have to appear right next to the code.

**Data: real.** The Palmer penguins again (`seaborn.load_dataset`), now as a classification task: the species from **bill length (mm) and body mass (g)** — the extremely different scales are chosen deliberately so that the scaling lesson lands (77 % → 95 % accuracy).

## Goal

Implement kNN entirely yourself (distance → k nearest neighbours → majority vote), verify it against scikit-learn, then sweep $k$ from 1 to 151 and see the bias-variance arc (overfitting on the left, underfitting on the right) as a curve and as a picture of decision boundaries.

## Prior knowledge

- Sections 1.2–1.4 of the module script (kNN, bias/variance, evaluation)
- numpy basics; `train_test_split` from module 03

## Tasks

1. Work through the notebook; to be implemented: `euclidean`, `knn_predict`, the standardisation (on the training data only — the leakage rule!) and the k sweep.
2. Mini checks must evaluate to `True`.
3. Describe the bias-variance plot in your own words before reading the reference interpretation.

## What should work in the end

- Unscaled (k=5): accuracy about 0.77 → scaled: about 0.95.
- The hand-written kNN and sklearn agree up to at most one test point (tie breaks).
- k sweep: k=1 → train 1.0 / test 0.92 (overfitting); the optimum around k = 5–31; k=151 → both about 0.75 (underfitting).
- Decision boundary plot: jagged at k=1, smooth at k=31.

## Solution

Fully executed reference solution: [`solution/solution.ipynb`](solution/solution.ipynb).

---
---

# Projekt 01 (basic) — kNN von Hand & der Bias-Variance-Tradeoff zum Anfassen (deutsche Fassung)

**Format:** Jupyter Notebook (`knn_from_scratch.ipynb`).
**Warum dieses Format?** Die zwei Kernlektionen (Skalierung, Bias-Variance-Bogen) sind visuelle Erkenntnisse — Accuracy-Kurven und Entscheidungsgrenzen müssen direkt neben dem Code entstehen.

**Daten: echt.** Wieder die Palmer-Pinguine (`seaborn.load_dataset`), jetzt als Klassifikationsaufgabe: Art aus **Schnabellänge (mm) und Körpermasse (g)** — die extrem verschiedenen Skalen sind bewusst gewählt, damit die Skalierungs-Lektion knallt (77 % → 95 % Accuracy).

## Ziel

kNN komplett selbst implementieren (Distanz → k nächste Nachbarn → Mehrheitsvotum), gegen scikit-learn verproben, dann $k$ von 1 bis 151 durchfahren und den Bias-Variance-Bogen (Overfitting links, Underfitting rechts) als Kurve und als Entscheidungsgrenzen-Bild sehen.

## Vorwissen

- Modul-Skript Abschnitte 1.2–1.4 (kNN, Bias/Variance, Evaluation)
- numpy-Basics; `train_test_split` aus Modul 03

## Aufgaben

1. Notebook durcharbeiten; zu implementieren: `euklid`, `knn_vorhersage`, die Standardisierung (nur auf Trainingsdaten — Leakage-Regel!) und der k-Sweep.
2. Mini-Checks müssen `True` ergeben.
3. Den Bias-Variance-Plot mit eigenen Worten beschreiben, bevor du die Musterinterpretation liest.

## Was am Ende funktionieren soll

- Unskaliert (k=5): Accuracy ≈ 0,77 → skaliert: ≈ 0,95.
- Hand-kNN und sklearn stimmen bis auf höchstens einen Testpunkt überein (Tie-Breaks).
- k-Sweep: k=1 → Train 1,0/Test 0,92 (Overfitting); Optimum bei k≈5–31; k=151 → beide ≈ 0,75 (Underfitting).
- Entscheidungsgrenzen-Plot: zackig bei k=1, glatt bei k=31.

## Lösung

Vollständig ausgeführte Musterlösung: [`solution/solution.ipynb`](solution/solution.ipynb).
