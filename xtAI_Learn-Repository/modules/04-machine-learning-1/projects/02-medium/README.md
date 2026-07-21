# Project 02 (medium) — The honest model race: pipelines, CV and tuning

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format:** Jupyter notebook (`model_race.ipynb`).
**Why this format?** Model comparisons live on tables and box plots right next to the code — and on the immediate visual check ("does the ROC curve match the number?").

**Data: real.** The **Breast Cancer Wisconsin (Diagnostic)** data set, directly available via `sklearn.datasets.load_breast_cancer()` (569 tumour samples, 30 numerical features from images of cell nuclei, target: *malignant* vs. *benign*, class ratio about 37/63 — real and slightly imbalanced). No download needed, no synthetic data required — ideal for practising the craft (pipelines, CV, tuning) on real but manageable data.

## Goal

You enter a "model competition" against yourself: preprocess several model families from the script (chapter 2) cleanly via a **pipeline**, compare them fairly with **stratified cross-validation**, tune the best two with **grid/randomised search**, and at the end do the honest final accounting **once** on the test set — including interpretation (permutation importance, calibration, learning curve).

Compared with project 01 there is markedly **less guidance**: you get intermediate goals and checks, but you are to derive the pipeline and grid search construction yourself from the script (sections 2.1–2.5).

## Prior knowledge

- The module script in full (especially 1.4–1.5 evaluation/metrics, 2.1–2.5 models/pipelines/tuning)
- Project 01 (kNN, bias-variance, scaling)
- The basic sklearn routine: `train_test_split`, `.fit`/`.predict` (modules 02/03)

## Tasks

1. **Load and explore**: `load_breast_cancer`, check the class distribution, stratified train/test split (put the test set away — do not touch it until step 6!).
2. **Baseline**: a `DummyClassifier` as the lower bound — every serious model has to beat it.
3. **Build pipelines**: for every model (logistic regression, kNN, SVM (RBF kernel), decision tree, random forest, gradient boosting) a `Pipeline` with `StandardScaler` + classifier. (Tree models do not need scaling — build them in consistently anyway, it does no harm and keeps the code uniform.)
4. **Fair comparison**: evaluate all pipelines with `StratifiedKFold(5)` and `cross_val_score` (scoring: `roc_auc`), present the results as a box plot. Which 2 models go into tuning?
5. **Hyperparameter tuning**: define a parameter grid for each of your top 2 models and tune it on the training data with `GridSearchCV` (or `RandomizedSearchCV`). Print the best CV result and the best parameters.
6. **The one-off test evaluation**: fit the best model finally on all of train, evaluate it **once** on the test set: confusion matrix, precision/recall/F1, ROC curve + AUC.
7. **Interpretation**: permutation importance on the test set (plot the top 10 features), briefly assess whether the most important features seem medically plausible (e.g. `worst radius`, `worst concave points`).
8. **Learning curve** for the final model: a bias or a variance problem?
9. Answer the reflection questions at the end of the notebook in your own words.

## What should work in the end

- The CV ROC-AUC lies between about 0.92 (the single decision tree — the only model clearly off the pace, exactly as script 2.2 predicts: single trees are unstable) and about 0.996 (logistic regression and SVM in front, gradient boosting just behind at about 0.992).
- The tuned model reaches ROC-AUC at least 0.98 and F1 at least 0.95 on the test set (reference: logistic regression wins the tuning with a CV of 0.9959 and reaches 0.9954 on the test set).
- Permutation importance highlights plausible `worst` features.
- You can justify in your own words why the test set was touched only once.

## Solution

Fully executed reference solution: [`solution/solution.ipynb`](solution/solution.ipynb). Please try it yourself first — this module lives on the craft, not on copying.

---
---

# Projekt 02 (medium) — Das ehrliche Modellrennen: Pipelines, CV & Tuning (deutsche Fassung)

**Format:** Jupyter Notebook (`model_race.ipynb`).
**Warum dieses Format?** Modellvergleiche leben von Tabellen und Boxplots direkt neben dem Code — und vom sofortigen visuellen Check ("stimmt die ROC-Kurve mit der Zahl überein?").

**Daten: echt.** Der **Breast Cancer Wisconsin (Diagnostic)**-Datensatz, direkt über `sklearn.datasets.load_breast_cancer()` verfügbar (569 Tumor-Proben, 30 numerische Features aus Zellkernbildern, Ziel: *malignant* (bösartig) vs. *benign* (gutartig), Klassenverhältnis ≈ 37/63 — real und leicht unausgewogen). Kein Download nötig, keine synthetischen Daten erforderlich — ideal, um das Handwerk (Pipelines, CV, Tuning) an echten, aber überschaubaren Daten zu üben.

## Ziel

Du trittst als "Modell-Wettbewerb" gegen dich selbst an: Mehrere Modellfamilien aus dem Skript (Kap. 2) sauber per **Pipeline** vorverarbeiten, per **stratifizierter Kreuzvalidierung** fair vergleichen, die zwei besten per **GridSearch/RandomizedSearch** tunen, und am Ende **einmal** auf dem Testsatz die ehrliche Endabrechnung machen — inklusive Interpretation (Permutation Importance, Kalibrierung, Lernkurve).

Im Vergleich zu Projekt 01 gibt es deutlich **weniger Anleitung**: Du bekommst Zwischenziele und Checks, aber die Pipeline-/GridSearch-Konstruktion sollst du selbst aus dem Skript (Abschnitte 2.1–2.5) ableiten.

## Vorwissen

- Modul-Skript komplett (v. a. 1.4–1.5 Evaluation/Metriken, 2.1–2.5 Modelle/Pipelines/Tuning)
- Projekt 01 (kNN, Bias-Variance, Skalierung)
- sklearn-Grundroutine: `train_test_split`, `.fit`/`.predict` (Module 02/03)

## Aufgaben

1. **Daten laden & explorieren**: `load_breast_cancer`, Klassenverteilung prüfen, stratifizierter Train/Test-Split (Test einmal weglegen — bis Schritt 6 nicht anfassen!).
2. **Baseline**: `DummyClassifier` als Untergrenze — jedes ernsthafte Modell muss das schlagen.
3. **Pipelines bauen**: Für jedes Modell (Logistische Regression, kNN, SVM (RBF-Kernel), Entscheidungsbaum, Random Forest, Gradient Boosting) eine `Pipeline` mit `StandardScaler` + Klassifikator. (Baum-Modelle brauchen keine Skalierung — bau sie trotzdem konsistent mit ein, das schadet nicht und hält den Code einheitlich.)
4. **Fairer Vergleich**: Alle Pipelines mit `StratifiedKFold(5)` und `cross_val_score` (Scoring: `roc_auc`) bewerten, Ergebnisse als Boxplot darstellen. Welche 2 Modelle gehen ins Tuning?
5. **Hyperparameter-Tuning**: Für deine Top-2-Modelle je ein Parameter-Grid definieren und mit `GridSearchCV` (oder `RandomizedSearchCV`) auf den Trainingsdaten tunen. Bestes CV-Ergebnis und beste Parameter ausgeben.
6. **Die einmalige Testauswertung**: Bestes Modell final auf ganz Train fitten, **einmal** auf dem Testsatz auswerten: Confusion Matrix, Precision/Recall/F1, ROC-Kurve + AUC.
7. **Interpretation**: Permutation Importance auf dem Testsatz (Top-10-Features plotten), kurz einordnen, ob die wichtigsten Features medizinisch plausibel wirken (z. B. `worst radius`, `worst concave points`).
8. **Lernkurve** für das finale Modell: Bias- oder Varianz-Problem?
9. Beantworte die Reflexionsfragen am Ende des Notebooks in eigenen Worten.

## Was am Ende funktionieren soll

- CV-ROC-AUC aller Modelle liegt zwischen ~0,95 (kNN unskaliert-Baseline wäre schlechter, aber mit Pipeline sollten alle über 0,95 liegen) und ~0,99 (Gradient Boosting/logistische Regression typischerweise vorne).
- Getuntes Modell erreicht auf dem Testsatz ROC-AUC ≥ 0,98 und F1 ≥ 0,95.
- Permutation Importance hebt plausible `worst`-Features hervor.
- Du kannst in eigenen Worten begründen, warum der Testsatz nur einmal angefasst wurde.

## Lösung

Vollständig ausgeführte Musterlösung: [`solution/solution.ipynb`](solution/solution.ipynb). Bitte erst selbst versuchen — das Modul lebt vom Handwerk, nicht vom Abschreiben.
