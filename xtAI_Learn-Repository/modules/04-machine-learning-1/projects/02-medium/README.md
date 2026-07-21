# Projekt 02 (medium) — Das ehrliche Modellrennen: Pipelines, CV & Tuning

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
