# Project 03 (final) — Income prediction on real census data

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format:** Jupyter notebook (`income_prediction.ipynb`).
**Why this format?** As in project 02, the model comparison, the cost-threshold plot, the fairness table and the importance plot belong visually together with the code that produces them.

**Data: real.** UCI **Adult / Census Income** (US census 1994, 48,842 people), loaded via `sklearn.datasets.fetch_openml("adult", version=2)` — no manual download needed, scikit-learn caches the data set locally in `~/scikit_learn_data/` (not part of the repository; the first run needs internet access). **A practical connection as required**: a real, classical ML task (predicting income from census attributes) with all the ugliness of real data — mixed numerical/categorical types, genuine missing values (`workclass`, `occupation`, `native-country`), class imbalance (about 24 % `>50K`) and sensitive attributes (`sex`, `race`) that invite a fairness analysis.

## Goal

Consolidate the module end to end: ColumnTransformer pipelines for mixed types, a fair model comparison under imbalance (PR-AUC instead of accuracy), tuning, and two topics that textbooks often skimp on but that are central in practice — **choosing a threshold by costs** and a **fairness check between subgroups**.

## Prior knowledge

- The complete module script, in particular 2.4 (pipelines/ColumnTransformer), 2.6 (imbalanced classes), 3.1 (interpretation)
- Project 02 (pipelines, CV, GridSearchCV)

## Tasks

1. **Load and explore**: `fetch_openml`, look at the missing values and the target distribution.
2. **Preparation**: encode the target variable as binary, remove `fnlwgt` (a census weight, not an attribute) and `education` (redundant with `education-num`), define the feature lists (numerical/categorical).
3. **Split**: stratified; do not touch the test set until step 6.
4. **Preprocessing pipeline**: a `ColumnTransformer` with imputation + scaling (numerical) and imputation + one-hot encoding (categorical, `sparse_output=False`, because `HistGradientBoostingClassifier` needs dense matrices).
5. **Baseline**: a `DummyClassifier`.
6. **Model comparison under imbalance**: three pipelines (logistic regression, random forest, gradient boosting), each with `class_weight="balanced"`, compared via `StratifiedKFold` + `scoring="average_precision"` (PR-AUC).
7. **Tuning**: refine the best model with `GridSearchCV` (scoring still PR-AUC).
8. **The one-off test evaluation**: classification report, PR curve, PR-AUC on the test set at the default threshold of 0.5.
9. **Choosing a threshold by a cost calculation**: a scenario with asymmetric costs (a false alarm 50 dollars vs. a missed case 200 dollars), find the cost-minimal threshold.
10. **Fairness check**: recall/false positive rate separately by `sex` at the chosen threshold — assess the pattern.
11. **Permutation importance** on the original raw features (not on the one-hot dummies).

## What should work in the end

- CV PR-AUC: logistic regression about 0.76, random forest about 0.71, gradient boosting about 0.83 (gradient boosting wins clearly).
- The tuned gradient boosting model reaches PR-AUC about 0.83 on the test set; at threshold 0.5: recall(`>50K`) about 0.87, precision about 0.60.
- The cost-minimal threshold lies at about 0.44 (moderately below 0.5 — `class_weight="balanced"` has already shifted the scores, so the cost threshold only fine-tunes).
- The fairness table shows a real gap: recall for `Female` (base rate `>50K` about 11 %) is about 0.80, for `Male` (base rate about 30 %) about 0.91 — a concrete result worth discussing, not an artefact.
- Permutation importance highlights `capital-gain`, `education-num`, `age`, `hours-per-week`, `marital-status`.

## Why real practice data (no design decision needed)

For this final project there was a freely available, established real data set with exactly the properties that make up the practical connection (messiness, imbalance, ethical questions) — hence no synthetic substitute data set was needed.

## Solution

Fully executed reference solution: [`solution/solution.ipynb`](solution/solution.ipynb). Please try it yourself first.

---
---

# Projekt 03 (final) — Einkommensvorhersage auf echten Zensusdaten (deutsche Fassung)

**Format:** Jupyter Notebook (`income_prediction.ipynb`).
**Warum dieses Format?** Wie in Projekt 02 gehören Modellvergleich, Kosten-Schwellen-Plot, Fairness-Tabelle und Importance-Plot visuell zusammen mit dem Code, der sie erzeugt.

**Daten: echt.** UCI **Adult / Census Income** (US-Zensus 1994, 48.842 Personen), geladen über `sklearn.datasets.fetch_openml("adult", version=2)` — kein manueller Download nötig, scikit-learn cached den Datensatz lokal in `~/scikit_learn_data/` (nicht Teil des Repos, erste Ausführung braucht Internetzugang). **Praxisbezug wie gefordert**: eine reale, klassische ML-Aufgabe (Einkommen aus Zensusmerkmalen vorhersagen) mit allen Unschönheiten echter Daten — gemischte numerische/kategoriale Typen, echte fehlende Werte (`workclass`, `occupation`, `native-country`), Klassenungleichgewicht (~24 % `>50K`) und sensible Attribute (`sex`, `race`), die eine Fairness-Betrachtung nahelegen.

## Ziel

Das Modul end-to-end konsolidieren: ColumnTransformer-Pipelines für gemischte Typen, fairer Modellvergleich unter Ungleichgewicht (PR-AUC statt Accuracy), Tuning, und zwei Themen, die in Lehrbüchern oft zu kurz kommen, aber in der Praxis zentral sind — **Schwellenwahl nach Kosten** und ein **Fairness-Check zwischen Untergruppen**.

## Vorwissen

- Das komplette Modul-Skript, insbesondere 2.4 (Pipelines/ColumnTransformer), 2.6 (unausgewogene Klassen), 3.1 (Interpretation)
- Projekt 02 (Pipelines, CV, GridSearchCV)

## Aufgaben

1. **Laden & explorieren**: `fetch_openml`, fehlende Werte und Zielverteilung ansehen.
2. **Aufbereitung**: Zielvariable binär codieren, `fnlwgt` (Zensus-Gewicht, kein Merkmal) und `education` (redundant zu `education-num`) entfernen, Feature-Listen (numerisch/kategorial) definieren.
3. **Split**: stratifiziert, Testsatz bis Schritt 6 nicht anfassen.
4. **Preprocessing-Pipeline**: `ColumnTransformer` mit Imputation + Skalierung (numerisch) bzw. Imputation + One-Hot-Encoding (kategorial, `sparse_output=False`, da `HistGradientBoostingClassifier` dichte Matrizen braucht).
5. **Baseline**: `DummyClassifier`.
6. **Modellvergleich unter Ungleichgewicht**: drei Pipelines (Logistische Regression, Random Forest, Gradient Boosting) je mit `class_weight="balanced"`, per `StratifiedKFold` + `scoring="average_precision"` (PR-AUC) vergleichen.
7. **Tuning**: Bestes Modell per `GridSearchCV` (Scoring weiterhin PR-AUC) verfeinern.
8. **Einmalige Testauswertung**: Classification Report, PR-Kurve, PR-AUC auf dem Testsatz bei Standard-Schwelle 0,5.
9. **Schwellenwahl per Kostenrechnung**: Szenario mit asymmetrischen Kosten (Fehlalarm 50 $ vs. übersehener Fall 200 $), kostenminimale Schwelle finden.
10. **Fairness-Check**: Recall/False-Positive-Rate getrennt nach `sex` bei der gewählten Schwelle — Muster einordnen.
11. **Permutation Importance** auf den Original-Rohfeatures (nicht auf One-Hot-Dummies).

## Was am Ende funktionieren soll

- CV-PR-AUC: Logistische Regression ≈ 0,76, Random Forest ≈ 0,71, Gradient Boosting ≈ 0,83 (Gradient Boosting gewinnt deutlich).
- Getuntes Gradient-Boosting-Modell erreicht auf dem Testsatz PR-AUC ≈ 0,83, bei Schwelle 0,5: Recall(`>50K`) ≈ 0,87, Precision ≈ 0,60.
- Kostenminimale Schwelle liegt bei ≈ 0,44 (moderat unter 0,5 — `class_weight="balanced"` hat die Scores schon vorverschoben, die Kostenschwelle korrigiert nur noch fein nach).
- Fairness-Tabelle zeigt eine reale Kluft: Recall für `Female` (Basisrate `>50K` ≈ 11 %) liegt bei ≈ 0,80, für `Male` (Basisrate ≈ 30 %) bei ≈ 0,91 — ein konkretes, diskussionswürdiges Ergebnis, kein Artefakt.
- Permutation Importance hebt `capital-gain`, `education-num`, `age`, `hours-per-week`, `marital-status` hervor.

## Warum echte Praxisdaten (keine Design-Entscheidung nötig)

Für dieses Abschlussprojekt gab es einen frei verfügbaren, etablierten echten Datensatz mit genau den Eigenschaften, die den Praxisbezug ausmachen (Unordentlichkeit, Ungleichgewicht, ethische Fragen) — daher kein synthetischer Ersatzdatensatz nötig.

## Lösung

Vollständig ausgeführte Musterlösung: [`solution/solution.ipynb`](solution/solution.ipynb). Bitte erst selbst versuchen.
