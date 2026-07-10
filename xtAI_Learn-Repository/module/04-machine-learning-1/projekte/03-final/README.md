# Projekt 03 (final) — Einkommensvorhersage auf echten Zensusdaten

**Format:** Jupyter Notebook (`einkommensvorhersage.ipynb`).
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

Vollständig ausgeführte Musterlösung: [`loesung/loesung.ipynb`](loesung/loesung.ipynb). Bitte erst selbst versuchen.
