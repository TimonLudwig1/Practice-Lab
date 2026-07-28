# 16 — Fraud Detection Under Extreme Class Imbalance

Difficulty: 🔴 Hard | Topic: Anomaly Detection / Imbalanced Classification

## 🎯 Project Goal
Detect fraudulent credit card transactions in a dataset where only 0.17% of examples are positive, comparing supervised and unsupervised approaches under a realistic operational constraint: a human review team that can inspect at most 200 flagged transactions per day.

## 📊 Dataset Description
**Credit Card Fraud Detection (ULB)** — 284,807 transactions over 2 days, 492 frauds. Features V1–V28 are PCA-anonymized, plus `Time` and `Amount`.
Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (free Kaggle account required).

## 📏 Evaluation Metric
- Primary: **AUPRC** (area under the precision-recall curve) — ROC-AUC is misleading at this imbalance and you should be able to explain why
- Operational: **precision and recall within the top-K alert budget** (K derived from the 200/day review capacity scaled to the dataset's time span)
- Split must respect time: train on the earlier portion, evaluate on the later portion

## 🏁 Success Criteria
- At least one supervised approach and one unsupervised/anomaly approach (trained without labels), evaluated on the identical time-based split
- AUPRC ≥ 0.80 for the best model
- A top-K alert analysis: with the review budget, how many frauds are caught, how much fraud value (€ via `Amount`) is recovered, and what is the analyst precision?
- An explicit comparison: when would the unsupervised approach be the right choice despite weaker metrics?
- A short note on why resampling techniques must never touch the evaluation set

---

# Deutsche Übersetzung

# 16 — Betrugserkennung bei extrem unausgeglichenen Klassen

Schwierigkeit: 🔴 Anspruchsvoll | Thema: Anomalieerkennung und unausgeglichene Klassifikation

## 🎯 Projektziel
Erkenne betrügerische Kreditkartentransaktionen in einem Datensatz, in dem nur 0,17 % der Beispiele positiv sind. Vergleiche überwachte und unüberwachte Verfahren unter der realistischen Bedingung, dass ein Prüferteam höchstens 200 markierte Transaktionen pro Tag untersuchen kann.

## 📊 Beschreibung des Datensatzes
**Credit Card Fraud Detection (ULB)** enthält 284.807 Transaktionen aus zwei Tagen, darunter 492 Betrugsfälle. Die Merkmale V1 bis V28 sind per PCA anonymisiert; zusätzlich gibt es `Time` und `Amount`.
Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. Ein kostenloses Kaggle-Konto wird benötigt.

## 📏 Bewertungsmetriken
- Primär: **AUPRC**, die Fläche unter der Precision-Recall-Kurve. ROC-AUC kann bei diesem Ungleichgewicht irreführend sein, was du erklären können solltest.
- Operativ: **Precision und Recall innerhalb des Top-K-Alarmbudgets**. K ergibt sich aus der Kapazität von 200 Prüfungen pro Tag, skaliert auf den Zeitraum des Datensatzes.
- Die Aufteilung muss die Zeit berücksichtigen: Trainiere auf dem früheren und bewerte auf dem späteren Teil.

## 🏁 Erfolgskriterien
- Mindestens ein überwachtes und ein ohne Labels trainiertes Anomalieverfahren auf derselben zeitbasierten Aufteilung
- AUPRC von mindestens 0,80 für das beste Modell
- Top-K-Analyse: Anzahl erkannter Betrugsfälle, geretteter Betragswert über `Amount` und Precision der Prüfer bei gegebenem Budget
- Expliziter Vergleich, wann das unüberwachte Verfahren trotz schwächerer Metriken die bessere Wahl wäre
- Kurzer Hinweis, warum Resampling-Verfahren niemals den Bewertungssatz verändern dürfen
