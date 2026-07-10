# Projekt 01 (basic) — kNN von Hand & der Bias-Variance-Tradeoff zum Anfassen

**Format:** Jupyter Notebook (`knn_von_hand.ipynb`).
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

Vollständig ausgeführte Musterlösung: [`loesung/loesung.ipynb`](loesung/loesung.ipynb).
