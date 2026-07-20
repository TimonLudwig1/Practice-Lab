# T27 – Logit-Modell für Vertragsverlängerungen

## Ausgangslage

Ein Streamingdienst untersucht, ob Kundinnen und Kunden ihren Vertrag verlängern. Die Verlängerungswahrscheinlichkeit hängt von einer angekündigten Preiserhöhung, der bisherigen Vertragsdauer und automatischer Zahlung ab. Das Logit-Modell hält vorhergesagte Wahrscheinlichkeiten innerhalb von null und eins, ist in seinen Koeffizienten aber auf der Log-Odds-Skala linear.

## Lernziele

Nach dem Projekt kannst du:

- ein Logit-Modell für ein binäres Outcome schätzen,
- den linearen Index in Log-Odds, Odds und Wahrscheinlichkeiten übersetzen,
- Koeffizienten und exponentierte Koeffizienten als Odds Ratios interpretieren,
- Szenariowahrscheinlichkeiten für konkrete Kovariaten berechnen,
- die S-Form der Wahrscheinlichkeitsfunktion erklären,
- Kalibrierung und Modellfit mit sinnvollen Diagnosen prüfen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Schätze `renewed` auf `price_increase_pct`, `loyalty_years` und `automatic_payment` mit einem Logit-Modell.
2. Gib Koeffizienten, Standardfehler, 95%-Konfidenzintervalle und Odds Ratios aus.
3. Interpretiere das Odds Ratio der Preiserhöhung für einen zusätzlichen Prozentpunkt. Unterscheide Odds ausdrücklich von Wahrscheinlichkeit.
4. Berechne für eine Person mit drei Loyalitätsjahren ohne automatische Zahlung den linearen Index, die Odds und die Verlängerungswahrscheinlichkeit bei mehreren Preiserhöhungen.
5. Zeige algebraisch und numerisch: \(p=\exp(\eta)/(1+\exp(\eta))\) und \(\eta=\log(p/(1-p))\).
6. Erzeuge Kalibrierungsdezile aus den vorhergesagten Wahrscheinlichkeiten und vergleiche vorhergesagte mit beobachteten Anteilen.
7. Vergleiche Brier Score und Log-Likelihood mit einer konstanten Basisprognose.
8. Erkläre anhand der Grafik, warum derselbe Logit-Koeffizient nicht überall dieselbe Änderung in Prozentpunkten bedeutet.

## Ausführen

```bash
python3 exercises/T27-logit-model/starter.py
```

Die Musterlösung erzeugt `data/subscription_renewals.csv` sowie `results/coefficients_and_odds_ratios.csv`, `results/scenario_predictions.csv`, `results/calibration_deciles.csv`, `results/model_diagnostics.csv` und `results/logit_model.png`:

```bash
python3 exercises/T27-logit-model/solution.py
```

## Denkfragen

- Was bedeutet ein Odds Ratio von 0,84 bei einer zusätzlichen Einheit von `price_increase_pct`?
- Weshalb ist ein Koeffizient von null äquivalent zu einem Odds Ratio von eins?
- Wo reagiert die vorhergesagte Wahrscheinlichkeit am stärksten auf eine Änderung des linearen Index?
- Warum kann ein Modell gut trennen, aber schlecht kalibriert sein?
