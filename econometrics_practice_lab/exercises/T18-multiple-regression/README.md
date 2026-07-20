# T18 – Wohnungsmieten mit mehreren Regressoren

## Ausgangslage

Für 350 Wohnungen werden Monatsmiete, Wohnfläche, Qualitätsindex, Distanz zum Zentrum und Gebäudealter beobachtet. Das multiple Modell isoliert den Zusammenhang einer Variable mit der Miete, während die übrigen beobachteten Merkmale konstant gehalten werden.

## Lernziele

Nach dem Projekt kannst du:

- eine multiple OLS-Regression formulieren und schätzen,
- Koeffizienten als ceteris-paribus-Zusammenhänge interpretieren,
- einfache und multiple Koeffizienten auseinanderhalten,
- den Frisch–Waugh–Lovell-Satz durch Residualisierung nachvollziehen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Untersuche Verteilungen und Korrelationen der vier Regressoren.
2. Schätze zunächst `rent_eur ~ area_sqm`, danach zusätzlich `quality_score`, `distance_center_km` und `building_age_years`.
3. Interpretiere alle multiplen Koeffizienten einschließlich Einheiten und ceteris-paribus-Bedingung.
4. Vergleiche den Flächenkoeffizienten zwischen einfachem und multiplem Modell. Erkläre, weshalb er sich ändern darf.
5. Residualisiere Wohnfläche auf die drei übrigen Regressoren und Miete auf dieselben Regressoren.
6. Regressiere die residualisierte Miete ohne Intercept auf die residualisierte Wohnfläche. Prüfe, dass die Steigung dem multiplen Flächenkoeffizienten entspricht.
7. Schätze ein Modell mit standardisierten Variablen, um die Koeffizientengrößen auf einer gemeinsamen Skala zu vergleichen.
8. Vergleiche R², adjustiertes R² und RMSE von einfachem und multiplem Modell, ohne daraus automatisch Kausalität abzuleiten.

## Ausführen

```bash
python3 exercises/T18-multiple-regression/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/coefficients.csv`, `results/model_comparison.csv` und `results/multiple_regression.png`:

```bash
python3 exercises/T18-multiple-regression/solution.py
```

## Denkfragen

- Was bedeutet „Distanz konstant halten“, obwohl zwei reale Wohnungen selten in allen anderen Merkmalen identisch sind?
- Kann ein multipler Koeffizient kausal interpretiert werden, wenn relevante unbeobachtete Merkmale fehlen?
- Weshalb ist der FWL-Plot ein partieller und kein gewöhnlicher Scatterplot?
