# T20 – Multikollinearität bei Gebäudeindikatoren

## Ausgangslage

Ein Isolationsindex und ein thermischer Effizienzscore messen nahezu dieselbe latente Gebäudequalität. Beide sollen Energiekosten erklären. Das gemeinsame Modell kann gut vorhersagen, hat aber Schwierigkeiten, den separaten Beitrag der beiden fast redundanten Regressoren präzise zu bestimmen.

## Lernziele

Nach dem Projekt kannst du:

- hohe, aber nicht perfekte Multikollinearität diagnostizieren,
- Variance Inflation Factors berechnen und interpretieren,
- aufgeblähte Standardfehler von Verzerrung unterscheiden,
- Koeffizienten- und Vorhersagestabilität getrennt beurteilen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Zeichne `insulation_index` gegen `thermal_score` und berechne ihre Korrelation.
2. Schätze drei Modelle: nur Isolationsindex, nur thermischer Score und beide gemeinsam.
3. Vergleiche Koeffizienten, Standardfehler, R² und RMSE.
4. Berechne für das gemeinsame Modell den VIF beider Regressoren.
5. Erkläre, weshalb die Einzelmodelle ungefähr den kombinierten Zusammenhang beider Indikatoren aufnehmen.
6. Ziehe 600 Bootstrap-Stichproben und speichere beide gemeinsamen Koeffizienten sowie ihre Summe.
7. Vergleiche die Bootstrap-Streuung der Einzelkoeffizienten mit der Streuung ihrer Summe.
8. Diskutiere, wann Multikollinearität für Vorhersage weniger problematisch als für getrennte inhaltliche Effekte ist.

## Ausführen

```bash
python3 exercises/T20-multicollinearity/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/vif.csv`, `results/model_coefficients.csv`, `results/bootstrap_summary.csv` und `results/multicollinearity.png`:

```bash
python3 exercises/T20-multicollinearity/solution.py
```

## Denkfragen

- Verletzt hohe Multikollinearität automatisch die Erwartungstreue von OLS?
- Ist das Löschen eines Regressors immer eine fachlich zulässige Lösung?
- Warum können zwei stark schwankende Koeffizienten gemeinsam eine relativ stabile Größe ergeben?
