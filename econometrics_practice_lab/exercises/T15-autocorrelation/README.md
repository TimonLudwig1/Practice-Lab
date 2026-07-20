# T15 – Autokorrelation in täglichen Bestellungen

## Ausgangslage

Die täglichen Bestellungen eines Onlineshops folgen einem linearen Wachstumstrend. Nicht beobachtete Nachfrageschocks verschwinden jedoch nicht sofort: Ein positiver Schock wirkt typischerweise in den folgenden Tagen weiter. Dadurch sind aufeinanderfolgende Regressionsfehler positiv korreliert.

## Lernziele

Nach dem Projekt kannst du:

- serielle Korrelation in zeitlich geordneten Residuen erkennen,
- Lag-Plot, Autokorrelationsfunktion und Durbin–Watson-Statistik lesen,
- erklären, warum unabhängige Beobachtungen bei Zeitreihen problematisch sein können,
- konventionelle und HAC-robuste Standardfehler vergleichen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Zeichne Bestellungen über die Zeit und schätze `orders ~ day`.
2. Stelle die Residuen in ihrer zeitlichen Reihenfolge dar. Suche nach längeren positiven oder negativen Runs.
3. Erzeuge Paare \((\hat u_{t-1},\hat u_t)\) und berechne die Lag-1-Korrelation.
4. Berechne die Residuen-ACF für Lags 1 bis 14 sowie die Durbin–Watson-Statistik.
5. Führe einen Ljung–Box-Test für Lag 7 durch und interpretiere seine Nullhypothese.
6. Vergleiche konventionelle und HAC-robuste Standardfehler mit `maxlags=7` für den Zeittrend.
7. Erkläre, weshalb Autokorrelation bei korrektem bedingtem Mittelwert nicht automatisch die OLS-Steigung verzerrt, aber die übliche Inferenz gefährdet.
8. Nenne Modelländerungen, die bei echter Anwendung vor einer reinen Standardfehlerkorrektur geprüft werden sollten.

## Ausführen

```bash
python3 exercises/T15-autocorrelation/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/autocorrelation_summary.csv`, `results/inference_comparison.csv` und `results/autocorrelation_diagnostics.png`:

```bash
python3 exercises/T15-autocorrelation/solution.py
```

## Denkfragen

- Welcher Durbin–Watson-Wert wäre bei fehlender Lag-1-Autokorrelation ungefähr zu erwarten?
- Reparieren HAC-Standardfehler eine fehlende Saisonvariable im bedingten Mittelwert?
- Warum würde zufälliges Mischen der Zeilen die sichtbare Zeitstruktur zerstören?
