# T37 – Within-Estimator für Unternehmensinvestitionen

## Ausgangslage

Unternehmen mit dauerhaft guten Investitionsmöglichkeiten besitzen im Mittel höheren Cashflow und investieren unabhängig davon mehr. Der Within-Estimator entfernt das Firmenmittel aus Outcome und Regressor. Eine Regression der zentrierten Größen liefert denselben Steigungskoeffizienten wie eine Regression mit Firmendummies.

## Lernziele

- die Within-Transformation \(x_{it}-\bar x_i\) durchführen,
- zeigen, dass zeitinvariante Firmenkomponenten verschwinden,
- den Within-Koeffizienten per OLS und Kovarianzquotient berechnen,
- die numerische Gleichheit von Within- und Dummy-Estimator verifizieren,
- den FE-Koeffizienten als Zusammenhang von Abweichungen vom Firmenmittel interpretieren.

## Aufgaben

1. Berechne Firmenmittel von Cashflow und Investition.
2. Erzeuge `cash_flow_within` und `investment_within`.
3. Prüfe, dass beide transformierten Variablen innerhalb jeder Firma Mittelwert null haben.
4. Schätze die Regression ohne Intercept auf den transformierten Daten.
5. Berechne den Koeffizienten manuell als \(\sum \tilde x\tilde y/\sum \tilde x^2\).
6. Schätze dasselbe Modell mit Firmendummies und vergleiche die Koeffizienten numerisch.
7. Vergleiche zusätzlich mit Pooled OLS und erkläre die Abweichung.

## Ausführen

```bash
python3 exercises/T37-within-estimator/starter.py
python3 exercises/T37-within-estimator/solution.py
```

Die Lösung erzeugt `data/firm_investment_panel.csv`, Ergebnistabellen und `results/within_estimator.png`.

## Denkfragen

- Was passiert bei der Within-Transformation mit einem konstanten Branchenindikator?
- Welche Firmen tragen wenig zur Schätzung bei?
- Warum braucht die transformierte Regression keinen Intercept?
