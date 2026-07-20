# T16 – Inferenz für Regressionskoeffizienten

## Ausgangslage

220 Filialen testen unterschiedlich große Rabattaktionen. Du schätzt den Zusammenhang zwischen Rabatthöhe und wöchentlichem Absatz und leitest Standardfehler, t-Tests, p-Werte und Konfidenzintervalle für die Steigung aus den OLS-Bausteinen her.

## Lernziele

Nach dem Projekt kannst du:

- den Standardfehler einer einfachen OLS-Steigung manuell berechnen,
- beliebige Nullhypothesen \(H_0:\beta_1=c\) testen,
- 95%- und 99%-Konfidenzintervalle konstruieren,
- Koeffizientenintervalle von Mittelwerts- und Vorhersageintervallen unterscheiden.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Schätze `weekly_units ~ discount_percentage` mit Intercept.
2. Berechne \(S_{xx}=\sum_i(x_i-\bar x)^2\), \(\hat\sigma^2=\sum_i\hat u_i^2/(n-2)\) und \(SE(\hat\beta_1)=\sqrt{\hat\sigma^2/S_{xx}}\) manuell.
3. Kontrolliere den Standardfehler mit `statsmodels`.
4. Teste zweiseitig \(H_0:\beta_1=0\) und \(H_0:\beta_1=3\). Berechne jeweils t-Wert, Freiheitsgrade und p-Wert.
5. Konstruiere ein 95%- und ein 99%-Konfidenzintervall für \(\beta_1\).
6. Zeige für beide Nullhypothesen, dass Testentscheidung und Lage relativ zum 95%-Intervall übereinstimmen.
7. Zeichne Regressionsgerade und 95%-Konfidenzband des bedingten Mittelwerts.
8. Erkläre, warum das Konfidenzband des Mittelwerts enger als ein Vorhersageintervall für eine neue Filiale wäre.

## Ausführen

```bash
python3 exercises/T16-coefficient-inference/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/coefficient_tests.csv`, `results/coefficient_intervals.csv` und `results/coefficient_inference.png`:

```bash
python3 exercises/T16-coefficient-inference/solution.py
```

## Denkfragen

- Warum steht im Zähler der t-Statistik \(\hat\beta_1-c\) statt nur \(\hat\beta_1\)?
- Was wird bei einem höheren Konfidenzniveau mit dem Intervall passieren?
- Bedeutet ein statistisch signifikanter Effekt automatisch einen wirtschaftlich großen Effekt?
