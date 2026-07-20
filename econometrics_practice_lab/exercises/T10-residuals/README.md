# T10 – Mietpreise durch Residuen verstehen

## Ausgangslage

Ein einfaches Modell erklärt monatliche Wohnungsmieten ausschließlich über die Wohnfläche. Residuen zeigen für jede Wohnung, wie weit die beobachtete Miete ober- oder unterhalb der Modellvorhersage liegt. Einige bewusst ungewöhnliche Angebote helfen dabei, Residuen als Diagnosewerkzeug zu verwenden.

## Lernziele

Nach dem Projekt kannst du:

- Fits und Residuen für jede Beobachtung berechnen,
- Vorzeichen und Größe eines Residuums in Originaleinheiten interpretieren,
- zentrale algebraische Eigenschaften von OLS-Residuen prüfen,
- große Residuen von einem generell falsch spezifizierten Modell unterscheiden.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Schätze `rent_eur` auf `floor_area_sqm` und speichere \(\hat y_i\) sowie \(\hat u_i=y_i-\hat y_i\).
2. Wähle drei Wohnungen aus und bestätige numerisch \(y_i=\hat y_i+\hat u_i\).
3. Interpretiere ein Residuum von +400 Euro und eines von −400 Euro in diesem Kontext.
4. Prüfe Summe und Mittelwert der Residuen sowie ihre Kovarianz mit Wohnfläche und Fits.
5. Identifiziere die fünf größten absoluten Residuen. Sind dies automatisch fehlerhafte Datenpunkte?
6. Zeichne ausgewählte Residuen als vertikale Abstände zwischen Beobachtung und Regressionsgerade.
7. Erstelle Residuen-gegen-Fits-Plot und Histogramm. Suche nach Asymmetrie, Ausreißern und systematischen Mustern.
8. Nenne mindestens drei ausgelassene Wohnungsmerkmale, die große Residuen plausibel erklären könnten.

## Ausführen

```bash
python3 exercises/T10-residuals/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/residual_table.csv`, `results/residual_summary.csv` und `results/residual_diagnostics.png`:

```bash
python3 exercises/T10-residuals/solution.py
```

## Denkfragen

- Ist ein positives Residuum dasselbe wie eine positive Steigung?
- Weshalb summieren sich OLS-Residuen nur bei einem Modell mit Intercept zu null?
- Kann ein Modell kleine Residuen in der Stichprobe und trotzdem schlechte Vorhersagen außerhalb der Stichprobe haben?
