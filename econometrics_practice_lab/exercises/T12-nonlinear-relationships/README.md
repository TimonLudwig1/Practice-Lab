# T12 – Nichtlinearität im Energieverbrauch erkennen

## Ausgangslage

Der tägliche Energieverbrauch eines Gebäudes ist sowohl an sehr kalten als auch an sehr heißen Tagen hoch. Ein einfaches lineares Modell kann diese U-Form nicht abbilden. Du nutzt Residuen, um die fehlende Struktur sichtbar zu machen, und vergleichst das Modell mit einer quadratischen Spezifikation.

## Lernziele

Nach dem Projekt kannst du:

- Nichtlinearität bereits im Scatterplot erkennen,
- systematische Residuenmuster als Spezifikationsproblem deuten,
- einen quadratischen Term korrekt in eine Regression aufnehmen,
- Fits und Residuen zweier funktionaler Formen vergleichen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Zeichne Energieverbrauch gegen Außentemperatur. Beschreibe die sichtbare funktionale Form.
2. Schätze dennoch ein lineares Modell `energy_kwh ~ temperature_c`.
3. Zeichne die linearen Residuen gegen Temperatur und berechne ihre Mittelwerte in Temperaturintervallen.
4. Erkläre, warum Residuen bei niedrigen und hohen Temperaturen überwiegend positiv sein sollten.
5. Ergänze `temperature_squared` und schätze ein quadratisches Modell.
6. Zeichne die vorhergesagte quadratische Kurve in Originaleinheiten.
7. Vergleiche RMSE, R² und Residuenplots beider Modelle.
8. Berechne den Temperaturwert des geschätzten Kurvenminimums \(-\hat\beta_1/(2\hat\beta_2)\) und interpretiere ihn vorsichtig.

## Ausführen

```bash
python3 exercises/T12-nonlinear-relationships/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/model_comparison.csv`, `results/residuals_by_temperature_bin.csv` und `results/nonlinearity_diagnostics.png`:

```bash
python3 exercises/T12-nonlinear-relationships/solution.py
```

## Denkfragen

- Kann die Steigung des quadratischen Modells mit nur einer Zahl beschrieben werden?
- Warum ist ein hohes R² allein kein Ersatz für einen Residuenplot?
- Welche Gefahr entsteht bei Vorhersagen weit außerhalb des beobachteten Temperaturbereichs?
