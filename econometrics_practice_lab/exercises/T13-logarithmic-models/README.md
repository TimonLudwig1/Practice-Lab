# T13 – Vier logarithmische Regressionsmodelle

## Ausgangslage

Vier synthetische Mini-Datensätze wurden mit unterschiedlichen funktionalen Formen erzeugt: Level-Level, Log-Level, Level-Log und Log-Log. Du schätzt jedes Modell passend und übersetzt denselben mathematischen Koeffizienten in vier unterschiedliche inhaltliche Aussagen.

## Lernziele

Nach dem Projekt kannst du:

- Level-Level-, Log-Level-, Level-Log- und Log-Log-Modelle unterscheiden,
- die richtige Variable auf der richtigen Seite logarithmieren,
- Koeffizienten als Einheiten-, Prozent- oder Elastizitätseffekt interpretieren,
- Approximationen von exakten prozentualen Änderungen abgrenzen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Ordne jedem Szenario die richtige Gleichung zu: \(Y=\beta_0+\beta_1X+u\), \(\log Y=\beta_0+\beta_1X+u\), \(Y=\beta_0+\beta_1\log X+u\) oder \(\log Y=\beta_0+\beta_1\log X+u\).
2. Prüfe vor jeder Transformation, dass die zu logarithmierende Variable strikt positiv ist.
3. Schätze alle vier Modelle und speichere Steigung, Standardfehler und R².
4. Interpretiere \(\hat\beta_1\) jeweils für eine Einheit mehr in \(X\) beziehungsweise für 1% mehr in \(X\).
5. Berechne im Log-Level-Modell zusätzlich die exakte prozentuale Änderung \(100(\exp(\hat\beta_1)-1)\).
6. Vergleiche diese mit der Approximation \(100\hat\beta_1\).
7. Interpretiere den Log-Log-Koeffizienten als Elastizität und prüfe sein Vorzeichen im Nachfrageszenario.
8. Zeichne Daten und Fits in Originaleinheiten. Erkläre, warum mehrere Modelle dort gekrümmt erscheinen, obwohl sie in den transformierten Variablen linear sind.

## Ausführen

```bash
python3 exercises/T13-logarithmic-models/starter.py
```

Die Musterlösung erzeugt `data/log_model_scenarios.csv`, `results/log_model_summary.csv` und `results/log_model_fits.png`:

```bash
python3 exercises/T13-logarithmic-models/solution.py
```

## Interpretationshilfe

| Modell | Interpretation von \(\beta_1\) |
|---|---|
| Level-Level | +1 Einheit \(X\) → \(\beta_1\) Einheiten \(Y\) |
| Log-Level | +1 Einheit \(X\) → ungefähr \(100\beta_1\)% in \(Y\) |
| Level-Log | +1% in \(X\) → ungefähr \(\beta_1/100\) Einheiten \(Y\) |
| Log-Log | +1% in \(X\) → ungefähr \(\beta_1\)% in \(Y\) |
