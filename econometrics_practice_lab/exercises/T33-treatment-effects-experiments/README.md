# T33 – Treatment-Effekte eines Mentoring-RCTs

## Ausgangslage

Ein Mentoringprogramm wird unter 700 Beschäftigten vollständig randomisiert. Das Outcome ist ein späterer Leistungsscore; ein Baseline-Score und ein vorab definierter High-Need-Indikator sind verfügbar. Das Treatment wirkt bei High-Need-Beschäftigten stärker.

## Lernziele

- den Average Treatment Effect per Mittelwertsdifferenz schätzen,
- den Standardfehler und ein Welch-Konfidenzintervall berechnen,
- die Äquivalenz von Mittelwertsdifferenz und Treatment-Koeffizient zeigen,
- Präzision durch vorab gemessene Kovariaten erhöhen,
- gruppenspezifische Effekte mit einer Interaktion schätzen,
- analytische und Randomisierungsinferenz vergleichen.

## Aufgaben

1. Schätze Treatment minus Kontrolle und berechne den Welch-Standardfehler samt 95%-KI.
2. Schätze eine Treatment-only-Regression und verifiziere die algebraische Gleichheit des Koeffizienten.
3. Verwende HC1-Standardfehler für die Regression.
4. Ergänze Baseline-Score und High-Need-Indikator. Vergleiche Punktschätzung und Präzision mit der unadjustierten Analyse.
5. Schätze ein Interaktionsmodell und leite die Effekte für Low Need und High Need samt Konfidenzintervallen ab.
6. Führe 2.000 Permutationen der Treatmentlabels unter der scharfen Nullhypothese durch.
7. Vergleiche Randomisierungs-p-Wert und analytische Inferenz.
8. Erkläre, weshalb Kovariatenadjustierung in einem korrekt randomisierten Experiment nicht zur Identifikation nötig ist.

## Ausführen

```bash
python3 exercises/T33-treatment-effects-experiments/starter.py
python3 exercises/T33-treatment-effects-experiments/solution.py
```

Die Lösung erzeugt `data/mentoring_rct.csv`, Ergebnistabellen und `results/treatment_effects.png`.

## Denkfragen

- Warum ändert Baseline-Adjustierung vor allem die Präzision?
- Welche Annahme testet die Permutation aller Treatmentlabels?
- Weshalb sollte eine Subgruppenanalyse vorab spezifiziert werden?
