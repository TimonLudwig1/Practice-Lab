# T31 – Balance Tests in einem Onlineexperiment

## Ausgangslage

In einem randomisierten Onlineexperiment werden Treatment und Kontrolle vor Beginn anhand mehrerer Pre-Treatment-Merkmale verglichen. Balance-Tabellen können Kodierungsfehler oder eine fehlerhafte Zuweisung sichtbar machen. Einzelne kleine p-Werte sind bei vielen Tests aber auch unter korrekter Randomisierung zu erwarten.

## Lernziele

- Pre-Treatment-Mittelwerte und standardisierte Mittelwertsdifferenzen berechnen,
- einzelne Balance-Tests und einen gemeinsamen F-Test durchführen,
- statistische Signifikanz von substanzieller Größe unterscheiden,
- das Problem multipler Balance-Tests durch wiederholte Randomisierung erkennen,
- Balance-Diagnostik korrekt interpretieren, ohne „Nichtsignifikanz“ als Beweis zu verwenden.

## Aufgaben

1. Erstelle für sechs Baseline-Variablen eine Tabelle mit Gruppenmitteln, Differenzen, SMD, t-Statistik und p-Wert.
2. Markiere SMDs außerhalb von ±0,10 als deskriptive Orientierung, nicht als formalen Beweis.
3. Regressiere den Treatmentindikator gemeinsam auf alle standardisierten Baseline-Merkmale und berichte den F-Test der gemeinsamen Nullhypothese.
4. Wiederhole die korrekte Zufallszuweisung 800-mal und führe jedes Mal alle sechs Einzeltests aus.
5. Bestimme pro Randomisierung den kleinsten p-Wert und die Zahl der p-Werte unter 0,05.
6. Schätze, wie oft mindestens ein scheinbar signifikanter Unterschied entsteht, obwohl Treatment zufällig zugewiesen wurde.
7. Erkläre, welche Befunde eher auf ein technisches Problem im Experiment hindeuten würden.

## Ausführen

```bash
python3 exercises/T31-balance-tests/starter.py
python3 exercises/T31-balance-tests/solution.py
```

Die Lösung erzeugt `data/online_experiment_baseline.csv`, drei Ergebnistabellen und `results/balance_tests.png`.

## Denkfragen

- Warum „verursacht“ das Treatment keine vor der Zuweisung gemessenen Merkmale?
- Ist ein p-Wert von 0,04 bei einem von zwanzig Balance-Tests überraschend?
- Weshalb hängt ein p-Wert von Stichprobengröße und Streuung ab, eine SMD aber anders?
