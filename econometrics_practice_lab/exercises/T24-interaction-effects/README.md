# T24 – Interaktionseffekte in einem Trainingsprogramm

## Ausgangslage

Beschäftigte werden zufällig einem Trainingsprogramm zugeteilt. Das Programm verbessert einen Leistungsscore, wirkt bei größerer Berufserfahrung aber stärker. Ein Interaktionsterm zwischen Treatment und Erfahrung erlaubt unterschiedliche Steigungen für Treatment- und Kontrollgruppe.

## Lernziele

Nach dem Projekt kannst du:

- ein Modell mit zwei Haupteffekten und Interaktion formulieren,
- jeden Koeffizienten als bedingten Effekt interpretieren,
- marginale Treatment-Effekte für konkrete Erfahrungswerte berechnen,
- Unsicherheit einer Koeffizientensumme mit der Kovarianzmatrix bestimmen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Zeichne Leistung gegen Erfahrung getrennt nach Treatmentstatus.
2. Schätze zuerst ein Modell ohne Interaktion und erkläre die darin erzwungene Parallelität.
3. Ergänze `treated × experience_years` und schätze das Interaktionsmodell.
4. Interpretiere Intercept, Erfahrungskoeffizient, Treatment-Haupteffekt und Interaktionskoeffizient jeweils an der passenden Bedingung.
5. Berechne den Treatment-Effekt \(\hat\beta_T+x\hat\beta_{T\times X}\) für 0, 5, 10 und 15 Erfahrungsjahre.
6. Berechne den Standardfehler jeder Summe mit Varianz, Kovarianz und Delta-Methode.
7. Zeichne den marginalen Treatment-Effekt über den gesamten beobachteten Erfahrungsbereich samt 95%-Konfidenzband.
8. Erkläre, warum ein nicht signifikanter Treatment-Haupteffekt nicht bedeuten würde, dass das Treatment für alle Erfahrungswerte wirkungslos ist.

## Ausführen

```bash
python3 exercises/T24-interaction-effects/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/model_comparison.csv`, `results/coefficients.csv`, `results/marginal_treatment_effects.csv` und `results/interaction_effects.png`:

```bash
python3 exercises/T24-interaction-effects/solution.py
```

## Denkfragen

- Welche Steigung gilt in der Treatmentgruppe?
- Wie würde Zentrieren der Erfahrung bei zehn Jahren die Bedeutung des Treatment-Haupteffekts verändern?
- Warum müssen Haupteffekte in einem Interaktionsmodell auch dann enthalten bleiben, wenn nur die Interaktion interessiert?
