# T36 – Individuelle Fixed Effects bei Beschäftigten

## Ausgangslage

120 Beschäftigte werden über sechs Halbjahre beobachtet. Dauerhafte Fähigkeit beeinflusst sowohl die typische Trainingsintensität als auch die Produktivität. Personenspezifische Fixed Effects erlauben jeder Person einen eigenen Intercept und identifizieren den Trainingskoeffizienten aus Veränderungen derselben Person.

## Lernziele

- individuelle Fixed Effects als personenspezifische Intercepts formulieren,
- eine Least-Squares-Dummy-Variable-Schätzung aufbauen,
- den gemeinsamen Steigungskoeffizienten von individuellen Niveaus trennen,
- die notwendige Normalisierung der Fixed Effects verstehen,
- erkennen, dass zeitinvariante Personenmerkmale absorbiert werden,
- prüfen, welche Personen tatsächlich Within-Variation zur Identifikation beitragen.

## Aufgaben

1. Schätze zunächst Pooled OLS für Produktivität auf Trainingsstunden.
2. Ergänze für alle bis auf eine Person Dummies und verwende nach Person geclusterte Standardfehler.
3. Rekonstruiere für jede Person den geschätzten Intercept relativ zur Referenzperson.
4. Zentriere geschätzte und wahre personenspezifische Intercepts und vergleiche sie.
5. Berechne die Within-Standardabweichung der Trainingsstunden je Person.
6. Zeige über den Matrixrang, warum ein zeitinvariantes Bildungsmerkmal neben vollständigen Personen-FE nicht separat identifiziert ist.
7. Interpretiere den Trainingskoeffizienten ausdrücklich als Within-Person-Zusammenhang.

## Ausführen

```bash
python3 exercises/T36-individual-fixed-effects/starter.py
python3 exercises/T36-individual-fixed-effects/solution.py
```

Die Lösung erzeugt `data/employee_panel.csv`, Ergebnistabellen und `results/individual_fixed_effects.png`.

## Denkfragen

- Warum sind absolute Fixed-Effect-Niveaus von der Normalisierung abhängig?
- Welche Person liefert keine Information über den Trainingseffekt?
- Kann man den Effekt von Bildungsabschluss schätzen, wenn sich dieser im Panel nie ändert?
