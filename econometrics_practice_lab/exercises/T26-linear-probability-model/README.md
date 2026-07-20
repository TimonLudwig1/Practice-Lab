# T26 – Linear Probability Model für Kursabschlüsse

## Ausgangslage

Eine Weiterbildungsplattform möchte erklären, wie wöchentliche Lernzeit mit dem erfolgreichen Kursabschluss zusammenhängt. Das Outcome `completed` ist binär. Ein OLS-Modell darauf ist ein Linear Probability Model (LPM): Seine Steigung misst direkt eine Änderung der vorhergesagten Wahrscheinlichkeit in Prozentpunkten.

## Lernziele

Nach dem Projekt kannst du:

- ein LPM mit OLS schätzen,
- Koeffizienten als Änderungen in Wahrscheinlichkeit beziehungsweise Prozentpunkten interpretieren,
- erklären, warum LPM-Residuen konstruktionsbedingt heteroskedastisch sind,
- klassische und heteroskedastizitätsrobuste Standardfehler vergleichen,
- Vorhersagen außerhalb des Intervalls von null bis eins diagnostizieren,
- zwischen einer einfachen lokalen Approximation und einem gültigen Wahrscheinlichkeitsmodell unterscheiden.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Schätze `completed` auf `study_hours_per_week` mit OLS.
2. Interpretiere den Intercept und die Steigung in Wahrscheinlichkeiten und Prozentpunkten. Sind beide Interpretationen für den gesamten Wertebereich plausibel?
3. Berechne klassische und HC1-robuste Standardfehler sowie 95%-Konfidenzintervalle.
4. Ermittle den kleinsten und größten Fit im Datensatz und den Anteil der Fits außerhalb von \([0,1]\).
5. Erweitere das Vorhersageraster leicht über den beobachteten Bereich. Zeige, wo die lineare Funktion negative Wahrscheinlichkeiten oder Werte über eins produziert.
6. Führe einen Breusch–Pagan-Test durch und stelle die quadrierten Residuen über gebinnte Fits dar.
7. Vergleiche den LPM-Fit mit beobachteten Abschlussquoten in Lernzeit-Bins.
8. Formuliere präzise, wann ein LPM trotz seiner Grenzen nützlich sein kann.

## Ausführen

```bash
python3 exercises/T26-linear-probability-model/starter.py
```

Die Musterlösung erzeugt `data/course_completion.csv` sowie `results/coefficients.csv`, `results/diagnostics.csv`, `results/binned_completion_rates.csv` und `results/linear_probability_model.png`:

```bash
python3 exercises/T26-linear-probability-model/solution.py
```

## Denkfragen

- Weshalb hängt die Residuenvarianz bei einem binären Outcome von der vorhergesagten Wahrscheinlichkeit ab?
- Ändern robuste Standardfehler die geschätzte Steigung?
- Warum ist eine Wahrscheinlichkeit von 1,08 keine sinnvolle Prognose, obwohl OLS sie rechnerisch zulässt?
- Was ist der Vorteil der direkten Prozentpunktinterpretation gegenüber einem Logit-Koeffizienten?
