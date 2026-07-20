# T30 – Randomisiertes Lerncoaching

## Ausgangslage

Eine Hochschule vergibt 300 von 600 Coachingplätzen per Zufall. Vor der Zuweisung stehen Baseline-Leistung, Motivation und digitaler Zugang fest. Der Datensatz enthält in der Simulation beide potenziellen Outcomes; beobachtet wird abhängig von der Zuweisung nur eines.

## Lernziele

- den Aufbau eines Randomized Controlled Trial erklären,
- Randomisierung von zufälliger Stichprobenziehung unterscheiden,
- verstehen, warum Zuweisung im Erwartungswert unabhängig von potenziellen Outcomes ist,
- endliche zufällige Ungleichgewichte von systematischer Selektion unterscheiden,
- mit wiederholter Randomisierung die Unverzerrtheit der Mittelwertsdifferenz nachvollziehen.

## Aufgaben

1. Prüfe, dass genau 300 Personen Treatment und 300 Kontrolle erhalten.
2. Erzeuge das beobachtete Outcome aus Zuweisung und potenziellen Outcomes.
3. Berechne den Sample ATE und die beobachtete Mittelwertsdifferenz.
4. Vergleiche Baseline-Merkmale mit standardisierten Mittelwertsdifferenzen.
5. Wiederhole die vollständige Zufallszuweisung 1.500-mal bei festgehaltenen potenziellen Outcomes.
6. Untersuche Mittelwert, Standardabweichung und Quantile der resultierenden Schätzwerte.
7. Zeige, dass der Mittelwert über viele Randomisierungen nahe am Sample ATE liegt.
8. Erkläre, weshalb eine einzelne Randomisierung keine exakt identischen Gruppen garantiert.

## Ausführen

```bash
python3 exercises/T30-randomized-experiments/starter.py
python3 exercises/T30-randomized-experiments/solution.py
```

Die Lösung erzeugt `data/randomized_coaching.csv`, Ergebnistabellen und `results/randomized_experiment.png`.

## Denkfragen

- Welche Größe wird durch die Randomisierung zufällig, welche bleibt fest?
- Warum ist ein zufälliges Baseline-Ungleichgewicht kein Beweis für fehlerhafte Randomisierung?
- Würde Randomisierung automatisch die Übertragbarkeit auf andere Hochschulen sichern?
