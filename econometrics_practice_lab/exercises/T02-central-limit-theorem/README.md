# T02 – Central Limit Theorem im Onlinehandel

## Ausgangslage

Ein Onlinehändler beobachtet stark rechtsschiefe Bestellwerte: Viele Bestellungen sind klein, einige wenige sehr groß. Du untersuchst, warum der Mittelwert einer ausreichend großen Stichprobe trotzdem näherungsweise normalverteilt sein kann.

## Lernziele

Nach dem Projekt kannst du:

- die Verteilung einzelner Beobachtungen von der Verteilung eines Stichprobenmittelwerts unterscheiden,
- das Central Limit Theorem durch Simulation sichtbar machen,
- den Standardfehler \(\sigma/\sqrt{n}\) empirisch überprüfen,
- erklären, weshalb „die Daten sind normalverteilt“ keine Voraussetzung für den CLT-Effekt ist.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Visualisiere die Population der Bestellwerte und beschreibe Mittelwert, Median und Schiefe.
2. Simuliere für \(n\in\{1,5,30,100\}\) jeweils 5.000 Stichprobenmittelwerte.
3. Berechne Mittelwert, Standardabweichung und Schiefe jeder Stichprobenverteilung.
4. Vergleiche die empirische Standardabweichung mit \(\sigma/\sqrt{n}\).
5. Zeichne Histogramme der vier Stichprobenverteilungen und lege jeweils die passende Normaldichte darüber.
6. Prüfe, welcher Anteil der simulierten Mittelwerte innerhalb von \(\mu\pm1{,}96\sigma/\sqrt{n}\) liegt.
7. Erkläre, was das CLT behauptet – und was es nicht behauptet.

## Ausführen

```bash
python3 exercises/T02-central-limit-theorem/starter.py
```

Die Musterlösung erzeugt `results/clt_summary.csv`, `results/population_distribution.png` und `results/clt_distributions.png`:

```bash
python3 exercises/T02-central-limit-theorem/solution.py
```

## Denkfragen

- Warum bleibt die Population schief, obwohl die Verteilung der Mittelwerte symmetrischer wird?
- Gilt die Normalapproximation bei jeder Ausgangsverteilung schon für \(n=30\)?
- Welche Rolle spielen Unabhängigkeit, identische Verteilung und eine endliche Varianz?
