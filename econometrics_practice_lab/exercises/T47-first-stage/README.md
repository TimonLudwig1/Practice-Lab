# T47 – Starke und schwache First Stages

## Ausgangslage

Zwei zufällige Einladungsvarianten sollen die Teilnahmeintensität an einem Coaching erhöhen. Die starke Variante verändert die Coachingstunden deutlich, die schwache nur minimal. Beide Instrumente sind exogen – aber ein schwaches Instrument liefert in endlichen Stichproben extrem instabile IV-Schätzer.

## Lernziele

- die First Stage als Regression des endogenen Treatments auf das Instrument schätzen,
- Instrumentenrelevanz mit Koeffizient, partiellem R² und F-Statistik beurteilen,
- statistische Signifikanz und praktische Stärke unterscheiden,
- die endliche-Stichproben-Instabilität schwacher Instrumente simulieren,
- verstehen, warum eine kleine First Stage den IV-Quotienten empfindlich macht.

## Aufgaben

1. Schätze für starke und schwache Einladung jeweils `coaching_hours ~ instrument`.
2. Vergleiche First-Stage-Koeffizient, Standardfehler, F-Statistik und R².
3. Erkläre die häufig verwendete Faustregel zur F-Statistik, ohne sie als universellen Beweis zu behandeln.
4. Wiederhole beide Datengenerierungsprozesse mindestens 300-mal.
5. Berechne in jeder Wiederholung den IV-Kovarianzquotienten.
6. Vergleiche Median, Streuung und Anteil extremer IV-Schätzungen.
7. Untersuche den Zusammenhang zwischen First-Stage-F und absolutem Schätzfehler.

## Ausführen

```bash
python3 exercises/T47-first-stage/starter.py
python3 exercises/T47-first-stage/solution.py
```

Die Lösung erzeugt Beispieldaten, Monte-Carlo-Tabellen und `results/first_stage_strength.png`.

## Denkfragen

- Warum ist ein nicht signifikanter First-Stage-Koeffizient problematisch, aber ein signifikanter noch nicht automatisch stark?
- Was passiert im IV-Quotienten, wenn der geschätzte Nenner nahe null liegt?
- Welche Verfahren wären bei wenigen oder schwachen Instrumenten robuster als gewöhnliche 2SLS-Inferenz?
