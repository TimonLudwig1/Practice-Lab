# T40 – DiD bei einer kommunalen Umweltzone

## Ausgangslage

Ab 2021 führen 30 von 60 Kommunen eine Umweltzone ein. Die behandelten Kommunen hatten bereits vorher höhere NO₂-Werte. Ein einfacher Niveauvergleich wäre deshalb irreführend. Differences-in-Differences nutzt stattdessen die **zusätzliche Veränderung** der Treatmentgruppe relativ zur Veränderung der Kontrollgruppe.

## Lernziele

- den DiD-Schätzer aus vier Gruppen-Zeit-Mittelwerten berechnen,
- dieselbe Größe als Koeffizienten einer Interaktion schätzen,
- Niveauunterschiede von unterschiedlichen Trends unterscheiden,
- die Parallel-Trends-Annahme mit Pre-Treatment-Daten untersuchen,
- geclusterte Standardfehler für wiederholte Kommunenbeobachtungen verwenden,
- ein DiD-Ergebnis in der Einheit des Outcomes interpretieren.

## Aufgaben

1. Prüfe Panelindex, Beobachtungsjahre und Treatmentbeginn.
2. Berechne die Mittelwerte von Treatment- und Kontrollgruppe vor und nach 2021.
3. Bilde zuerst die beiden Zeitdifferenzen und danach deren Differenz.
4. Schätze `no2 ~ treated + post + treated × post`. Welcher Koeffizient ist der DiD-Effekt?
5. Verwende nach Kommune geclusterte Standardfehler und ein 95%-Konfidenzintervall.
6. Schätze in den Jahren vor 2021 gruppenspezifische lineare Trends. Was würde ein deutlicher Interaktionseffekt bedeuten?
7. Zeichne die jährlichen Gruppenmittel und markiere den Beginn der Umweltzone.

## Ausführen

```bash
python3 exercises/T40-differences-in-differences/starter.py
python3 exercises/T40-differences-in-differences/solution.py
```

Die Lösung erzeugt `data/municipality_panel.csv`, mehrere Ergebnistabellen und `results/did_environment_zone.png`.

## Denkfragen

- Warum müssen die Gruppen vor der Behandlung nicht dasselbe Niveau besitzen?
- Weshalb kann ein optisch paralleler Pre-Trend die Annahme plausibel machen, aber nicht beweisen?
- Welche zeitgleich mit der Umweltzone auftretende gruppenspezifische Veränderung würde DiD verzerren?
