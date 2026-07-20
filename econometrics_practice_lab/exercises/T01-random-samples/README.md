# T01 – Stadtmobilitäts-Stichproben

## Ausgangslage

Eine Stadtverwaltung möchte die durchschnittliche Pendelzeit ihrer erwachsenen Bevölkerung schätzen. Eine synthetische Vollpopulation steht nur in diesem Lab zur Verfügung; in einer echten Erhebung würdest du lediglich eine Stichprobe beobachten. Dadurch kannst du hier kontrolliert untersuchen, wie gut Zufallsstichproben die Wahrheit treffen.

## Lernziele

Nach dem Projekt kannst du:

- eine einfache Zufallsstichprobe ohne Zurücklegen ziehen,
- Population, einzelne Stichprobe und Stichprobenverteilung auseinanderhalten,
- erklären, warum zwei korrekte Zufallsstichproben unterschiedliche Ergebnisse liefern,
- den Effekt der Stichprobengröße auf Standardfehler und Präzision quantifizieren.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Erzeuge die Population und beschreibe Größe, Mittelwert, Standardabweichung und Verteilung der Pendelzeit.
2. Ziehe mit drei unterschiedlichen Seeds je eine Zufallsstichprobe von 100 Personen. Vergleiche die geschätzten Mittelwerte mit dem Populationsmittel.
3. Ziehe für \(n\in\{50,200,1000\}\) jeweils 2.000 wiederholte Stichproben und speichere jeden Stichprobenmittelwert.
4. Berechne pro Stichprobengröße den mittleren Schätzwert, Bias, empirische Standardabweichung und den theoretischen Standardfehler \(s/\sqrt{n}\).
5. Visualisiere die drei Stichprobenverteilungen mit derselben x-Achse und markiere den wahren Populationsmittelwert.
6. Formuliere in drei Sätzen: Was ändert ein größeres \(n\), was ändert es nicht, und warum ist ein einzelner Stichprobenmittelwert fast nie exakt gleich dem Populationsmittel?

## Ausführen

```bash
python3 exercises/T01-random-samples/starter.py
```

Die Musterlösung erzeugt `results/sampling_summary.csv` und `results/sample_means.png`:

```bash
python3 exercises/T01-random-samples/solution.py
```

## Denkfragen

- Ist der Stichprobenmittelwert in diesem Experiment erwartungstreu?
- Wie sollte sich der Standardfehler verändern, wenn \(n\) vervierfacht wird?
- Garantiert ein großes \(n\), dass die Datenerhebung frei von Bias ist?
