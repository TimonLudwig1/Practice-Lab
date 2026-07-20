# 02-medium — Graph-Aufgabenkatalog

Dieses Projekt bündelt vier wiederkehrende Traversierungsmuster. Alle Verfahren
arbeiten deterministisch nach der Einfügereihenfolge und können ihre Schritte in
eine Trace-Liste schreiben. Die Traces zeigen nicht nur *was* herauskommt,
sondern *warum*.

## 1. Zyklus in einem ungerichteten Graphen

DFS merkt sich den Parent. Eine Kante zu einem bereits besuchten Knoten ist nur
dann ein Zyklusbeweis, wenn dieser Knoten nicht der Parent ist.

```text
Graph: A--B, B--C, C--A

01 Neue Komponente ab 'A'
02 Besuche 'A'; Parent=None
03 Baumkante 'A'--'B'
04 Besuche 'B'; Parent='A'
05 Baumkante 'B'--'C'
06 Besuche 'C'; Parent='B'
07 Zyklus über 'C'--'A'
```

Ein ungerichteter Graph muss symmetrisch angegeben werden: Steht `A -> B` in
der Adjazenzliste, muss auch `B -> A` vorkommen. Ein asymmetrischer Eingang wird
mit `ValueError` abgelehnt.

## 2. Zyklus in einem gerichteten Graphen

Die DFS verwendet drei Farben:

- **weiß (0):** noch nicht besucht,
- **grau (1):** aktuell auf dem Rekursionspfad,
- **schwarz (2):** vollständig abgeschlossen.

Nur eine Kante zu einem grauen Knoten ist eine Rückkante und damit ein
gerichteter Zyklus.

```text
Graph: A -> B -> C -> A

01 'A': weiß -> grau
02 Baumkante 'A'->'B'
03 'B': weiß -> grau
04 Baumkante 'B'->'C'
05 'C': weiß -> grau
06 Rückkante 'C'->'A': Zyklus
```

## 3. Topologische Sortierung

Die Kahn-Variante berechnet alle In-Degrees und startet mit den Knoten vom Grad
null. Jede entnommene Aufgabe entfernt gedanklich ihre ausgehenden Kanten. Bleiben
am Ende Knoten übrig, enthält der Graph einen Zyklus und ist kein DAG.

```text
extract -> clean -> train -> evaluate

01 Startqueue: ['extract']
02 Entnimm 'extract'; Ordnung=['extract']
03 In-Degree 'clean' -> 0
04 Füge 'clean' zur Queue hinzu
05 Entnimm 'clean'; Ordnung=['extract', 'clean']
... bis alle Knoten verarbeitet sind
```

Zusätzlich enthält das Projekt die DFS-Variante. Sie hängt einen Knoten erst
beim Abschluss an und dreht diese Abschlussliste am Ende um. Beide Varianten
prüfen die DAG-Voraussetzung und lösen bei einem Zyklus `CycleError` aus. Dessen
Attribut `trace` bewahrt auch den Fehlerpfad.

## 4. Bipartitheit per BFS-Färbung

Jede neue Komponente startet mit Farbe 0. Jeder unbemalte Nachbar erhält die
Gegenfarbe. Treffen zwei gleich gefärbte Endpunkte aufeinander, ist keine
Zweifärbung möglich.

```text
Quadrat A--B--C--D--A

01 Neue Komponente: färbe 'A' mit 0
02 Entnimm 'A' (Farbe 0)
03 Färbe 'B' mit 1
04 Färbe 'D' mit 1
05 Entnimm 'B' (Farbe 1)
06 Färbe 'C' mit 0
```

Ein ungerichteter Graph ist genau dann bipartit, wenn er keinen ungeraden Zyklus
enthält. Das Verfahren verarbeitet auch isolierte und getrennte Komponenten.

## 5. Inselzählung als impliziter Grid-Graph

Jede Landzelle (`1`) ist ein Knoten. Kanten zu oben, rechts, unten und links
werden nicht gespeichert, sondern bei Bedarf aus den Koordinaten erzeugt.
Diagonalen zählen nicht als Verbindung.

```text
Grid:             Trace der ersten Insel:
1 1 0             01 Insel 1 startet bei (0, 0)
1 0 0             02 Besuche (0, 0); Größe=1
0 0 1             03 Entdecke (0, 1) und füge es ein
                  04 Entdecke (1, 0) und füge es ein
                  05 Besuche (0, 1); Größe=2
                  06 Besuche (1, 0); Größe=3
                  07 Insel 1 abgeschlossen; Größe=3
```

`analyze_islands` liefert Anzahl und Größen in zeilenweiser
Entdeckungsreihenfolge. `count_islands` ist der kompakte Wrapper nur für die
Anzahl. Das Eingabegrid bleibt unverändert.

## Auswahl des Musters

| Problem | Zustand | Zentrale Prüfung |
|---|---|---|
| ungerichteter Zyklus | `visited`, Parent | besuchter Nachbar ≠ Parent |
| gerichteter Zyklus | drei Farben | Kante zu grauem Knoten |
| Toposort nach Kahn | In-Degree, Queue | alle Knoten entnommen? |
| Toposort per DFS | drei Farben, Abschlussliste | keine Rückkante? |
| Bipartitheit | Farbe 0/1, Queue | Nachbarn verschieden? |
| Inseln | Koordinate, `visited`, Queue | gültige 4er-Nachbarzelle? |

Alle Verfahren laufen in `O(V + E)`. Für ein Grid mit `R` Zeilen und `C`
Spalten entspricht das `O(R * C)`, weil jede Zelle höchstens einmal besucht und
jede ihrer vier Richtungen höchstens einmal geprüft wird.

## Ausführen

Im Projektordner:

```bash
python3 demo.py
python3 -m pytest -q
```
