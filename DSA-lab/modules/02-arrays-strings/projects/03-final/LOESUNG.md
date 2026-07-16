# Lösung und Auswertung

Die Referenzimplementierungen stehen in `sensor_toolkit.py`, der reproduzierbare
Vergleich in `benchmark.py`. Diese Erklärung konzentriert sich auf Invarianten,
Korrektheit und die Interpretation der Messung.

## 1. Datenmodell

Die Zeitreihe kombiniert vier Komponenten:

```text
Messwert = Baseline + Drift + periodischer Verlauf + Rauschen
```

Erst danach werden an bekannten Indizes große Offsets addiert. Das ist für einen
fairen Test wichtig: Mit demselben Seed können eine unveränderte und eine
manipulierte Reihe erzeugt werden. Ihre Differenz ist an den ausgewählten Stellen
exakt der injizierte Offset.

Eine lokale Instanz von `random.Random(seed)` kapselt den Zufallszustand. Andere
Teile eines größeren Programms können dadurch nicht versehentlich die Datenfolge
verändern.

## 2. Rolling Sum

Für das erste Fenster wird die Summe einmal vollständig aufgebaut. Danach hält
`rolling_sum` diese Invariante:

> Vor dem Anhängen eines Mittelwerts ist `rolling_sum` die Summe genau des
> aktuellen Fensters.

Beim Schritt nach rechts verlässt `values[right - window]` das Fenster und
`values[right]` tritt ein. Damit sind pro neuem Fenster nur zwei Zugriffe und zwei
arithmetische Änderungen nötig.

```text
Werte:             2   4   8   6   5
Fensterbreite:     3

Summe [2,4,8]:    14  -> 4,6667
-2 +6:            18  -> 6,0000
-4 +5:            19  -> 6,3333
```

Komplexität:

- Zeit: O(n), weil jeder Wert konstant oft verwendet wird,
- Ergebnis: O(n - w + 1),
- zusätzlicher Arbeitszustand ohne Ergebnis: O(1).

Gleitkommaarithmetik ist nicht assoziativ. NumPy und Python können deshalb in den
letzten Bits abweichen, obwohl beide dasselbe mathematische Ergebnis berechnen.
Der Benchmark prüft eine enge Toleranz statt bitweiser Gleichheit.

## 3. Prefix-Sum-Index

Die führende Null macht halboffene Bereiche natürlich:

```text
Werte:       [4, -1, 7, 3, 2]
Prefix:    [0,  4, 3, 10, 13, 15]
```

Für `[1, 4)` gilt:

```text
prefix[4] - prefix[1] = 13 - 4 = 9
```

Beide Prefix-Werte enthalten den Bereich vor `start`; die Subtraktion entfernt
ihn. Die Invariante lautet:

> `prefix[i]` ist die Summe von `values[0:i]`.

Der Aufbau kostet O(n) Zeit und O(n) Speicher. Eine Query führt zwei Indexzugriffe
und eine Subtraktion aus, also O(1). Für `q` Abfragen kostet der Batch O(q). Gegen
das erneute Summieren jedes Bereichs lohnt sich der Index besonders bei vielen
oder überlappenden Abfragen.

## 4. Z-Score-Erkennung

Die Implementierung verwendet drei lineare Phasen:

1. Mittelwert bestimmen,
2. quadrierte Abweichungen für die Populationsvarianz summieren,
3. Z-Score jedes Werts berechnen und Grenzverletzungen sammeln.

Jede Phase ist O(n); ihre Summe bleibt O(n). Die Ergebnisliste braucht O(a) für
`a` erkannte Ausreißer, der übrige Arbeitszustand O(1).

Bei einer konstanten Reihe ist die Standardabweichung null. Ein Z-Score wäre dann
nicht definiert. Die fachlich sinnvolle Entscheidung für dieses Toolkit lautet:
Keine Abweichung bedeutet keine Ausreißer, also eine leere Ergebnisliste.

Der Schwellwertvergleich ist inklusiv. Ein Z-Score mit Betrag exakt gleich dem
Schwellwert wird aufgenommen. Das Verhalten ist durch Tests festgelegt und im
NumPy-Ausdruck identisch.

## 5. Warum die Pipeline Snapshots zurückgibt

`AnalysisResult` speichert Tuples. Dadurch beschreibt das Ergebnis den Zustand
zum Analysezeitpunkt und kann nicht versehentlich durch Anhängen oder Überschreiben
verändert werden. Die Originalmesswerte müssen dafür nicht dauerhaft kopiert
werden; gespeichert werden nur die tatsächlich angeforderten Resultate.

Der Prefix-Index wird einmal aufgebaut und für alle Bereichsabfragen wiederverwendet.
Ein beliebiges Iterable von Bereichen wird genau im Batch konsumiert, weshalb
auch ein Generator korrekt funktioniert.

## 6. Validierungsstrategie des Benchmarks

Eine schnellere Funktion ist nutzlos, wenn sie etwas anderes berechnet. Daher
erfolgt die Validierung vor der Zeitmessung:

```text
Python-Listen ─┐
               ├─ Form prüfen ─ Werte prüfen ─ erst dann messen
NumPy ─────────┘
```

Für numerische Arrays wird der größte absolute Fehler protokolliert und zusätzlich
`numpy.allclose` mit enger Toleranz geprüft. Für Ausreißer werden die Indexarrays
exakt verglichen. Ein Unterschied beendet den Lauf mit einem Fehler statt eine
irreführende Performancezahl zu erzeugen.

## 7. Was der NumPy-Vorteil bedeutet

Die Listenalgorithmen sind bereits asymptotisch optimal: Rolling Mean,
Prefix-Aufbau und Ausreißererkennung sind linear; Batch-Abfragen sind linear in
der Anzahl der Queries. NumPy ändert diese Big-O-Klassen nicht.

Der praktische Unterschied liegt in den konstanten Faktoren:

- Ein Python-Listenplatz verweist auf ein separates Python-Objekt.
- Eine Python-Schleife führt pro Element Interpreterarbeit und dynamische
  Typprüfungen aus.
- Ein NumPy-Array speichert gleichartige Werte zusammenhängend.
- NumPy-Schleifen laufen in kompiliertem Code und nutzen den CPU-Cache besser.
- Vektorisierte Indexierung verarbeitet viele Queries in einem Bibliotheksaufruf.

Das erklärt, warum zwei O(n)-Algorithmen bei großem `n` deutlich verschiedene
Laufzeiten haben können. Big-O beschreibt das Wachstum, nicht die absolute Zeit.

## 8. Fairness und Grenzen der Messung

Der Benchmark misst den Median mehrerer Wiederholungen, um einzelne Störungen zu
dämpfen. Die gleichen Daten, Fenster, Queries und Schwellwerte werden beiden
Varianten gegeben. Ein Warm-up oder sehr viele Wiederholungen könnte die
Stabilität weiter erhöhen, verlängert aber den Lernlauf.

Die Operationen werden bewusst separat gemessen:

- Prefix-Aufbau enthält die einmaligen Vorbereitungskosten.
- Range Queries verwenden bereits aufgebaute Indizes auf beiden Seiten.
- Ergebnisvalidierung liegt außerhalb der gemessenen Callables.
- Datengenerierung und Plotten zählen nicht zur Algorithmuslaufzeit.

Trotzdem ist das Ergebnis keine allgemeine Hardwarestudie. CPU-Last,
Bibliotheksversionen, Speicherhierarchie und Eingabegröße beeinflussen die Zahlen.
Der CSV-Bericht hält deshalb Rohzeiten und nicht nur Speedup-Faktoren fest.

## 9. Fachliche Grenzen der Ausreißerregel

Der globale Z-Score setzt voraus, dass ein gemeinsamer Mittelwert und eine
gemeinsame Streuung die gesamte Reihe sinnvoll beschreiben. Das ist bei starkem
Trend, wechselnder Sensorvarianz oder multimodalen Zuständen oft falsch.

Mögliche Erweiterungen wären:

- lokale Z-Scores in einem Rolling Window,
- Median Absolute Deviation als robuste Streuung,
- saisonale Zerlegung vor der Erkennung,
- getrennte Schwellen für positive und negative Abweichungen.

Diese Methoden sind nicht Teil des Projekts, weil hier Array-Zugriffe, Rolling
Windows, Prefix Sums und Vektorisierung isoliert nachvollziehbar bleiben sollen.

## Komplexitätsübersicht

| Operation | Aufbau/Zeit | Query-Zeit | Extra-Speicher ohne Ergebnis |
|---|---:|---:|---:|
| Sensordaten erzeugen | O(n) | — | O(1) |
| Moving Average | O(n) | — | O(1) |
| Prefix Sum | O(n) | O(1) | O(n) |
| Batch mit q Ranges | — | O(q) | O(1) |
| Z-Score-Ausreißer | O(n) | — | O(1) |

Die erzeugte Messreihe und geforderte Ergebnislisten sind in der letzten Spalte
nicht als zusätzlicher Arbeitszustand gezählt.
