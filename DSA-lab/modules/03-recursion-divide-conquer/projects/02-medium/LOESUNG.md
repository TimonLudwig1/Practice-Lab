# Lösung und Laufzeitanalyse

## Binäre Potenzierung

Der entscheidende Schritt ist die Wiederverwendung des Halbresultats:

```text
half = power(base, exponent // 2)
result = half * half
bei ungeradem Exponenten zusätzlich result *= base
```

Pro Frame entsteht genau ein Kind. Der Exponent halbiert sich, daher gibt es
`floor(log₂ |e|) + O(1)` Frames. Nichtrekursive Arbeit pro Frame ist konstant.

Würde `power(base, e // 2)` zweimal im Ausdruck stehen, entstünde
`T(e)=2T(e/2)+Θ(1)=Θ(e)`. Der mathematisch identische Ausdruck hätte damit eine
schlechtere algorithmische Struktur.

Negative Exponenten verändern den Rekursionsbaum nicht: Zuerst wird die positive
Potenz des Betrags berechnet, danach einmal der Kehrwert gebildet.

## Maximum Subarray

Jedes zusammenhängende Teilarray relativ zur Mitte liegt zwingend links, rechts
oder schneidet die Mitte. Die drei Kandidaten sind deshalb vollständig.

Für den Crossing-Kandidaten läuft ein Zeiger von der Mitte nach links und hält
den besten Suffix. Ein zweiter läuft nach rechts und hält den besten Prefix. Ihre
Verkettung ist das beste Teilarray, das die Grenze überquert.

Die rekursive Korrektheit folgt aus:

1. Die Kinder liefern per Induktionsannahme die besten Kandidaten ihrer Hälften.
2. Die linearen Scans liefern den besten Crossing-Kandidaten.
3. Das Maximum dieser erschöpfenden Kategorien ist global optimal.

Das Tie-Breaking ist Teil der Spezifikation, nicht Kosmetik. `_result_key` ordnet
nach Summe, negativem Start und negativer Länge. Ein normales `max` wählt damit
größere Summe, kleineren Start und anschließend kürzere Länge.

Die Combine-Arbeit ist Θ(n). Daraus folgt
`T(n)=2T(n/2)+Θ(n)=Θ(n log n)`. Der Call Stack ist Θ(log n); die Ergebnisobjekte
und temporären Aufrufe ändern diese Tiefe nicht.

Ein linearer Kadane-Algorithmus wäre für dieses Einzelproblem schneller. Die
D&C-Variante ist hier bewusst gewählt, um Kategorien, Crossing-Combine und
Rekursionsbäume zu üben.

## Inversionen zählen

Inversionen zerfallen ebenfalls in drei disjunkte Gruppen:

- vollständig links,
- vollständig rechts,
- ein Wert links und ein Wert rechts.

Die Kinder zählen die ersten beiden Gruppen. Beim sortierten Merge werden
Split-Inversionen sichtbar. Gilt `right[j] < left[i]`, dann ist `right[j]` auch
kleiner als `left[i+1:]`, weil links sortiert ist. Alle diese Paare können sofort
addiert werden.

Bei Gleichheit wird links gewählt. Gleiche Werte erfüllen nicht `>` und sind
daher keine Inversionen.

Jede Ebene merged insgesamt n Elemente, und es gibt Θ(log n) Ebenen. Zeit:
Θ(n log n). Die sortierten Teillisten benötigen Θ(n) zusätzlichen Speicher; der
Stack ist Θ(log n).

## Trace-Ausgaben lesen

Einrückung entspricht Rekursionstiefe. Eintrittszeilen dokumentieren Divide,
Rückgabe- oder Merge-Zeilen dokumentieren Combine. Die letzte Zeile jedes Traces
ist die Entscheidung des Wurzelproblems. Damit lässt sich prüfen, ob eine
theoretische Rekurrenz den tatsächlich ausgeführten Code beschreibt.

## Übersicht

| Algorithmus | Rekurrenz | Zeit | Stack | weiterer Speicher |
|---|---|---:|---:|---:|
| Binäre Potenz | `T(e)=T(e/2)+Θ(1)` | Θ(log |e|) | Θ(log |e|) | O(1) |
| Maximum Subarray | `T(n)=2T(n/2)+Θ(n)` | Θ(n log n) | Θ(log n) | Θ(log n)* |
| Inversionen | `T(n)=2T(n/2)+Θ(n)` | Θ(n log n) | Θ(log n) | Θ(n) |

`*` Ohne optionalen Trace und abgesehen von Resultatobjekten der aktiven Frames.
