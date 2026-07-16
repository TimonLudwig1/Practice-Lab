# 02-medium — Divide-and-Conquer-Werkzeugkasten

## Ziel

Dieses Projekt wendet Divide and Conquer auf drei unterschiedlich strukturierte
Probleme an. Du sollst nicht nur funktionierenden Code schreiben, sondern für
jeden Algorithmus die Teilprobleme, die Combine-Phase und die Arbeit pro
Rekursionsebene erklären können.

Alle drei Funktionen können optional einen eingerückten Trace füllen. So lässt
sich der tatsächlich ausgeführte Rekursionsbaum mit der theoretischen Rekurrenz
vergleichen. Python-Skripte mit Tests und einer gemeinsamen Demo sind dafür
übersichtlicher als ein Notebook.

## Aufgabe 1 — Binäre Potenzierung

```python
binary_power(base, exponent, trace=None) -> Number
```

Nutze:

```text
x^(2k)   = (x^k)²
x^(2k+1) = (x^k)² · x
```

Berechne `x^k` nur einmal und quadriere das Ergebnis. Zwei identische rekursive
Aufrufe würden die Laufzeit unnötig wieder verzweigen. Unterstütze positive,
negative und den Exponenten null sowie reelle und komplexe Basen.

Rekursionsbaum für `3^13`:

```text
e=13
└─ e=6
   └─ e=3
      └─ e=1
         └─ e=0
```

Rekurrenz: `T(e) = T(floor(e/2)) + Θ(1)`, also Θ(log |e|) Zeit und Stack.

## Aufgabe 2 — Maximum Subarray mit Divide and Conquer

```python
maximum_subarray(values, trace=None) -> SubarrayResult
```

Das beste nichtleere Teilarray liegt entweder:

1. vollständig in der linken Hälfte,
2. vollständig in der rechten Hälfte,
3. über der Teilungsgrenze.

Die Combine-Phase bestimmt den besten linken Suffix und rechten Prefix. Definiere
Gleichstände deterministisch: größere Summe, dann früherer Start, dann kürzere
Länge.

Segmentbaum für acht Werte:

```text
[0:8)
├─ [0:4)
│  ├─ [0:2)
│  └─ [2:4)
└─ [4:8)
   ├─ [4:6)
   └─ [6:8)
```

Auf jeder Ebene untersuchen alle Crossing-Berechnungen zusammen Θ(n) Werte. Bei
Θ(log n) Ebenen ergibt sich `T(n) = 2T(n/2) + Θ(n) = Θ(n log n)`.

## Aufgabe 3 — Inversionen beim Merge zählen

Eine Inversion ist ein Indexpaar `(i, j)` mit `i < j`, aber
`values[i] > values[j]`.

```python
count_inversions(values, trace=None) -> tuple[sorted_copy, count]
```

Sortiere beide Hälften rekursiv. Wird beim Merge ein Wert aus der rechten Hälfte
vor `left[left_index]` gewählt, ist er kleiner als alle noch nicht verbrauchten
linken Werte. Dadurch entstehen auf einmal
`len(left) - left_index` Split-Inversionen.

Beispielbaum:

```text
[2, 4, 1, 3]
├─ [2, 4] → [2, 4], 0
└─ [1, 3] → [1, 3], 0
Merge: 1 steht vor 2 und 4 → +2
Ergebnis: [1, 2, 3, 4], 2
```

Rekurrenz und Laufzeit entsprechen Merge Sort: Θ(n log n). Eine direkte Prüfung
aller Paare wäre Θ(n²).

## Ausführen

```bash
python3 -m unittest -v test_divide_conquer.py
python3 tree_demo.py
```

Es werden keine externen Bibliotheken benötigt.

## Analysefragen

1. Warum darf `binary_power` das Halbresultat nicht zweimal rekursiv berechnen?
2. Weshalb muss ein Maximum-Subarray eine der drei Kategorien erfüllen?
3. Warum zählt die Merge-Regel mehrere Inversionen in O(1) auf einmal?
4. Welche Algorithmen haben einen verzweigten Baum, und welcher nur eine Kette?
5. Wie unterscheiden sich Baumgröße und Stack-Tiefe bei den drei Verfahren?

## Fertig, wenn …

- binäre Potenzierung mit logarithmischer Rekursionstiefe arbeitet,
- Maximum Subarray korrekte Grenzen, Summe und Tie-Breaking liefert,
- die Inversionszählung mit Duplikaten korrekt umgeht und die Eingabe nicht
  verändert,
- alle drei optionalen Traces vollständige Wurzelentscheidungen zeigen,
- normale Fälle, Randfälle und ungültige Eingaben getestet sind,
- Tests und vollständige Tree-Demo fehlerfrei laufen,
- du die drei Rekurrenzen aus ihren tatsächlichen Bäumen begründen kannst.
