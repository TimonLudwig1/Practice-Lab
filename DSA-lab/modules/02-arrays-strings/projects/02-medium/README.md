# 02-medium — Array-Pattern-Katalog

## Ziel

Dieses Projekt übersetzt wiederkehrende Array- und String-Probleme in zehn
konkrete Lösungsmuster. Es geht nicht darum, einzelne Lösungen auswendig zu
lernen. Du sollst anhand der Eingabeform und der geforderten Operation erkennen,
welches Pattern passt, welche Invariante es trägt und welche Zeit-/Speicherkosten
daraus entstehen.

Das Projekt verwendet bewusst normale Python-Skripte mit `unittest`. So bleiben
Mutation, Rückgabewerte und Fehlerfälle präzise prüfbar; für einen Katalog aus
vielen kleinen Algorithmen ist dieses Format übersichtlicher als ein Notebook.

## Die zehn Aufgaben

| Nr. | Aufgabe | Kern-Pattern | Zielkomplexität |
|---:|---|---|---|
| 1 | Array nach rechts rotieren | Drei Umkehrungen, in-place | O(n) Zeit, O(1) Extra-Speicher |
| 2 | Sortierte Arrays zusammenführen | Zwei Zeiger von rechts | O(n + m), O(1) Extra-Speicher |
| 3 | Wiederholte Bereichssummen | Prefix Sum | O(n) Aufbau, O(1) je Query |
| 4 | Anagramm prüfen | Frequency Map | O(n) Zeit, O(k) Speicher |
| 5 | Sortiertes Array deduplizieren | Read-/Write-Pointer | O(n), O(1) Extra-Speicher |
| 6 | Nullen stabil nach hinten schieben | Read-/Write-Pointer | O(n), O(1) Extra-Speicher |
| 7 | Product Except Self | Prefix-/Suffix-Produkte | O(n), ohne Division |
| 8 | Längstes eindeutiges Teilstück | Sliding Window | O(n) Zeit, O(k) Speicher |
| 9 | Matrix spiralförmig lesen | Schrumpfende Grenzen | O(r · c) Zeit |
| 10 | Zeichenläufe komprimieren | Read-/Write-Pointer | O(n), in-place |

Dabei steht `k` für die Zahl verschiedener Zeichen, `r` für Zeilen und `c` für
Spalten.

## Aufgabenstellung

Arbeite die Aufgaben zunächst nur mit dieser README durch. Implementiere die
angegebenen Signaturen in `pattern_catalog.py` und nutze die Tests als
ausführbare Spezifikation.

### 1. Rotation in-place

```python
rotate_right_in_place(values, steps) -> None
```

- Verändere dasselbe Sequenzobjekt.
- Verwende keinen zweiten Puffer proportional zur Eingabelänge.
- Normalisiere Schritte größer als die Länge.
- Interpretiere negative Schritte als Rotation nach links.

Hinweis: Eine Rechtsrotation um `k` lässt sich durch drei Umkehrungen zerlegen.

### 2. Sortierte Arrays in-place zusammenführen

```python
merge_sorted_in_place(target, valid_count, other) -> None
```

`target[:valid_count]` und `other` sind sortiert. Hinter dem gültigen Bereich von
`target` liegen genau `len(other)` Pufferplätze. Führe beide Folgen in `target`
zusammen, ohne die noch nicht gelesenen Werte zu überschreiben.

Hinweis: Beginne am rechten Rand, wo der freie Platz liegt.

### 3. Prefix-Sum-Index

```python
prefix = PrefixSum.from_values(values)
prefix.range_sum(start, end)
```

- Der Abfragebereich ist halboffen: `[start, end)`.
- Aufbau soll O(n), jede Abfrage O(1) kosten.
- Leere Bereiche ergeben `0`.
- Ungültige Grenzen werden abgewiesen.
- Eine spätere Mutation der Eingabeliste darf den Index nicht ändern.

### 4. Anagramm-Check

```python
are_anagrams(left, right) -> bool
```

Vergleiche exakte Unicode-Codepoints einschließlich Groß-/Kleinschreibung und
Leerzeichen. Verwende eine selbst gepflegte Häufigkeitstabelle statt Sortieren.

### 5. Deduplizierung eines sortierten Arrays

```python
new_length = remove_duplicates_sorted(values)
```

Überschreibe das Array mit seinen eindeutigen Werten und entferne den Rest. Die
Reihenfolge bleibt erhalten. Weise unsortierte Eingaben ab.

### 6. Nullen stabil verschieben

```python
non_zero_count = move_zeros_to_end(values)
```

Verschiebe alle numerischen Nullen ans Ende. Die relative Reihenfolge aller
Nichtnullwerte muss erhalten bleiben. Der Rückgabewert ist ihre Anzahl.

### 7. Product Except Self

```python
product_except_self(values) -> list
```

An Position `i` steht das Produkt aller Eingabewerte außer `values[i]`.
Division ist verboten, und Nullen müssen korrekt behandelt werden. Nutze einen
Vorwärts- und einen Rückwärtslauf.

### 8. Längstes eindeutiges Teilstück

```python
longest_unique_substring(text) -> str
```

Finde mit einem Sliding Window den längsten zusammenhängenden Teilstring ohne
Zeichenwiederholung. Bei Gleichstand gewinnt das früheste Vorkommen.

### 9. Spiral-Traversierung

```python
spiral_order(matrix) -> list
```

Lies eine rechteckige Matrix im Uhrzeigersinn, ohne Elemente doppelt zu besuchen.
Leere Formen sind erlaubt, gezackte Matrizen nicht.

### 10. Run-Length-Kompression in-place

```python
new_length = compress_runs_in_place(characters)
```

Komprimiere aufeinanderfolgende gleiche Zeichen. Aus `aaabb` wird `a3b2`, ein
einzelnes Zeichen erhält keine `1`. Mehrstellige Lauflängen werden in einzelne
Ziffern geschrieben.

## Empfohlener Arbeitsablauf

1. Wähle eine Aufgabe und notiere vor dem Coden die Invariante.
2. Ergänze zuerst je einen normalen Fall, einen Randfall und einen Fehlerfall als
   Test.
3. Implementiere die lineare Lösung.
4. Begründe Zeit und Extra-Speicher anhand der tatsächlichen Schleifen.
5. Führe alle Tests aus und vergleiche erst danach mit `LOESUNG.md`.

## Ausführen

Alle Tests:

```bash
python3 -m unittest -v test_pattern_catalog.py
```

Je ein repräsentatives Beispiel pro Pattern:

```bash
python3 catalog_demo.py
```

Es werden ausschließlich Module aus der Python-Standardbibliothek verwendet.

## Projektstruktur

```text
02-medium/
├── README.md
├── LOESUNG.md
├── pattern_catalog.py
├── catalog_demo.py
└── test_pattern_catalog.py
```

## Fertig, wenn …

- alle zehn Funktionen beziehungsweise Klassen ohne Abkürzung implementiert sind,
- alle Mutationseffekte und Rückgabewerte der Signaturen entsprechen,
- normale Fälle, leere/minimale Eingaben, Duplikate und ungültige Eingaben
  getestet sind,
- alle Tests ohne Fehler laufen,
- du für jedes Pattern die Invariante sowie Zeit- und Speicherkomplexität
  erklären kannst,
- du bei einem neuen Problem begründen kannst, ob In-place, Prefix Sum, zwei
  Zeiger, Sliding Window oder eine Frequency Map passt.
