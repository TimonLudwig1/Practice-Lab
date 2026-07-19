# Projekt 01-basic: Sortier-Werkstatt

## Ziel

In dieser Werkstatt werden Bubble, Selection, Insertion, Merge und Quicksort von
Grund auf implementiert. Alle Funktionen besitzen dieselbe API, verändern die
Eingabe nicht und unterstützen einen `key`-Parameter. Optionale Metriken und
Traces machen Vergleiche, Schreiboperationen, Tausche und Rekursion sichtbar.

Python-Skripte mit pytest sind für dieses Projekt geeigneter als ein Notebook:
Fünf Implementierungen können über dieselben Property-Tests geprüft werden, und
die Demo bleibt ein separater beobachtbarer Ablauf. Zufallsfälle werden mit
festem Seed erzeugt und sind dadurch reproduzierbare Regressionstests.

## Aufgabenstellung

1. Implementiere Bubble Sort mit Early Exit nach einem tauschfreien Pass.
2. Implementiere Selection Sort mit einem fertigen Präfix.
3. Implementiere stabilen Insertion Sort durch Verschieben statt wiederholtem
   Tauschen.
4. Implementiere stabilen Merge Sort mit rekursivem Teilen und linearem Merge.
5. Implementiere Quicksort mit Lomuto-Partitionierung und letztem Element als
   Pivot. Mache den Worst Case dieser Pivotwahl messbar.
6. Sammle pro Lauf Vergleiche, Tausche, Schreiboperationen und Rekursionstiefe.
7. Prüfe alle Verfahren gegen `sorted()` auf strukturierten und mindestens 200
   reproduzierbaren Zufallsarrays.

## Gemeinsame Schnittstelle

```python
result = algorithm(values, key=None, metrics=None, trace=None)
```

- `values` bleibt unverändert; zurückgegeben wird eine sortierte Kopie.
- `key` bildet Elemente auf vergleichbare Sortierschlüssel ab.
- `metrics` sammelt algorithmische Operationen, keine instabilen Laufzeiten.
- `trace` erhält nach relevanten Mutationen unveränderliche Zustandskopien.

## Invarianten

| Verfahren | Invariante |
|---|---|
| Bubble | Das rechte Suffix enthält die größten Elemente endgültig sortiert. |
| Selection | Das linke Präfix enthält die kleinsten Elemente endgültig sortiert. |
| Insertion | Das linke Präfix ist sortiert und enthält dieselben Präfixelemente. |
| Merge | Der Merge-Ausgang enthält stets die kleinsten betrachteten Elemente. |
| Quick | Links der Partitionierungsgrenze liegen nur Keys `<= pivot`. |

Jedes Ergebnis muss zusätzlich eine Permutation der Eingabe sein. Der Vergleich
mit `sorted()` prüft beide Eigenschaften gleichzeitig, solange die Eingabe nicht
mutiert wurde.

## Eigenschaftsprofil

| Verfahren | Best | Average/Worst | stabil | innerer Zusatzspeicher |
|---|---:|---:|---|---:|
| Bubble | O(n) | O(n²) | ja | O(1) |
| Selection | O(n²) | O(n²) | nein | O(1) |
| Insertion | O(n) | O(n²) | ja | O(1) |
| Merge | O(n log n) | O(n log n) | ja | O(n) |
| Quick | O(n log n) | Average O(n log n), Worst O(n²) | nein | Stack O(log n), Worst O(n) |

Die Lern-API kopiert die Eingabe bewusst. „Innerer Zusatzspeicher“ beschreibt
das eigentliche Verfahren nach dieser Kopie.

## Ausführen

Im Projektordner:

```bash
python demo.py
python -m pytest -q
```

Die Demo sortiert dasselbe Array mit allen Verfahren, vergleicht ihre Zähler und
zeigt anschließend einen vollständigen Insertion-Trace.

## Hinweise

- Nutze ausschließlich `<` für fachliche Key-Vergleiche. So funktionieren auch
  Objekte, die nur eine Less-than-Ordnung definieren.
- Merge bleibt stabil, wenn bei Gleichheit die linke Seite gewählt wird.
- Insertion bleibt stabil, wenn nur strikt größere Keys verschoben werden.
- Der Pivotindex darf in den rekursiven Quicksort-Bereichen nicht erneut
  enthalten sein.
- Ein Property-Test benötigt einen gespeicherten Seed; sonst ist ein Fehlerlauf
  nicht zuverlässig reproduzierbar.
- Ein schneller Lauf beweist keine Korrektheit. Prüfe das Ergebnis vor späteren
  Benchmarks immer gegen `sorted()`.

## Fertig, wenn …

- alle fünf Verfahren leere, sortierte, umgekehrte und duplikatreiche Arrays
  korrekt sortieren,
- keine Implementierung die Eingabe verändert,
- `key` für Strings und Datensätze funktioniert,
- Bubble, Insertion und Merge die Reihenfolge gleicher Keys erhalten,
- Metriken Best- beziehungsweise Worst-Case-Unterschiede sichtbar machen,
- Traces die entscheidenden Zustandsänderungen enthalten,
- alle Seed-basierten Zufallsfälle mit `sorted()` übereinstimmen und
- die vollständige pytest-Suite erfolgreich ist.
