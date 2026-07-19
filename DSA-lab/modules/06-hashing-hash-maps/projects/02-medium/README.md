# Projekt 02-medium: Hash-Pattern-Katalog

## Ziel

Dieses Projekt macht vier wiederkehrende Hash-Patterns an vollständigen
Problemlösungen sichtbar: Komplement-Lookup, Gruppierung über kanonische Keys,
Frequency Counting und zustandsbehaftete Duplikaterkennung. Jede optimierte
Lösung steht neben einer absichtlich naiven Variante mit linearer Suche.

Python-Skripte und pytest eignen sich hier besser als ein Notebook, weil alle
Implementierungen dieselbe testbare API besitzen. `benchmark.py` führt beide
Strategien auf identischen, per Seed reproduzierbaren Eingaben aus, prüft zuerst
Ergebnisgleichheit und schreibt anschließend die Laufzeiten als CSV.

## Aufgabenstellung

1. Löse Two Sum in einem Durchlauf mit einer Map `value -> erster Index`.
2. Gruppiere Anagramme über eine kanonische, hashable Signatur.
3. Finde das erste eindeutige Zeichen über Frequency Counting und einen zweiten
   Durchlauf in Originalreihenfolge.
4. Implementiere einen zustandsbehafteten Stream-Detektor, der jedes Element als
   neu oder bereits gesehen markiert und Zähler führt.
5. Implementiere für jedes Problem eine naive Referenz ohne Hash-Lookup.
6. Vergleiche beide Varianten bei wachsenden Eingabegrößen und bestätige vor der
   Messung, dass ihre Ergebnisse identisch sind.

## Die vier Muster

| Problem | Hash-Key | gespeicherter Wert/Zustand | Zeit erwartet |
|---|---|---|---:|
| Two Sum | bereits gesehener Zahlenwert | erster Index | O(n) |
| Anagramme | sortierte Zeichen | Gruppe von Wörtern | O(n · k log k) |
| erstes eindeutiges Zeichen | Zeichen | Häufigkeit | O(n) |
| Stream-Duplikate | gesamtes Element | bereits gesehen | O(n) gesamt |

Bei Anagrammen bezeichnet `k` die Wortlänge. Das Sortieren der Zeichen bleibt
auch mit Hash Map bestehen; vermieden wird die lineare Suche nach einer bereits
existierenden Gruppe.

## Simulation: Two Sum

Für `[3, 2, 4]` und Ziel 6:

| Index | Wert | Komplement | bisherige Map | Aktion |
|---:|---:|---:|---|---|
| 0 | 3 | 3 | `{}` | `3 -> 0` speichern |
| 1 | 2 | 4 | `{3: 0}` | `2 -> 1` speichern |
| 2 | 4 | 2 | `{3: 0, 2: 1}` | Indexpaar `(1, 2)` liefern |

Die Komplementprüfung geschieht vor dem Speichern des aktuellen Werts. So wird
ein einzelnes Element nicht zweimal verwendet. Beim Wert 3 wären zwei getrennte
Vorkommen nötig, um Ziel 6 zu bilden.

## Stream-Invariante

Vor dem Verarbeiten eines Elements enthält `seen` exakt die unterschiedlichen
Elemente des bereits konsumierten Präfixes. `process(item)` liefert genau dann
`True`, wenn `item` in diesem Set liegt. Für exakte Erkennung wächst der Speicher
mit der Zahl verschiedener Elemente; ein unbegrenzter Stream benötigt daher eine
fachliche Strategie für Fenster, Ablaufzeiten oder approximative Filter.

## Komplexitätsvergleich

| Pattern | Hash-basiert | Naiv | Zusatzspeicher |
|---|---:|---:|---:|
| Two Sum | erwartet O(n) | O(n²) | O(n) |
| Anagrammgruppen | O(n · k log k) erwartet | bis O(n² + n · k log k) | O(nk) |
| erstes eindeutiges Zeichen | erwartet O(n) | O(n²) | O(k) |
| Stream-Duplikate | erwartet O(n) | O(n²) | O(k) |

`k` steht in der letzten Spalte für die Zahl verschiedener Werte. Die
Optimierung tauscht zusätzlichen Speicher gegen vermiedene wiederholte Suche.

## Ausführen

Im Projektordner:

```bash
python demo.py
python benchmark.py
python -m pytest -q
```

Die Messreihe landet in `output/benchmark.csv`. Kleine Laufzeiten schwanken;
entscheidend ist die Wachstumstendenz über größere `item_count`-Werte.

## Hinweise

- Der Hash-Key muss die fachliche Gleichheit ausdrücken. Für Anagramme ist das
  die sortierte Zeichenfolge, nicht das Originalwort.
- Ein Dictionary bewahrt zwar Einfügereihenfolge, ist aber keine sortierte Map.
- Frequency Counting braucht zwei Phasen, wenn das erste Element in
  Originalreihenfolge gesucht wird.
- Ein exakter Stream-Detektor kann alte Werte nicht vergessen, ohne seine
  Aussage zu verändern.
- Prüfe Optimierungen immer gegen eine einfache Referenz, bevor du Laufzeiten
  interpretierst.

## Fertig, wenn …

- alle vier hash-basierten Lösungen ihre Randfälle korrekt behandeln,
- die naiven Referenzen dieselben fachlichen Ergebnisse liefern,
- Two Sum kein einzelnes Element doppelt verwendet,
- Anagramme über einen reproduzierbaren kanonischen Key gruppiert werden,
- der Stream-Detektor Zustand, Reset und Zähler korrekt verwaltet,
- der Benchmark beide Strategien über wachsende Eingaben vergleicht,
- Demo und vollständige Messreihe laufen und
- alle Tests mit `python -m pytest -q` erfolgreich sind.
