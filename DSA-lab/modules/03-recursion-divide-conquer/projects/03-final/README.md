# 03-final — Rekursive Dateisystem-Analyse

## Ziel

Dieses Abschlussprojekt überträgt Rekursion auf eine Struktur, die tatsächlich
rekursiv aufgebaut ist: einen Verzeichnisbaum. Ein Generator erzeugt mit festem
Seed einen reproduzierbaren Baum aus Ordnern und Dateien. Anschließend liefern
eine rekursive Traversierung und eine iterative Depth-First-Traversierung mit
explizitem Stack exakt dieselbe Analyse.

Das Skriptformat ist bewusst gewählt: Es erzeugt echte Verzeichnisse in einem
ignorierten Datenordner, kann Pfade und Dateigrößen realistisch mit `pathlib`
untersuchen und lässt sich vollständig automatisiert testen und benchmarken.

## Analysierte Kennzahlen

Beide Traversierungen bestimmen:

- Gesamtgröße aller regulären Dateien,
- Zahl der Dateien und Verzeichnisse,
- maximale Verzeichnistiefe,
- Dateien, Verzeichnisse und Bytes pro Tiefe,
- Bytes gruppiert nach Dateiendung,
- Dateien, deren Name oder relativer Pfad ein `fnmatch`-Pattern erfüllt.

Symlinks werden bewusst ignoriert. Dadurch können weder Zyklen noch Dateien
außerhalb des gewählten Wurzelbaums in die Analyse gelangen.

## Aufgabe 1 — Reproduzierbaren Baum generieren

```python
generate_tree(
    root,
    seed=...,
    max_depth=...,
    max_subdirectories=...,
    max_files_per_directory=...,
    overwrite=False,
) -> GenerationSummary
```

Anforderungen:

- Lokale `random.Random`-Instanz mit festem Standard-Seed verwenden.
- Verzeichniszahl, Dateizahl und Tiefe klar begrenzen.
- Eindeutige Namen und zufällige, reproduzierbare Dateigrößen erzeugen.
- Dateien per `truncate` anlegen; ihr Inhalt ist für die Strukturanalyse egal.
- Ein kanonisch sortiertes Manifest zurückgeben.
- Bestehende nichtleere Ziele nur bei explizitem `overwrite=True` ersetzen.
- Niemals ein Dateisystem-Wurzelverzeichnis als Ziel akzeptieren.

## Aufgabe 2 — Rekursive Analyse

```python
analyze_recursive(root, pattern="*") -> TreeAnalysis
```

Ein Aufruf von `visit(directory, depth)` verarbeitet genau ein Verzeichnis. Für
jeden Unterordner ruft er sich mit `depth + 1` auf. Die Fortschrittsgarantie ist
nicht eine kleiner werdende Zahl, sondern der Abstieg in einen endlichen,
zyklenfreien Teilbaum.

Invariante:

> Nach Rückkehr aus `visit(directory, depth)` sind alle regulären Dateien und
> echten Unterordner dieses Teilbaums genau einmal in den Aggregaten enthalten.

## Aufgabe 3 — Iterative Analyse

```python
analyze_iterative(root, pattern="*") -> TreeAnalysis
```

Ersetze den impliziten Call Stack durch:

```text
stack = [(root, 0)]
solange stack nicht leer:
    directory, depth = stack.pop()
    directory auswerten
    Unterordner mit depth + 1 auf stack legen
```

Beide Varianten müssen dasselbe unveränderliche `TreeAnalysis` liefern. Dazu
werden Dictionaries und Treffer am Ende kanonisch sortiert; die Reihenfolge von
`Path.iterdir()` darf das Ergebnis nicht beeinflussen.

## Aufgabe 4 — Pattern-Suche und Tiefenstatistik

Das Pattern wird case-sensitive auf den Dateinamen und den relativen POSIX-Pfad
angewendet. Beispiele:

```text
*.csv              alle CSV-Dateinamen
report*            Namen mit Präfix report
raw_0001/*         Dateien direkt unter diesem relativen Pfadpattern
```

Die Tiefe bezeichnet das enthaltende Verzeichnis: Dateien direkt unter der
Wurzel liegen auf Tiefe `0`; ein Unterordner der Wurzel liegt auf Tiefe `1`.

## Aufgabe 5 — Vergleichsmessung

`benchmark.py` erzeugt den Standardbaum neu, analysiert ihn einmal mit beiden
Strategien und vergleicht die vollständigen Ergebnisobjekte. Nur bei exakter
Gleichheit beginnen die Zeitmessungen. Gemessen wird der Median mehrerer Läufe.

Der Quotient lautet:

```text
rekursive Laufzeit / iterative Laufzeit
```

Ein Wert größer als `1` bedeutet in diesem konkreten Lauf eine schnellere
iterative Variante. Es gibt keine garantierte Richtung: Dateisystem-Caches,
Interpreterkosten, Baumform und Plattform beeinflussen die Messung. Beide
Verfahren besuchen dieselben Knoten und haben dieselbe asymptotische Laufzeit.

## Ausführen

Alle Tests:

```bash
python3 -m unittest -v test_filesystem_analysis.py
```

Nur den Baum erzeugen:

```bash
python3 generate_tree.py
```

Vollständigen Vergleich ausführen:

```bash
python3 benchmark.py
```

Der Benchmark erzeugt `results/traversal_benchmark.csv`. Der Datenbaum und der
Bericht sind reproduzierbare, ignorierte Laufzeitartefakte.

## Komplexität

Seien `D` Verzeichnisse, `F` Dateien und `H` die maximale Tiefe.

| Variante | Zeit | Traversal-Speicher |
|---|---:|---:|
| rekursiv | O(D + F) | O(H) Call Stack |
| iterativ | O(D + F) | O(B) expliziter Stack |

`B` ist die maximale Zahl noch offener Verzeichnisse. Bei Depth First Search ist
sie von Verzweigung und Tiefe abhängig. Ergebnislisten und Aggregattabellen
kommen bei beiden Varianten zusätzlich hinzu.

## Fertig, wenn …

- derselbe Seed dasselbe Manifest erzeugt,
- Grenzen, Überschreiben und gefährliche Ziele validiert werden,
- beide Traversierungen jede reguläre Datei genau einmal zählen,
- Gesamtgrößen, Tiefenstatistik, Extensions und Pattern-Treffer korrekt sind,
- Symlinks ignoriert und Ergebnisreihenfolgen deterministisch sind,
- rekursive und iterative Resultate exakt übereinstimmen,
- Tests, Generator und vollständiger Benchmark fehlerfrei laufen,
- du impliziten Call Stack und expliziten Stack anhand des Codes vergleichen
  kannst.
