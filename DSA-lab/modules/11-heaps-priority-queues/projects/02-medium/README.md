# Projekt 02-medium: Heap-Pattern-Katalog

## Ziel

Dieses Projekt löst drei Aufgaben, bei denen nicht eine Heap-Klasse, sondern die
richtige **Modellierung des Heap-Inhalts** entscheidend ist:

1. Top-K Frequent hält nur die aktuell besten `k` Häufigkeiten.
2. k-Way-Merge hält nur den aktuellen Kopf jeder sortierten Folge.
3. Running Median teilt den Stream auf einen Max- und einen Min-Heap auf.

Das Format aus importierbarem Modul, Demo, Tests und Benchmark-Skript macht
Korrektheit und Laufzeit getrennt sichtbar. `heapq` ist hier erlaubt, weil das
Lernziel nach dem Eigenbau aus 01-basic auf den höheren Mustern liegt.

## Dateien

- `heap_patterns.py`: die drei produktionsnahen Muster
- `demo.py`: kleine, vollständig nachvollziehbare Beispiele
- `test_heap_patterns.py`: Kantenfälle und reproduzierbare Property-Tests
- `run_benchmarks.py`: Vergleiche mit einfachen Referenzlösungen
- `output/benchmark.csv`: Messergebnisse des vollständigen Standardlaufs

## Muster 1: Top-K Frequent

Nach dem Zählen gibt es `u` eindeutige Werte. Der Min-Heap enthält höchstens
`k` Kandidaten. Seine Wurzel ist der schlechteste aktuelle Gewinner: kleinste
Häufigkeit, bei Gleichstand die späteste erste Fundstelle. Nur ein besserer
Kandidat darf diese Grenze ersetzen.

```text
Stream: a, b, a, c, b, a       k = 2
Counts: a:3, b:2, c:1
Heap hält am Ende: a:3, b:2
Ausgabe: [(a, 3), (b, 2)]
```

Die erste Fundstelle ist ein expliziter Tie-Breaker. Dadurch ist die Ausgabe
deterministisch und die Nutzobjekte müssen untereinander nicht vergleichbar sein.

Komplexität: `O(n)` zum Zählen plus `O(u log k)` für die Kandidaten. Die Hash Map
benötigt `O(u)`, der Heap `O(min(k, u))` Speicher. Eine Vollsortierung aller
eindeutigen Werte kostet dagegen `O(u log u)`.

## Muster 2: k-Way-Merge

Aus jeder nicht leeren, bereits sortierten Folge liegt genau ein Kopf im
Min-Heap. Nach einem Pop wird nur aus derselben Folge der Nachfolger geladen:

```text
A: 1, 7, 10       Heap: (1,A), (2,B), (4,C)
B: 2, 3, 11       Pop (1,A), Push (7,A)
C: 4, 8           Pop (2,B), Push (3,B)
```

Der Sequenzindex dient bei gleichen Werten als Tie-Breaker und identifiziert den
Iterator, der weitergeschaltet werden muss. Die Eingaben dürfen Generatoren sein;
sie werden nicht vollständig in den Speicher kopiert.

Bei insgesamt `n` Werten und `k` Folgen entstehen `n` Pop- und höchstens `n`
Push-Operationen auf einem Heap der Größe `k`: `O(n log k)` Zeit und `O(k)`
Zusatzspeicher ohne Ergebnisliste.

## Muster 3: Running Median

Der kleinere Teil des Streams liegt als negierter Max-Heap `lower` vor, der
größere als Min-Heap `upper`. Nach jedem Insert gelten:

```text
len(lower) == len(upper)
oder
len(lower) == len(upper) + 1

max(lower) <= min(upper)
```

Bei ungerader Anzahl ist `max(lower)` der Median. Bei gerader Anzahl ist der
Mittelwert aus `max(lower)` und `min(upper)` der Median. Ein Insert kostet
`O(log n)`, die Abfrage `O(1)` und der exakte Streamzustand `O(n)` Speicher.

## Aufgabenstellung

1. Formuliere für jedes Muster in einem Satz, **welche Elemente im Heap liegen**
   und **was die Wurzel bedeutet**.
2. Simuliere Top-K Frequent mit mehreren Häufigkeitsgleichständen. Entferne den
   Tie-Breaker testweise und erkläre, was verloren geht.
3. Merge vier sortierte Folgen von Hand. Notiere nach jeder Ausgabe den
   vollständigen Heapzustand.
4. Führe Running Median für eine absteigende Folge aus. Prüfe nach jedem Wert
   Ordnungs- und Größeninvariante.
5. Führe den Benchmark aus. Erkläre Ergebnisse aus den Komplexitäten, nicht nur
   aus einzelnen Zeitwerten.
6. Ergänze für jedes Muster mindestens einen eigenen Property-Test.

## Benchmark richtig lesen

Der Benchmark nutzt den festen Seed `20260720`, prüft Ergebnisgleichheit und
speichert je Fall die beste von drei Messungen. Verglichen werden:

| Muster | Heap-Variante | Referenz |
|---|---|---|
| Top-K | begrenzter Min-Heap | alle eindeutigen Werte sortieren |
| k-Way-Merge | Heap der Folgenköpfe | konkatenieren und mit Timsort sortieren |
| Running Median | zwei Heaps | jeden Präfix neu sortieren |

Die Referenz für k-Way-Merge kann in CPython trotz schwächerer asymptotischer
Garantie schnell sein: `sorted` ist in C implementiert und erkennt vorhandene
Runs. Der Vergleich demonstriert deshalb keine pauschale Siegerliste. Relevant
sind Skalierung, Streaming-Fähigkeit und Speichergrenzen.

Der vollständige Standardlauf ergab folgende charakteristische Endpunkte:

| Muster | Größe | Heap-Variante | Referenz | Referenz / Heap |
|---|---:|---:|---:|---:|
| Top-K | 100.000 Werte | 0,009819 s | 0,006753 s | 0,69x |
| k-Way-Merge | 64 Folgen / 40.000 Werte | 0,005937 s | 0,001848 s | 0,31x |
| Running Median | 3.000 Präfixe | 0,001020 s | 0,128169 s | 125,62x |

Alle 18 Messzeilen stimmen inhaltlich mit ihrer Referenz überein. Top-K und
k-Way-Merge verlieren bei diesen speicherinternen Größen gegen hochoptimierte
C-Sortierung; ihr Vorteil liegt in begrenzter Heap-Größe und inkrementeller
Verarbeitung. Beim Running Median wird die bessere Skalierung bereits deutlich:
Der Speedup wächst im Lauf von 21,30x über 60,05x auf 125,62x. Zeitmessungen
sind maschinenabhängig; die CSV ist die maßgebliche Ausgabe dieses Laufs.

## Ausführen

Im Projektordner:

```bash
python3 demo.py
python3 run_benchmarks.py
python3 -m pytest -q
```

## Fertig, wenn …

- Top-K nur `k` Kandidaten hält und Gleichstände deterministisch löst,
- k-Way-Merge Listen, leere Folgen und Generatoren korrekt verarbeitet,
- Running Median nach jedem Insert beide Invarianten erfüllt,
- alle drei Ergebnisse mit unabhängigen Referenzen übereinstimmen,
- die Laufzeiten mit `n`, `u`, `k` und der Heap-Größe begründet werden,
- Demo, Benchmark, Syntaxprüfung und alle Tests fehlerfrei durchlaufen.
