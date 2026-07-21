# 02-medium — Minimum Spanning Tree Workshop

This workshop implements Kruskal's and Prim's algorithms from scratch and runs
them on the exact same reproducible random graphs. The comparison checks both
correctness and local runtime behavior.

## Goals

- implement Kruskal with edge sorting and Union-Find,
- implement Prim with an adjacency list and a minimum heap,
- generate connected simple graphs from explicit seeds,
- verify every returned tree independently,
- compare total MST weights, selected edge sets, and median runtimes,
- distinguish correctness evidence from noisy benchmark observations.

## Seeded graph generation

`generate_connected_graph` first creates a random spanning tree. Each new vertex
is linked to one earlier vertex, so the graph is connected by construction. It
then samples unused vertex pairs without replacement until the requested edge
count is reached.

The same immutable edge tuple is passed to both algorithms. This matters: a
benchmark would not be meaningful if Kruskal and Prim received different random
graphs.

## Algorithms

### Kruskal

1. Sort all edges by weight.
2. Start with one Union-Find component per vertex.
3. Select an edge only when its endpoints are in different components.
4. Stop after `V - 1` selected edges.

Typical complexity: `O(E log E)`.

### Prim

1. Start from one vertex.
2. Store all frontier edges in a minimum heap.
3. Select the cheapest edge leading to an unvisited vertex.
4. Add that vertex's outgoing edges to the heap.

Typical complexity with an adjacency list: `O(E log V)`.

## Correctness checks

Every result is validated independently. A spanning tree must:

- contain exactly `V - 1` edges,
- contain no cycle,
- connect all vertices,
- use only valid endpoints.

Kruskal and Prim must also return the same total weight. Their edge sets do not
have to be identical: tied weights can permit several different minimum spanning
trees.

## Benchmark cases

| Vertices | Edges | Seed |
|---:|---:|---:|
| 25 | 60 | 1301 |
| 75 | 250 | 1302 |
| 150 | 600 | 1303 |
| 300 | 1,500 | 1304 |

Each algorithm is repeated seven times per graph; the CSV stores the median in
microseconds. These numbers describe one local run, not a universal winner.

## Files

- `mst_workshop.py`: graph generator, Kruskal, Prim, verifier, benchmark logic
- `run_workshop.py`: complete benchmark and report generation
- `test_mst_workshop.py`: algorithm, property, boundary, and artifact tests
- `results/mst_comparison.csv`: machine-readable comparison
- `results/REPORT.md`: concise English interpretation

## Run

From this project directory:

```bash
python3 run_workshop.py
python3 -m pytest -q
```

---

# Deutsch — 02-medium: Werkstatt für minimale Spannbäume

Diese Werkstatt implementiert die Algorithmen von Kruskal und Prim vollständig
selbst und führt sie auf exakt denselben reproduzierbaren Zufallsgraphen aus. Der
Vergleich prüft sowohl die Korrektheit als auch das lokale Laufzeitverhalten.

## Ziele

- Kruskal mit Kantensortierung und Union-Find implementieren,
- Prim mit Adjazenzliste und Min-Heap implementieren,
- zusammenhängende einfache Graphen aus expliziten Seeds erzeugen,
- jeden ausgegebenen Baum unabhängig verifizieren,
- MST-Gesamtgewichte, gewählte Kantenmengen und Medianlaufzeiten vergleichen,
- Korrektheitsbelege von schwankenden Benchmark-Beobachtungen unterscheiden.

## Seed-basierte Graphgenerierung

`generate_connected_graph` erzeugt zunächst einen zufälligen Spannbaum. Jeder
neue Knoten wird mit einem früheren Knoten verbunden; der Graph ist daher
konstruktiv zusammenhängend. Danach werden unbenutzte Knotenpaare ohne
Zurücklegen gewählt, bis die gewünschte Kantenzahl erreicht ist.

Beide Algorithmen erhalten dasselbe unveränderliche Kanten-Tupel. Das ist
entscheidend: Ein Benchmark wäre nicht aussagekräftig, wenn Kruskal und Prim
verschiedene Zufallsgraphen bekämen.

## Algorithmen

### Kruskal

1. Alle Kanten nach Gewicht sortieren.
2. Mit einer Union-Find-Komponente pro Knoten beginnen.
3. Eine Kante nur wählen, wenn ihre Endpunkte in verschiedenen Komponenten sind.
4. Nach `V - 1` gewählten Kanten stoppen.

Typische Komplexität: `O(E log E)`.

### Prim

1. Bei einem Knoten starten.
2. Alle Randkanten in einem Min-Heap speichern.
3. Die billigste Kante zu einem unbesuchten Knoten wählen.
4. Die ausgehenden Kanten dieses Knotens in den Heap aufnehmen.

Typische Komplexität mit Adjazenzliste: `O(E log V)`.

## Korrektheitsprüfungen

Jedes Ergebnis wird unabhängig validiert. Ein Spannbaum muss:

- genau `V - 1` Kanten enthalten,
- zyklenfrei sein,
- alle Knoten verbinden,
- ausschließlich gültige Endpunkte verwenden.

Kruskal und Prim müssen außerdem dasselbe Gesamtgewicht liefern. Ihre
Kantenmengen müssen nicht identisch sein: Gleiche Gewichte können mehrere
verschiedene minimale Spannbäume erlauben.

## Benchmark-Fälle

| Knoten | Kanten | Seed |
|---:|---:|---:|
| 25 | 60 | 1301 |
| 75 | 250 | 1302 |
| 150 | 600 | 1303 |
| 300 | 1.500 | 1304 |

Jeder Algorithmus wird pro Graph siebenmal wiederholt; die CSV speichert den
Median in Mikrosekunden. Diese Zahlen beschreiben einen lokalen Lauf, keinen
universellen Gewinner.

## Dateien

- `mst_workshop.py`: Graphgenerator, Kruskal, Prim, Verifikation und Benchmarklogik
- `run_workshop.py`: vollständiger Benchmark und Berichtserzeugung
- `test_mst_workshop.py`: Algorithmus-, Property-, Randfall- und Artefakttests
- `results/mst_comparison.csv`: maschinenlesbarer Vergleich
- `results/REPORT.md`: kompakte englische Interpretation

## Ausführen

In diesem Projektordner:

```bash
python3 run_workshop.py
python3 -m pytest -q
```
