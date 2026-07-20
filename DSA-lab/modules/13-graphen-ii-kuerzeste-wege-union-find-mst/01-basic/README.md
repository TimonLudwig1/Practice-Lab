# 01-basic — Dijkstra & Union-Find bauen

Dieses Projekt implementiert zwei zentrale Graphbausteine vollständig selbst:

- Dijkstra mit einer Min-Priority-Queue für nichtnegative Kantengewichte,
- Union-Find mit Path Compression und Union by Rank.

Die Implementierungen verwenden nur die Python-Standardbibliothek. Der
handkonstruierte A–F-Graph entspricht der vollständigen Handrechnung aus der
Theorie.

## Dijkstra-Beispiel

```text
Kanten:
A–B:4, A–C:2, B–C:1, B–D:5, C–D:8,
C–E:10, D–E:2, D–F:6, E–F:3
```

Erwartete Distanzen ab A:

| Knoten | Distanz | Vorgänger |
|---|---:|---|
| A | 0 | – |
| B | 3 | C |
| C | 2 | A |
| D | 8 | B |
| E | 10 | D |
| F | 13 | E |

Der rekonstruierte Weg zu F lautet:

```text
A -> C -> B -> D -> E -> F    Kosten: 13
```

`DijkstraResult` enthält schreibgeschützte Distanzen und Vorgänger, die
Fixierreihenfolge sowie einen vollständigen Trace aus `settle`, `relax` und
`stale`. Ein monotoner Zähler im Heap verhindert, dass Python bei gleichen
Distanzen beliebige Knotentypen miteinander vergleichen muss.

## Union-Find

Jedes Element startet als eigene Komponente. `union(a, b)` hängt die Wurzel mit
kleinerem Rank unter die größere. Bei gleichem Rank wird eine Wurzel gewählt und
ihr Rank erhöht. `find(x)` hängt auf dem Rückweg alle besuchten Elemente direkt
unter die Wurzel.

```python
from algorithms import UnionFind

groups = UnionFind(["A", "B", "C", "D"])
groups.union("A", "B")
groups.union("C", "D")
groups.union("A", "C")

assert groups.connected("B", "D")
assert groups.component_size("A") == 4
```

Mit beiden Optimierungen sind Union und Find amortisiert nahezu `O(1)`; formal
`O(α(n))` mit der extrem langsam wachsenden inversen Ackermann-Funktion.

## Komplexität

| Verfahren/Operation | Zeit | Zusatzspeicher |
|---|---:|---:|
| Dijkstra, Adjazenzliste + Heap | `O((V+E) log V)` | `O(V+E)` |
| Union-Find `find`/`union` amortisiert | `O(α(n))` | – |
| Union-Find gesamt | – | `O(n)` |

## Randfälle

- Dijkstra lehnt negative Gewichte ausdrücklich ab.
- Unerreichbare Knoten behalten Distanz `∞` und liefern keinen Pfad.
- Nullgewichte sind erlaubt.
- Veraltete Heap-Einträge werden übersprungen.
- Union-Find meldet redundante Vereinigungen mit `False`.
- Unbekannte Elemente und Knoten lösen verständliche `KeyError`s aus.

## Ausführen

Im Projektordner:

```bash
python3 demo.py
python3 -m pytest -q
```
