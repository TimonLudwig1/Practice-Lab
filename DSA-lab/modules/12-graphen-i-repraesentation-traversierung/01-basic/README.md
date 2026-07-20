# 01-basic — Graph-Grundgerüst

Dieses Projekt implementiert einen ungewichteten Graphen von Grund auf. Die
Adjazenzliste verwendet Dictionaries: Knoten und Kanten sind dadurch im Mittel
in `O(1)` auffindbar, während die Einfügereihenfolge reproduzierbare BFS- und
DFS-Ergebnisse liefert.

## Enthalten

- generische `Graph`-Klasse für beliebige hashbare Knoten,
- gerichtete und ungerichtete Kanten,
- automatische Aufnahme neuer Knoten beim Einfügen einer Kante,
- BFS mit `deque`,
- rekursive und iterative DFS mit gleicher deterministischer Reihenfolge,
- Zusammenhangskomponenten für ungerichtete Graphen,
- schreibgeschützte Adjazenz-Momentaufnahme und interne Invariantenprüfung,
- Demo und Tests auf handkonstruierten Graphen.

## Beispielgraph

```text
        A
       / \
      B   C
      |   |
      D   E
          |
          F
```

Die Kanten werden in der Reihenfolge `A-B`, `A-C`, `B-D`, `C-E`, `E-F`
eingefügt. Deshalb entstehen ab `A` diese stabilen Traversierungen:

```text
BFS:           A -> B -> C -> D -> E -> F
DFS rekursiv:  A -> B -> D -> C -> E -> F
DFS iterativ:  A -> B -> D -> C -> E -> F
```

BFS arbeitet Ebene für Ebene mit einer Queue. DFS verfolgt zunächst den ersten
Pfad bis zum Ende. Die iterative DFS legt Nachbarn in umgekehrter Reihenfolge
auf den Stack, damit beim Entfernen derselbe Nachbar zuerst besucht wird wie in
der rekursiven Variante.

## Wichtige API

```python
from graph import Graph

graph: Graph[str] = Graph(directed=False)
graph.add_edge("A", "B")
graph.add_edge("A", "C")

print(graph.bfs("A"))           # ('A', 'B', 'C')
print(graph.dfs_recursive("A")) # ('A', 'B', 'C')
print(graph.dfs_iterative("A")) # ('A', 'B', 'C')
```

`connected_components()` ist absichtlich nur für ungerichtete Graphen
definiert. Bei gerichteten Graphen müsste zuerst präzisiert werden, ob schwache
oder starke Zusammenhangskomponenten gemeint sind.

## Komplexität

| Operation | Laufzeit | Zusatzspeicher |
|---|---:|---:|
| Knoten einfügen | erwartet `O(1)` | `O(1)` |
| Kante einfügen/prüfen | erwartet `O(1)` | `O(1)` |
| Nachbarn eines Knotens lesen | `O(deg(v))` | `O(deg(v))` für das Tupel |
| BFS | `O(V + E)` | `O(V)` |
| DFS | `O(V + E)` | `O(V)` |
| Alle Komponenten | `O(V + E)` | `O(V)` |

Bei ungerichteten Graphen steht jede gewöhnliche Kante in zwei
Nachbarschaftslisten. Das ändert nur den konstanten Faktor; die asymptotische
Grenze bleibt `O(V + E)`.

## Ausführen

Im Ordner dieses Projekts:

```bash
python3 demo.py
python3 -m pytest -q
```
