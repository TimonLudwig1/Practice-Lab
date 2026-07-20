# Modul 12: Graphen I — Repräsentation & Traversierung

Ein Baum beschreibt eine Hierarchie: Jeder Knoten besitzt außer der Wurzel genau
einen Elternknoten. Viele reale Beziehungen sind weniger ordentlich. Eine Person
kennt mehrere Menschen, ein Softwarepaket hängt von mehreren Paketen ab, Straßen
verbinden Orte in beide Richtungen und eine Datenpipeline verzweigt und vereinigt
sich. Das gemeinsame Denkmodell ist ein **Graph**.

Graphalgorithmen beginnen selten mit einer komplizierten Formel. Zuerst muss
klar sein, was Knoten und Kanten bedeuten, ob Kanten eine Richtung besitzen und
welche Zustände eine Traversierung bereits gesehen hat. Dieses Modul entwickelt
deshalb erst die Modellierung, simuliert BFS und DFS auf demselben Graphen und
formalisiert danach Anwendungen wie Komponenten, Zyklenerkennung, topologische
Sortierung und Bipartitheit.

## Lernziele

Nach diesem Modul kannst du:

- reale Probleme als Knoten und Kanten modellieren,
- gerichtete, ungerichtete und gewichtete Graphen unterscheiden,
- Adjazenzliste und Adjazenzmatrix nach Kosten und Dichte auswählen,
- BFS Ebene für Ebene simulieren und ungewichtete kürzeste Pfade rekonstruieren,
- DFS rekursiv und mit explizitem Stack implementieren,
- Entdeckungs- und Abschlusszeiten konzeptionell deuten,
- Zusammenhangskomponenten bestimmen,
- Zyklen in gerichteten und ungerichteten Graphen erkennen,
- topologische Reihenfolgen mit Kahn und DFS erzeugen und DAGs prüfen,
- Bipartitheit durch Zweifärbung testen,
- Laufzeiten konsequent als Funktion von `V` und `E` angeben.

---

# Teil I — Intuition: Beziehungen statt Reihenfolge

## 1. Graphen als universelles Modell

Ein Graph `G = (V, E)` besteht aus einer Knotenmenge `V` (vertices) und einer
Kantenmenge `E` (edges). Die mathematische Form ist knapp; die eigentliche
Modellierungsarbeit steckt in der Bedeutung:

| Domäne | Knoten | Kante |
|---|---|---|
| soziales Netzwerk | Person | folgt / kennt |
| Paketmanager | Softwarepaket | hängt ab von |
| Straßennetz | Kreuzung | Straße |
| Feature-Graph | Feature | statistische/kausale Beziehung |
| Datenpipeline | Task | muss vorher abgeschlossen sein |
| Wissensgraph | Entität | typisierte Relation |

Dieselben Algorithmen funktionieren in allen Zeilen, sobald die Semantik sauber
in Knoten und Kanten übersetzt wurde. Ein Graph ist daher weniger eine spezielle
Datenstruktur als eine universelle Sprache für Beziehungen.

```python
people = {
    "Ada": {"Grace", "Linus"},
    "Grace": {"Ada"},
    "Linus": {"Ada"},
}
assert "Grace" in people["Ada"]
assert "Ada" in people["Grace"]
```

## 2. Gerichtete und ungerichtete Kanten

Eine ungerichtete Kante `{u, v}` verbindet beide Richtungen. „Ada kennt Grace“
kann so modelliert werden, wenn Bekanntschaft symmetrisch gemeint ist. Eine
gerichtete Kante `(u, v)` ist ein Pfeil `u -> v`. „Task u muss vor Task v laufen“
oder „Ada folgt Grace“ ist nicht automatisch symmetrisch.

```text
ungerichtet: A -- B          gerichtet: A --> B

A ist Nachbar von B          B ist Out-Neighbor von A
und B von A.                 A ist In-Neighbor von B.
```

Eine falsche Richtungsentscheidung ändert Erreichbarkeit, Zyklen und Grade. Sie
ist kein Implementierungsdetail.

## 3. Gewichte und Labels

Eine Kante kann zusätzliche Daten tragen:

- Distanz oder Reisezeit im Straßennetz,
- Kosten einer Aktion,
- Stärke einer Korrelation,
- Typ einer Wissensrelation,
- Kapazität einer Netzwerkverbindung.

Ein gewichteter Graph speichert typischerweise `(neighbor, weight)`. BFS findet
nur dann kürzeste Wege, wenn jede Kante dieselben Kosten besitzt. Beliebige
nichtnegative Gewichte führen später zu Dijkstra.

## 4. Zentrale Terminologie

- **Nachbarn** eines Knotens sind direkt über eine Kante verbunden.
- Der **Grad** in einem ungerichteten Graphen ist die Zahl inzidenter Kanten.
- Im gerichteten Graphen unterscheidet man **In-Degree** und **Out-Degree**.
- Ein **Walk** darf Knoten und Kanten wiederholen.
- Ein **Pfad** wiederholt üblicherweise keine Knoten.
- Ein **Zyklus** beginnt und endet am selben Knoten und besitzt dazwischen eine
  nichtleere Kantenfolge.
- Zwei Knoten sind **verbunden**, wenn ein Pfad zwischen ihnen existiert.
- Ein ungerichteter Graph ist **zusammenhängend**, wenn jedes Knotenpaar verbunden
  ist; sonst zerfällt er in Zusammenhangskomponenten.
- Ein gerichteter azyklischer Graph heißt **DAG** (Directed Acyclic Graph).

Bei gerichteten Graphen ist „Komponente“ mehrdeutig. **Schwach zusammenhängend**
ignoriert Pfeilrichtungen; **stark zusammenhängend** verlangt Wege in beide
Richtungen. Die einfache Komponentensuche dieses Moduls bezieht sich auf
ungerichtete Graphen.

---

# Teil II — Ein gemeinsamer Beispielgraph

## 5. Der Graph für alle Simulationen

Wir verwenden einen ungerichteten Graphen mit sieben Knoten. Die alphabetische
Nachbarreihenfolge macht Traces deterministisch:

```text
        A
       / \
      B   C
     / \   \
    D   E---F       G ist isoliert
```

Kanten: `A-B`, `A-C`, `B-D`, `B-E`, `C-F`, `E-F`. Der Graph besitzt die
Komponenten `{A,B,C,D,E,F}` und `{G}`. Zwischen `A` und `F` existieren unter
anderem die Pfade `A-C-F` und `A-B-E-F`; BFS wird den kürzeren finden.

```python
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
    "G": [],
}

assert sum(len(neighbors) for neighbors in graph.values()) // 2 == 6
assert set(graph) == set("ABCDEFG")
```

Die Summe aller Grade ist bei ungerichteten Graphen `2|E|`, weil jede Kante in
beiden Adjazenzlisten erscheint. Das ist das Handshaking Lemma.

---

# Teil III — Repräsentationen

## 6. Adjazenzliste

Eine Adjazenzliste ordnet jedem Knoten seine ausgehenden Nachbarn zu. In Python
ist `dict[node, list/set]` naheliegend. Eine Liste erhält Reihenfolge und erlaubt
parallele Kanten; ein Set verhindert Duplikate und testet Nachbarschaft im Mittel
in `O(1)`, besitzt aber keine fachliche Reihenfolge.

Speicherbedarf: `O(V + E)` für gerichtete und ebenfalls `O(V + E)` für
ungerichtete Graphen, obwohl dort jede Kante zweimal gespeichert wird.

```python
def add_undirected_edge(adjacency: dict[str, list[str]], u: str, v: str) -> None:
    """Add both directions while preserving explicit vertices."""
    adjacency.setdefault(u, []).append(v)
    adjacency.setdefault(v, []).append(u)


small: dict[str, list[str]] = {}
add_undirected_edge(small, "A", "B")
add_undirected_edge(small, "B", "C")
assert small == {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
```

Ein isolierter Knoten muss explizit als Key mit leerer Nachbarliste gespeichert
werden. Sonst verschwindet er aus Traversierung und Komponentenanzahl.

## 7. Adjazenzmatrix

Nummeriere `V` Knoten von `0` bis `V-1`. Eine Matrix `M` speichert `M[u][v] = 1`,
wenn die Kante existiert. Bei Gewichten steht dort das Gewicht und ein separater
Sentinel repräsentiert „keine Kante“.

```python
def adjacency_matrix(
    adjacency: dict[str, list[str]],
) -> tuple[list[str], list[list[int]]]:
    """Convert a simple graph to a zero/one adjacency matrix."""
    vertices = sorted(adjacency)
    index = {vertex: position for position, vertex in enumerate(vertices)}
    matrix = [[0] * len(vertices) for _ in vertices]
    for source, neighbors in adjacency.items():
        for target in neighbors:
            matrix[index[source]][index[target]] = 1
    return vertices, matrix


vertices, matrix = adjacency_matrix({"A": ["B"], "B": ["A", "C"], "C": ["B"]})
assert vertices == ["A", "B", "C"]
assert matrix == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
```

Die Matrix benötigt immer `O(V²)` Speicher. Kantenexistenz ist `O(1)`, aber alle
Nachbarn eines Knotens aufzuzählen kostet `O(V)`, weil eine ganze Zeile gelesen
wird.

## 8. Kostenvergleich und Wahl

| Operation | Adjazenzliste (Set) | Adjazenzmatrix |
|---|---:|---:|
| Speicher | `O(V + E)` | `O(V²)` |
| Kante `u -> v` testen | durchschnittlich `O(1)` | `O(1)` |
| alle Nachbarn von `u` | `O(deg(u))` | `O(V)` |
| Kante einfügen | durchschnittlich `O(1)` | `O(1)` |
| alle Kanten traversieren | `O(V + E)` | `O(V²)` |

Ein Graph ist **sparse**, wenn `E` viel kleiner als `V²` ist; reale Netzwerke
sind häufig sparse. Dann ist die Adjazenzliste Standard. Eine Matrix eignet sich
für dichte Graphen, sehr häufige Edge Queries oder matrixbasierte numerische
Verfahren. Die Datenstrukturwahl hängt von Dichte und Operationen ab, nicht nur
von Bequemlichkeit.

---

# Teil IV — BFS: Schichtweise mit einer Queue

## 9. Intuition

Breadth-First Search breitet sich wie eine Welle vom Start aus. Zuerst besucht
sie Distanz 0, dann alle Knoten mit einer Kante Abstand, danach Distanz 2 usw.
Eine FIFO-Queue erzwingt genau diese Reihenfolge.

Wichtig ist der Zeitpunkt der Markierung: Ein Knoten wird **beim Enqueue** als
entdeckt markiert. Erst beim Dequeue zu markieren kann denselben Knoten mehrfach
in die Queue legen.

## 10. Vollständige Simulation ab A

| Schritt | Dequeue | neu entdeckt | Queue danach | Distanz |
|---:|---|---|---|---|
| Start | — | `A` | `[A]` | `A:0` |
| 1 | `A` | `B,C` | `[B,C]` | `B:1,C:1` |
| 2 | `B` | `D,E` | `[C,D,E]` | `D:2,E:2` |
| 3 | `C` | `F` | `[D,E,F]` | `F:2` |
| 4 | `D` | — | `[E,F]` | — |
| 5 | `E` | — (`F` entdeckt) | `[F]` | — |
| 6 | `F` | — | `[]` | — |

`G` wird nicht erreicht, weil BFS nur die Startkomponente traversiert.

```python
from collections import deque


def bfs(adjacency: dict[str, list[str]], start: str) -> tuple[list[str], dict[str, int], dict[str, str | None]]:
    """Return BFS order, edge distances, and parent tree."""
    if start not in adjacency:
        raise KeyError(start)
    queue = deque([start])
    distance = {start: 0}
    parent: dict[str, str | None] = {start: None}
    order: list[str] = []
    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        for neighbor in adjacency[vertex]:
            if neighbor not in distance:
                distance[neighbor] = distance[vertex] + 1
                parent[neighbor] = vertex
                queue.append(neighbor)
    return order, distance, parent


bfs_order, distances, bfs_parent = bfs(graph, "A")
assert bfs_order == ["A", "B", "C", "D", "E", "F"]
assert distances == {"A": 0, "B": 1, "C": 1, "D": 2, "E": 2, "F": 2}
```

## 11. Kürzeste ungewichtete Pfade

Wenn BFS einen Knoten erstmals entdeckt, kommt es aus der frühestmöglichen
Schicht. Ein kürzerer Pfad müsste aus einer früheren Schicht kommen und wäre
bereits verarbeitet worden. Daher ist die erste Distanz minimal in Anzahl Kanten.
Parent-Zeiger rekonstruieren einen konkreten kürzesten Pfad rückwärts.

```python
def reconstruct_path(
    parent: dict[str, str | None], target: str
) -> list[str] | None:
    """Reconstruct a parent-tree path to target."""
    if target not in parent:
        return None
    path: list[str] = []
    current: str | None = target
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


assert reconstruct_path(bfs_parent, "F") == ["A", "C", "F"]
assert reconstruct_path(bfs_parent, "G") is None
```

BFS garantiert die Pfadlänge, nicht einen einzigartigen Pfad. Bei mehreren
gleich kurzen Pfaden entscheidet die Nachbarreihenfolge, welcher Parent gesetzt
wird.

## 12. BFS-Komplexität

Mit Adjazenzliste wird jeder erreichbare Knoten einmal enqueued und dequeued.
Jede ausgehende Adjazenz wird einmal betrachtet. Für den gesamten Graphen:

```text
Zeit:    O(V + E)
Speicher O(V) für Queue, visited, distance und parent
```

Bei ungerichteten Graphen erscheint jede Kante zweimal, doch `2E` ist weiterhin
`O(E)`. Mit Adjazenzmatrix kostet dieselbe Traversierung `O(V²)`.

---

# Teil V — DFS: Einen Pfad bis zum Ende verfolgen

## 13. Rekursive DFS

Depth-First Search folgt einem Nachbarn so tief wie möglich. Erst wenn dort kein
unbesuchter Ausgang bleibt, kehrt sie zurück. Der Call Stack speichert den offenen
Pfad.

Simulation ab `A` mit der gegebenen Reihenfolge:

```text
discover A
  discover B
    discover D
    finish D
    discover E
      discover F
        discover C
        finish C
      finish F
    finish E
  finish B
finish A
```

```python
def dfs_recursive(adjacency: dict[str, list[str]], start: str) -> list[str]:
    """Return recursive DFS preorder."""
    if start not in adjacency:
        raise KeyError(start)
    visited: set[str] = set()
    order: list[str] = []

    def visit(vertex: str) -> None:
        visited.add(vertex)
        order.append(vertex)
        for neighbor in adjacency[vertex]:
            if neighbor not in visited:
                visit(neighbor)

    visit(start)
    return order


assert dfs_recursive(graph, "A") == ["A", "B", "D", "E", "F", "C"]
```

## 14. Iterative DFS mit explizitem Stack

Ein Stack ersetzt Rekursionsframes. Wenn die iterative Ausgabe dieselbe
Nachbarreihenfolge wie rekursive DFS besitzen soll, werden Nachbarn in
**umgekehrter** Reihenfolge gepusht: Der zuletzt gepushte liegt oben und wird
zuerst verarbeitet.

```python
def dfs_iterative(adjacency: dict[str, list[str]], start: str) -> list[str]:
    """Return DFS preorder with an explicit stack."""
    if start not in adjacency:
        raise KeyError(start)
    stack = [start]
    visited: set[str] = set()
    order: list[str] = []
    while stack:
        vertex = stack.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        order.append(vertex)
        for neighbor in reversed(adjacency[vertex]):
            if neighbor not in visited:
                stack.append(neighbor)
    return order


assert dfs_iterative(graph, "A") == dfs_recursive(graph, "A")
```

Hier wird beim Pop markiert, weshalb derselbe Knoten vorübergehend mehrfach im
Stack liegen kann. Alternativ markiert man beim Push und speichert komplexere
Frames, wenn Abschlussereignisse benötigt werden.

## 15. Entdeckungs- und Abschlusszeiten

DFS erzeugt zwei Ereignisse pro Knoten:

- **discover**: Der Knoten wird erstmals betreten.
- **finish**: Alle ausgehenden Kanten und Nachkommen sind verarbeitet.

Ein globaler Zähler liefert Intervalle `[discover[v], finish[v]]`. Im DFS-Wald
sind diese Intervalle entweder verschachtelt (Ancestor/Descendant) oder disjunkt.

```python
def dfs_timestamps(adjacency: dict[str, list[str]]) -> tuple[dict[str, int], dict[str, int]]:
    """Create discovery and finish timestamps for a complete DFS forest."""
    discovered: dict[str, int] = {}
    finished: dict[str, int] = {}
    time = 0

    def visit(vertex: str) -> None:
        nonlocal time
        time += 1
        discovered[vertex] = time
        for neighbor in adjacency[vertex]:
            if neighbor not in discovered:
                visit(neighbor)
        time += 1
        finished[vertex] = time

    for vertex in adjacency:
        if vertex not in discovered:
            visit(vertex)
    return discovered, finished


discovered, finished = dfs_timestamps(graph)
assert discovered["A"] < discovered["B"] < finished["B"] < finished["A"]
assert len(discovered) == len(finished) == 7
```

Diese Zeiten erklären später Kantenklassifikation, Zyklenerkennung und
topologische Sortierung. Die Laufzeit bleibt `O(V + E)`, der Stack kann im
Worst Case `O(V)` tief werden. Für sehr tiefe Graphen vermeidet iterative DFS
Pythons Recursion Limit.

## 16. BFS oder DFS?

| Frage | Typisches Werkzeug | Warum |
|---|---|---|
| minimale Kantenanzahl ab Start | BFS | Verarbeitung nach Distanzschichten |
| Erreichbarkeit irgendeines Ziels | beide | beide besuchen Komponente |
| Komponenten | beide | Neustart pro unbesuchtem Knoten |
| tiefe Struktur / Backtracking | DFS | offener Pfad im Stack |
| Zyklus im gerichteten Graphen | DFS-Farben | Kante zu aktivem Pfad |
| topologische Reihenfolge | Kahn/BFS oder DFS | In-Degree bzw. Finish Order |
| bipartit / minimale Färbeschicht | BFS | Farben propagieren schichtweise |

DFS ist nicht grundsätzlich schneller als BFS; beide sind mit Adjazenzliste
`O(V + E)`. Die Wahl richtet sich nach der benötigten Struktur des Ergebnisses.

---

# Teil VI — Anwendungen

## 17. Zusammenhangskomponenten

Eine einzelne Traversierung besucht genau eine ungerichtete Komponente. Starte
für jeden noch unbesuchten Knoten erneut und sammle die jeweils erreichte Menge.

```python
def connected_components(adjacency: dict[str, list[str]]) -> list[set[str]]:
    """Return connected components of an undirected graph."""
    unseen = set(adjacency)
    components: list[set[str]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[str] = set()
        unseen.remove(start)
        while stack:
            vertex = stack.pop()
            component.add(vertex)
            for neighbor in adjacency[vertex]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


assert connected_components(graph) == [set("ABCDEF"), {"G"}]
```

Jeder Knoten wird genau einer Komponente zugeordnet. Gesamtzeit `O(V + E)`,
nicht „Anzahl Komponenten mal Graphgröße“, weil jede Kante insgesamt nur in
ihrer Komponente betrachtet wird.

## 18. Zyklus in ungerichteten Graphen

Beim Traversieren sieht man zwangsläufig die Kante zurück zum Parent; sie ist
kein Zyklus. Eine Kante zu einem bereits besuchten Nachbarn, der **nicht** Parent
ist, schließt dagegen einen Zyklus.

```python
def has_undirected_cycle(adjacency: dict[str, list[str]]) -> bool:
    """Detect a cycle in every component of a simple undirected graph."""
    visited: set[str] = set()

    def visit(vertex: str, parent: str | None) -> bool:
        visited.add(vertex)
        for neighbor in adjacency[vertex]:
            if neighbor not in visited:
                if visit(neighbor, vertex):
                    return True
            elif neighbor != parent:
                return True
        return False

    return any(visit(vertex, None) for vertex in adjacency if vertex not in visited)


assert has_undirected_cycle(graph)  # A-B-E-F-C-A
assert not has_undirected_cycle({"A": ["B"], "B": ["A", "C"], "C": ["B"]})
```

Bei Multigraphen mit parallelen Kanten braucht die Parent-Prüfung Kanten-IDs;
die einfache Version nimmt einen simple graph an.

## 19. Zyklus in gerichteten Graphen: drei Farben

Eine gerichtete Kante zu irgendeinem besuchten Knoten beweist keinen Zyklus. Nur
eine Kante zu einem Knoten, der aktuell auf dem offenen DFS-Pfad liegt, ist eine
Back Edge. Drei Farben modellieren den Zustand:

- `WHITE`: unbesucht,
- `GRAY`: entdeckt, aber nicht abgeschlossen,
- `BLACK`: abgeschlossen.

```python
def has_directed_cycle(adjacency: dict[str, list[str]]) -> bool:
    """Detect a directed cycle with DFS colors."""
    color = {vertex: "WHITE" for vertex in adjacency}

    def visit(vertex: str) -> bool:
        color[vertex] = "GRAY"
        for neighbor in adjacency[vertex]:
            if color[neighbor] == "GRAY":
                return True
            if color[neighbor] == "WHITE" and visit(neighbor):
                return True
        color[vertex] = "BLACK"
        return False

    return any(color[v] == "WHITE" and visit(v) for v in adjacency)


dag = {"extract": ["clean"], "clean": ["train"], "train": [], "report": []}
cyclic = {"A": ["B"], "B": ["C"], "C": ["A"]}
assert not has_directed_cycle(dag)
assert has_directed_cycle(cyclic)
```

## 20. Topologische Sortierung

Eine topologische Reihenfolge ordnet jeden Pfeil `u -> v` so, dass `u` vor `v`
steht. Sie existiert genau für DAGs. Mehrere gültige Reihenfolgen sind normal.

### Kahn: In-Degree und Queue

Kahn startet mit allen Knoten mit In-Degree 0. Wird ein Knoten ausgegeben,
werden seine ausgehenden Kanten gedanklich entfernt. Nachbarn, deren In-Degree
auf 0 fällt, werden bereit.

```text
extract -> clean -> train -> evaluate
                    \-----> deploy (nach evaluate)

Start: extract hat In-Degree 0.
Nach extract: clean wird 0.
Nach clean: train wird 0.
...
```

```python
from collections import deque


def topological_kahn(adjacency: dict[str, list[str]]) -> list[str]:
    """Return a topological order or raise for a directed cycle."""
    indegree = {vertex: 0 for vertex in adjacency}
    for neighbors in adjacency.values():
        for neighbor in neighbors:
            indegree[neighbor] += 1
    ready = deque(sorted(v for v, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while ready:
        vertex = ready.popleft()
        order.append(vertex)
        for neighbor in adjacency[vertex]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                ready.append(neighbor)
    if len(order) != len(adjacency):
        raise ValueError("graph contains a directed cycle")
    return order


pipeline = {
    "extract": ["clean"],
    "clean": ["train"],
    "train": ["evaluate"],
    "evaluate": ["deploy"],
    "deploy": [],
}
assert topological_kahn(pipeline) == ["extract", "clean", "train", "evaluate", "deploy"]
```

Bleiben Knoten übrig, aber die Ready Queue ist leer, besitzt der Restgraph einen
Zyklus. Die Längenprüfung ist damit zugleich DAG-Prüfung.

### DFS: Umgekehrte Abschlussreihenfolge

In einem DAG wird ein Knoten erst abgeschlossen, nachdem alle von ihm
erreichbaren Nachfolger abgeschlossen sind. Append beim Finish und anschließendes
Reverse liefert daher eine topologische Reihenfolge. GRAY erkennt Zyklen.

```python
def topological_dfs(adjacency: dict[str, list[str]]) -> list[str]:
    """Topologically sort a DAG by reverse DFS finish order."""
    color = {vertex: "WHITE" for vertex in adjacency}
    finished: list[str] = []

    def visit(vertex: str) -> None:
        color[vertex] = "GRAY"
        for neighbor in adjacency[vertex]:
            if color[neighbor] == "GRAY":
                raise ValueError("graph contains a directed cycle")
            if color[neighbor] == "WHITE":
                visit(neighbor)
        color[vertex] = "BLACK"
        finished.append(vertex)

    for vertex in adjacency:
        if color[vertex] == "WHITE":
            visit(vertex)
    finished.reverse()
    return finished


def is_topological(order: list[str], adjacency: dict[str, list[str]]) -> bool:
    position = {vertex: index for index, vertex in enumerate(order)}
    return len(position) == len(adjacency) and all(
        position[source] < position[target]
        for source, neighbors in adjacency.items()
        for target in neighbors
    )


assert is_topological(topological_dfs(pipeline), pipeline)
```

Beide Verfahren kosten `O(V + E)`. Kahn macht parallele „bereite“ Aufgaben
sichtbar; DFS ist eng mit Abschlusszeiten verwandt.

## 21. Bipartitheit durch Färbung

Ein ungerichteter Graph ist bipartit, wenn seine Knoten in zwei Mengen zerfallen,
sodass jede Kante die Mengen verbindet. BFS oder DFS propagiert Farben `0/1`:
Jeder Nachbar erhält die Gegenfarbe. Eine Kante zwischen gleich gefärbten Knoten
beweist einen Konflikt.

```python
from collections import deque


def bipartite_coloring(adjacency: dict[str, list[str]]) -> dict[str, int] | None:
    """Return a two-coloring, or None if an odd cycle exists."""
    color: dict[str, int] = {}
    for start in adjacency:
        if start in color:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency[vertex]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[vertex]
                    queue.append(neighbor)
                elif color[neighbor] == color[vertex]:
                    return None
    return color


square = {"A": ["B", "D"], "B": ["A", "C"], "C": ["B", "D"], "D": ["A", "C"]}
triangle = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
assert bipartite_coloring(square) is not None
assert bipartite_coloring(triangle) is None
```

Ein Graph ist genau dann bipartit, wenn er keinen ungeraden Zyklus enthält.
Selbstschleifen sind sofort ein Konflikt. Auch isolierte Knoten erhalten eine
beliebige Farbe.

---

# Teil VII — Formalisierung und Korrektheit

## 22. Traversierungsinvarianten

### BFS

Beim Dequeue eines Knotens `v` ist `distance[v]` die minimale Kantenanzahl vom
Start. Die Queue enthält höchstens zwei aufeinanderfolgende Distanzschichten;
Knoten der kleineren Schicht stehen zuerst.

### DFS

Der rekursive Call Stack beziehungsweise explizite Frame Stack beschreibt einen
Pfad im DFS-Wald. GRAY-Knoten sind exakt die offenen Frames. Beim Finish sind
alle ausgehenden Kanten des Knotens untersucht.

### Visited-Menge

Ohne `visited` kann eine zyklische Struktur endlos traversiert werden. Ein Baum
braucht oft keine Menge, weil der Parent die Rückkehr strukturell kontrolliert;
ein allgemeiner Graph schon.

## 23. Warum `O(V + E)` und nicht `O(V * E)`?

Die verschachtelten Schleifen wirken auf den ersten Blick multiplikativ:

```text
for vertex:
    for neighbor in adjacency[vertex]:
```

Doch die inneren Längen sind unterschiedlich. Ihre Summe ist bei gerichteten
Graphen `E`, bei ungerichteten `2E`. Daher ist die gesamte Nachbararbeit linear
in der Zahl gespeicherter Adjazenzen: `O(V + E)`.

## 24. Traversierungswald

Bei einem nicht zusammenhängenden Graphen erzeugt der Neustart pro unbesuchtem
Knoten einen Wald aus Parent-Bäumen. Tree Edges sind die Kanten, über die ein
Knoten erstmals entdeckt wurde. Andere Kanten gehören zum Originalgraphen, aber
nicht zum Traversierungsbaum.

Die konkrete Waldform hängt von Start- und Nachbarreihenfolge ab. Erreichbarkeit,
Komponenten und die Existenz von Zyklen dürfen davon nicht abhängen.

---

# Teil VIII — Fallstricke

## 25. Typische Fehler

### Nur Knoten mit Kanten speichern

Dann verschwinden isolierte Knoten und Komponentenzahlen sind falsch.

### Ungerichtete Kante nur einmal eintragen

Die Repräsentation verhält sich anschließend gerichtet. Eine Graphklasse sollte
die Symmetrie zentral sicherstellen.

### BFS erst beim Dequeue markieren

Mehrere Eltern können denselben Nachbarn enqueueen. Markiere bei Entdeckung.

### Iterative DFS ohne umgekehrtes Pushen

Sie bleibt korrekt, liefert aber eine andere Reihenfolge als rekursive DFS. Tests
müssen wissen, ob Reihenfolge Teil des Vertrags ist.

### Parent-Kante als ungerichteten Zyklus werten

Jeder DFS-Schritt sieht seinen Parent erneut. Nur ein anderer besuchter Nachbar
beweist den Zyklus.

### Gerichtete und ungerichtete Zykluslogik mischen

Im gerichteten Graphen ist der Parent-Sonderfall unzureichend. Dort zählt der
aktive GRAY-Pfad.

### Topologisch sortieren, ohne DAG zu prüfen

Ein zyklischer Graph besitzt keine topologische Reihenfolge. Kahn prüft die
Ausgabelänge; DFS muss GRAY-Kanten abweisen.

### Unbestimmte Set-Reihenfolge als exakte Ausgabe testen

Viele Graphprobleme besitzen mehrere korrekte Traversierungen oder topologische
Reihenfolgen. Entweder sortiere Nachbarn für Determinismus oder teste semantische
Eigenschaften statt einer einzigen Liste.

## 26. Selbstschleifen, parallele Kanten, fehlende Endpunkte

Eine robuste API muss entscheiden:

- Sind `u -> u`-Selbstschleifen erlaubt?
- Sind parallele Kanten erlaubt oder werden Sets verwendet?
- Legt `add_edge(u, v)` fehlende Knoten automatisch an?
- Wird das Entfernen einer unbekannten Kante ignoriert oder abgewiesen?
- Darf ein Nachbar erscheinen, der kein eigener Key ist?

Diese Entscheidungen beeinflussen Grad, Zyklenerkennung und Speicher. Sie gehören
in den API-Vertrag und in Tests.

---

# Teil IX — Data Science und Transfer

## 27. Graphen im Data-Science-Alltag

- **Feature Graphs:** Knoten sind Features, Kanten markieren hohe Korrelation
  oder bekannte Abhängigkeit. Komponenten zeigen Gruppen redundanter Variablen.
- **Data Lineage:** Tabellen und Transformationen bilden einen gerichteten Graphen.
  Erreichbarkeit beantwortet, welche Outputs von einer Quelle abhängen.
- **Recommendation:** Nutzer und Produkte bilden oft einen bipartiten Graphen.
- **Knowledge Graphs:** typisierte, gerichtete Kanten verbinden Entitäten.
- **Experiment-DAGs:** Topologische Reihenfolge bestimmt ausführbare Schritte.
- **Grid-Daten:** Bilder, Karten und Masken werden implizite Graphen; Nachbarn
  entstehen aus vier oder acht Richtungen, ohne alle Kanten zu speichern.

```python
def count_islands(grid: list[list[int]]) -> int:
    """Count four-neighbor components of ones in a rectangular grid."""
    if not grid:
        return 0
    rows, columns = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    islands = 0
    for row in range(rows):
        for column in range(columns):
            if grid[row][column] == 0 or (row, column) in seen:
                continue
            islands += 1
            stack = [(row, column)]
            seen.add((row, column))
            while stack:
                r, c = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < columns
                        and grid[nr][nc] == 1
                        and (nr, nc) not in seen
                    ):
                        seen.add((nr, nc))
                        stack.append((nr, nc))
    return islands


assert count_islands([[1, 1, 0], [0, 1, 0], [1, 0, 1]]) == 3
```

Der Grid-Graph wird **implizit** repräsentiert: Nachbarn werden bei Bedarf aus
Koordinaten berechnet. Das spart eine separate Adjazenzliste.

## 28. Entscheidungsrezept

1. Was sind Knoten und Kanten?
2. Ist Richtung fachlich relevant?
3. Tragen Kanten Gewichte oder Labels?
4. Ist der Graph sparse oder dense?
5. Benötige ich minimale Kantenanzahl (BFS) oder tiefe Struktur (DFS)?
6. Muss der gesamte Graph oder nur eine Startkomponente verarbeitet werden?
7. Ist eine konkrete Reihenfolge wichtig oder nur eine Eigenschaft?
8. Welche Kantenfälle erlaubt der API-Vertrag?

## 29. Kontrollfragen

1. Warum ist ein Baum ein Graph, aber nicht jeder Graph ein Baum?
2. Wie unterscheiden sich In-Degree und Out-Degree?
3. Wann ist eine Matrix trotz `O(V²)` sinnvoll?
4. Warum findet BFS kürzeste ungewichtete Pfade?
5. Warum werden iterative DFS-Nachbarn oft umgekehrt gepusht?
6. Weshalb muss eine Traversierung pro Komponente neu starten?
7. Wie unterscheiden sich ungerichtete und gerichtete Zyklenerkennung?
8. Was beweist eine leere Kahn-Queue bei noch nicht ausgegebenen Knoten?
9. Warum ist ein Dreieck nicht bipartit?
10. Was bedeutet `O(V + E)` konkret für eine Adjazenzliste?

### Kurzantworten

1. Ein Baum ist zusammenhängend und azyklisch mit eindeutigen Parent-Pfaden;
   allgemeine Graphen dürfen Zyklen, mehrere Pfade und Komponenten besitzen.
2. Eingehende versus ausgehende Pfeile.
3. Bei dichten Graphen oder sehr vielen konstanten Kantenabfragen.
4. Die FIFO-Queue verarbeitet Knoten in nicht abnehmender Distanzschicht.
5. Damit der erste Listen-Nachbar oben auf dem LIFO-Stack landet.
6. Eine Traversierung kann keine Kante zwischen getrennten Komponenten erfinden.
7. Parent-Ausnahme versus Kante zu GRAY auf dem aktiven DFS-Pfad.
8. Der verbleibende gerichtete Teilgraph enthält einen Zyklus.
9. Ein ungerader Zyklus kann nicht konsistent alternierend zweigefärbt werden.
10. Jeder Knoten und jede gespeicherte Adjazenz wird konstant oft bearbeitet.

## 30. Kompakte Gesamtübersicht

```text
Graph G=(V,E)
|
+-- Repräsentation
|   +-- Adjazenzliste  O(V+E), ideal für sparse Graphen
|   `-- Matrix         O(V²), konstante Edge Query
|
+-- BFS (Queue)
|   +-- Ebenen / Distanz
|   +-- ungewichteter kürzester Pfad
|   `-- Zweifärbung
|
+-- DFS (Call Stack / Stack)
|   +-- tiefe Pfade
|   +-- discover / finish
|   +-- Zyklenerkennung
|   `-- reverse Finish Order
|
`-- kompletter Traversierungswald
    +-- Komponenten
    +-- DAG-Prüfung / Toposort
    `-- Erreichbarkeit
```

## 31. Ausblick auf die Projekte

**01-basic** baut eine Graphklasse mit gerichteter/ungerichteter Adjazenzliste,
BFS, rekursiver und iterativer DFS sowie Komponenten. **02-medium** isoliert
Zyklen, beide topologischen Methoden, Bipartitheit und Grid-Inseln mit Traces.
**03-final** analysiert einen reproduzierbaren Pipeline-DAG: Reihenfolge,
kritische Knoten und Ausfallauswirkungen.

Die zentrale Einsicht lautet:

> BFS und DFS sind keine zwei Listen von Besuchsreihenfolgen. Sie sind Gerüste,
> an die Parent-, Distanz-, Farb-, Zeit- und Aggregationszustände angehängt werden,
> um eine ganze Familie von Graphproblemen zu lösen.
