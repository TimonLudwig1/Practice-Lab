"""Ein deterministisches Graph-Grundgerüst auf Basis einer Adjazenzliste.

Die Nachbarschaften werden als Dictionaries gespeichert. Dadurch bleiben Knoten
hash-basiert schnell auffindbar und die Einfügereihenfolge ist zugleich stabil.
Das macht insbesondere Traversierungsbeispiele und Tests reproduzierbar.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Mapping
from types import MappingProxyType
from typing import Generic, TypeVar


Vertex = TypeVar("Vertex", bound=Hashable)


class Graph(Generic[Vertex]):
    """Ein ungewichteter gerichteter oder ungerichteter Graph.

    Knoten dürfen beliebige hashbare Python-Objekte sein. ``add_edge`` legt noch
    nicht vorhandene Endpunkte automatisch an. Parallele Kanten werden bewusst
    nicht gespeichert; eine Selbstschleife zählt als genau eine Kante.
    """

    def __init__(self, *, directed: bool = False) -> None:
        self._directed = directed
        self._adjacency: dict[Vertex, dict[Vertex, None]] = {}
        self._edge_count = 0

    @property
    def directed(self) -> bool:
        """Gibt an, ob Kanten eine Richtung besitzen."""

        return self._directed

    @property
    def vertex_count(self) -> int:
        """Anzahl der Knoten in O(1)."""

        return len(self._adjacency)

    @property
    def edge_count(self) -> int:
        """Anzahl der logischen Kanten in O(1)."""

        return self._edge_count

    @property
    def vertices(self) -> tuple[Vertex, ...]:
        """Alle Knoten in Einfügereihenfolge als unveränderliches Tupel."""

        return tuple(self._adjacency)

    @property
    def adjacency(self) -> Mapping[Vertex, tuple[Vertex, ...]]:
        """Eine schreibgeschützte Momentaufnahme der Adjazenzliste."""

        snapshot = {
            vertex: tuple(neighbors) for vertex, neighbors in self._adjacency.items()
        }
        return MappingProxyType(snapshot)

    def __len__(self) -> int:
        return self.vertex_count

    def __contains__(self, vertex: object) -> bool:
        return vertex in self._adjacency

    def add_vertex(self, vertex: Vertex) -> bool:
        """Fügt einen Knoten ein und meldet, ob er neu war."""

        if vertex in self._adjacency:
            return False
        self._adjacency[vertex] = {}
        return True

    def add_edge(self, source: Vertex, target: Vertex) -> bool:
        """Fügt eine Kante ein und meldet, ob sie neu war.

        Bei ungerichteten Graphen wird die symmetrische Adjazenz automatisch
        ergänzt. Die logische Kante wird dennoch nur einmal gezählt.
        """

        self.add_vertex(source)
        self.add_vertex(target)

        if target in self._adjacency[source]:
            return False

        self._adjacency[source][target] = None
        if not self._directed and source != target:
            self._adjacency[target][source] = None
        self._edge_count += 1
        return True

    def has_edge(self, source: Vertex, target: Vertex) -> bool:
        """Prüft in erwarteter O(1)-Zeit, ob ``source -> target`` existiert."""

        return source in self._adjacency and target in self._adjacency[source]

    def neighbors(self, vertex: Vertex) -> tuple[Vertex, ...]:
        """Gibt die Nachbarn eines Knotens in Einfügereihenfolge zurück."""

        self._require_vertex(vertex)
        return tuple(self._adjacency[vertex])

    def bfs(self, start: Vertex) -> tuple[Vertex, ...]:
        """Durchläuft den erreichbaren Teilgraphen schichtweise mit einer Queue.

        Laufzeit: O(V + E), zusätzlicher Speicher: O(V), jeweils bezogen auf den
        vom Startknoten erreichbaren Teilgraphen.
        """

        self._require_vertex(start)
        visited = {start}
        queue = deque([start])
        order: list[Vertex] = []

        while queue:
            vertex = queue.popleft()
            order.append(vertex)
            for neighbor in self._adjacency[vertex]:
                if neighbor not in visited:
                    # Beim Einreihen markieren: So gelangt jeder Knoten höchstens
                    # einmal in die Queue.
                    visited.add(neighbor)
                    queue.append(neighbor)

        return tuple(order)

    def dfs_recursive(self, start: Vertex) -> tuple[Vertex, ...]:
        """Durchläuft den erreichbaren Teilgraphen rekursiv in die Tiefe."""

        self._require_vertex(start)
        visited: set[Vertex] = set()
        order: list[Vertex] = []

        def visit(vertex: Vertex) -> None:
            visited.add(vertex)
            order.append(vertex)
            for neighbor in self._adjacency[vertex]:
                if neighbor not in visited:
                    visit(neighbor)

        visit(start)
        return tuple(order)

    def dfs_iterative(self, start: Vertex) -> tuple[Vertex, ...]:
        """Durchläuft iterativ mit einem expliziten Stack.

        Nachbarn werden rückwärts auf den Stack gelegt. Daher stimmt die
        deterministische Reihenfolge mit der rekursiven Variante überein.
        """

        self._require_vertex(start)
        visited: set[Vertex] = set()
        stack = [start]
        order: list[Vertex] = []

        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            order.append(vertex)

            unvisited_neighbors = [
                neighbor
                for neighbor in self._adjacency[vertex]
                if neighbor not in visited
            ]
            stack.extend(reversed(unvisited_neighbors))

        return tuple(order)

    def connected_components(self) -> tuple[frozenset[Vertex], ...]:
        """Bestimmt die Zusammenhangskomponenten eines ungerichteten Graphen.

        Die Komponenten erscheinen in der Reihenfolge ihres zuerst eingefügten
        Knotens. Für gerichtete Graphen ist "Zusammenhang" mehrdeutig; deshalb
        verlangt diese Basismethode ausdrücklich einen ungerichteten Graphen.
        """

        if self._directed:
            raise ValueError(
                "connected_components ist nur für ungerichtete Graphen definiert"
            )

        visited: set[Vertex] = set()
        components: list[frozenset[Vertex]] = []

        for start in self._adjacency:
            if start in visited:
                continue
            component = frozenset(self.bfs(start))
            visited.update(component)
            components.append(component)

        return tuple(components)

    def is_connected(self) -> bool:
        """Prüft Zusammenhang; der leere Graph gilt hier als zusammenhängend."""

        if self._directed:
            raise ValueError("is_connected ist nur für ungerichtete Graphen definiert")
        return len(self.connected_components()) <= 1

    def validate(self) -> None:
        """Prüft interne Repräsentations- und Zählerinvarianten.

        Die Methode liefert bei Erfolg nichts und löst bei einem internen Fehler
        einen ``AssertionError`` aus. Sie ist vor allem für Lernzwecke und Tests da.
        """

        if self._directed:
            counted_edges = sum(len(neighbors) for neighbors in self._adjacency.values())
        else:
            counted_edges = 0
            seen: set[frozenset[Vertex]] = set()
            for source, neighbors in self._adjacency.items():
                for target in neighbors:
                    assert source in self._adjacency[target]
                    edge = frozenset((source, target))
                    if edge not in seen:
                        seen.add(edge)
                        counted_edges += 1

        assert counted_edges == self._edge_count

    def _require_vertex(self, vertex: Vertex) -> None:
        if vertex not in self._adjacency:
            raise KeyError(f"Unbekannter Knoten: {vertex!r}")


def example_graph() -> Graph[str]:
    """Erzeugt den handkonstruierten Beispielgraphen aus README und Demo."""

    graph: Graph[str] = Graph()
    for source, target in (
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "E"),
        ("E", "F"),
    ):
        graph.add_edge(source, target)
    return graph
