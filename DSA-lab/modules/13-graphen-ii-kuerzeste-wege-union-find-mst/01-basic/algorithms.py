"""Dijkstra und Union-Find als eigenständige, getestete Grundbausteine."""

from __future__ import annotations

import heapq
import math
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from itertools import count
from types import MappingProxyType
from typing import Generic, TypeVar


Vertex = TypeVar("Vertex", bound=Hashable)
Element = TypeVar("Element", bound=Hashable)


class WeightedGraph(Generic[Vertex]):
    """Gewichteter gerichteter oder ungerichteter Adjazenzlisten-Graph."""

    def __init__(self, *, directed: bool = False) -> None:
        self._directed = directed
        self._adjacency: dict[Vertex, dict[Vertex, float]] = {}
        self._edge_count = 0

    @property
    def directed(self) -> bool:
        return self._directed

    @property
    def vertices(self) -> tuple[Vertex, ...]:
        return tuple(self._adjacency)

    @property
    def vertex_count(self) -> int:
        return len(self._adjacency)

    @property
    def edge_count(self) -> int:
        return self._edge_count

    @property
    def adjacency(self) -> Mapping[Vertex, Mapping[Vertex, float]]:
        """Liefert eine vollständig schreibgeschützte Momentaufnahme."""

        snapshot = {
            vertex: MappingProxyType(dict(neighbors))
            for vertex, neighbors in self._adjacency.items()
        }
        return MappingProxyType(snapshot)

    def __len__(self) -> int:
        return self.vertex_count

    def __contains__(self, vertex: object) -> bool:
        return vertex in self._adjacency

    def add_vertex(self, vertex: Vertex) -> bool:
        if vertex in self._adjacency:
            return False
        self._adjacency[vertex] = {}
        return True

    def add_edge(self, source: Vertex, target: Vertex, weight: float) -> bool:
        """Fügt eine Kante ein oder aktualisiert ihr Gewicht.

        Rückgabewert ``True`` bedeutet, dass eine neue logische Kante entstand;
        ``False`` bedeutet, dass eine vorhandene Kante aktualisiert wurde.
        Negative Gewichte dürfen gespeichert werden, werden aber von Dijkstra
        ausdrücklich abgelehnt.
        """

        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("Kantengewicht muss eine reelle Zahl sein")
        if not math.isfinite(weight):
            raise ValueError("Kantengewicht muss endlich sein")

        self.add_vertex(source)
        self.add_vertex(target)
        is_new = target not in self._adjacency[source]
        numeric_weight = float(weight)
        self._adjacency[source][target] = numeric_weight
        if not self._directed and source != target:
            self._adjacency[target][source] = numeric_weight
        if is_new:
            self._edge_count += 1
        return is_new

    def has_edge(self, source: Vertex, target: Vertex) -> bool:
        return source in self._adjacency and target in self._adjacency[source]

    def neighbors(self, vertex: Vertex) -> Mapping[Vertex, float]:
        self._require_vertex(vertex)
        return MappingProxyType(dict(self._adjacency[vertex]))

    def weight(self, source: Vertex, target: Vertex) -> float:
        self._require_vertex(source)
        try:
            return self._adjacency[source][target]
        except KeyError as error:
            raise KeyError(f"Kante existiert nicht: {source!r}->{target!r}") from error

    def path_cost(self, path: Iterable[Vertex]) -> float:
        vertices = tuple(path)
        if not vertices:
            raise ValueError("Ein Pfad muss mindestens einen Knoten enthalten")
        for vertex in vertices:
            self._require_vertex(vertex)
        return sum(
            self.weight(source, target)
            for source, target in zip(vertices, vertices[1:])
        )

    def _require_vertex(self, vertex: Vertex) -> None:
        if vertex not in self._adjacency:
            raise KeyError(f"Unbekannter Knoten: {vertex!r}")


@dataclass(frozen=True)
class DijkstraEvent(Generic[Vertex]):
    """Ein reproduzierbarer Schritt des Algorithmus."""

    action: str
    vertex: Vertex
    neighbor: Vertex | None
    distance: float
    previous_distance: float | None = None


@dataclass(frozen=True)
class DijkstraResult(Generic[Vertex]):
    """Unveränderliche Distanzen, Vorgänger und Trace eines Dijkstra-Laufs."""

    start: Vertex
    distances: Mapping[Vertex, float]
    predecessors: Mapping[Vertex, Vertex | None]
    settled_order: tuple[Vertex, ...]
    trace: tuple[DijkstraEvent[Vertex], ...]

    def path_to(self, target: Vertex) -> tuple[Vertex, ...] | None:
        if target not in self.distances:
            raise KeyError(f"Unbekannter Zielknoten: {target!r}")
        if math.isinf(self.distances[target]):
            return None

        reversed_path: list[Vertex] = []
        current: Vertex | None = target
        while current is not None:
            reversed_path.append(current)
            if current == self.start:
                reversed_path.reverse()
                return tuple(reversed_path)
            current = self.predecessors[current]
        raise RuntimeError("Vorgängerkette erreicht den Start nicht")


def dijkstra(graph: WeightedGraph[Vertex], start: Vertex) -> DijkstraResult[Vertex]:
    """Berechnet kürzeste Wege bei nichtnegativen Gewichten mit einem Min-Heap."""

    if start not in graph:
        raise KeyError(f"Unbekannter Startknoten: {start!r}")
    for source in graph.vertices:
        for target, weight in graph.neighbors(source).items():
            if weight < 0:
                raise ValueError(
                    f"Dijkstra erlaubt keine negativen Gewichte: "
                    f"{source!r}->{target!r} hat {weight}"
                )

    distances = {vertex: math.inf for vertex in graph.vertices}
    predecessors: dict[Vertex, Vertex | None] = {
        vertex: None for vertex in graph.vertices
    }
    distances[start] = 0.0
    sequence = count()
    heap: list[tuple[float, int, Vertex]] = [(0.0, next(sequence), start)]
    settled_order: list[Vertex] = []
    trace: list[DijkstraEvent[Vertex]] = []

    while heap:
        distance, _, vertex = heapq.heappop(heap)
        if distance != distances[vertex]:
            trace.append(DijkstraEvent("stale", vertex, None, distance))
            continue

        settled_order.append(vertex)
        trace.append(DijkstraEvent("settle", vertex, None, distance))

        for neighbor, weight in graph.neighbors(vertex).items():
            candidate = distance + weight
            if candidate < distances[neighbor]:
                old_distance = distances[neighbor]
                distances[neighbor] = candidate
                predecessors[neighbor] = vertex
                heapq.heappush(heap, (candidate, next(sequence), neighbor))
                trace.append(
                    DijkstraEvent(
                        "relax",
                        vertex,
                        neighbor,
                        candidate,
                        old_distance,
                    )
                )

    return DijkstraResult(
        start=start,
        distances=MappingProxyType(distances),
        predecessors=MappingProxyType(predecessors),
        settled_order=tuple(settled_order),
        trace=tuple(trace),
    )


class UnionFind(Generic[Element]):
    """Disjoint Set Union mit Path Compression und Union by Rank."""

    def __init__(self, elements: Iterable[Element] = ()) -> None:
        self._parent: dict[Element, Element] = {}
        self._rank: dict[Element, int] = {}
        self._size: dict[Element, int] = {}
        self._component_count = 0
        for element in elements:
            self.add(element)

    @property
    def component_count(self) -> int:
        return self._component_count

    @property
    def elements(self) -> tuple[Element, ...]:
        return tuple(self._parent)

    @property
    def parents(self) -> Mapping[Element, Element]:
        return MappingProxyType(dict(self._parent))

    @property
    def ranks(self) -> Mapping[Element, int]:
        return MappingProxyType(dict(self._rank))

    def __len__(self) -> int:
        return len(self._parent)

    def __contains__(self, element: object) -> bool:
        return element in self._parent

    def add(self, element: Element) -> bool:
        if element in self._parent:
            return False
        self._parent[element] = element
        self._rank[element] = 0
        self._size[element] = 1
        self._component_count += 1
        return True

    def find(self, element: Element) -> Element:
        """Findet die Wurzel und komprimiert den gesamten Suchpfad rekursiv."""

        if element not in self._parent:
            raise KeyError(f"Unbekanntes Element: {element!r}")
        if self._parent[element] != element:
            self._parent[element] = self.find(self._parent[element])
        return self._parent[element]

    def union(self, first: Element, second: Element) -> bool:
        """Vereinigt nach Rank und meldet, ob zwei Komponenten verschmolzen."""

        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return False

        if self._rank[root_first] < self._rank[root_second]:
            root_first, root_second = root_second, root_first
        self._parent[root_second] = root_first
        self._size[root_first] += self._size.pop(root_second)
        if self._rank[root_first] == self._rank[root_second]:
            self._rank[root_first] += 1
        self._component_count -= 1
        return True

    def connected(self, first: Element, second: Element) -> bool:
        return self.find(first) == self.find(second)

    def component_size(self, element: Element) -> int:
        return self._size[self.find(element)]

    def components(self) -> tuple[frozenset[Element], ...]:
        groups: dict[Element, set[Element]] = {}
        for element in self._parent:
            root = self.find(element)
            groups.setdefault(root, set()).add(element)
        return tuple(frozenset(group) for group in groups.values())

    def validate(self) -> None:
        """Prüft Wurzeln, Rank-/Size-Metadaten und Komponentenzähler."""

        assert set(self._parent) == set(self._rank)
        roots = {self.find(element) for element in self._parent}
        assert roots == set(self._size)
        assert self._component_count == len(roots)
        assert sum(self._size.values()) == len(self._parent)
        for root in roots:
            assert self._parent[root] == root
            assert self._size[root] == sum(
                1 for element in self._parent if self.find(element) == root
            )


def hand_calculation_graph() -> WeightedGraph[str]:
    """Erzeugt den A–F-Beispielgraphen der Theorie-Handrechnung."""

    graph: WeightedGraph[str] = WeightedGraph()
    for source, target, weight in (
        ("A", "B", 4),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 5),
        ("C", "D", 8),
        ("C", "E", 10),
        ("D", "E", 2),
        ("D", "F", 6),
        ("E", "F", 3),
    ):
        graph.add_edge(source, target, weight)
    return graph
