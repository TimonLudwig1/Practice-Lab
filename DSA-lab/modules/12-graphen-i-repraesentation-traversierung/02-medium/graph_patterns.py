"""Klassische Graphmuster mit optionalen, reproduzierbaren Traversierungs-Traces."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from typing import TypeVar


Vertex = TypeVar("Vertex", bound=Hashable)
Adjacency = Mapping[Vertex, Iterable[Vertex]]
Trace = MutableSequence[str] | None


class CycleError(ValueError):
    """Eine topologische Sortierung scheitert an einem gerichteten Zyklus."""

    def __init__(self, message: str, trace: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.trace = tuple(trace)


@dataclass(frozen=True)
class IslandReport:
    """Ergebnis einer vollständigen Inselanalyse."""

    count: int
    sizes: tuple[int, ...]


def _record(trace: Trace, message: str) -> None:
    if trace is not None:
        trace.append(message)


def _normalize(adjacency: Adjacency[Vertex]) -> dict[Vertex, tuple[Vertex, ...]]:
    """Nimmt Zielknoten auf und entfernt doppelte Nachbarn stabil."""

    normalized: dict[Vertex, dict[Vertex, None]] = {}
    for source, neighbors in adjacency.items():
        normalized.setdefault(source, {})
        for target in neighbors:
            normalized.setdefault(target, {})
            normalized[source].setdefault(target, None)
    return {vertex: tuple(neighbors) for vertex, neighbors in normalized.items()}


def _require_undirected(
    adjacency: Adjacency[Vertex],
) -> dict[Vertex, tuple[Vertex, ...]]:
    normalized = _normalize(adjacency)
    for source, neighbors in normalized.items():
        for target in neighbors:
            if source not in normalized[target]:
                raise ValueError(
                    "Ungerichtete Adjazenzliste ist nicht symmetrisch: "
                    f"{source!r} -> {target!r} hat keine Rückkante"
                )
    return normalized


def has_undirected_cycle(adjacency: Adjacency[Vertex], trace: Trace = None) -> bool:
    """Erkennt per DFS und Parent-Kante einen Zyklus im ungerichteten Graphen."""

    graph = _require_undirected(adjacency)
    visited: set[Vertex] = set()

    def visit(vertex: Vertex, parent: Vertex | None) -> bool:
        visited.add(vertex)
        _record(trace, f"Besuche {vertex!r}; Parent={parent!r}")

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                _record(trace, f"Baumkante {vertex!r}--{neighbor!r}")
                if visit(neighbor, vertex):
                    return True
            elif neighbor != parent:
                _record(trace, f"Zyklus über {vertex!r}--{neighbor!r}")
                return True

        _record(trace, f"Schließe {vertex!r} ab")
        return False

    for start in graph:
        if start not in visited:
            _record(trace, f"Neue Komponente ab {start!r}")
            if visit(start, None):
                return True
    return False


def has_directed_cycle(adjacency: Adjacency[Vertex], trace: Trace = None) -> bool:
    """Erkennt einen gerichteten Zyklus mittels Weiß-Grau-Schwarz-DFS."""

    graph = _normalize(adjacency)
    # 0 = weiß/unbesucht, 1 = grau/aktiv, 2 = schwarz/abgeschlossen
    color = {vertex: 0 for vertex in graph}

    def visit(vertex: Vertex) -> bool:
        color[vertex] = 1
        _record(trace, f"{vertex!r}: weiß -> grau")

        for neighbor in graph[vertex]:
            if color[neighbor] == 1:
                _record(trace, f"Rückkante {vertex!r}->{neighbor!r}: Zyklus")
                return True
            if color[neighbor] == 0:
                _record(trace, f"Baumkante {vertex!r}->{neighbor!r}")
                if visit(neighbor):
                    return True

        color[vertex] = 2
        _record(trace, f"{vertex!r}: grau -> schwarz")
        return False

    for start in graph:
        if color[start] == 0 and visit(start):
            return True
    return False


def topological_sort_kahn(
    adjacency: Adjacency[Vertex], trace: Trace = None
) -> tuple[Vertex, ...]:
    """Sortiert einen DAG mit In-Degrees und Queue nach Kahn.

    Bei mehreren gleichzeitig verfügbaren Knoten entscheidet die stabile
    Einfügereihenfolge der Eingabe. Ein Zyklus löst ``CycleError`` aus.
    """

    graph = _normalize(adjacency)
    in_degree = {vertex: 0 for vertex in graph}
    for neighbors in graph.values():
        for neighbor in neighbors:
            in_degree[neighbor] += 1

    queue = deque(vertex for vertex in graph if in_degree[vertex] == 0)
    _record(trace, f"Startqueue: {list(queue)!r}")
    order: list[Vertex] = []

    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        _record(trace, f"Entnimm {vertex!r}; Ordnung={order!r}")

        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            _record(trace, f"In-Degree {neighbor!r} -> {in_degree[neighbor]}")
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                _record(trace, f"Füge {neighbor!r} zur Queue hinzu")

    if len(order) != len(graph):
        remaining = [vertex for vertex in graph if in_degree[vertex] > 0]
        _record(trace, f"Restknoten mit In-Degree > 0: {remaining!r}")
        raise CycleError(
            f"Topologische Sortierung unmöglich; Zyklus bei {remaining!r}",
            trace or (),
        )

    return tuple(order)


def topological_sort_dfs(
    adjacency: Adjacency[Vertex], trace: Trace = None
) -> tuple[Vertex, ...]:
    """Sortiert einen DAG über die umgekehrte DFS-Abschlussreihenfolge."""

    graph = _normalize(adjacency)
    color = {vertex: 0 for vertex in graph}
    finished: list[Vertex] = []

    def visit(vertex: Vertex) -> None:
        color[vertex] = 1
        _record(trace, f"Öffne {vertex!r}")
        for neighbor in graph[vertex]:
            if color[neighbor] == 1:
                _record(trace, f"Rückkante {vertex!r}->{neighbor!r}: Zyklus")
                raise CycleError(
                    f"Topologische Sortierung unmöglich; Rückkante "
                    f"{vertex!r}->{neighbor!r}",
                    trace or (),
                )
            if color[neighbor] == 0:
                visit(neighbor)
        color[vertex] = 2
        finished.append(vertex)
        _record(trace, f"Schließe {vertex!r}; Abschlussliste={finished!r}")

    for start in graph:
        if color[start] == 0:
            visit(start)

    finished.reverse()
    _record(trace, f"Drehe Abschlussliste um: {finished!r}")
    return tuple(finished)


def is_valid_topological_order(
    adjacency: Adjacency[Vertex], order: Iterable[Vertex]
) -> bool:
    """Prüft Vollständigkeit, Eindeutigkeit und jede Kantenrichtung einer Ordnung."""

    graph = _normalize(adjacency)
    sequence = tuple(order)
    if len(sequence) != len(graph) or len(set(sequence)) != len(sequence):
        return False
    if set(sequence) != set(graph):
        return False

    position = {vertex: index for index, vertex in enumerate(sequence)}
    return all(
        position[source] < position[target]
        for source, neighbors in graph.items()
        for target in neighbors
    )


def bipartite_coloring(
    adjacency: Adjacency[Vertex], trace: Trace = None
) -> dict[Vertex, int] | None:
    """Färbt einen ungerichteten Graphen per BFS mit 0/1 oder liefert ``None``."""

    graph = _require_undirected(adjacency)
    colors: dict[Vertex, int] = {}

    for start in graph:
        if start in colors:
            continue
        colors[start] = 0
        queue = deque([start])
        _record(trace, f"Neue Komponente: färbe {start!r} mit 0")

        while queue:
            vertex = queue.popleft()
            _record(trace, f"Entnimm {vertex!r} (Farbe {colors[vertex]})")
            for neighbor in graph[vertex]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[vertex]
                    queue.append(neighbor)
                    _record(trace, f"Färbe {neighbor!r} mit {colors[neighbor]}")
                elif colors[neighbor] == colors[vertex]:
                    _record(
                        trace,
                        f"Konflikt {vertex!r}--{neighbor!r}: beide Farbe "
                        f"{colors[vertex]}",
                    )
                    return None

    return colors


def is_bipartite(adjacency: Adjacency[Vertex]) -> bool:
    """Bequemer Wahrheitswert-Wrapper um ``bipartite_coloring``."""

    return bipartite_coloring(adjacency) is not None


def analyze_islands(
    grid: Sequence[Sequence[int | bool]], trace: Trace = None
) -> IslandReport:
    """Zählt 4-fach verbundene Inseln und ermittelt ihre Größen per BFS."""

    cells = tuple(tuple(row) for row in grid)
    if not cells:
        return IslandReport(count=0, sizes=())

    width = len(cells[0])
    if any(len(row) != width for row in cells):
        raise ValueError("Das Grid muss rechteckig sein")
    if any(value not in (0, 1, False, True) for row in cells for value in row):
        raise ValueError("Das Grid darf nur 0/Wasser und 1/Land enthalten")
    if width == 0:
        return IslandReport(count=0, sizes=())

    height = len(cells)
    visited: set[tuple[int, int]] = set()
    sizes: list[int] = []

    for row in range(height):
        for column in range(width):
            start = (row, column)
            if cells[row][column] != 1 or start in visited:
                continue

            island_number = len(sizes) + 1
            queue = deque([start])
            visited.add(start)
            size = 0
            _record(trace, f"Insel {island_number} startet bei {start}")

            while queue:
                current_row, current_column = queue.popleft()
                size += 1
                _record(
                    trace,
                    f"Besuche {(current_row, current_column)}; Größe={size}",
                )

                for row_delta, column_delta in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                    next_row = current_row + row_delta
                    next_column = current_column + column_delta
                    neighbor = (next_row, next_column)
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and cells[next_row][next_column] == 1
                        and neighbor not in visited
                    ):
                        visited.add(neighbor)
                        queue.append(neighbor)
                        _record(trace, f"Entdecke {neighbor} und füge es ein")

            sizes.append(size)
            _record(trace, f"Insel {island_number} abgeschlossen; Größe={size}")

    return IslandReport(count=len(sizes), sizes=tuple(sizes))


def count_islands(grid: Sequence[Sequence[int | bool]]) -> int:
    """Liefert nur die Anzahl 4-fach verbundener Inseln."""

    return analyze_islands(grid).count
