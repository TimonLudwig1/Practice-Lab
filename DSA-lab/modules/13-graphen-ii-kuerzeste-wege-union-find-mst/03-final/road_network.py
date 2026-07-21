"""Synthetic weighted road network and a from-scratch Dijkstra router."""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from itertools import count


Node = tuple[int, int]
RoadKey = tuple[Node, Node]


def canonical_road(first: Node, second: Node) -> RoadKey:
    if first == second:
        raise ValueError("A road needs two distinct endpoints")
    return (first, second) if first < second else (second, first)


@dataclass(frozen=True, order=True)
class Road:
    first: Node
    second: Node
    travel_time: float

    def __post_init__(self) -> None:
        if self.first >= self.second:
            raise ValueError("Road endpoints must be canonical and distinct")
        if isinstance(self.travel_time, bool) or not isinstance(
            self.travel_time, (int, float)
        ):
            raise TypeError("travel_time must be numeric")
        if not math.isfinite(self.travel_time) or self.travel_time <= 0:
            raise ValueError("travel_time must be finite and positive")

    @property
    def key(self) -> RoadKey:
        return (self.first, self.second)


class RoadNetwork:
    """Immutable rectangular node set with undirected weighted roads."""

    def __init__(self, rows: int, columns: int, roads: tuple[Road, ...]) -> None:
        if rows <= 0 or columns <= 0:
            raise ValueError("rows and columns must be positive")
        self.rows = rows
        self.columns = columns
        self.nodes = tuple(
            (row, column)
            for row in range(rows)
            for column in range(columns)
        )
        node_set = set(self.nodes)
        adjacency: dict[Node, dict[Node, float]] = {
            node: {} for node in self.nodes
        }
        seen: set[RoadKey] = set()
        for road in roads:
            if road.first not in node_set or road.second not in node_set:
                raise ValueError(f"Road endpoint outside network: {road.key!r}")
            if road.key in seen:
                raise ValueError(f"Duplicate road: {road.key!r}")
            seen.add(road.key)
            adjacency[road.first][road.second] = float(road.travel_time)
            adjacency[road.second][road.first] = float(road.travel_time)
        self.roads = roads
        self._adjacency = adjacency
        self._road_keys = frozenset(seen)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def road_count(self) -> int:
        return len(self.roads)

    def contains_node(self, node: Node) -> bool:
        return node in self._adjacency

    def contains_road(self, road: RoadKey) -> bool:
        return canonical_road(*road) in self._road_keys

    def neighbors(self, node: Node) -> tuple[tuple[Node, float], ...]:
        if node not in self._adjacency:
            raise KeyError(f"Unknown node: {node!r}")
        return tuple(self._adjacency[node].items())

    def travel_time(self, first: Node, second: Node) -> float:
        if first not in self._adjacency:
            raise KeyError(f"Unknown node: {first!r}")
        try:
            return self._adjacency[first][second]
        except KeyError as error:
            raise KeyError(f"Road does not exist: {first!r}<->{second!r}") from error

    def path_cost(self, path: tuple[Node, ...]) -> float:
        if not path:
            raise ValueError("A path must contain at least one node")
        return sum(
            self.travel_time(first, second)
            for first, second in zip(path, path[1:])
        )


@dataclass(frozen=True)
class RouteResult:
    start: Node
    target: Node
    path: tuple[Node, ...] | None
    travel_time: float
    settled_nodes: int

    @property
    def reached(self) -> bool:
        return self.path is not None

    @property
    def hop_count(self) -> int | None:
        return None if self.path is None else len(self.path) - 1


@dataclass(frozen=True)
class ClosureScenario:
    name: str
    closed_roads: frozenset[RoadKey] = frozenset()
    closed_nodes: frozenset[Node] = frozenset()


@dataclass(frozen=True)
class ScenarioResult:
    scenario: ClosureScenario
    route: RouteResult
    delay: float
    delay_percent: float


def generate_grid_network(
    rows: int,
    columns: int,
    *,
    seed: int,
    minimum_time: float = 1.0,
    maximum_time: float = 6.0,
) -> RoadNetwork:
    """Generate all orthogonal grid roads with seeded positive travel times."""

    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    if minimum_time <= 0 or maximum_time < minimum_time:
        raise ValueError("Travel-time range must be positive and ordered")
    generator = random.Random(seed)
    roads: list[Road] = []

    for row in range(rows):
        for column in range(columns):
            node = (row, column)
            if column + 1 < columns:
                weight = round(generator.uniform(minimum_time, maximum_time), 3)
                roads.append(Road(node, (row, column + 1), weight))
            if row + 1 < rows:
                weight = round(generator.uniform(minimum_time, maximum_time), 3)
                roads.append(Road(node, (row + 1, column), weight))

    return RoadNetwork(rows, columns, tuple(roads))


def shortest_route(
    network: RoadNetwork,
    start: Node,
    target: Node,
    *,
    closed_roads: frozenset[RoadKey] = frozenset(),
    closed_nodes: frozenset[Node] = frozenset(),
) -> RouteResult:
    """Run Dijkstra without using NetworkX or another algorithm library."""

    if not network.contains_node(start):
        raise KeyError(f"Unknown start node: {start!r}")
    if not network.contains_node(target):
        raise KeyError(f"Unknown target node: {target!r}")
    unknown_nodes = closed_nodes.difference(network.nodes)
    if unknown_nodes:
        raise ValueError(f"Unknown closed nodes: {sorted(unknown_nodes)!r}")

    normalized_closed_roads: set[RoadKey] = set()
    for road in closed_roads:
        normalized = canonical_road(*road)
        if not network.contains_road(normalized):
            raise ValueError(f"Unknown closed road: {normalized!r}")
        normalized_closed_roads.add(normalized)

    if start in closed_nodes or target in closed_nodes:
        return RouteResult(start, target, None, math.inf, 0)
    if start == target:
        return RouteResult(start, target, (start,), 0.0, 1)

    distances = {node: math.inf for node in network.nodes}
    predecessors: dict[Node, Node | None] = {node: None for node in network.nodes}
    distances[start] = 0.0
    tie_breaker = count()
    heap: list[tuple[float, int, Node]] = [(0.0, next(tie_breaker), start)]
    settled = 0

    while heap:
        distance, _, node = heapq.heappop(heap)
        if distance != distances[node]:
            continue
        settled += 1
        if node == target:
            break

        for neighbor, weight in network.neighbors(node):
            if neighbor in closed_nodes:
                continue
            if canonical_road(node, neighbor) in normalized_closed_roads:
                continue
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                predecessors[neighbor] = node
                heapq.heappush(
                    heap,
                    (candidate, next(tie_breaker), neighbor),
                )

    if math.isinf(distances[target]):
        return RouteResult(start, target, None, math.inf, settled)

    reversed_path: list[Node] = []
    current: Node | None = target
    while current is not None:
        reversed_path.append(current)
        if current == start:
            break
        current = predecessors[current]
    reversed_path.reverse()
    path = tuple(reversed_path)
    return RouteResult(start, target, path, distances[target], settled)


def route_roads(path: tuple[Node, ...] | None) -> tuple[RoadKey, ...]:
    if path is None:
        return ()
    return tuple(canonical_road(first, second) for first, second in zip(path, path[1:]))


def default_scenarios(
    network: RoadNetwork,
    start: Node,
    target: Node,
) -> tuple[ClosureScenario, ...]:
    """Create baseline, one-road, and barrier scenarios from the actual network."""

    baseline = shortest_route(network, start, target)
    if baseline.path is None or len(baseline.path) < 2:
        raise ValueError("Baseline route must contain at least one road")
    baseline_roads = route_roads(baseline.path)
    middle_road = baseline_roads[len(baseline_roads) // 2]

    barrier_column = network.columns // 2
    barrier = frozenset(
        canonical_road((row, barrier_column - 1), (row, barrier_column))
        for row in range(1, network.rows)
    )
    return (
        ClosureScenario("baseline"),
        ClosureScenario("single_route_road", frozenset({middle_road})),
        ClosureScenario("north_gap_barrier", barrier),
    )


def analyze_scenarios(
    network: RoadNetwork,
    start: Node,
    target: Node,
    scenarios: tuple[ClosureScenario, ...],
) -> tuple[ScenarioResult, ...]:
    if not scenarios or scenarios[0].name != "baseline":
        raise ValueError("The first scenario must be named 'baseline'")

    results: list[ScenarioResult] = []
    baseline_time: float | None = None
    for scenario in scenarios:
        route = shortest_route(
            network,
            start,
            target,
            closed_roads=scenario.closed_roads,
            closed_nodes=scenario.closed_nodes,
        )
        if baseline_time is None:
            if not route.reached:
                raise ValueError("Baseline route is unreachable")
            baseline_time = route.travel_time
        if route.reached:
            delay = route.travel_time - baseline_time
            delay_percent = 0.0 if baseline_time == 0 else 100 * delay / baseline_time
        else:
            delay = math.inf
            delay_percent = math.inf
        results.append(ScenarioResult(scenario, route, delay, delay_percent))
    return tuple(results)
