"""Minimum spanning tree workshop: Kruskal, Prim, seeded graphs, benchmarks."""

from __future__ import annotations

import csv
import heapq
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path


class DisconnectedGraphError(ValueError):
    """Raised when a single spanning tree cannot cover all vertices."""


@dataclass(frozen=True, order=True)
class Edge:
    """Canonical undirected weighted edge with ``source < target``."""

    weight: float
    source: int
    target: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, int) or not isinstance(self.target, int):
            raise TypeError("Edge endpoints must be integers")
        if self.source < 0 or self.target < 0:
            raise ValueError("Edge endpoints must be non-negative")
        if self.source >= self.target:
            raise ValueError("Edges must be canonical with source < target")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TypeError("Edge weight must be numeric")
        if not math.isfinite(self.weight):
            raise ValueError("Edge weight must be finite")


@dataclass(frozen=True)
class MSTResult:
    algorithm: str
    edges: tuple[Edge, ...]
    total_weight: float


@dataclass(frozen=True)
class BenchmarkCase:
    vertex_count: int
    edge_count: int
    seed: int


@dataclass(frozen=True)
class BenchmarkResult:
    vertex_count: int
    edge_count: int
    seed: int
    mst_weight: float
    kruskal_median_us: float
    prim_median_us: float
    same_edge_set: bool


class UnionFind:
    """Small internal DSU with path compression and union by rank."""

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError("UnionFind size must be non-negative")
        self.parent = list(range(size))
        self.rank = [0] * size
        self.component_count = size

    def find(self, item: int) -> int:
        if not 0 <= item < len(self.parent):
            raise IndexError(item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: int, second: int) -> bool:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return False
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1
        self.component_count -= 1
        return True


def _validate_graph(vertex_count: int, edges: tuple[Edge, ...]) -> None:
    if vertex_count < 0:
        raise ValueError("vertex_count must be non-negative")
    for edge in edges:
        if edge.target >= vertex_count:
            raise ValueError(
                f"Edge {edge.source}-{edge.target} exceeds vertex range "
                f"0..{vertex_count - 1}"
            )


def kruskal(vertex_count: int, edges: tuple[Edge, ...]) -> MSTResult:
    """Build an MST by globally selecting safe edges in ascending order."""

    _validate_graph(vertex_count, edges)
    if vertex_count <= 1:
        return MSTResult("kruskal", (), 0.0)

    dsu = UnionFind(vertex_count)
    selected: list[Edge] = []
    for edge in sorted(edges):
        if dsu.union(edge.source, edge.target):
            selected.append(edge)
            if len(selected) == vertex_count - 1:
                break

    if len(selected) != vertex_count - 1:
        raise DisconnectedGraphError("Graph is disconnected; no spanning tree exists")
    return MSTResult(
        "kruskal",
        tuple(selected),
        sum(edge.weight for edge in selected),
    )


def prim(
    vertex_count: int,
    edges: tuple[Edge, ...],
    *,
    start: int = 0,
) -> MSTResult:
    """Build an MST by growing one tree through its cheapest frontier edge."""

    _validate_graph(vertex_count, edges)
    if vertex_count == 0:
        if start != 0:
            raise IndexError(start)
        return MSTResult("prim", (), 0.0)
    if not 0 <= start < vertex_count:
        raise IndexError(start)
    if vertex_count == 1:
        return MSTResult("prim", (), 0.0)

    adjacency: list[list[tuple[float, int, int]]] = [
        [] for _ in range(vertex_count)
    ]
    for edge in edges:
        adjacency[edge.source].append((edge.weight, edge.source, edge.target))
        adjacency[edge.target].append((edge.weight, edge.target, edge.source))

    visited = {start}
    heap = list(adjacency[start])
    heapq.heapify(heap)
    selected: list[Edge] = []

    while heap and len(visited) < vertex_count:
        weight, source, target = heapq.heappop(heap)
        if target in visited:
            continue
        visited.add(target)
        selected.append(Edge(weight, min(source, target), max(source, target)))
        for candidate in adjacency[target]:
            if candidate[2] not in visited:
                heapq.heappush(heap, candidate)

    if len(visited) != vertex_count:
        raise DisconnectedGraphError("Graph is disconnected; no spanning tree exists")
    return MSTResult("prim", tuple(selected), sum(edge.weight for edge in selected))


def generate_connected_graph(
    vertex_count: int,
    edge_count: int,
    *,
    seed: int,
    max_weight: int = 100,
) -> tuple[Edge, ...]:
    """Generate a simple connected graph reproducibly.

    A random spanning tree guarantees connectivity. Remaining edges are sampled
    without replacement from all still unused vertex pairs.
    """

    if vertex_count < 0:
        raise ValueError("vertex_count must be non-negative")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    minimum_edges = max(0, vertex_count - 1)
    maximum_edges = vertex_count * (vertex_count - 1) // 2
    if not minimum_edges <= edge_count <= maximum_edges:
        raise ValueError(
            f"edge_count must be between {minimum_edges} and {maximum_edges}"
        )

    generator = random.Random(seed)
    edges: list[Edge] = []
    used_pairs: set[tuple[int, int]] = set()

    for target in range(1, vertex_count):
        source = generator.randrange(target)
        pair = (source, target)
        used_pairs.add(pair)
        edges.append(Edge(generator.randint(1, max_weight), source, target))

    candidates = [
        (source, target)
        for source in range(vertex_count)
        for target in range(source + 1, vertex_count)
        if (source, target) not in used_pairs
    ]
    generator.shuffle(candidates)
    for source, target in candidates[: edge_count - minimum_edges]:
        edges.append(Edge(generator.randint(1, max_weight), source, target))

    return tuple(edges)


def verify_spanning_tree(vertex_count: int, edges: tuple[Edge, ...]) -> bool:
    """Check edge count, acyclicity, endpoint range, and connectivity."""

    if vertex_count < 0:
        return False
    if vertex_count == 0:
        return not edges
    if len(edges) != vertex_count - 1:
        return False
    try:
        _validate_graph(vertex_count, edges)
    except ValueError:
        return False
    dsu = UnionFind(vertex_count)
    for edge in edges:
        if not dsu.union(edge.source, edge.target):
            return False
    return dsu.component_count == 1


def benchmark_case(case: BenchmarkCase, *, repeats: int = 7) -> BenchmarkResult:
    """Run both algorithms repeatedly on one identical immutable edge tuple."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    edges = generate_connected_graph(
        case.vertex_count,
        case.edge_count,
        seed=case.seed,
    )
    kruskal_times: list[int] = []
    prim_times: list[int] = []
    kruskal_result: MSTResult | None = None
    prim_result: MSTResult | None = None

    for _ in range(repeats):
        start = time.perf_counter_ns()
        kruskal_result = kruskal(case.vertex_count, edges)
        kruskal_times.append(time.perf_counter_ns() - start)

        start = time.perf_counter_ns()
        prim_result = prim(case.vertex_count, edges)
        prim_times.append(time.perf_counter_ns() - start)

    assert kruskal_result is not None and prim_result is not None
    if not math.isclose(kruskal_result.total_weight, prim_result.total_weight):
        raise AssertionError("Kruskal and Prim disagree on MST weight")
    if not verify_spanning_tree(case.vertex_count, kruskal_result.edges):
        raise AssertionError("Kruskal returned an invalid spanning tree")
    if not verify_spanning_tree(case.vertex_count, prim_result.edges):
        raise AssertionError("Prim returned an invalid spanning tree")

    return BenchmarkResult(
        vertex_count=case.vertex_count,
        edge_count=case.edge_count,
        seed=case.seed,
        mst_weight=kruskal_result.total_weight,
        kruskal_median_us=statistics.median(kruskal_times) / 1_000,
        prim_median_us=statistics.median(prim_times) / 1_000,
        same_edge_set=set(kruskal_result.edges) == set(prim_result.edges),
    )


DEFAULT_CASES = (
    BenchmarkCase(25, 60, 1301),
    BenchmarkCase(75, 250, 1302),
    BenchmarkCase(150, 600, 1303),
    BenchmarkCase(300, 1_500, 1304),
)


def run_benchmark(
    cases: tuple[BenchmarkCase, ...] = DEFAULT_CASES,
    *,
    repeats: int = 7,
) -> tuple[BenchmarkResult, ...]:
    return tuple(benchmark_case(case, repeats=repeats) for case in cases)


def write_benchmark_csv(results: tuple[BenchmarkResult, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "vertices",
                "edges",
                "seed",
                "mst_weight",
                "kruskal_median_us",
                "prim_median_us",
                "same_edge_set",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result.vertex_count,
                    result.edge_count,
                    result.seed,
                    f"{result.mst_weight:.3f}",
                    f"{result.kruskal_median_us:.3f}",
                    f"{result.prim_median_us:.3f}",
                    str(result.same_edge_set).lower(),
                )
            )
