"""Tests for graph generation, both MST algorithms, and benchmark artifacts."""

import csv
import math
from pathlib import Path

import pytest

from mst_workshop import (
    BenchmarkCase,
    DisconnectedGraphError,
    Edge,
    UnionFind,
    benchmark_case,
    generate_connected_graph,
    kruskal,
    prim,
    run_benchmark,
    verify_spanning_tree,
    write_benchmark_csv,
)
from run_workshop import write_report


THEORY_EDGES = (
    Edge(4, 0, 1),
    Edge(2, 0, 2),
    Edge(1, 1, 2),
    Edge(5, 1, 3),
    Edge(8, 2, 3),
    Edge(10, 2, 4),
    Edge(2, 3, 4),
    Edge(6, 3, 5),
    Edge(3, 4, 5),
)


class TestEdge:
    def test_valid_edge(self) -> None:
        assert Edge(2.5, 1, 3) == Edge(2.5, 1, 3)

    @pytest.mark.parametrize(("source", "target"), [(1, 1), (2, 1), (0, 0)])
    def test_non_canonical_edge_rejected(self, source: int, target: int) -> None:
        with pytest.raises(ValueError, match="canonical"):
            Edge(1, source, target)

    @pytest.mark.parametrize(("source", "target"), [(-1, 2), (0, -1)])
    def test_negative_endpoint_rejected(self, source: int, target: int) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Edge(1, source, target)

    def test_non_integer_endpoint_rejected(self) -> None:
        with pytest.raises(TypeError, match="integers"):
            Edge(1, 0.0, 1)  # type: ignore[arg-type]

    @pytest.mark.parametrize("weight", [math.inf, -math.inf, math.nan])
    def test_non_finite_weight_rejected(self, weight: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            Edge(weight, 0, 1)

    @pytest.mark.parametrize("weight", [True, "1", None])
    def test_non_numeric_weight_rejected(self, weight: object) -> None:
        with pytest.raises(TypeError, match="numeric"):
            Edge(weight, 0, 1)  # type: ignore[arg-type]

    def test_negative_weight_is_valid_for_mst(self) -> None:
        assert Edge(-4, 0, 1).weight == -4

    def test_order_is_weight_then_endpoints(self) -> None:
        edges = [Edge(2, 0, 1), Edge(1, 1, 2), Edge(1, 0, 2)]
        assert sorted(edges) == [Edge(1, 0, 2), Edge(1, 1, 2), Edge(2, 0, 1)]


class TestUnionFind:
    def test_initial_components(self) -> None:
        structure = UnionFind(4)
        assert structure.parent == [0, 1, 2, 3]
        assert structure.component_count == 4

    def test_union_and_redundant_union(self) -> None:
        structure = UnionFind(3)
        assert structure.union(0, 1)
        assert not structure.union(1, 0)
        assert structure.component_count == 2

    def test_path_compression_and_rank(self) -> None:
        structure = UnionFind(4)
        structure.union(0, 1)
        structure.union(2, 3)
        structure.union(0, 2)
        root = structure.find(3)
        assert structure.parent[3] == root
        assert structure.rank[root] == 2

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            UnionFind(-1)

    @pytest.mark.parametrize("item", [-1, 3])
    def test_find_out_of_range(self, item: int) -> None:
        with pytest.raises(IndexError):
            UnionFind(3).find(item)


class TestKnownGraph:
    def test_kruskal_theory_weight(self) -> None:
        result = kruskal(6, THEORY_EDGES)
        assert result.algorithm == "kruskal"
        assert result.total_weight == 13
        assert result.edges == (
            Edge(1, 1, 2),
            Edge(2, 0, 2),
            Edge(2, 3, 4),
            Edge(3, 4, 5),
            Edge(5, 1, 3),
        )

    @pytest.mark.parametrize("start", range(6))
    def test_prim_theory_weight_from_every_start(self, start: int) -> None:
        result = prim(6, THEORY_EDGES, start=start)
        assert result.algorithm == "prim"
        assert result.total_weight == 13
        assert verify_spanning_tree(6, result.edges)

    def test_both_algorithms_return_valid_trees(self) -> None:
        assert verify_spanning_tree(6, kruskal(6, THEORY_EDGES).edges)
        assert verify_spanning_tree(6, prim(6, THEORY_EDGES).edges)

    def test_input_is_not_modified(self) -> None:
        before = THEORY_EDGES
        kruskal(6, THEORY_EDGES)
        prim(6, THEORY_EDGES)
        assert THEORY_EDGES == before

    def test_parallel_edges_choose_cheaper_one(self) -> None:
        edges = (Edge(9, 0, 1), Edge(2, 0, 1), Edge(3, 1, 2), Edge(8, 0, 2))
        assert kruskal(3, edges).total_weight == 5
        assert prim(3, edges).total_weight == 5

    def test_negative_weights_supported(self) -> None:
        edges = (Edge(-3, 0, 1), Edge(2, 1, 2), Edge(5, 0, 2))
        assert kruskal(3, edges).total_weight == -1
        assert prim(3, edges).total_weight == -1


class TestBoundaryCases:
    @pytest.mark.parametrize("algorithm", [kruskal, prim])
    def test_empty_graph(self, algorithm) -> None:  # type: ignore[no-untyped-def]
        result = algorithm(0, ())
        assert result.edges == ()
        assert result.total_weight == 0

    @pytest.mark.parametrize("algorithm", [kruskal, prim])
    def test_single_vertex(self, algorithm) -> None:  # type: ignore[no-untyped-def]
        result = algorithm(1, ())
        assert result.edges == ()
        assert result.total_weight == 0

    @pytest.mark.parametrize("algorithm", [kruskal, prim])
    def test_disconnected_graph_rejected(self, algorithm) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(DisconnectedGraphError, match="disconnected"):
            algorithm(4, (Edge(1, 0, 1), Edge(1, 2, 3)))

    @pytest.mark.parametrize("algorithm", [kruskal, prim])
    def test_negative_vertex_count_rejected(self, algorithm) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="non-negative"):
            algorithm(-1, ())

    @pytest.mark.parametrize("algorithm", [kruskal, prim])
    def test_endpoint_out_of_range_rejected(self, algorithm) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="exceeds"):
            algorithm(2, (Edge(1, 0, 2),))

    @pytest.mark.parametrize("start", [-1, 3])
    def test_prim_start_out_of_range(self, start: int) -> None:
        with pytest.raises(IndexError):
            prim(3, (Edge(1, 0, 1), Edge(1, 1, 2)), start=start)

    def test_empty_graph_nonzero_start_rejected(self) -> None:
        with pytest.raises(IndexError):
            prim(0, (), start=1)


class TestGenerator:
    @pytest.mark.parametrize(
        ("vertices", "edges"),
        [(0, 0), (1, 0), (2, 1), (10, 9), (10, 20), (10, 45)],
    )
    def test_exact_size(self, vertices: int, edges: int) -> None:
        generated = generate_connected_graph(vertices, edges, seed=42)
        assert len(generated) == edges
        assert len({(edge.source, edge.target) for edge in generated}) == edges

    @pytest.mark.parametrize(
        ("vertices", "edges"),
        [(2, 1), (10, 9), (10, 20), (30, 100)],
    )
    def test_generated_graph_is_connected(self, vertices: int, edges: int) -> None:
        generated = generate_connected_graph(vertices, edges, seed=43)
        assert verify_spanning_tree(vertices, kruskal(vertices, generated).edges)

    def test_same_seed_is_identical(self) -> None:
        assert generate_connected_graph(20, 50, seed=99) == generate_connected_graph(
            20, 50, seed=99
        )

    def test_different_seed_changes_graph(self) -> None:
        assert generate_connected_graph(20, 50, seed=99) != generate_connected_graph(
            20, 50, seed=100
        )

    @pytest.mark.parametrize(("vertices", "edges"), [(3, 1), (3, 4), (0, 1)])
    def test_impossible_edge_count_rejected(self, vertices: int, edges: int) -> None:
        with pytest.raises(ValueError, match="edge_count"):
            generate_connected_graph(vertices, edges, seed=1)

    def test_negative_vertex_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            generate_connected_graph(-1, 0, seed=1)

    def test_non_positive_max_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            generate_connected_graph(3, 2, seed=1, max_weight=0)

    def test_weight_range(self) -> None:
        generated = generate_connected_graph(30, 100, seed=7, max_weight=5)
        assert all(1 <= edge.weight <= 5 for edge in generated)


class TestVerifier:
    def test_empty_tree_valid(self) -> None:
        assert verify_spanning_tree(0, ())

    def test_single_vertex_tree_valid(self) -> None:
        assert verify_spanning_tree(1, ())

    def test_wrong_edge_count_invalid(self) -> None:
        assert not verify_spanning_tree(3, (Edge(1, 0, 1),))

    def test_cycle_invalid(self) -> None:
        edges = (Edge(1, 0, 1), Edge(1, 1, 2), Edge(1, 0, 2))
        assert not verify_spanning_tree(4, edges)

    def test_out_of_range_invalid(self) -> None:
        assert not verify_spanning_tree(2, (Edge(1, 0, 2),))

    def test_negative_vertex_count_invalid(self) -> None:
        assert not verify_spanning_tree(-1, ())


class TestBenchmarkAndArtifacts:
    def test_single_benchmark_case(self) -> None:
        result = benchmark_case(BenchmarkCase(20, 50, 123), repeats=2)
        assert result.vertex_count == 20
        assert result.edge_count == 50
        assert result.seed == 123
        assert result.mst_weight > 0
        assert result.kruskal_median_us > 0
        assert result.prim_median_us > 0

    def test_repeats_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="repeats"):
            benchmark_case(BenchmarkCase(5, 4, 1), repeats=0)

    def test_multiple_cases(self) -> None:
        cases = (BenchmarkCase(10, 15, 1), BenchmarkCase(15, 30, 2))
        results = run_benchmark(cases, repeats=1)
        assert tuple(result.seed for result in results) == (1, 2)
        assert all(result.mst_weight > 0 for result in results)

    def test_csv_schema_and_rows(self, tmp_path: Path) -> None:
        results = run_benchmark((BenchmarkCase(10, 15, 1),), repeats=1)
        path = tmp_path / "benchmark.csv"
        write_benchmark_csv(results, path)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert set(rows[0]) == {
            "vertices",
            "edges",
            "seed",
            "mst_weight",
            "kruskal_median_us",
            "prim_median_us",
            "same_edge_set",
        }

    def test_report_contains_interpretation(self, tmp_path: Path) -> None:
        results = run_benchmark((BenchmarkCase(10, 15, 1),), repeats=1)
        path = tmp_path / "REPORT.md"
        write_report(results, path)
        content = path.read_text(encoding="utf-8")
        assert "# Seeded MST Workshop Results" in content
        assert "Kruskal" in content
        assert "Prim" in content
        assert "## Interpretation" in content
        assert "identical total weight" in content
