"""Tests für gewichteten Graphen, Dijkstra und Union-Find."""

import math
from types import MappingProxyType

import pytest

from algorithms import (
    DijkstraEvent,
    UnionFind,
    WeightedGraph,
    dijkstra,
    hand_calculation_graph,
)


class TestWeightedGraph:
    def test_empty_graph(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph()
        assert not graph.directed
        assert len(graph) == 0
        assert graph.vertex_count == 0
        assert graph.edge_count == 0
        assert graph.vertices == ()

    def test_directed_flag(self) -> None:
        assert WeightedGraph[str](directed=True).directed

    def test_add_vertex_reports_if_new(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph()
        assert graph.add_vertex("A")
        assert not graph.add_vertex("A")
        assert "A" in graph

    def test_add_edge_adds_endpoints(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph()
        assert graph.add_edge("A", "B", 2)
        assert graph.vertices == ("A", "B")
        assert graph.edge_count == 1

    def test_undirected_edge_is_symmetric(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph()
        graph.add_edge("A", "B", 2.5)
        assert graph.weight("A", "B") == 2.5
        assert graph.weight("B", "A") == 2.5

    def test_directed_edge_is_not_symmetric(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph(directed=True)
        graph.add_edge("A", "B", 2)
        assert graph.has_edge("A", "B")
        assert not graph.has_edge("B", "A")

    def test_update_edge_does_not_change_count(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph()
        graph.add_edge("A", "B", 2)
        assert not graph.add_edge("A", "B", 7)
        assert graph.edge_count == 1
        assert graph.weight("A", "B") == graph.weight("B", "A") == 7

    def test_reverse_update_is_same_undirected_edge(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph()
        graph.add_edge("A", "B", 2)
        assert not graph.add_edge("B", "A", 3)
        assert graph.edge_count == 1
        assert graph.weight("A", "B") == 3

    @pytest.mark.parametrize("directed", [False, True])
    def test_self_loop_counts_once(self, directed: bool) -> None:
        graph: WeightedGraph[str] = WeightedGraph(directed=directed)
        graph.add_edge("A", "A", 0)
        assert graph.edge_count == 1
        assert graph.neighbors("A") == {"A": 0.0}

    @pytest.mark.parametrize("weight", [math.inf, -math.inf, math.nan])
    def test_non_finite_weight_rejected(self, weight: float) -> None:
        with pytest.raises(ValueError, match="endlich"):
            WeightedGraph[str]().add_edge("A", "B", weight)

    @pytest.mark.parametrize("weight", [True, "2", None])
    def test_non_numeric_weight_rejected(self, weight: object) -> None:
        with pytest.raises(TypeError, match="reelle Zahl"):
            WeightedGraph[str]().add_edge("A", "B", weight)  # type: ignore[arg-type]

    def test_negative_weight_can_be_represented(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph(directed=True)
        graph.add_edge("A", "B", -2)
        assert graph.weight("A", "B") == -2

    def test_neighbors_are_read_only(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph()
        graph.add_edge("A", "B", 2)
        neighbors = graph.neighbors("A")
        assert isinstance(neighbors, MappingProxyType)
        with pytest.raises(TypeError):
            neighbors["B"] = 3  # type: ignore[index]

    def test_adjacency_is_deep_read_only_snapshot(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph()
        graph.add_edge("A", "B", 2)
        snapshot = graph.adjacency
        graph.add_edge("A", "C", 3)
        assert tuple(snapshot) == ("A", "B")
        assert snapshot["A"] == {"B": 2.0}
        with pytest.raises(TypeError):
            snapshot["A"]["B"] = 9  # type: ignore[index]

    def test_missing_vertex_and_edge_raise(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph()
        graph.add_vertex("A")
        with pytest.raises(KeyError, match="Unbekannter Knoten"):
            graph.neighbors("X")
        with pytest.raises(KeyError, match="Kante existiert nicht"):
            graph.weight("A", "X")

    def test_path_cost(self) -> None:
        graph = hand_calculation_graph()
        assert graph.path_cost(("A", "C", "B", "D")) == 8
        assert graph.path_cost(("A",)) == 0

    def test_empty_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="mindestens einen"):
            hand_calculation_graph().path_cost(())

    def test_path_with_missing_edge_rejected(self) -> None:
        with pytest.raises(KeyError, match="Kante existiert nicht"):
            hand_calculation_graph().path_cost(("A", "F"))


class TestDijkstra:
    def test_hand_calculation_distances(self) -> None:
        result = dijkstra(hand_calculation_graph(), "A")
        assert dict(result.distances) == {
            "A": 0,
            "B": 3,
            "C": 2,
            "D": 8,
            "E": 10,
            "F": 13,
        }

    def test_hand_calculation_predecessors(self) -> None:
        result = dijkstra(hand_calculation_graph(), "A")
        assert dict(result.predecessors) == {
            "A": None,
            "B": "C",
            "C": "A",
            "D": "B",
            "E": "D",
            "F": "E",
        }

    def test_hand_calculation_settled_order(self) -> None:
        result = dijkstra(hand_calculation_graph(), "A")
        assert result.settled_order == ("A", "C", "B", "D", "E", "F")

    @pytest.mark.parametrize(
        ("target", "expected_path", "expected_cost"),
        [
            ("A", ("A",), 0),
            ("B", ("A", "C", "B"), 3),
            ("D", ("A", "C", "B", "D"), 8),
            ("F", ("A", "C", "B", "D", "E", "F"), 13),
        ],
    )
    def test_reconstructed_paths(
        self,
        target: str,
        expected_path: tuple[str, ...],
        expected_cost: float,
    ) -> None:
        graph = hand_calculation_graph()
        result = dijkstra(graph, "A")
        path = result.path_to(target)
        assert path == expected_path
        assert graph.path_cost(path) == expected_cost  # type: ignore[arg-type]

    def test_trace_contains_relaxation_and_stale_entry(self) -> None:
        result = dijkstra(hand_calculation_graph(), "A")
        assert all(isinstance(event, DijkstraEvent) for event in result.trace)
        assert any(event.action == "relax" for event in result.trace)
        assert any(event.action == "stale" for event in result.trace)

    def test_distances_and_predecessors_are_read_only(self) -> None:
        result = dijkstra(hand_calculation_graph(), "A")
        with pytest.raises(TypeError):
            result.distances["A"] = 5  # type: ignore[index]
        with pytest.raises(TypeError):
            result.predecessors["A"] = "B"  # type: ignore[index]

    def test_unreachable_vertex(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph(directed=True)
        graph.add_edge("A", "B", 2)
        graph.add_vertex("X")
        result = dijkstra(graph, "A")
        assert math.isinf(result.distances["X"])
        assert result.path_to("X") is None
        assert "X" not in result.settled_order

    def test_directed_edges_respected(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph(directed=True)
        graph.add_edge("A", "B", 1)
        graph.add_edge("C", "A", 1)
        result = dijkstra(graph, "A")
        assert result.path_to("B") == ("A", "B")
        assert result.path_to("C") is None

    def test_zero_weight_edges(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph()
        graph.add_edge("A", "B", 0)
        graph.add_edge("B", "C", 0)
        result = dijkstra(graph, "A")
        assert result.distances["C"] == 0
        assert result.path_to("C") == ("A", "B", "C")

    def test_negative_weight_rejected(self) -> None:
        graph: WeightedGraph[str] = WeightedGraph(directed=True)
        graph.add_edge("A", "B", -1)
        with pytest.raises(ValueError, match="keine negativen"):
            dijkstra(graph, "A")

    def test_unknown_start_rejected(self) -> None:
        with pytest.raises(KeyError, match="Unbekannter Startknoten"):
            dijkstra(WeightedGraph[str](), "A")

    def test_unknown_target_rejected(self) -> None:
        result = dijkstra(hand_calculation_graph(), "A")
        with pytest.raises(KeyError, match="Unbekannter Zielknoten"):
            result.path_to("X")

    def test_equal_distance_with_non_comparable_vertices(self) -> None:
        class Token:
            pass

        start, left, right = Token(), Token(), Token()
        graph: WeightedGraph[Token] = WeightedGraph(directed=True)
        graph.add_edge(start, left, 1)
        graph.add_edge(start, right, 1)
        result = dijkstra(graph, start)
        assert result.distances[left] == result.distances[right] == 1

    def test_graph_is_not_mutated(self) -> None:
        graph = hand_calculation_graph()
        before = graph.adjacency
        dijkstra(graph, "A")
        assert graph.adjacency == before


class TestUnionFind:
    def test_empty(self) -> None:
        structure: UnionFind[str] = UnionFind()
        assert len(structure) == 0
        assert structure.component_count == 0
        assert structure.elements == ()
        assert structure.components() == ()
        structure.validate()

    def test_initial_singletons(self) -> None:
        structure = UnionFind("ABC")
        assert len(structure) == 3
        assert structure.component_count == 3
        assert structure.components() == (
            frozenset({"A"}),
            frozenset({"B"}),
            frozenset({"C"}),
        )

    def test_duplicate_add_ignored(self) -> None:
        structure = UnionFind(["A", "A"])
        assert len(structure) == 1
        assert not structure.add("A")
        assert structure.component_count == 1

    def test_membership(self) -> None:
        structure = UnionFind([1, 2])
        assert 1 in structure
        assert 3 not in structure

    def test_union_reduces_component_count(self) -> None:
        structure = UnionFind("ABC")
        assert structure.union("A", "B")
        assert structure.component_count == 2
        assert structure.connected("A", "B")

    def test_redundant_union_reports_false(self) -> None:
        structure = UnionFind("AB")
        structure.union("A", "B")
        assert not structure.union("B", "A")
        assert structure.component_count == 1

    def test_union_by_rank(self) -> None:
        structure = UnionFind("ABCD")
        structure.union("A", "B")
        structure.union("C", "D")
        root_a = structure.find("A")
        root_c = structure.find("C")
        assert structure.ranks[root_a] == structure.ranks[root_c] == 1
        structure.union("A", "C")
        root = structure.find("A")
        assert structure.ranks[root] == 2

    def test_path_compression(self) -> None:
        structure = UnionFind("ABCDEFGH")
        structure.union("A", "B")
        structure.union("C", "D")
        structure.union("A", "C")
        structure.union("E", "F")
        structure.union("G", "H")
        structure.union("E", "G")
        structure.union("A", "E")
        root = structure.find("H")
        assert structure.parents["H"] == root
        assert structure.connected("H", "B")

    def test_component_sizes(self) -> None:
        structure = UnionFind("ABCDE")
        structure.union("A", "B")
        structure.union("B", "C")
        assert structure.component_size("A") == 3
        assert structure.component_size("C") == 3
        assert structure.component_size("D") == 1

    def test_components_are_stable_by_first_root(self) -> None:
        structure = UnionFind("ABCDEF")
        structure.union("A", "B")
        structure.union("C", "D")
        structure.union("E", "F")
        assert structure.components() == (
            frozenset({"A", "B"}),
            frozenset({"C", "D"}),
            frozenset({"E", "F"}),
        )

    @pytest.mark.parametrize("method", ["find", "component_size"])
    def test_unknown_element_in_unary_method(self, method: str) -> None:
        structure: UnionFind[str] = UnionFind()
        with pytest.raises(KeyError, match="Unbekanntes Element"):
            getattr(structure, method)("X")

    @pytest.mark.parametrize("method", ["union", "connected"])
    def test_unknown_element_in_binary_method(self, method: str) -> None:
        structure = UnionFind(["A"])
        with pytest.raises(KeyError, match="Unbekanntes Element"):
            getattr(structure, method)("A", "X")

    def test_snapshots_are_read_only(self) -> None:
        structure = UnionFind("AB")
        with pytest.raises(TypeError):
            structure.parents["A"] = "B"  # type: ignore[index]
        with pytest.raises(TypeError):
            structure.ranks["A"] = 9  # type: ignore[index]

    def test_full_theory_simulation(self) -> None:
        structure = UnionFind("ABCDEF")
        assert structure.union("A", "B")
        assert structure.union("C", "D")
        assert structure.union("A", "C")
        assert structure.union("E", "F")
        assert not structure.union("B", "D")
        assert set(structure.components()) == {
            frozenset("ABCD"),
            frozenset("EF"),
        }
        assert structure.component_count == 2
        structure.validate()

    def test_mixed_hashable_elements(self) -> None:
        structure: UnionFind[object] = UnionFind([1, "A", (2, 3)])
        structure.union(1, "A")
        assert structure.connected(1, "A")
        assert not structure.connected(1, (2, 3))

    def test_validate_after_many_operations(self) -> None:
        structure = UnionFind(range(20))
        for value in range(1, 20):
            structure.union(0, value)
        assert structure.component_count == 1
        assert structure.component_size(19) == 20
        structure.validate()
