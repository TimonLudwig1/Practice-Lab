"""Tests auf bewusst kleinen, handkonstruierten Beispielgraphen."""

from types import MappingProxyType

import pytest

from graph import Graph, example_graph


class TestRepresentation:
    def test_new_graph_is_empty_and_undirected(self) -> None:
        graph: Graph[str] = Graph()
        assert not graph.directed
        assert len(graph) == 0
        assert graph.vertex_count == 0
        assert graph.edge_count == 0
        assert graph.vertices == ()

    def test_directed_flag(self) -> None:
        assert Graph[str](directed=True).directed

    def test_add_vertex_reports_if_new(self) -> None:
        graph: Graph[str] = Graph()
        assert graph.add_vertex("A")
        assert not graph.add_vertex("A")
        assert graph.vertices == ("A",)

    def test_membership(self) -> None:
        graph: Graph[int] = Graph()
        graph.add_vertex(7)
        assert 7 in graph
        assert 8 not in graph

    def test_edge_adds_both_vertices(self) -> None:
        graph: Graph[str] = Graph()
        assert graph.add_edge("A", "B")
        assert graph.vertices == ("A", "B")
        assert graph.vertex_count == 2

    def test_undirected_edge_is_symmetric(self) -> None:
        graph: Graph[str] = Graph()
        graph.add_edge("A", "B")
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "A")
        assert graph.neighbors("A") == ("B",)
        assert graph.neighbors("B") == ("A",)

    def test_directed_edge_is_not_symmetric(self) -> None:
        graph: Graph[str] = Graph(directed=True)
        graph.add_edge("A", "B")
        assert graph.has_edge("A", "B")
        assert not graph.has_edge("B", "A")

    @pytest.mark.parametrize("directed", [False, True])
    def test_duplicate_edge_is_ignored(self, directed: bool) -> None:
        graph: Graph[str] = Graph(directed=directed)
        assert graph.add_edge("A", "B")
        assert not graph.add_edge("A", "B")
        assert graph.edge_count == 1

    def test_reverse_undirected_edge_is_same_edge(self) -> None:
        graph: Graph[str] = Graph()
        graph.add_edge("A", "B")
        assert not graph.add_edge("B", "A")
        assert graph.edge_count == 1

    @pytest.mark.parametrize("directed", [False, True])
    def test_self_loop_counts_once(self, directed: bool) -> None:
        graph: Graph[str] = Graph(directed=directed)
        graph.add_edge("A", "A")
        assert graph.neighbors("A") == ("A",)
        assert graph.edge_count == 1

    def test_directed_opposite_edges_count_separately(self) -> None:
        graph: Graph[str] = Graph(directed=True)
        graph.add_edge("A", "B")
        graph.add_edge("B", "A")
        assert graph.edge_count == 2

    def test_neighbor_order_is_insertion_order(self) -> None:
        graph: Graph[str] = Graph()
        graph.add_edge("A", "C")
        graph.add_edge("A", "B")
        graph.add_edge("A", "D")
        assert graph.neighbors("A") == ("C", "B", "D")

    def test_missing_neighbors_raises_key_error(self) -> None:
        graph: Graph[str] = Graph()
        with pytest.raises(KeyError, match="Unbekannter Knoten"):
            graph.neighbors("A")

    def test_has_edge_with_missing_vertex_is_false(self) -> None:
        graph: Graph[str] = Graph()
        assert not graph.has_edge("A", "B")

    def test_adjacency_is_read_only_snapshot(self) -> None:
        graph: Graph[str] = Graph()
        graph.add_edge("A", "B")
        snapshot = graph.adjacency
        assert isinstance(snapshot, MappingProxyType)
        with pytest.raises(TypeError):
            snapshot["A"] = ()  # type: ignore[index]
        graph.add_edge("A", "C")
        assert snapshot["A"] == ("B",)

    def test_hashable_non_string_vertices(self) -> None:
        graph: Graph[tuple[int, int]] = Graph()
        graph.add_edge((0, 0), (0, 1))
        assert graph.bfs((0, 0)) == ((0, 0), (0, 1))


class TestBFS:
    def test_bfs_visits_level_by_level(self) -> None:
        assert example_graph().bfs("A") == ("A", "B", "C", "D", "E", "F")

    def test_bfs_from_inner_vertex(self) -> None:
        assert example_graph().bfs("C") == ("C", "A", "E", "B", "F", "D")

    def test_bfs_stays_in_start_component(self) -> None:
        graph: Graph[str] = Graph()
        graph.add_edge("A", "B")
        graph.add_edge("X", "Y")
        assert graph.bfs("A") == ("A", "B")

    def test_bfs_on_isolated_vertex(self) -> None:
        graph: Graph[str] = Graph()
        graph.add_vertex("solo")
        assert graph.bfs("solo") == ("solo",)

    def test_bfs_respects_edge_direction(self) -> None:
        graph: Graph[str] = Graph(directed=True)
        graph.add_edge("A", "B")
        graph.add_edge("C", "A")
        assert graph.bfs("A") == ("A", "B")
        assert graph.bfs("C") == ("C", "A", "B")

    def test_bfs_handles_cycle_without_duplicates(self) -> None:
        graph: Graph[str] = Graph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")
        assert graph.bfs("A") == ("A", "B", "C")

    def test_bfs_handles_self_loop(self) -> None:
        graph: Graph[str] = Graph()
        graph.add_edge("A", "A")
        assert graph.bfs("A") == ("A",)

    def test_bfs_missing_start_raises(self) -> None:
        with pytest.raises(KeyError, match="Unbekannter Knoten"):
            Graph[str]().bfs("A")


class TestDFS:
    def test_recursive_dfs_follows_first_branch(self) -> None:
        assert example_graph().dfs_recursive("A") == (
            "A",
            "B",
            "D",
            "C",
            "E",
            "F",
        )

    def test_iterative_dfs_matches_recursive_order(self) -> None:
        graph = example_graph()
        assert graph.dfs_iterative("A") == graph.dfs_recursive("A")

    @pytest.mark.parametrize("method_name", ["dfs_recursive", "dfs_iterative"])
    def test_dfs_stays_in_start_component(self, method_name: str) -> None:
        graph: Graph[str] = Graph()
        graph.add_edge("A", "B")
        graph.add_edge("X", "Y")
        traversal = getattr(graph, method_name)
        assert traversal("A") == ("A", "B")

    @pytest.mark.parametrize("method_name", ["dfs_recursive", "dfs_iterative"])
    def test_dfs_on_isolated_vertex(self, method_name: str) -> None:
        graph: Graph[str] = Graph()
        graph.add_vertex("solo")
        assert getattr(graph, method_name)("solo") == ("solo",)

    @pytest.mark.parametrize("method_name", ["dfs_recursive", "dfs_iterative"])
    def test_dfs_handles_cycle_without_duplicates(self, method_name: str) -> None:
        graph: Graph[str] = Graph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")
        assert getattr(graph, method_name)("A") == ("A", "B", "C")

    @pytest.mark.parametrize("method_name", ["dfs_recursive", "dfs_iterative"])
    def test_dfs_respects_edge_direction(self, method_name: str) -> None:
        graph: Graph[str] = Graph(directed=True)
        graph.add_edge("A", "B")
        graph.add_edge("C", "A")
        assert getattr(graph, method_name)("A") == ("A", "B")

    @pytest.mark.parametrize("method_name", ["dfs_recursive", "dfs_iterative"])
    def test_dfs_missing_start_raises(self, method_name: str) -> None:
        graph: Graph[str] = Graph()
        with pytest.raises(KeyError, match="Unbekannter Knoten"):
            getattr(graph, method_name)("A")


class TestConnectedComponents:
    def test_three_components_including_isolated_vertex(self) -> None:
        graph: Graph[str] = Graph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("X", "Y")
        graph.add_vertex("solo")
        assert graph.connected_components() == (
            frozenset({"A", "B", "C"}),
            frozenset({"X", "Y"}),
            frozenset({"solo"}),
        )

    def test_empty_graph_has_no_components(self) -> None:
        assert Graph[str]().connected_components() == ()

    def test_single_component(self) -> None:
        assert example_graph().connected_components() == (
            frozenset({"A", "B", "C", "D", "E", "F"}),
        )

    def test_components_are_ordered_by_first_vertex(self) -> None:
        graph: Graph[str] = Graph()
        graph.add_vertex("Z")
        graph.add_edge("A", "B")
        assert graph.connected_components() == (
            frozenset({"Z"}),
            frozenset({"A", "B"}),
        )

    @pytest.mark.parametrize("method_name", ["connected_components", "is_connected"])
    def test_directed_graph_rejected(self, method_name: str) -> None:
        graph: Graph[str] = Graph(directed=True)
        graph.add_edge("A", "B")
        with pytest.raises(ValueError, match="ungerichtete"):
            getattr(graph, method_name)()

    def test_empty_and_singleton_are_connected(self) -> None:
        empty: Graph[str] = Graph()
        singleton: Graph[str] = Graph()
        singleton.add_vertex("A")
        assert empty.is_connected()
        assert singleton.is_connected()

    def test_disconnected_graph_is_not_connected(self) -> None:
        graph: Graph[str] = Graph()
        graph.add_vertex("A")
        graph.add_vertex("B")
        assert not graph.is_connected()


class TestInvariants:
    @pytest.mark.parametrize("directed", [False, True])
    def test_validate_accepts_mixed_graph(self, directed: bool) -> None:
        graph: Graph[str] = Graph(directed=directed)
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("C", "C")
        graph.add_vertex("isolated")
        graph.validate()

    def test_example_graph_counts(self) -> None:
        graph = example_graph()
        assert graph.vertex_count == 6
        assert graph.edge_count == 5
        graph.validate()
