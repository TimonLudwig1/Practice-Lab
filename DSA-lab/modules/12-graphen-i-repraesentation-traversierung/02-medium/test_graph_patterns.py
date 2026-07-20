"""Tests für Zykluserkennung, Toposort, Färbung und Inselzählung."""

import pytest

from graph_patterns import (
    CycleError,
    IslandReport,
    analyze_islands,
    bipartite_coloring,
    count_islands,
    has_directed_cycle,
    has_undirected_cycle,
    is_bipartite,
    is_valid_topological_order,
    topological_sort_dfs,
    topological_sort_kahn,
)


class TestUndirectedCycle:
    def test_empty_graph_has_no_cycle(self) -> None:
        assert not has_undirected_cycle({})

    def test_tree_has_no_cycle(self) -> None:
        tree = {"A": ("B", "C"), "B": ("A",), "C": ("A",)}
        assert not has_undirected_cycle(tree)

    def test_triangle_has_cycle(self) -> None:
        triangle = {
            "A": ("B", "C"),
            "B": ("A", "C"),
            "C": ("A", "B"),
        }
        assert has_undirected_cycle(triangle)

    def test_self_loop_is_cycle(self) -> None:
        assert has_undirected_cycle({"A": ("A",)})

    def test_cycle_in_later_component(self) -> None:
        graph = {
            "A": ("B",),
            "B": ("A",),
            "X": ("Y", "Z"),
            "Y": ("X", "Z"),
            "Z": ("X", "Y"),
        }
        assert has_undirected_cycle(graph)

    def test_isolated_vertices_have_no_cycle(self) -> None:
        assert not has_undirected_cycle({"A": (), "B": ()})

    def test_asymmetric_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="nicht symmetrisch"):
            has_undirected_cycle({"A": ("B",), "B": ()})

    def test_trace_contains_parent_and_cycle_edge(self) -> None:
        trace: list[str] = []
        graph = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
        assert has_undirected_cycle(graph, trace)
        assert any("Parent" in line for line in trace)
        assert any("Zyklus" in line for line in trace)


class TestDirectedCycle:
    def test_empty_graph_has_no_cycle(self) -> None:
        assert not has_directed_cycle({})

    def test_dag_has_no_cycle(self) -> None:
        assert not has_directed_cycle({"A": ("B", "C"), "B": ("D",), "C": ("D",)})

    def test_three_cycle_detected(self) -> None:
        assert has_directed_cycle({"A": ("B",), "B": ("C",), "C": ("A",)})

    def test_self_loop_detected(self) -> None:
        assert has_directed_cycle({"A": ("A",)})

    def test_opposite_edges_are_cycle(self) -> None:
        assert has_directed_cycle({"A": ("B",), "B": ("A",)})

    def test_cross_edge_to_black_vertex_is_not_cycle(self) -> None:
        graph = {"A": ("B", "C"), "B": ("D",), "C": ("D",), "D": ()}
        assert not has_directed_cycle(graph)

    def test_neighbor_only_vertex_is_included(self) -> None:
        assert not has_directed_cycle({"A": ("B",)})

    def test_trace_uses_three_colors(self) -> None:
        trace: list[str] = []
        assert not has_directed_cycle({"A": ("B",)}, trace)
        assert any("weiß -> grau" in line for line in trace)
        assert any("grau -> schwarz" in line for line in trace)


DAG = {
    "fetch": ("clean",),
    "clean": ("train", "report"),
    "train": ("evaluate",),
    "evaluate": ("deploy",),
    "report": (),
    "deploy": (),
}


class TestTopologicalSort:
    @pytest.mark.parametrize("sorter", [topological_sort_kahn, topological_sort_dfs])
    def test_result_is_valid(self, sorter) -> None:  # type: ignore[no-untyped-def]
        order = sorter(DAG)
        assert is_valid_topological_order(DAG, order)

    def test_kahn_order_is_deterministic(self) -> None:
        assert topological_sort_kahn(DAG) == (
            "fetch",
            "clean",
            "train",
            "report",
            "evaluate",
            "deploy",
        )

    def test_kahn_includes_isolated_vertex(self) -> None:
        graph = {"isolated": (), "A": ("B",), "B": ()}
        assert topological_sort_kahn(graph) == ("isolated", "A", "B")

    def test_neighbor_only_vertex_is_included(self) -> None:
        graph = {"A": ("B",)}
        assert topological_sort_kahn(graph) == ("A", "B")

    def test_empty_graph_returns_empty_order(self) -> None:
        assert topological_sort_kahn({}) == ()
        assert topological_sort_dfs({}) == ()

    @pytest.mark.parametrize("sorter", [topological_sort_kahn, topological_sort_dfs])
    def test_cycle_raises(self, sorter) -> None:  # type: ignore[no-untyped-def]
        cyclic = {"A": ("B",), "B": ("C",), "C": ("A",)}
        with pytest.raises(CycleError, match="Topologische Sortierung unmöglich"):
            sorter(cyclic)

    def test_kahn_cycle_error_carries_trace(self) -> None:
        trace: list[str] = []
        with pytest.raises(CycleError) as error:
            topological_sort_kahn({"A": ("A",)}, trace)
        assert error.value.trace == tuple(trace)
        assert "Restknoten" in trace[-1]

    def test_dfs_cycle_error_carries_trace(self) -> None:
        trace: list[str] = []
        with pytest.raises(CycleError) as error:
            topological_sort_dfs({"A": ("A",)}, trace)
        assert error.value.trace == tuple(trace)
        assert "Rückkante" in trace[-1]

    def test_validator_accepts_different_valid_order(self) -> None:
        assert is_valid_topological_order(
            {"A": ("C",), "B": ("C",), "C": ()},
            ("B", "A", "C"),
        )

    @pytest.mark.parametrize(
        "order",
        [
            ("B", "A"),  # Kante verletzt
            ("A",),  # unvollständig
            ("A", "A", "B"),  # Duplikat
            ("A", "B", "X"),  # fremder Knoten
        ],
    )
    def test_validator_rejects_invalid_orders(self, order: tuple[str, ...]) -> None:
        assert not is_valid_topological_order({"A": ("B",), "B": ()}, order)

    def test_kahn_trace_shows_in_degree_and_queue(self) -> None:
        trace: list[str] = []
        topological_sort_kahn({"A": ("B",)}, trace)
        assert trace[0].startswith("Startqueue")
        assert any("In-Degree" in line for line in trace)
        assert any("Queue" in line for line in trace[1:])

    def test_dfs_trace_ends_with_reversal(self) -> None:
        trace: list[str] = []
        topological_sort_dfs({"A": ("B",)}, trace)
        assert trace[-1].startswith("Drehe Abschlussliste")


class TestBipartite:
    def test_empty_graph_is_bipartite(self) -> None:
        assert bipartite_coloring({}) == {}
        assert is_bipartite({})

    def test_path_is_bipartite(self) -> None:
        path = {"A": ("B",), "B": ("A", "C"), "C": ("B",)}
        assert bipartite_coloring(path) == {"A": 0, "B": 1, "C": 0}

    def test_even_cycle_is_bipartite(self) -> None:
        square = {
            "A": ("B", "D"),
            "B": ("A", "C"),
            "C": ("B", "D"),
            "D": ("A", "C"),
        }
        assert is_bipartite(square)

    def test_odd_cycle_is_not_bipartite(self) -> None:
        triangle = {
            "A": ("B", "C"),
            "B": ("A", "C"),
            "C": ("A", "B"),
        }
        assert bipartite_coloring(triangle) is None
        assert not is_bipartite(triangle)

    def test_self_loop_is_not_bipartite(self) -> None:
        assert not is_bipartite({"A": ("A",)})

    def test_disconnected_components_all_colored(self) -> None:
        graph = {"A": ("B",), "B": ("A",), "X": (), "Y": ()}
        assert bipartite_coloring(graph) == {"A": 0, "B": 1, "X": 0, "Y": 0}

    def test_asymmetric_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="nicht symmetrisch"):
            bipartite_coloring({"A": ("B",), "B": ()})

    def test_trace_reports_color_conflict(self) -> None:
        triangle = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
        trace: list[str] = []
        assert bipartite_coloring(triangle, trace) is None
        assert any("Färbe" in line for line in trace)
        assert "Konflikt" in trace[-1]


class TestIslands:
    def test_empty_grid(self) -> None:
        assert analyze_islands(()) == IslandReport(0, ())

    def test_grid_with_zero_columns(self) -> None:
        assert analyze_islands(((), ())) == IslandReport(0, ())

    def test_only_water(self) -> None:
        assert count_islands(((0, 0), (0, 0))) == 0

    def test_single_land_cell(self) -> None:
        assert analyze_islands(((1,),)) == IslandReport(1, (1,))

    def test_all_land_is_one_island(self) -> None:
        assert analyze_islands(((1, 1), (1, 1))) == IslandReport(1, (4,))

    def test_diagonal_cells_are_separate(self) -> None:
        assert analyze_islands(((1, 0), (0, 1))) == IslandReport(2, (1, 1))

    def test_multiple_islands_and_sizes(self) -> None:
        grid = (
            (1, 1, 0, 0, 0),
            (1, 0, 0, 1, 1),
            (0, 0, 1, 0, 0),
            (1, 1, 0, 0, 1),
        )
        assert analyze_islands(grid) == IslandReport(5, (3, 2, 1, 2, 1))
        assert count_islands(grid) == 5

    def test_boolean_grid_supported(self) -> None:
        assert analyze_islands(((True, False), (True, True))) == IslandReport(1, (3,))

    def test_input_is_not_modified(self) -> None:
        grid = [[1, 0], [1, 1]]
        original = [row[:] for row in grid]
        assert count_islands(grid) == 1
        assert grid == original

    def test_ragged_grid_rejected(self) -> None:
        with pytest.raises(ValueError, match="rechteckig"):
            analyze_islands(((1, 0), (1,)))

    @pytest.mark.parametrize("bad_value", [-1, 2, "1", None])
    def test_non_binary_value_rejected(self, bad_value: object) -> None:
        with pytest.raises(ValueError, match="nur 0/Wasser und 1/Land"):
            analyze_islands(((1, bad_value),))  # type: ignore[arg-type]

    def test_trace_contains_discovery_and_completion(self) -> None:
        trace: list[str] = []
        analyze_islands(((1, 1),), trace)
        assert trace[0].startswith("Insel 1 startet")
        assert any("Entdecke" in line for line in trace)
        assert trace[-1].startswith("Insel 1 abgeschlossen")
