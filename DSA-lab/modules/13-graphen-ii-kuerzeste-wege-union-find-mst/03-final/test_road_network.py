"""Tests for custom routing, closure analysis, outputs, and visualization."""

import csv
import math
from pathlib import Path

import pytest

from reporting import plot_scenarios, write_report, write_roads_csv, write_scenarios_csv
from road_network import (
    ClosureScenario,
    Road,
    RoadNetwork,
    analyze_scenarios,
    canonical_road,
    default_scenarios,
    generate_grid_network,
    route_roads,
    shortest_route,
)


def square_network() -> RoadNetwork:
    return RoadNetwork(
        2,
        2,
        (
            Road((0, 0), (0, 1), 1),
            Road((0, 0), (1, 0), 2),
            Road((0, 1), (1, 1), 1),
            Road((1, 0), (1, 1), 1),
        ),
    )


class TestRoad:
    def test_canonical_road_orders_endpoints(self) -> None:
        assert canonical_road((2, 1), (0, 3)) == ((0, 3), (2, 1))

    def test_canonical_road_rejects_loop(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            canonical_road((1, 1), (1, 1))

    def test_valid_road(self) -> None:
        road = Road((0, 0), (0, 1), 2.5)
        assert road.key == ((0, 0), (0, 1))
        assert road.travel_time == 2.5

    @pytest.mark.parametrize(
        ("first", "second"),
        [((0, 1), (0, 0)), ((1, 1), (1, 1))],
    )
    def test_non_canonical_road_rejected(self, first, second) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="canonical"):
            Road(first, second, 1)

    @pytest.mark.parametrize("travel_time", [0, -1, math.inf, math.nan])
    def test_invalid_travel_time_rejected(self, travel_time: float) -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            Road((0, 0), (0, 1), travel_time)

    @pytest.mark.parametrize("travel_time", [True, "1", None])
    def test_non_numeric_travel_time_rejected(self, travel_time: object) -> None:
        with pytest.raises(TypeError, match="numeric"):
            Road((0, 0), (0, 1), travel_time)  # type: ignore[arg-type]


class TestRoadNetwork:
    def test_counts_and_nodes(self) -> None:
        network = square_network()
        assert network.rows == 2
        assert network.columns == 2
        assert network.node_count == 4
        assert network.road_count == 4
        assert network.nodes == ((0, 0), (0, 1), (1, 0), (1, 1))

    def test_neighbors_are_symmetric(self) -> None:
        network = square_network()
        assert ((0, 1), 1.0) in network.neighbors((0, 0))
        assert ((0, 0), 1.0) in network.neighbors((0, 1))

    def test_contains_node_and_road(self) -> None:
        network = square_network()
        assert network.contains_node((1, 1))
        assert not network.contains_node((2, 2))
        assert network.contains_road(((0, 1), (0, 0)))

    def test_travel_time_and_path_cost(self) -> None:
        network = square_network()
        assert network.travel_time((0, 0), (0, 1)) == 1
        assert network.path_cost(((0, 0), (0, 1), (1, 1))) == 2
        assert network.path_cost(((0, 0),)) == 0

    def test_empty_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            square_network().path_cost(())

    def test_missing_node_and_road_rejected(self) -> None:
        network = square_network()
        with pytest.raises(KeyError, match="Unknown node"):
            network.neighbors((3, 3))
        with pytest.raises(KeyError, match="Road does not exist"):
            network.travel_time((0, 0), (1, 1))

    @pytest.mark.parametrize(("rows", "columns"), [(0, 1), (1, 0), (-1, 2)])
    def test_invalid_dimensions_rejected(self, rows: int, columns: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            RoadNetwork(rows, columns, ())

    def test_outside_endpoint_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            RoadNetwork(1, 2, (Road((0, 0), (0, 2), 1),))

    def test_duplicate_road_rejected(self) -> None:
        road = Road((0, 0), (0, 1), 1)
        with pytest.raises(ValueError, match="Duplicate"):
            RoadNetwork(1, 2, (road, road))


class TestGenerator:
    @pytest.mark.parametrize(("rows", "columns"), [(1, 1), (2, 3), (5, 7)])
    def test_expected_counts(self, rows: int, columns: int) -> None:
        network = generate_grid_network(rows, columns, seed=1)
        expected_roads = rows * (columns - 1) + (rows - 1) * columns
        assert network.node_count == rows * columns
        assert network.road_count == expected_roads

    def test_same_seed_is_identical(self) -> None:
        first = generate_grid_network(5, 6, seed=10)
        second = generate_grid_network(5, 6, seed=10)
        assert first.roads == second.roads

    def test_different_seed_changes_weights(self) -> None:
        assert generate_grid_network(5, 6, seed=10).roads != generate_grid_network(
            5, 6, seed=11
        ).roads

    def test_weight_range(self) -> None:
        network = generate_grid_network(
            10,
            10,
            seed=5,
            minimum_time=2,
            maximum_time=3,
        )
        assert all(2 <= road.travel_time <= 3 for road in network.roads)

    @pytest.mark.parametrize(("rows", "columns"), [(0, 2), (2, 0)])
    def test_invalid_dimensions_rejected(self, rows: int, columns: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            generate_grid_network(rows, columns, seed=1)

    @pytest.mark.parametrize(
        ("minimum", "maximum"),
        [(0, 1), (-1, 2), (3, 2)],
    )
    def test_invalid_weight_range_rejected(self, minimum: float, maximum: float) -> None:
        with pytest.raises(ValueError, match="positive and ordered"):
            generate_grid_network(
                2,
                2,
                seed=1,
                minimum_time=minimum,
                maximum_time=maximum,
            )


class TestRouting:
    def test_shortest_route_on_square(self) -> None:
        result = shortest_route(square_network(), (0, 0), (1, 1))
        assert result.reached
        assert result.path == ((0, 0), (0, 1), (1, 1))
        assert result.travel_time == 2
        assert result.hop_count == 2
        assert result.settled_nodes <= 4

    def test_single_closure_forces_alternative(self) -> None:
        result = shortest_route(
            square_network(),
            (0, 0),
            (1, 1),
            closed_roads=frozenset({((0, 0), (0, 1))}),
        )
        assert result.path == ((0, 0), (1, 0), (1, 1))
        assert result.travel_time == 3

    def test_reversed_closed_road_is_normalized(self) -> None:
        result = shortest_route(
            square_network(),
            (0, 0),
            (1, 1),
            closed_roads=frozenset({((0, 1), (0, 0))}),
        )
        assert result.travel_time == 3

    def test_closed_node_forces_alternative(self) -> None:
        result = shortest_route(
            square_network(),
            (0, 0),
            (1, 1),
            closed_nodes=frozenset({(0, 1)}),
        )
        assert result.path == ((0, 0), (1, 0), (1, 1))

    def test_closed_start_or_target_is_unreachable(self) -> None:
        for node in ((0, 0), (1, 1)):
            result = shortest_route(
                square_network(),
                (0, 0),
                (1, 1),
                closed_nodes=frozenset({node}),
            )
            assert not result.reached
            assert result.path is None
            assert math.isinf(result.travel_time)
            assert result.hop_count is None

    def test_all_exits_closed_is_unreachable(self) -> None:
        result = shortest_route(
            square_network(),
            (0, 0),
            (1, 1),
            closed_roads=frozenset(
                {
                    ((0, 0), (0, 1)),
                    ((0, 0), (1, 0)),
                }
            ),
        )
        assert not result.reached
        assert result.settled_nodes == 1

    def test_start_equals_target(self) -> None:
        result = shortest_route(square_network(), (0, 0), (0, 0))
        assert result.path == ((0, 0),)
        assert result.travel_time == 0
        assert result.hop_count == 0
        assert result.settled_nodes == 1

    @pytest.mark.parametrize("kind", ["start", "target"])
    def test_unknown_endpoint_rejected(self, kind: str) -> None:
        start = (9, 9) if kind == "start" else (0, 0)
        target = (9, 9) if kind == "target" else (1, 1)
        with pytest.raises(KeyError, match="Unknown"):
            shortest_route(square_network(), start, target)

    def test_unknown_closed_node_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown closed nodes"):
            shortest_route(
                square_network(),
                (0, 0),
                (1, 1),
                closed_nodes=frozenset({(9, 9)}),
            )

    def test_unknown_closed_road_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown closed road"):
            shortest_route(
                square_network(),
                (0, 0),
                (1, 1),
                closed_roads=frozenset({((0, 0), (1, 1))}),
            )

    def test_path_cost_matches_route_distance(self) -> None:
        network = generate_grid_network(8, 9, seed=20)
        result = shortest_route(network, (7, 0), (7, 8))
        assert result.path is not None
        assert math.isclose(network.path_cost(result.path), result.travel_time)

    def test_route_road_conversion(self) -> None:
        assert route_roads(((0, 0), (0, 1), (1, 1))) == (
            ((0, 0), (0, 1)),
            ((0, 1), (1, 1)),
        )
        assert route_roads(None) == ()


class TestScenarios:
    def test_default_scenarios(self) -> None:
        network = generate_grid_network(8, 10, seed=30)
        scenarios = default_scenarios(network, (7, 0), (7, 9))
        assert tuple(scenario.name for scenario in scenarios) == (
            "baseline",
            "single_route_road",
            "north_gap_barrier",
        )
        assert len(scenarios[1].closed_roads) == 1
        assert len(scenarios[2].closed_roads) == 7

    def test_scenario_delays(self) -> None:
        network = generate_grid_network(8, 10, seed=30)
        scenarios = default_scenarios(network, (7, 0), (7, 9))
        results = analyze_scenarios(network, (7, 0), (7, 9), scenarios)
        assert results[0].delay == 0
        assert results[0].delay_percent == 0
        assert all(result.route.reached for result in results)
        assert all(result.delay >= 0 for result in results)
        assert results[2].delay > 0

    def test_unreachable_scenario_uses_infinity(self) -> None:
        scenarios = (
            ClosureScenario("baseline"),
            ClosureScenario(
                "blocked",
                frozenset(
                    {
                        ((0, 0), (0, 1)),
                        ((0, 0), (1, 0)),
                    }
                ),
            ),
        )
        results = analyze_scenarios(square_network(), (0, 0), (1, 1), scenarios)
        assert math.isinf(results[1].delay)
        assert math.isinf(results[1].delay_percent)

    def test_baseline_must_be_first(self) -> None:
        with pytest.raises(ValueError, match="first scenario"):
            analyze_scenarios(
                square_network(),
                (0, 0),
                (1, 1),
                (ClosureScenario("other"),),
            )

    def test_baseline_must_be_reachable(self) -> None:
        blocked = ClosureScenario(
            "baseline",
            frozenset({((0, 0), (0, 1)), ((0, 0), (1, 0))}),
        )
        with pytest.raises(ValueError, match="Baseline route"):
            analyze_scenarios(square_network(), (0, 0), (1, 1), (blocked,))


class TestOutputs:
    def scenario_results(self):  # type: ignore[no-untyped-def]
        network = generate_grid_network(5, 6, seed=40)
        scenarios = default_scenarios(network, (4, 0), (4, 5))
        return network, analyze_scenarios(network, (4, 0), (4, 5), scenarios)

    def test_roads_csv(self, tmp_path: Path) -> None:
        network, _ = self.scenario_results()
        path = tmp_path / "roads.csv"
        write_roads_csv(network, path)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == network.road_count
        assert set(rows[0]) == {"first", "second", "travel_time"}

    def test_scenarios_csv(self, tmp_path: Path) -> None:
        _, results = self.scenario_results()
        path = tmp_path / "scenarios.csv"
        write_scenarios_csv(results, path)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert rows[0]["scenario"] == "baseline"
        assert rows[0]["reached"] == "true"
        assert rows[0]["delay"] == "0.000"

    def test_report_is_english_and_complete(self, tmp_path: Path) -> None:
        network, results = self.scenario_results()
        path = tmp_path / "REPORT.md"
        write_report(network, results, path, seed=40)
        content = path.read_text(encoding="utf-8")
        assert "# Synthetic Road Network Routing Report" in content
        assert "Seed: `40`" in content
        assert "## Closure impact" in content
        assert "## Interpretation" in content
        assert "custom Dijkstra implementation" in content

    def test_plot_is_nonempty_png(self, tmp_path: Path) -> None:
        network, results = self.scenario_results()
        path = tmp_path / "routes.png"
        plot_scenarios(network, results, path)
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 10_000
