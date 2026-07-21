"""Run the complete seeded road-network and closure analysis."""

from pathlib import Path

from reporting import plot_scenarios, write_report, write_roads_csv, write_scenarios_csv
from road_network import analyze_scenarios, default_scenarios, generate_grid_network


SEED = 1313
ROWS = 12
COLUMNS = 16
START = (ROWS - 1, 0)
TARGET = (ROWS - 1, COLUMNS - 1)


def main() -> None:
    project_directory = Path(__file__).resolve().parent
    results_directory = project_directory / "results"
    network = generate_grid_network(ROWS, COLUMNS, seed=SEED)
    scenarios = default_scenarios(network, START, TARGET)
    results = analyze_scenarios(network, START, TARGET, scenarios)

    roads_csv = results_directory / "road_network.csv"
    scenarios_csv = results_directory / "route_scenarios.csv"
    report = results_directory / "REPORT.md"
    figure = results_directory / "route_closures.png"

    write_roads_csv(network, roads_csv)
    write_scenarios_csv(results, scenarios_csv)
    write_report(network, results, report, seed=SEED)
    plot_scenarios(network, results, figure)

    print(f"Seed: {SEED}")
    print(f"Network: {network.node_count} nodes, {network.road_count} roads")
    print("scenario            roads_closed  time     delay     settled")
    for result in results:
        print(
            f"{result.scenario.name:<19} "
            f"{len(result.scenario.closed_roads):>12}  "
            f"{result.route.travel_time:>7.3f}  "
            f"{result.delay_percent:>+7.2f}%  "
            f"{result.route.settled_nodes:>7}"
        )
    print("\nGenerated:")
    for path in (roads_csv, scenarios_csv, report, figure):
        print(f"  {path.relative_to(project_directory)}")


if __name__ == "__main__":
    main()
