"""CSV, Markdown, and matplotlib outputs for the routing analysis."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from road_network import RoadNetwork, ScenarioResult, canonical_road, route_roads


def node_label(node: tuple[int, int]) -> str:
    return f"r{node[0]}c{node[1]}"


def road_label(road: tuple[tuple[int, int], tuple[int, int]]) -> str:
    return f"{node_label(road[0])}-{node_label(road[1])}"


def write_roads_csv(network: RoadNetwork, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("first", "second", "travel_time"))
        for road in network.roads:
            writer.writerow(
                (node_label(road.first), node_label(road.second), f"{road.travel_time:.3f}")
            )


def write_scenarios_csv(results: tuple[ScenarioResult, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "scenario",
                "reached",
                "travel_time",
                "delay",
                "delay_percent",
                "hops",
                "settled_nodes",
                "closed_roads",
                "closed_nodes",
                "path",
            )
        )
        for result in results:
            route = result.route
            writer.writerow(
                (
                    result.scenario.name,
                    str(route.reached).lower(),
                    "inf" if math.isinf(route.travel_time) else f"{route.travel_time:.3f}",
                    "inf" if math.isinf(result.delay) else f"{result.delay:.3f}",
                    "inf" if math.isinf(result.delay_percent) else f"{result.delay_percent:.3f}",
                    "" if route.hop_count is None else route.hop_count,
                    route.settled_nodes,
                    "|".join(road_label(road) for road in sorted(result.scenario.closed_roads)),
                    "|".join(node_label(node) for node in sorted(result.scenario.closed_nodes)),
                    "|".join(node_label(node) for node in route.path or ()),
                )
            )


def write_report(
    network: RoadNetwork,
    results: tuple[ScenarioResult, ...],
    path: Path,
    *,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = results[0].route
    lines = [
        "# Synthetic Road Network Routing Report",
        "",
        f"- Seed: `{seed}`",
        f"- Grid: **{network.rows} × {network.columns}**",
        f"- Nodes: **{network.node_count}**",
        f"- Roads: **{network.road_count}**",
        f"- Start: `{node_label(baseline.start)}`",
        f"- Target: `{node_label(baseline.target)}`",
        "- Routing algorithm: **custom Dijkstra implementation**",
        "",
        "## Closure impact",
        "",
        "| Scenario | Closed roads | Travel time | Delay | Delay | Hops | Settled nodes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        route = result.route
        if route.reached:
            lines.append(
                f"| `{result.scenario.name}` | {len(result.scenario.closed_roads)} | "
                f"{route.travel_time:.3f} | {result.delay:.3f} | "
                f"{result.delay_percent:.1f} % | {route.hop_count} | "
                f"{route.settled_nodes} |"
            )
        else:
            lines.append(
                f"| `{result.scenario.name}` | {len(result.scenario.closed_roads)} | "
                f"unreachable | ∞ | ∞ | – | {route.settled_nodes} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The single-road closure tests local resilience: the route must leave "
            "its previously optimal path but can usually rejoin nearby. The barrier "
            "scenario closes every east-west crossing at one column except the "
            "northern gap. It forces a structural detour rather than a small local "
            "adjustment.",
            "",
            "The plot uses matplotlib only for presentation. Route computation, "
            "closure filtering, path reconstruction, and all reported metrics come "
            "from the custom graph implementation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_scenarios(
    network: RoadNetwork,
    results: tuple[ScenarioResult, ...],
    path: Path,
) -> None:
    """Render aligned small multiples with roads, routes, and closures."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1,
        len(results),
        figsize=(5.8 * len(results), 5.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(results) == 1:
        axes = [axes]

    for axis, result in zip(axes, results):
        closed_roads = {
            canonical_road(*road) for road in result.scenario.closed_roads
        }
        route_edge_set = set(route_roads(result.route.path))

        for road in network.roads:
            x_values = (road.first[1], road.second[1])
            y_values = (-road.first[0], -road.second[0])
            if road.key in closed_roads:
                axis.plot(
                    x_values,
                    y_values,
                    color="#d62728",
                    linewidth=2.2,
                    linestyle=(0, (2, 2)),
                    zorder=2,
                )
                midpoint_x = sum(x_values) / 2
                midpoint_y = sum(y_values) / 2
                axis.scatter(
                    [midpoint_x],
                    [midpoint_y],
                    marker="x",
                    s=24,
                    color="#d62728",
                    linewidths=1.2,
                    zorder=5,
                )
            elif road.key in route_edge_set:
                axis.plot(
                    x_values,
                    y_values,
                    color="#1f77b4",
                    linewidth=3.2,
                    zorder=4,
                )
            else:
                axis.plot(
                    x_values,
                    y_values,
                    color="#b8b8b8",
                    linewidth=0.7,
                    alpha=0.75,
                    zorder=1,
                )

        start = result.route.start
        target = result.route.target
        axis.scatter(
            [start[1]],
            [-start[0]],
            marker="o",
            s=60,
            facecolor="#2ca02c",
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )
        axis.scatter(
            [target[1]],
            [-target[0]],
            marker="s",
            s=60,
            facecolor="#9467bd",
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )
        if result.route.reached:
            subtitle = (
                f"time {result.route.travel_time:.1f}  |  "
                f"delay {result.delay_percent:+.1f}%"
            )
        else:
            subtitle = "unreachable"
        axis.set_title(result.scenario.name.replace("_", " ") + "\n" + subtitle)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    legend = (
        Line2D([0], [0], color="#b8b8b8", linewidth=1.2, label="open road"),
        Line2D([0], [0], color="#1f77b4", linewidth=3.2, label="shortest route"),
        Line2D([0], [0], color="#d62728", linestyle=(0, (2, 2)), label="closed road"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", label="start"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#9467bd", label="target"),
    )
    figure.legend(handles=legend, loc="outside lower center", ncol=5, frameon=False)
    figure.suptitle("Shortest routes under road closures", fontsize=14)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
