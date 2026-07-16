"""Visualize capacity growth and amortized append costs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dynamic_array import DynamicArray

DEFAULT_APPENDS = 64
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class AppendMeasurement:
    """Operation-model data for one append."""

    append_number: int
    length: int
    capacity: int
    copied_elements: int
    actual_cost: int
    cumulative_average_cost: float


def run_experiment(
    append_count: int,
    initial_capacity: int = 1,
) -> list[AppendMeasurement]:
    """Append integers and record deterministic operation costs."""
    if append_count < 1:
        raise ValueError("append_count must be positive")
    if initial_capacity < 1:
        raise ValueError("initial_capacity must be at least 1")

    array: DynamicArray[int] = DynamicArray(initial_capacity)
    measurements = []
    cumulative_cost = 0

    for value in range(append_count):
        event_count_before = len(array.growth_events)
        array.append(value)

        copied_elements = 0
        if len(array.growth_events) > event_count_before:
            copied_elements = array.growth_events[-1].copied_elements

        actual_cost = 1 + copied_elements
        cumulative_cost += actual_cost
        append_number = value + 1
        measurements.append(
            AppendMeasurement(
                append_number=append_number,
                length=len(array),
                capacity=array.capacity,
                copied_elements=copied_elements,
                actual_cost=actual_cost,
                cumulative_average_cost=(
                    cumulative_cost / append_number
                ),
            )
        )

    return measurements


def write_csv(
    measurements: Sequence[AppendMeasurement],
    destination: Path,
) -> None:
    """Write experiment measurements to CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "append_number",
                "length",
                "capacity",
                "copied_elements",
                "actual_cost",
                "cumulative_average_cost",
            ]
        )
        for point in measurements:
            writer.writerow(
                [
                    point.append_number,
                    point.length,
                    point.capacity,
                    point.copied_elements,
                    point.actual_cost,
                    f"{point.cumulative_average_cost:.8f}",
                ]
            )


def create_plot(
    measurements: Sequence[AppendMeasurement],
    destination: Path,
) -> None:
    """Create capacity and append-cost plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    append_numbers = [point.append_number for point in measurements]

    figure, (capacity_axis, cost_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
    )
    capacity_axis.step(
        append_numbers,
        [point.capacity for point in measurements],
        where="post",
        linewidth=2,
        color="tab:blue",
        label="Buffer capacity",
    )
    capacity_axis.plot(
        append_numbers,
        [point.length for point in measurements],
        linestyle="--",
        color="tab:gray",
        label="Logical length",
    )
    capacity_axis.set_ylabel("Positions")
    capacity_axis.set_title("Geometric buffer growth")
    capacity_axis.grid(True, linestyle=":", alpha=0.65)
    capacity_axis.legend()

    cost_axis.vlines(
        append_numbers,
        0,
        [point.actual_cost for point in measurements],
        color="tab:red",
        alpha=0.55,
        label="Actual cost per append",
    )
    cost_axis.plot(
        append_numbers,
        [point.cumulative_average_cost for point in measurements],
        color="tab:green",
        linewidth=2,
        label="Cumulative average cost",
    )
    cost_axis.axhline(
        3,
        color="tab:purple",
        linestyle=":",
        label="Upper amortized bound (< 3)",
    )
    cost_axis.set_xlabel("Append number")
    cost_axis.set_ylabel("Abstract operation units")
    cost_axis.set_title("Rare resize spikes, bounded amortized average")
    cost_axis.grid(True, linestyle=":", alpha=0.65)
    cost_axis.legend()

    figure.suptitle("DynamicArray: capacity and amortized append cost")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def print_summary(measurements: Sequence[AppendMeasurement]) -> None:
    """Print resize points and final amortized cost."""
    resize_points = [
        point for point in measurements if point.copied_elements > 0
    ]

    print("Resize events")
    print("=" * 58)
    for point in resize_points:
        print(
            f"append={point.append_number:4d}, "
            f"capacity={point.capacity:4d}, "
            f"copied={point.copied_elements:4d}, "
            f"actual_cost={point.actual_cost:4d}"
        )

    final = measurements[-1]
    total_copies = sum(point.copied_elements for point in measurements)
    print("\nSummary")
    print(f"  appends:                 {final.append_number}")
    print(f"  final capacity:          {final.capacity}")
    print(f"  total resize copies:     {total_copies}")
    print(
        f"  cumulative average cost: "
        f"{final.cumulative_average_cost:.4f}"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Visualize dynamic-array growth and amortized append cost."
    )
    parser.add_argument("--appends", type=int, default=DEFAULT_APPENDS)
    parser.add_argument("--initial-capacity", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the experiment and persist its artifacts."""
    arguments = parse_arguments()
    try:
        measurements = run_experiment(
            arguments.appends,
            arguments.initial_capacity,
        )
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from error

    csv_path = arguments.output_dir / "growth_log.csv"
    plot_path = arguments.output_dir / "capacity_and_costs.png"
    write_csv(measurements, csv_path)
    create_plot(measurements, plot_path)
    print_summary(measurements)

    print("\nArtifacts")
    print(f"  CSV:  {csv_path}")
    print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
