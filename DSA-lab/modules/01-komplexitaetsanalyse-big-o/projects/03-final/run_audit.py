"""Benchmark the baseline and optimized customer aggregation pipelines."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable, Sequence

from audit_pipeline import (
    CustomerSummary,
    Event,
    inefficient_pipeline,
    load_events,
    optimized_pipeline,
    write_summaries,
)
from data.generate_data import DEFAULT_SEED, generate_events

Pipeline = Callable[[Sequence[Event]], list[CustomerSummary]]

DEFAULT_ROWS = 8_000
DEFAULT_SIZES = (500, 1_000, 2_000, 4_000, 8_000)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "events.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results"


@dataclass(frozen=True)
class AuditMeasurement:
    """Before/after timing for one input size."""

    size: int
    baseline_seconds: float
    optimized_seconds: float
    customer_count: int

    @property
    def speedup(self) -> float:
        """Return baseline runtime divided by optimized runtime."""
        return self.baseline_seconds / self.optimized_seconds


_AUDIT_SINK = 0


def median_runtime(
    pipeline: Pipeline,
    events: Sequence[Event],
    repeats: int,
) -> float:
    """Measure median runtime and retain the output length."""
    global _AUDIT_SINK

    durations = []
    for _ in range(repeats):
        start = perf_counter()
        summaries = pipeline(events)
        durations.append(perf_counter() - start)
        _AUDIT_SINK ^= len(summaries)
    return median(durations)


def validate_benchmark(
    events: Sequence[Event],
    sizes: Sequence[int],
    repeats: int,
) -> None:
    """Raise ValueError for an invalid audit configuration."""
    if not sizes:
        raise ValueError("at least one benchmark size is required")
    if any(size < 1 for size in sizes):
        raise ValueError("all benchmark sizes must be positive")
    if any(left >= right for left, right in zip(sizes, sizes[1:])):
        raise ValueError("benchmark sizes must be strictly increasing")
    if sizes[-1] > len(events):
        raise ValueError(
            f"largest benchmark size {sizes[-1]} exceeds "
            f"available rows {len(events)}"
        )
    if repeats < 1:
        raise ValueError("repeats must be positive")


def benchmark_pipelines(
    events: Sequence[Event],
    sizes: Sequence[int],
    repeats: int,
) -> list[AuditMeasurement]:
    """Verify equality and benchmark both implementations."""
    validate_benchmark(events, sizes, repeats)
    measurements = []

    for size in sizes:
        subset = events[:size]
        baseline_result = inefficient_pipeline(subset)
        optimized_result = optimized_pipeline(subset)
        if baseline_result != optimized_result:
            raise AssertionError(
                f"pipeline outputs differ for input size {size}"
            )

        baseline_seconds = median_runtime(
            inefficient_pipeline,
            subset,
            repeats,
        )
        optimized_seconds = median_runtime(
            optimized_pipeline,
            subset,
            repeats,
        )
        measurements.append(
            AuditMeasurement(
                size=size,
                baseline_seconds=baseline_seconds,
                optimized_seconds=optimized_seconds,
                customer_count=len(optimized_result),
            )
        )

    return measurements


def write_measurements(
    measurements: Sequence[AuditMeasurement],
    destination: Path,
) -> None:
    """Write before/after timings and speedups to CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "input_rows",
                "customer_count",
                "baseline_seconds",
                "optimized_seconds",
                "speedup",
            ]
        )
        for point in measurements:
            writer.writerow(
                [
                    point.size,
                    point.customer_count,
                    f"{point.baseline_seconds:.12g}",
                    f"{point.optimized_seconds:.12g}",
                    f"{point.speedup:.6f}",
                ]
            )


def write_report(
    measurements: Sequence[AuditMeasurement],
    destination: Path,
    seed: int,
    repeats: int,
) -> None:
    """Write a reproducible Markdown audit report."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    largest = measurements[-1]
    lines = [
        "# Performance-Audit — Messergebnis",
        "",
        "## Konfiguration",
        "",
        f"- Seed: {seed}",
        f"- Wiederholungen pro Variante und Größe: {repeats}",
        "- Messgrenze: reine Aggregation; CSV-Erzeugung und Laden ausgeschlossen",
        "- Korrektheitsbedingung: vollständige Ergebnisgleichheit vor jeder Messung",
        "",
        "## Vorher/Nachher",
        "",
        "| Zeilen | Kunden | Baseline (s) | Optimiert (s) | Speedup |",
        "|---:|---:|---:|---:|---:|",
    ]
    for point in measurements:
        lines.append(
            f"| {point.size} | {point.customer_count} | "
            f"{point.baseline_seconds:.6f} | "
            f"{point.optimized_seconds:.6f} | {point.speedup:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## Automatische Zusammenfassung",
            "",
            (
                f"Bei {largest.size} Ereignissen verarbeitet die optimierte "
                f"Pipeline dieselbe Ausgabe in {largest.optimized_seconds:.6f} "
                f"Sekunden statt {largest.baseline_seconds:.6f} Sekunden. "
                f"Das entspricht einem Speedup von {largest.speedup:.2f}x."
            ),
            "",
            "Die Baseline verwendet Listen-Membership und wiederholte "
            "Vollscans. Ihre Laufzeit ist O(nu) und damit bei u in O(n) "
            "quadratisch. Die optimierte Variante aggregiert per Hash Map in "
            "erwarteter O(n)-Zeit und sortiert u Ausgaben in O(u log u).",
            "",
            "## Eigene Interpretation",
            "",
            "Ergänze hier Verdopplungsfaktoren, praktische Relevanz, Grenzen "
            "des synthetischen Benchmarks und mögliche nächste Schritte.",
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_plot(
    measurements: Sequence[AuditMeasurement],
    destination: Path,
) -> None:
    """Plot runtime growth and measured speedup."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    sizes = [point.size for point in measurements]

    figure, (runtime_axis, speedup_axis) = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
    )
    runtime_axis.plot(
        sizes,
        [point.baseline_seconds for point in measurements],
        marker="o",
        linewidth=2,
        label="Baseline",
    )
    runtime_axis.plot(
        sizes,
        [point.optimized_seconds for point in measurements],
        marker="o",
        linewidth=2,
        label="Optimized",
    )
    runtime_axis.set_xscale("log", base=2)
    runtime_axis.set_yscale("log")
    runtime_axis.set_xlabel("Input rows n")
    runtime_axis.set_ylabel("Median runtime (seconds)")
    runtime_axis.set_title("Before/after runtime growth")
    runtime_axis.grid(True, which="both", linestyle=":", alpha=0.65)
    runtime_axis.legend()

    speedup_axis.plot(
        sizes,
        [point.speedup for point in measurements],
        marker="o",
        color="tab:green",
        linewidth=2,
    )
    speedup_axis.set_xscale("log", base=2)
    speedup_axis.set_xlabel("Input rows n")
    speedup_axis.set_ylabel("Speedup factor")
    speedup_axis.set_title("Measured speedup")
    speedup_axis.grid(True, which="both", linestyle=":", alpha=0.65)

    figure.suptitle("Customer aggregation performance audit")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def print_summary(measurements: Sequence[AuditMeasurement]) -> None:
    """Print the audit table to the terminal."""
    print("\nPerformance audit")
    print("=" * 78)
    print(
        f"{'rows':>8} {'customers':>10} {'baseline':>14} "
        f"{'optimized':>14} {'speedup':>10}"
    )
    for point in measurements:
        print(
            f"{point.size:8d} {point.customer_count:10d} "
            f"{point.baseline_seconds:13.6f}s "
            f"{point.optimized_seconds:13.6f}s "
            f"{point.speedup:9.2f}x"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Audit a deliberately inefficient aggregation pipeline."
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="reuse data-path instead of regenerating it",
    )
    return parser.parse_args()


def main() -> None:
    """Generate data, run the audit, and persist all artifacts."""
    arguments = parse_arguments()
    if arguments.rows < 1:
        raise SystemExit("Configuration error: rows must be positive")

    if not arguments.reuse_data:
        generate_events(arguments.data_path, arguments.rows, arguments.seed)
        print(
            f"Generated {arguments.rows} events with seed {arguments.seed} "
            f"at {arguments.data_path}."
        )
    elif not arguments.data_path.exists():
        raise SystemExit(
            f"Configuration error: data file does not exist: "
            f"{arguments.data_path}"
        )

    events = load_events(arguments.data_path)
    try:
        measurements = benchmark_pipelines(
            events,
            arguments.sizes,
            arguments.repeats,
        )
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from error

    output_dir = arguments.output_dir
    summary_path = output_dir / "customer_summary.csv"
    measurements_path = output_dir / "performance_audit.csv"
    plot_path = output_dir / "performance_audit.png"
    report_path = output_dir / "AUDIT_REPORT.md"

    summaries = optimized_pipeline(events)
    write_summaries(summaries, summary_path)
    write_measurements(measurements, measurements_path)
    create_plot(measurements, plot_path)
    write_report(
        measurements,
        report_path,
        arguments.seed,
        arguments.repeats,
    )
    print_summary(measurements)

    print("\nArtifacts")
    for artifact in (
        summary_path,
        measurements_path,
        plot_path,
        report_path,
    ):
        print(f"  {artifact}")


if __name__ == "__main__":
    main()
