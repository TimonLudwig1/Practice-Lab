"""Benchmark five functions with growth from constant to quadratic.

The curve labels are intentionally neutral. Learners should derive the
complexity classes from the implementations before inspecting the measurements.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable, Mapping, Sequence

WorkFunction = Callable[[int], int]


def curve_a(size: int) -> int:
    """Perform a fixed amount of work independent of size."""
    operations = 0
    for _ in range(32):
        operations += 1
    return operations


def curve_b(size: int) -> int:
    """Repeatedly halve the remaining problem size."""
    operations = 0
    remaining = size
    while remaining > 1:
        for _ in range(32):
            operations += 1
        remaining //= 2
    return operations


def curve_c(size: int) -> int:
    """Visit one item for every unit of input."""
    operations = 0
    for _ in range(size):
        for _ in range(8):
            operations += 1
    return operations


def curve_d(size: int) -> int:
    """Perform a halving process for every input item."""
    operations = 0
    for _ in range(size):
        remaining = size
        while remaining > 1:
            operations += 1
            remaining //= 2
    return operations


def curve_e(size: int) -> int:
    """Visit every unordered pair of distinct positions."""
    operations = 0
    for left in range(size):
        for _ in range(left):
            operations += 1
    return operations


FUNCTIONS: Mapping[str, WorkFunction] = {
    "A": curve_a,
    "B": curve_b,
    "C": curve_c,
    "D": curve_d,
    "E": curve_e,
}

DEFAULT_SIZES = (128, 256, 512, 1_024, 2_048, 4_096)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a complete benchmark series."""

    sizes: tuple[int, ...] = DEFAULT_SIZES
    repeats: int = 5
    min_sample_seconds: float = 0.01
    max_iterations: int = 1_048_576

    def validate(self) -> None:
        """Raise ValueError if the configuration cannot produce valid data."""
        if not self.sizes:
            raise ValueError("at least one input size is required")
        if any(size < 2 for size in self.sizes):
            raise ValueError("all input sizes must be at least 2")
        if any(left >= right for left, right in zip(self.sizes, self.sizes[1:])):
            raise ValueError("input sizes must be strictly increasing")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")
        if self.min_sample_seconds <= 0:
            raise ValueError("min_sample_seconds must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True)
class Measurement:
    """One median runtime measurement."""

    label: str
    size: int
    seconds_per_call: float
    iterations_per_sample: int


_BENCHMARK_SINK = 0


def _run_batch(function: WorkFunction, size: int, iterations: int) -> float:
    """Run one timed batch and preserve the computed result."""
    global _BENCHMARK_SINK

    checksum = 0
    start = perf_counter()
    for _ in range(iterations):
        checksum ^= function(size)
    elapsed = perf_counter() - start
    _BENCHMARK_SINK ^= checksum
    return elapsed


def calibrate_iterations(
    function: WorkFunction,
    size: int,
    min_sample_seconds: float,
    max_iterations: int,
) -> int:
    """Choose enough calls to make one timing sample measurable."""
    iterations = 1

    while True:
        elapsed = _run_batch(function, size, iterations)
        if elapsed >= min_sample_seconds or iterations >= max_iterations:
            return iterations
        iterations = min(iterations * 2, max_iterations)


def measure_runtime(
    function: WorkFunction,
    size: int,
    config: BenchmarkConfig,
) -> tuple[float, int]:
    """Return median seconds per call and calibrated batch size."""
    iterations = calibrate_iterations(
        function,
        size,
        config.min_sample_seconds,
        config.max_iterations,
    )

    samples = [
        _run_batch(function, size, iterations) / iterations
        for _ in range(config.repeats)
    ]
    return median(samples), iterations


def run_benchmarks(
    functions: Mapping[str, WorkFunction],
    config: BenchmarkConfig,
) -> list[Measurement]:
    """Measure every function at every configured input size."""
    config.validate()
    measurements = []

    for label, function in functions.items():
        for size in config.sizes:
            seconds_per_call, iterations = measure_runtime(function, size, config)
            measurements.append(
                Measurement(
                    label=label,
                    size=size,
                    seconds_per_call=seconds_per_call,
                    iterations_per_sample=iterations,
                )
            )

    return measurements


def estimate_log_log_slope(measurements: Sequence[Measurement]) -> float:
    """Estimate the slope of log(runtime) against log(input size)."""
    if len(measurements) < 2:
        raise ValueError("at least two measurements are required")

    x_values = [math.log(point.size) for point in measurements]
    y_values = [math.log(point.seconds_per_call) for point in measurements]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)

    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    )
    denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
    if denominator == 0:
        raise ValueError("input sizes must not all be equal")
    return numerator / denominator


def group_by_label(
    measurements: Sequence[Measurement],
) -> dict[str, list[Measurement]]:
    """Group measurements while preserving their input order."""
    grouped: dict[str, list[Measurement]] = {}
    for point in measurements:
        grouped.setdefault(point.label, []).append(point)
    return grouped


def write_csv(measurements: Sequence[Measurement], destination: Path) -> None:
    """Write raw benchmark measurements to a CSV file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["curve", "input_size", "seconds_per_call", "iterations_per_sample"]
        )
        for point in measurements:
            writer.writerow(
                [
                    point.label,
                    point.size,
                    f"{point.seconds_per_call:.12g}",
                    point.iterations_per_sample,
                ]
            )


def create_plot(measurements: Sequence[Measurement], destination: Path) -> None:
    """Create a log-log runtime plot using a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))

    for label, points in group_by_label(measurements).items():
        axis.plot(
            [point.size for point in points],
            [point.seconds_per_call for point in points],
            marker="o",
            linewidth=2,
            label=f"Curve {label}",
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Input size n")
    axis.set_ylabel("Median runtime per call (seconds)")
    axis.set_title("Runtime growth of five unknown complexity classes")
    axis.grid(True, which="both", linestyle=":", alpha=0.65)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def print_summary(measurements: Sequence[Measurement]) -> None:
    """Print runtimes, doubling ratios, and empirical slopes."""
    print("\nEmpirical results")
    print("=" * 76)

    for label, points in group_by_label(measurements).items():
        runtimes = [point.seconds_per_call for point in points]
        ratios = [
            current / previous
            for previous, current in zip(runtimes, runtimes[1:])
            if previous > 0
        ]
        ratio_text = ", ".join(f"{ratio:.2f}" for ratio in ratios)
        slope = estimate_log_log_slope(points)

        print(f"Curve {label}")
        print(
            "  runtimes: "
            + ", ".join(
                f"n={point.size}: {point.seconds_per_call:.3e}s"
                for point in points
            )
        )
        print(f"  doubling ratios: {ratio_text}")
        print(f"  estimated log-log slope: {slope:.3f}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Measure five unknown runtime growth curves."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="strictly increasing input sizes (default: powers of two 128..4096)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="timing samples per function and size (default: 5)",
    )
    parser.add_argument(
        "--min-sample-ms",
        type=float,
        default=10.0,
        help="target duration for one calibrated sample in ms (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for CSV and PNG output",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark lab and persist its results."""
    arguments = parse_arguments()
    config = BenchmarkConfig(
        sizes=tuple(arguments.sizes),
        repeats=arguments.repeats,
        min_sample_seconds=arguments.min_sample_ms / 1_000,
    )

    print(
        f"Measuring {len(FUNCTIONS)} curves across {len(config.sizes)} sizes. "
        "Please wait..."
    )
    measurements = run_benchmarks(FUNCTIONS, config)

    csv_path = arguments.output_dir / "measurements.csv"
    plot_path = arguments.output_dir / "runtime_growth.png"
    write_csv(measurements, csv_path)
    create_plot(measurements, plot_path)
    print_summary(measurements)

    print("\nArtifacts")
    print(f"  CSV:  {csv_path}")
    print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
