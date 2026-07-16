"""Investigate ten Python functions with visible and hidden runtime costs."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable, Mapping, Sequence

CaseFunction = Callable[[int], int]


def case_01(size: int) -> int:
    """Read one fixed position from a fixed-size tuple."""
    values = (10, 20, 30, 40, 50)
    return values[3]


def case_02(size: int) -> int:
    """Count how often an integer can be halved."""
    steps = 0
    remaining = size
    while remaining > 1:
        remaining //= 2
        steps += 1
    return steps


def case_03(size: int) -> int:
    """Process the same input in two consecutive passes."""
    result = 0
    for value in range(size):
        result += value
    for _ in range(size):
        result += 1
    return result


def case_04(size: int) -> int:
    """Scan a sequence of geometrically shrinking ranges."""
    operations = 0
    remaining = size
    while remaining > 0:
        for _ in range(remaining):
            operations += 1
        remaining //= 2
    return operations


def case_05(size: int) -> int:
    """Repeat membership checks against a hash-based container."""
    allowed = set(range(size))
    matches = 0
    for value in range(size):
        if value in allowed:
            matches += 1
    return matches


def case_06(size: int) -> int:
    """Repeat unsuccessful membership checks against a list."""
    values = list(range(size))
    misses = 0
    for _ in range(size):
        if -1 not in values:
            misses += 1
    return misses


def case_07(size: int) -> int:
    """Consume a list by repeatedly copying a suffix."""
    values = list(range(size))
    result = 0
    while values:
        result += values[0]
        values = values[1:]
    return result


def case_08(size: int) -> int:
    """Build a list by repeatedly inserting at the front."""
    values: list[int] = []
    for value in range(size):
        values.insert(0, value)
    return values[0] + len(values)


def case_09(size: int) -> int:
    """Build a string while retaining every previous prefix."""
    result = ""
    history = []
    for _ in range(size):
        history.append(result)
        result = result + "x"
    return len(result) + len(history)


def case_10(size: int) -> int:
    """Generate deterministic irregular values and sort them."""
    state = 42
    values = []
    for _ in range(size):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
        values.append(state)

    ordered = sorted(values)
    return ordered[0] ^ ordered[-1]


CASES: Mapping[str, CaseFunction] = {
    "01": case_01,
    "02": case_02,
    "03": case_03,
    "04": case_04,
    "05": case_05,
    "06": case_06,
    "07": case_07,
    "08": case_08,
    "09": case_09,
    "10": case_10,
}

DEFAULT_SIZES = (64, 128, 256, 512, 1_024, 2_048, 4_096)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration shared by all measured cases."""

    sizes: tuple[int, ...] = DEFAULT_SIZES
    repeats: int = 5
    min_sample_seconds: float = 0.005
    max_iterations: int = 1_048_576

    def validate(self) -> None:
        """Raise ValueError for invalid benchmark settings."""
        if len(self.sizes) < 2:
            raise ValueError("at least two input sizes are required")
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
    """One median runtime measured for one case and size."""

    case_id: str
    size: int
    seconds_per_call: float
    iterations_per_sample: int


_BENCHMARK_SINK = 0


def _run_batch(function: CaseFunction, size: int, iterations: int) -> float:
    """Time a batch and retain its computed checksum."""
    global _BENCHMARK_SINK

    checksum = 0
    start = perf_counter()
    for _ in range(iterations):
        checksum ^= function(size)
    elapsed = perf_counter() - start
    _BENCHMARK_SINK ^= checksum
    return elapsed


def calibrate_iterations(
    function: CaseFunction,
    size: int,
    config: BenchmarkConfig,
) -> int:
    """Find a batch size that is long enough for stable timing."""
    iterations = 1
    while True:
        elapsed = _run_batch(function, size, iterations)
        if (
            elapsed >= config.min_sample_seconds
            or iterations >= config.max_iterations
        ):
            return iterations
        iterations = min(iterations * 2, config.max_iterations)


def measure_case(
    function: CaseFunction,
    size: int,
    config: BenchmarkConfig,
) -> tuple[float, int]:
    """Return median seconds per call and the calibrated batch size."""
    iterations = calibrate_iterations(function, size, config)
    samples = [
        _run_batch(function, size, iterations) / iterations
        for _ in range(config.repeats)
    ]
    return median(samples), iterations


def run_benchmarks(
    cases: Mapping[str, CaseFunction],
    config: BenchmarkConfig,
) -> list[Measurement]:
    """Measure every selected case for every input size."""
    config.validate()
    measurements = []

    for case_id, function in cases.items():
        for size in config.sizes:
            seconds_per_call, iterations = measure_case(function, size, config)
            measurements.append(
                Measurement(
                    case_id=case_id,
                    size=size,
                    seconds_per_call=seconds_per_call,
                    iterations_per_sample=iterations,
                )
            )

    return measurements


def group_by_case(
    measurements: Sequence[Measurement],
) -> dict[str, list[Measurement]]:
    """Group measurements without changing their order."""
    grouped: dict[str, list[Measurement]] = {}
    for point in measurements:
        grouped.setdefault(point.case_id, []).append(point)
    return grouped


def estimate_log_log_slope(points: Sequence[Measurement]) -> float:
    """Estimate a power-law exponent through linear regression in log space."""
    if len(points) < 2:
        raise ValueError("at least two measurements are required")

    x_values = [math.log(point.size) for point in points]
    y_values = [math.log(point.seconds_per_call) for point in points]
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


def write_csv(measurements: Sequence[Measurement], destination: Path) -> None:
    """Write all raw measurements to CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["case", "input_size", "seconds_per_call", "iterations_per_sample"]
        )
        for point in measurements:
            writer.writerow(
                [
                    point.case_id,
                    point.size,
                    f"{point.seconds_per_call:.12g}",
                    point.iterations_per_sample,
                ]
            )


def create_normalized_plot(
    measurements: Sequence[Measurement],
    destination: Path,
) -> None:
    """Plot runtime growth relative to each case's first measurement."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 7))

    for case_id, points in group_by_case(measurements).items():
        baseline = points[0].seconds_per_call
        axis.plot(
            [point.size for point in points],
            [point.seconds_per_call / baseline for point in points],
            marker="o",
            linewidth=1.8,
            label=f"Case {case_id}",
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("Input size n")
    axis.set_ylabel("Runtime relative to the first input size")
    axis.set_title("Complexity detective: normalized runtime growth")
    axis.grid(True, which="both", linestyle=":", alpha=0.65)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def print_summary(measurements: Sequence[Measurement]) -> None:
    """Print empirical slopes and total growth factors."""
    print("\nEmpirical detective report")
    print("=" * 66)

    for case_id, points in group_by_case(measurements).items():
        slope = estimate_log_log_slope(points)
        growth = points[-1].seconds_per_call / points[0].seconds_per_call
        print(
            f"Case {case_id}: slope={slope:6.3f}, "
            f"runtime growth={growth:9.2f}x"
        )


def select_cases(case_ids: Sequence[str]) -> dict[str, CaseFunction]:
    """Return selected cases or raise ValueError for unknown identifiers."""
    unknown = [case_id for case_id in case_ids if case_id not in CASES]
    if unknown:
        raise ValueError(f"unknown case identifiers: {', '.join(unknown)}")
    return {case_id: CASES[case_id] for case_id in case_ids}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Measure ten Python functions with hidden complexity costs."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(CASES),
        metavar="ID",
        help="case identifiers to measure (default: 01 through 10)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="strictly increasing input sizes (default: powers of two 64..4096)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="timing samples per case and size (default: 5)",
    )
    parser.add_argument(
        "--min-sample-ms",
        type=float,
        default=5.0,
        help="target duration for one calibrated sample in ms (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for CSV and PNG output",
    )
    return parser.parse_args()


def main() -> None:
    """Run selected detective cases and write their artifacts."""
    arguments = parse_arguments()
    try:
        selected_cases = select_cases(arguments.cases)
        config = BenchmarkConfig(
            sizes=tuple(arguments.sizes),
            repeats=arguments.repeats,
            min_sample_seconds=arguments.min_sample_ms / 1_000,
        )
        config.validate()
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from error

    print(
        f"Measuring {len(selected_cases)} cases across "
        f"{len(config.sizes)} sizes. Please wait..."
    )
    measurements = run_benchmarks(selected_cases, config)

    csv_path = arguments.output_dir / "measurements.csv"
    plot_path = arguments.output_dir / "normalized_growth.png"
    write_csv(measurements, csv_path)
    create_normalized_plot(measurements, plot_path)
    print_summary(measurements)

    print("\nArtifacts")
    print(f"  CSV:  {csv_path}")
    print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
