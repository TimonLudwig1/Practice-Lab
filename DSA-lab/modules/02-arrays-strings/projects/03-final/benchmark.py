"""Validate and benchmark the list toolkit against NumPy equivalents."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from generate_sensor_data import DEFAULT_SEED, SensorDataset, generate_sensor_data
from generate_sensor_data import write_sensor_csv
from sensor_toolkit import PrefixSumIndex, detect_zscore_outliers, moving_average


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing and numerical agreement for one Python/NumPy operation pair."""

    operation: str
    python_seconds: float
    numpy_seconds: float
    speedup: float
    max_abs_error: float


def _timed(function: Callable[[], Any], repetitions: int) -> float:
    """Return the median runtime of a no-argument callable."""
    samples: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _make_ranges(size: int, query_count: int, seed: int) -> list[tuple[int, int]]:
    """Build deterministic non-empty half-open query ranges."""
    random_source = random.Random(seed)
    ranges: list[tuple[int, int]] = []
    for _ in range(query_count):
        start = random_source.randrange(size)
        max_width = min(500, size - start)
        end = start + random_source.randint(1, max_width)
        ranges.append((start, end))
    return ranges


def _max_error(left: Sequence[float], right: np.ndarray[Any, Any]) -> float:
    """Return the largest absolute difference between equally shaped results."""
    left_array = np.asarray(left, dtype=np.float64)
    if left_array.shape != right.shape:
        raise AssertionError(f"shape mismatch: {left_array.shape} != {right.shape}")
    if left_array.size == 0:
        return 0.0
    return float(np.max(np.abs(left_array - right)))


def _result(
    operation: str,
    python_seconds: float,
    numpy_seconds: float,
    max_abs_error: float,
) -> BenchmarkResult:
    """Create a benchmark row and guard against an invalid zero duration."""
    if python_seconds <= 0.0 or numpy_seconds <= 0.0:
        raise RuntimeError("timer resolution was insufficient")
    return BenchmarkResult(
        operation=operation,
        python_seconds=python_seconds,
        numpy_seconds=numpy_seconds,
        speedup=python_seconds / numpy_seconds,
        max_abs_error=max_abs_error,
    )


def run_benchmark(
    dataset: SensorDataset,
    *,
    window: int = 64,
    query_count: int = 20_000,
    threshold: float = 4.0,
    repetitions: int = 5,
    query_seed: int = DEFAULT_SEED + 1,
) -> list[BenchmarkResult]:
    """Validate equivalent outputs and measure four algorithmic operations."""
    if not isinstance(query_count, int) or isinstance(query_count, bool):
        raise TypeError("query_count must be an integer")
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise TypeError("repetitions must be an integer")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    values = dataset.readings
    array = np.asarray(values, dtype=np.float64)
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    ranges = _make_ranges(len(values), query_count, query_seed)
    starts = np.fromiter((start for start, _ in ranges), dtype=np.int64)
    ends = np.fromiter((end for _, end in ranges), dtype=np.int64)

    python_moving = moving_average(values, window)
    numpy_moving = np.convolve(array, kernel, mode="valid")
    moving_error = _max_error(python_moving, numpy_moving)
    if not np.allclose(python_moving, numpy_moving, rtol=1e-10, atol=1e-10):
        raise AssertionError("moving-average implementations disagree")

    python_prefix = PrefixSumIndex.from_readings(values)
    numpy_prefix = np.concatenate(([0.0], np.cumsum(array)))
    prefix_error = _max_error(python_prefix.prefix_values, numpy_prefix)
    if not np.allclose(
        python_prefix.prefix_values, numpy_prefix, rtol=1e-12, atol=1e-9
    ):
        raise AssertionError("prefix-sum implementations disagree")

    python_ranges = python_prefix.batch_range_sums(ranges)
    numpy_ranges = numpy_prefix[ends] - numpy_prefix[starts]
    range_error = _max_error(python_ranges, numpy_ranges)
    if not np.allclose(python_ranges, numpy_ranges, rtol=1e-10, atol=1e-8):
        raise AssertionError("range-query implementations disagree")

    python_outliers = detect_zscore_outliers(values, threshold)
    numpy_zscores = (array - np.mean(array)) / np.std(array)
    numpy_outlier_indices = np.flatnonzero(np.abs(numpy_zscores) >= threshold)
    python_outlier_indices = np.asarray(
        [outlier.index for outlier in python_outliers], dtype=np.int64
    )
    if not np.array_equal(python_outlier_indices, numpy_outlier_indices):
        raise AssertionError("outlier implementations disagree")

    rows = [
        _result(
            "moving_average",
            _timed(lambda: moving_average(values, window), repetitions),
            _timed(lambda: np.convolve(array, kernel, mode="valid"), repetitions),
            moving_error,
        ),
        _result(
            "prefix_build",
            _timed(lambda: PrefixSumIndex.from_readings(values), repetitions),
            _timed(
                lambda: np.concatenate(([0.0], np.cumsum(array))), repetitions
            ),
            prefix_error,
        ),
        _result(
            "range_queries",
            _timed(lambda: python_prefix.batch_range_sums(ranges), repetitions),
            _timed(lambda: numpy_prefix[ends] - numpy_prefix[starts], repetitions),
            range_error,
        ),
        _result(
            "outlier_detection",
            _timed(lambda: detect_zscore_outliers(values, threshold), repetitions),
            _timed(
                lambda: np.flatnonzero(
                    np.abs((array - np.mean(array)) / np.std(array)) >= threshold
                ),
                repetitions,
            ),
            0.0,
        ),
    ]
    return rows


def write_benchmark_csv(results: Sequence[BenchmarkResult], output_path: Path) -> None:
    """Write benchmark rows to a machine-readable CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "operation",
                "python_seconds",
                "numpy_seconds",
                "numpy_speedup",
                "max_abs_error",
            )
        )
        for row in results:
            writer.writerow(
                (
                    row.operation,
                    f"{row.python_seconds:.9f}",
                    f"{row.numpy_seconds:.9f}",
                    f"{row.speedup:.3f}",
                    f"{row.max_abs_error:.12g}",
                )
            )


def create_benchmark_plot(
    dataset: SensorDataset,
    results: Sequence[BenchmarkResult],
    output_path: Path,
    *,
    window: int,
    threshold: float,
) -> None:
    """Plot the synthetic series and Python/NumPy runtime comparison."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = dataset.readings
    averages = moving_average(values, window)
    outliers = detect_zscore_outliers(values, threshold)

    figure, (series_axis, runtime_axis) = plt.subplots(2, 1, figsize=(12, 8))
    sample_step = max(1, len(values) // 5_000)
    sampled_indices = range(0, len(values), sample_step)
    series_axis.plot(
        sampled_indices,
        values[::sample_step],
        color="#6384a8",
        linewidth=0.7,
        alpha=0.65,
        label="Messwerte",
    )
    average_indices = range(window - 1, len(values), sample_step)
    series_axis.plot(
        average_indices,
        averages[::sample_step],
        color="#153f66",
        linewidth=1.4,
        label=f"Moving Average (Fenster {window})",
    )
    series_axis.scatter(
        [outlier.index for outlier in outliers],
        [outlier.value for outlier in outliers],
        color="#c43b3b",
        marker="x",
        s=45,
        label=f"|z| ≥ {threshold:g}",
        zorder=3,
    )
    series_axis.set_title("Synthetische Sensor-Zeitreihe")
    series_axis.set_xlabel("Messindex")
    series_axis.set_ylabel("Temperatur")
    series_axis.grid(alpha=0.25)
    series_axis.legend(loc="best")

    positions = np.arange(len(results))
    width = 0.36
    runtime_axis.bar(
        positions - width / 2,
        [row.python_seconds for row in results],
        width,
        label="Python-Listen",
        color="#d27c35",
    )
    runtime_axis.bar(
        positions + width / 2,
        [row.numpy_seconds for row in results],
        width,
        label="NumPy",
        color="#4e9b6f",
    )
    runtime_axis.set_yscale("log")
    runtime_axis.set_xticks(positions, [row.operation for row in results])
    runtime_axis.set_ylabel("Median-Laufzeit in Sekunden (log)")
    runtime_axis.set_title("Gleiche Ergebnisse, unterschiedliche Ausführungskosten")
    runtime_axis.grid(axis="y", alpha=0.25)
    runtime_axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Run the full reproducible benchmark and write all artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=20_000)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()

    dataset = generate_sensor_data(arguments.size, seed=arguments.seed)
    results = run_benchmark(
        dataset,
        window=arguments.window,
        query_count=arguments.queries,
        threshold=arguments.threshold,
        repetitions=arguments.repetitions,
        query_seed=arguments.seed + 1,
    )

    write_sensor_csv(dataset, Path("data/sensor_readings.csv"))
    write_benchmark_csv(results, Path("results/benchmark_results.csv"))
    create_benchmark_plot(
        dataset,
        results,
        Path("results/sensor_and_runtime_comparison.png"),
        window=arguments.window,
        threshold=arguments.threshold,
    )

    print(f"Readings: {len(dataset.readings):,}")
    print(f"Injected anomalies: {dataset.anomaly_indices}")
    print("\nOperation                 Python (s)   NumPy (s)   NumPy-Faktor   Max. Fehler")
    for row in results:
        print(
            f"{row.operation:24} {row.python_seconds:10.6f} "
            f"{row.numpy_seconds:10.6f} {row.speedup:13.2f} "
            f"{row.max_abs_error:13.3e}"
        )


if __name__ == "__main__":
    main()
