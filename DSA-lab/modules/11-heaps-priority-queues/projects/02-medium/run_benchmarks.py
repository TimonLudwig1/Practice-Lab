"""Reproducible timing comparisons for the three heap patterns."""

from __future__ import annotations

import csv
import random
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from heap_patterns import RunningMedian, merge_sorted, top_k_frequent


R = TypeVar("R")
SEED = 20260720
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "benchmark.csv"


@dataclass(frozen=True)
class BenchmarkRow:
    pattern: str
    size: int
    variant: str
    seconds: float
    results_equal: bool


def timed(function: Callable[[], R], repeats: int = 3) -> tuple[float, R]:
    """Return the best duration and its result."""

    best = float("inf")
    best_result: R | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        duration = time.perf_counter() - start
        if duration < best:
            best = duration
            best_result = result
    assert best_result is not None
    return best, best_result


def top_k_reference(values: list[int], k: int) -> list[tuple[int, int]]:
    """Reference using a full sort of all unique values."""

    counts = Counter(values)
    first: dict[int, int] = {}
    for index, value in enumerate(values):
        first.setdefault(value, index)
    ordered = sorted(counts, key=lambda value: (-counts[value], first[value]))
    return [(value, counts[value]) for value in ordered[:k]]


def all_prefix_medians(values: list[float]) -> list[float]:
    """Compute all prefix medians with the two-heap structure."""

    tracker = RunningMedian()
    result: list[float] = []
    for value in values:
        tracker.add(value)
        result.append(tracker.median())
    return result


def naive_prefix_medians(values: list[float]) -> list[float]:
    """Reference that sorts every observed prefix from scratch."""

    result: list[float] = []
    prefix: list[float] = []
    for value in values:
        prefix.append(value)
        ordered = sorted(prefix)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            result.append(ordered[middle])
        else:
            result.append((ordered[middle - 1] + ordered[middle]) / 2.0)
    return result


def benchmark_top_k(rng: random.Random) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for size in (5_000, 25_000, 100_000):
        values = [rng.randrange(max(100, size // 20)) for _ in range(size)]
        heap_time, heap_result = timed(lambda: top_k_frequent(values, 25))
        sort_time, sort_result = timed(lambda: top_k_reference(values, 25))
        equal = heap_result == sort_result
        rows.extend(
            [
                BenchmarkRow("top_k", size, "bounded_heap", heap_time, equal),
                BenchmarkRow("top_k", size, "full_sort", sort_time, equal),
            ]
        )
    return rows


def benchmark_merge(rng: random.Random) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    total = 40_000
    for sequence_count in (4, 16, 64):
        width = total // sequence_count
        sequences = [
            sorted(rng.randrange(total * 4) for _ in range(width))
            for _ in range(sequence_count)
        ]
        heap_time, heap_result = timed(lambda: merge_sorted(sequences))
        sort_time, sort_result = timed(
            lambda: sorted(value for sequence in sequences for value in sequence)
        )
        equal = heap_result == sort_result
        rows.extend(
            [
                BenchmarkRow("k_way_merge", sequence_count, "head_heap", heap_time, equal),
                BenchmarkRow("k_way_merge", sequence_count, "concatenate_sort", sort_time, equal),
            ]
        )
    return rows


def benchmark_median(rng: random.Random) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for size in (500, 1_500, 3_000):
        values = [rng.uniform(-1000.0, 1000.0) for _ in range(size)]
        heap_time, heap_result = timed(lambda: all_prefix_medians(values))
        sort_time, sort_result = timed(lambda: naive_prefix_medians(values))
        equal = heap_result == sort_result
        rows.extend(
            [
                BenchmarkRow("running_median", size, "two_heaps", heap_time, equal),
                BenchmarkRow("running_median", size, "sort_each_prefix", sort_time, equal),
            ]
        )
    return rows


def run_benchmarks() -> list[BenchmarkRow]:
    """Execute all cases and write a stable CSV report."""

    rng = random.Random(SEED)
    rows = benchmark_top_k(rng) + benchmark_merge(rng) + benchmark_median(rng)
    if not all(row.results_equal for row in rows):
        raise AssertionError("a heap result differs from its reference")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("pattern", "size", "variant", "seconds", "results_equal"))
        for row in rows:
            writer.writerow(
                (
                    row.pattern,
                    row.size,
                    row.variant,
                    f"{row.seconds:.8f}",
                    str(row.results_equal).lower(),
                )
            )
    return rows


def main() -> None:
    rows = run_benchmarks()
    print(f"wrote {len(rows)} benchmark rows to {OUTPUT_PATH}")
    for index in range(0, len(rows), 2):
        heap_row, reference_row = rows[index : index + 2]
        ratio = reference_row.seconds / heap_row.seconds
        print(
            f"{heap_row.pattern:>14} size={heap_row.size:>6}: "
            f"{heap_row.variant}={heap_row.seconds:.6f}s, "
            f"{reference_row.variant}={reference_row.seconds:.6f}s, "
            f"reference/heap={ratio:.2f}x"
        )


if __name__ == "__main__":
    main()
