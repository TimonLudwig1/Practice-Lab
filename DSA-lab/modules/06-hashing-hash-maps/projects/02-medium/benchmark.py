"""Benchmark hash patterns against equivalent naive implementations."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from time import perf_counter_ns
from typing import Any, Callable

from patterns import (
    duplicate_flags_hash,
    duplicate_flags_naive,
    first_unique_index_hash,
    first_unique_index_naive,
    group_anagrams_hash,
    group_anagrams_naive,
    two_sum_hash,
    two_sum_naive,
)


# The fixed seed makes every strategy receive the same anagram workloads and
# makes the checked result equality reproducible across runs.
DEFAULT_SEED = 60602


@dataclass(frozen=True)
class BenchmarkRow:
    pattern: str
    strategy: str
    item_count: int
    elapsed_ms: float


def _timed(function: Callable[..., Any], *arguments: Any) -> tuple[Any, float]:
    start = perf_counter_ns()
    result = function(*arguments)
    return result, (perf_counter_ns() - start) / 1_000_000


def _anagram_words(size: int, rng: random.Random) -> list[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    words: list[str] = []
    while len(words) < size:
        base = "".join(rng.sample(alphabet, 8))
        words.append(base)
        if len(words) < size:
            words.append(base[::-1])
    return words


def _measure_pair(
    pattern: str,
    size: int,
    hash_function: Callable[..., Any],
    naive_function: Callable[..., Any],
    *arguments: Any,
) -> list[BenchmarkRow]:
    hash_result, hash_ms = _timed(hash_function, *arguments)
    naive_result, naive_ms = _timed(naive_function, *arguments)
    assert hash_result == naive_result
    return [
        BenchmarkRow(pattern, "hash", size, hash_ms),
        BenchmarkRow(pattern, "naive", size, naive_ms),
    ]


def run_benchmark(
    sizes: tuple[int, ...] = (100, 300, 600, 1_000),
    *,
    seed: int = DEFAULT_SEED,
) -> list[BenchmarkRow]:
    """Benchmark all four patterns over growing deterministic inputs."""
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("sizes must contain positive integers")
    rng = random.Random(seed)
    rows: list[BenchmarkRow] = []

    for size in sizes:
        unique_numbers = list(range(size))
        rows.extend(
            _measure_pair(
                "two_sum",
                size,
                two_sum_hash,
                two_sum_naive,
                unique_numbers,
                -1,
            )
        )

        words = _anagram_words(size, rng)
        rows.extend(
            _measure_pair(
                "group_anagrams",
                size,
                group_anagrams_hash,
                group_anagrams_naive,
                words,
            )
        )

        repeated_text = "a" * size
        rows.extend(
            _measure_pair(
                "first_unique",
                size,
                first_unique_index_hash,
                first_unique_index_naive,
                repeated_text,
            )
        )

        rows.extend(
            _measure_pair(
                "stream_duplicates",
                size,
                duplicate_flags_hash,
                duplicate_flags_naive,
                unique_numbers,
            )
        )
    return rows


def write_csv(path: Path, rows: list[BenchmarkRow]) -> Path:
    if not rows:
        raise ValueError("rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(asdict(rows[0])),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def main() -> None:
    rows = run_benchmark()
    output_path = write_csv(Path(__file__).parent / "output" / "benchmark.csv", rows)
    print("pattern             strategy items elapsed_ms")
    print("-" * 48)
    for row in rows:
        print(
            f"{row.pattern:<20}{row.strategy:<9}"
            f"{row.item_count:>6}{row.elapsed_ms:>11.4f}"
        )
    print(f"\nCSV written to {output_path}")


if __name__ == "__main__":
    main()
