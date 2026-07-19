"""Run six reproducible reference-versus-pattern benchmarks."""

from __future__ import annotations

import csv
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from patterns import (
    filter_copy,
    filter_in_place,
    longest_unique_substring,
    longest_unique_substring_brute,
    max_container,
    max_container_brute,
    minimum_length,
    minimum_length_brute,
    pair_sum_brute,
    pair_sum_sorted,
    rolling_sums,
    rolling_sums_brute,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
CSV_PATH = OUTPUT_DIR / "benchmark.csv"
REPORT_PATH = OUTPUT_DIR / "REPORT.md"
SEED = 90901
REPETITIONS = 3


@dataclass(frozen=True)
class BenchmarkResult:
    """One deterministic case with measured median runtimes."""

    pattern: str
    exercise: str
    size: int
    brute_ms: float
    optimized_ms: float
    speedup: float
    equivalent: bool
    note: str


def _median_milliseconds(operation: Callable[[], object]) -> float:
    samples = []
    for _ in range(REPETITIONS):
        start = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(samples)


def _measure(
    pattern: str,
    exercise: str,
    size: int,
    brute: Callable[[], object],
    optimized: Callable[[], object],
    note: str,
) -> BenchmarkResult:
    brute_result = brute()
    optimized_result = optimized()
    equivalent = brute_result == optimized_result
    if not equivalent:
        raise RuntimeError(f"result mismatch in {exercise}")
    brute_ms = _median_milliseconds(brute)
    optimized_ms = _median_milliseconds(optimized)
    speedup = brute_ms / optimized_ms if optimized_ms else float("inf")
    return BenchmarkResult(
        pattern,
        exercise,
        size,
        brute_ms,
        optimized_ms,
        speedup,
        equivalent,
        note,
    )


def run_benchmarks() -> list[BenchmarkResult]:
    """Build seeded workloads and measure all six exercise pairs."""

    rng = random.Random(SEED)
    results: list[BenchmarkResult] = []

    pair_values = list(range(2_500))
    results.append(
        _measure(
            "two_pointers",
            "pair_sum",
            len(pair_values),
            lambda: pair_sum_brute(pair_values, 10_000),
            lambda: pair_sum_sorted(pair_values, 10_000),
            "O(n²) versus O(n); absent target forces the full search",
        )
    )

    heights = [rng.randrange(0, 10_000) for _ in range(2_200)]
    results.append(
        _measure(
            "two_pointers",
            "max_container",
            len(heights),
            lambda: max_container_brute(heights),
            lambda: max_container(heights),
            "O(n²) pair enumeration versus O(n) boundary elimination",
        )
    )

    filter_values = [rng.randrange(-100, 101) for _ in range(100_000)]
    keep = lambda value: value >= 0
    results.append(
        _measure(
            "two_pointers",
            "in_place_filter",
            len(filter_values),
            lambda: filter_copy(filter_values, keep),
            lambda: _filtered_in_place(filter_values, keep),
            "both O(n); write pointer avoids an additional result list",
        )
    )

    window_values = [rng.randrange(-20, 21) for _ in range(5_000)]
    width = 500
    results.append(
        _measure(
            "sliding_window",
            "rolling_sums",
            len(window_values),
            lambda: rolling_sums_brute(window_values, width),
            lambda: rolling_sums(window_values, width),
            "O(nk) recomputation versus O(n) entering/leaving updates",
        )
    )

    unique_text = "".join(chr(0x1000 + index) for index in range(1_500))
    results.append(
        _measure(
            "sliding_window",
            "longest_unique_substring",
            len(unique_text),
            lambda: longest_unique_substring_brute(unique_text),
            lambda: longest_unique_substring(unique_text),
            "O(n²) starts versus O(n) last-seen window",
        )
    )

    positive_values = [1] * 2_500
    results.append(
        _measure(
            "sliding_window",
            "minimum_length",
            len(positive_values),
            lambda: minimum_length_brute(positive_values, 2_501),
            lambda: minimum_length(positive_values, 2_501),
            "O(n²) absent-target scan versus O(n) monotone window",
        )
    )
    return results


def _filtered_in_place(values: list[int], keep: Callable[[int], bool]) -> list[int]:
    copy = list(values)
    filter_in_place(copy, keep)
    return copy


def _write_csv(results: list[BenchmarkResult]) -> None:
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "pattern",
                "exercise",
                "size",
                "brute_ms",
                "optimized_ms",
                "speedup",
                "equivalent",
                "note",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result.pattern,
                    result.exercise,
                    result.size,
                    f"{result.brute_ms:.6f}",
                    f"{result.optimized_ms:.6f}",
                    f"{result.speedup:.3f}",
                    result.equivalent,
                    result.note,
                )
            )


def _write_report(results: list[BenchmarkResult]) -> None:
    rows = "\n".join(
        f"| {result.exercise} | {result.size:,} | {result.brute_ms:.3f} | "
        f"{result.optimized_ms:.3f} | {result.speedup:.2f}x |"
        for result in results
    ).replace(",", ".")
    text = f"""# Benchmark-Bericht: Muster-Grundübungen

Alle Fälle verwenden Seed `{SEED}`, drei Wiederholungen und den Median. Vor der
Messung wird die Ergebnisgleichheit jeder Referenz-/Musterlösung geprüft.

| Aufgabe | n | Brute Force (ms) | Muster (ms) | Faktor |
|---|---:|---:|---:|---:|
{rows}

Die quadratischen Referenzen für Paarsumme, Wassercontainer, eindeutigen
Substring und unerreichbare Zielsumme zeigen den strukturellen Vorteil der
gerichteten Zeigerbewegung. Bei Rolling Sums wird jedes überlappende Fenster
nicht erneut summiert, sondern mit Eintritt und Austritt aktualisiert.

Die In-Place-Filterung ist bewusst kein asymptotischer Zeitgewinn: Beide
Varianten sind `O(n)`. Der Schreibzeiger reduziert den zusätzlichen Speicher von
`O(n)` auf `O(1)` abgesehen von der übergebenen Liste. Kleine Laufzeitunterschiede
sind hier Implementierungsdetails und nicht das Lernziel.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    """Run, persist, and summarize the standard benchmark."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_benchmarks()
    _write_csv(results)
    _write_report(results)
    print("exercise                  brute ms   pattern ms   speedup")
    print("-" * 62)
    for result in results:
        print(
            f"{result.exercise:<25} {result.brute_ms:>9.3f} "
            f"{result.optimized_ms:>12.3f} {result.speedup:>9.2f}x"
        )
    print(f"Artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
