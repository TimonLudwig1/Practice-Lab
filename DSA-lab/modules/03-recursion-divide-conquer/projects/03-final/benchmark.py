"""Compare recursive and explicit-stack traversal on the same generated tree."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from filesystem_analysis import TreeAnalysis, analyze_iterative, analyze_recursive
from generate_tree import DEFAULT_SEED, generate_tree


@dataclass(frozen=True)
class TraversalBenchmark:
    """Median timings for two equivalent directory-tree traversals."""

    recursive_seconds: float
    iterative_seconds: float
    recursive_over_iterative: float
    repetitions: int
    analysis: TreeAnalysis


def _median_runtime(function: Callable[[], object], repetitions: int) -> float:
    """Return the median runtime across repeated calls."""
    samples: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def run_benchmark(
    root: Path, *, pattern: str = "*.csv", repetitions: int = 25
) -> TraversalBenchmark:
    """Validate identical analyses, then benchmark both traversal strategies."""
    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise TypeError("repetitions must be an integer")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    recursive_result = analyze_recursive(root, pattern)
    iterative_result = analyze_iterative(root, pattern)
    if recursive_result != iterative_result:
        raise AssertionError("recursive and iterative analyses disagree")

    recursive_seconds = _median_runtime(
        lambda: analyze_recursive(root, pattern), repetitions
    )
    iterative_seconds = _median_runtime(
        lambda: analyze_iterative(root, pattern), repetitions
    )
    if recursive_seconds <= 0.0 or iterative_seconds <= 0.0:
        raise RuntimeError("timer resolution was insufficient")

    return TraversalBenchmark(
        recursive_seconds=recursive_seconds,
        iterative_seconds=iterative_seconds,
        recursive_over_iterative=recursive_seconds / iterative_seconds,
        repetitions=repetitions,
        analysis=recursive_result,
    )


def write_benchmark_csv(result: TraversalBenchmark, output_path: Path) -> None:
    """Write timings and analyzed tree dimensions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "recursive_seconds",
                "iterative_seconds",
                "recursive_over_iterative",
                "repetitions",
                "directories",
                "files",
                "bytes",
                "max_depth",
                "matches",
            )
        )
        writer.writerow(
            (
                f"{result.recursive_seconds:.9f}",
                f"{result.iterative_seconds:.9f}",
                f"{result.recursive_over_iterative:.4f}",
                result.repetitions,
                result.analysis.directory_count,
                result.analysis.file_count,
                result.analysis.total_bytes,
                result.analysis.max_depth,
                len(result.analysis.matches),
            )
        )


def main() -> None:
    """Generate the standard tree and run the complete comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/synthetic_tree"))
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()

    generation = generate_tree(arguments.root, seed=arguments.seed, overwrite=True)
    result = run_benchmark(
        arguments.root,
        pattern=arguments.pattern,
        repetitions=arguments.repetitions,
    )
    write_benchmark_csv(result, Path("results/traversal_benchmark.csv"))

    print(f"Generated directories: {generation.directory_count}")
    print(f"Generated files:       {generation.file_count}")
    print(f"Analyzed bytes:        {result.analysis.total_bytes}")
    print(f"Maximum depth:         {result.analysis.max_depth}")
    print(f"Pattern matches:       {len(result.analysis.matches)}")
    print(f"Recursive median:      {result.recursive_seconds:.6f} s")
    print(f"Iterative median:      {result.iterative_seconds:.6f} s")
    print(f"Recursive / iterative: {result.recursive_over_iterative:.3f}")


if __name__ == "__main__":
    main()
