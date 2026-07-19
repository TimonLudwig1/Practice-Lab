"""Run, visualize and interpret the systematic sorting benchmark."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from statistics import median
import tempfile
from time import perf_counter_ns


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dsa_sort_benchmark_matplotlib")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmark_algorithms import ALGORITHMS, ComparisonCounter  # noqa: E402
from workloads import DEFAULT_SEED, INPUT_TYPES, generate_workloads  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"
DEFAULT_SIZES = (100, 200, 400, 800, 1_600)


@dataclass(frozen=True)
class BenchmarkRow:
    algorithm: str
    input_type: str
    size: int
    repetitions: int
    median_ms: float
    comparisons: int


def measure_algorithm(
    name: str,
    values: list[int],
    *,
    repetitions: int = 3,
) -> BenchmarkRow:
    """Validate repeated runs and return median time plus deterministic comparisons."""
    if name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {name}")
    if repetitions <= 0:
        raise ValueError("repetitions must be greater than zero")
    expected = sorted(values)
    elapsed_values: list[float] = []
    comparison_counts: list[int] = []
    algorithm = ALGORITHMS[name]

    for _ in range(repetitions):
        counter = ComparisonCounter()
        start = perf_counter_ns()
        result = algorithm(values, counter)
        elapsed_values.append((perf_counter_ns() - start) / 1_000_000)
        comparison_counts.append(counter.comparisons)
        if result != expected:
            raise AssertionError(f"{name} produced an incorrect result")

    if len(set(comparison_counts)) != 1:
        raise AssertionError(f"{name} comparison count changed between runs")
    return BenchmarkRow(
        algorithm=name,
        input_type="",
        size=len(values),
        repetitions=repetitions,
        median_ms=median(elapsed_values),
        comparisons=comparison_counts[0],
    )


def run_benchmark(
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    *,
    repetitions: int = 3,
    seed: int = DEFAULT_SEED,
) -> list[BenchmarkRow]:
    """Measure all algorithms for every size and input type."""
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("sizes must contain positive integers")
    rows: list[BenchmarkRow] = []
    for size in sizes:
        workloads = generate_workloads(size, seed=seed)
        for input_type in INPUT_TYPES:
            values = workloads[input_type]
            for name in ALGORITHMS:
                measured = measure_algorithm(name, values, repetitions=repetitions)
                rows.append(
                    BenchmarkRow(
                        algorithm=measured.algorithm,
                        input_type=input_type,
                        size=measured.size,
                        repetitions=measured.repetitions,
                        median_ms=measured.median_ms,
                        comparisons=measured.comparisons,
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


def create_plot(path: Path, rows: list[BenchmarkRow]) -> Path:
    """Create one log-log runtime panel for every workload shape."""
    if not rows:
        raise ValueError("rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {
        "bubble": "#4C78A8",
        "selection": "#F58518",
        "insertion": "#54A24B",
        "merge": "#E45756",
        "quick_3way": "#B279A2",
        "python_timsort": "#72B7B2",
    }

    for axis, input_type in zip(axes.flat, INPUT_TYPES):
        for algorithm in ALGORITHMS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row.input_type == input_type and row.algorithm == algorithm
                ),
                key=lambda row: row.size,
            )
            axis.plot(
                [row.size for row in selected],
                [row.median_ms for row in selected],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=algorithm,
                color=colors[algorithm],
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(input_type.replace("_", " ").title())
        axis.set_xlabel("Elemente n")
        axis.set_ylabel("Medianlaufzeit (ms)")
        axis.grid(alpha=0.25, which="both")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=6)
    figure.suptitle("Sortierverfahren nach Eingabeform", fontsize=15)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def write_report(path: Path, rows: list[BenchmarkRow]) -> Path:
    """Write a theory-linked interpretation using the largest measured size."""
    largest_size = max(row.size for row in rows)
    selected = [row for row in rows if row.size == largest_size]
    lookup = {
        (row.input_type, row.algorithm): row for row in selected
    }

    table_rows: list[str] = []
    for input_type in INPUT_TYPES:
        fastest = min(
            (lookup[(input_type, algorithm)] for algorithm in ALGORITHMS),
            key=lambda row: row.median_ms,
        )
        insertion = lookup[(input_type, "insertion")]
        selection = lookup[(input_type, "selection")]
        table_rows.append(
            f"| {input_type} | {fastest.algorithm} | {fastest.median_ms:.3f} | "
            f"{insertion.comparisons} | {selection.comparisons} |"
        )

    content = f"""# Ergebnisbericht: Sortier-Benchmark

Alle Algorithmen wurden vor jeder Zeitübernahme gegen `sorted()` geprüft. Die
Tabelle zeigt den größten Lauf mit n = {largest_size}; Zeiten sind Medianwerte
aus den im CSV dokumentierten Wiederholungen.

| Eingabe | schnellste gemessene Variante | Median ms | Insertion-Vergleiche | Selection-Vergleiche |
|---|---|---:|---:|---:|
{chr(10).join(table_rows)}

## Interpretation gegen die Theorie

Selection Sort führt für dieselbe Größe unabhängig von der Eingabeform stets
nahezu n(n-1)/2 Vergleiche aus. Seine Kurven reagieren deshalb kaum auf bereits
vorhandene Ordnung. Bubble Sort kann nur dann früh abbrechen, wenn ein kompletter
Pass ohne Tausch bleibt.

Insertion Sort nutzt vorhandene Ordnung direkt: Seine Arbeit hängt von der Zahl
der Inversionen ab. Auf umgekehrten Eingaben nähert es sich dem quadratischen
Worst Case, während es auf fast sortierten Folgen deutlich weniger Vergleiche
benötigt.

Merge Sort bleibt über alle Formen bei O(n log n). Der 3-Wege-Quicksort isoliert
gleiche Werte gemeinsam und vermeidet deshalb die typische Degeneration bei
vielen Duplikaten. Python-Timsort erkennt natürliche Runs und ist besonders für
fast sortierte reale Daten optimiert.

Absolute Millisekunden hängen von Hardware und Python-Version ab. Für die
algorithmische Bewertung sind Kurvenform und Vergleichszahlen belastbarer als
ein einzelner Geschwindigkeitsfaktor.
"""
    path.write_text(content, encoding="utf-8")
    return path


def run_pipeline(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[BenchmarkRow]:
    rows = run_benchmark()
    write_csv(output_dir / "benchmark.csv", rows)
    create_plot(output_dir / "sorting_benchmark.png", rows)
    write_report(output_dir / "REPORT.md", rows)
    return rows


def main() -> None:
    rows = run_pipeline()
    largest = max(row.size for row in rows)
    print(f"Created {len(rows)} measurements; largest n={largest}")
    print("input type       fastest at largest n     median ms")
    print("-" * 58)
    for input_type in INPUT_TYPES:
        candidates = [
            row for row in rows if row.size == largest and row.input_type == input_type
        ]
        fastest = min(candidates, key=lambda row: row.median_ms)
        print(f"{input_type:<17}{fastest.algorithm:<25}{fastest.median_ms:>10.3f}")
    print(f"\nArtifacts written to {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
