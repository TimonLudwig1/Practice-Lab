"""Run the reproducible classifier-threshold optimization case study."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

from generate_data import DEFAULT_SEED, generate_score_data
from threshold_optimizer import (
    OptimizationResult,
    ThresholdIndex,
    optimize_threshold_binary,
    optimize_threshold_exhaustive,
    optimize_threshold_grid,
    read_score_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_PATH = DATA_DIR / "scores.csv"
TRACE_PATH = OUTPUT_DIR / "binary_search_trace.csv"
COMPARISON_PATH = OUTPUT_DIR / "method_comparison.csv"
PLOT_PATH = OUTPUT_DIR / "threshold_metrics.png"
REPORT_PATH = OUTPUT_DIR / "RUN_REPORT.md"
RECORD_COUNT = 5_000
MAX_FALSE_POSITIVE_RATE = 0.05
GRID_INTERVALS = 1_000


def _write_trace(result: OptimizationResult) -> None:
    with TRACE_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "low_index",
                "high_index",
                "middle_index",
                "threshold",
                "recall",
                "false_positive_rate",
                "precision",
                "feasible",
                "next_low_index",
                "next_high_index",
            )
        )
        for step in result.trace:
            writer.writerow(
                (
                    step.low_index,
                    step.high_index,
                    step.middle_index,
                    f"{step.metrics.threshold:.8f}",
                    f"{step.metrics.recall:.8f}",
                    f"{step.metrics.false_positive_rate:.8f}",
                    f"{step.metrics.precision:.8f}",
                    step.feasible,
                    step.next_low_index,
                    step.next_high_index,
                )
            )


def _write_comparison(results: tuple[OptimizationResult, ...]) -> None:
    with COMPARISON_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "method",
                "threshold",
                "recall",
                "false_positive_rate",
                "precision",
                "true_positives",
                "false_positives",
                "evaluations",
                "candidate_count",
            )
        )
        for result in results:
            metrics = result.metrics
            writer.writerow(
                (
                    result.method,
                    f"{metrics.threshold:.8f}",
                    f"{metrics.recall:.8f}",
                    f"{metrics.false_positive_rate:.8f}",
                    f"{metrics.precision:.8f}",
                    metrics.true_positives,
                    metrics.false_positives,
                    result.evaluations,
                    result.candidate_count,
                )
            )


def _write_plot(index: ThresholdIndex, selected_threshold: float) -> None:
    cache_directory = Path(tempfile.gettempdir()) / "dsa-lab-plot-cache-module-08"
    mpl_directory = cache_directory / "matplotlib"
    xdg_directory = cache_directory / "xdg"
    mpl_directory.mkdir(parents=True, exist_ok=True)
    xdg_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_directory))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_directory))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    thresholds = [position / 250 for position in range(251)]
    metrics = [index.evaluate(threshold) for threshold in thresholds]
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.plot(thresholds, [item.recall for item in metrics], label="Recall")
    axis.plot(
        thresholds,
        [item.false_positive_rate for item in metrics],
        label="False Positive Rate",
    )
    axis.plot(thresholds, [item.precision for item in metrics], label="Precision")
    axis.axhline(
        MAX_FALSE_POSITIVE_RATE,
        color="black",
        linestyle="--",
        linewidth=1,
        label="FPR-Grenze",
    )
    axis.axvline(
        selected_threshold,
        color="tab:red",
        linestyle=":",
        linewidth=2,
        label="Binary-Search-Schwelle",
    )
    axis.set(
        title="Klassifikationsmetriken nach Schwellenwert",
        xlabel="Schwellenwert (Score ≥ Schwelle wird positiv)",
        ylabel="Metrikwert",
        xlim=(0, 1),
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(PLOT_PATH, dpi=160)
    plt.close(figure)


def _write_report(
    index: ThresholdIndex,
    binary: OptimizationResult,
    grid: OptimizationResult,
    exhaustive: OptimizationResult,
) -> None:
    metrics = binary.metrics
    evaluation_factor = exhaustive.evaluations / binary.evaluations
    grid_factor = grid.evaluations / binary.evaluations
    record_count = f"{index.record_count:,}".replace(",", ".")
    positive_count = f"{index.positive_count:,}".replace(",", ".")
    negative_count = f"{index.negative_count:,}".replace(",", ".")
    exact_candidates = f"{binary.candidate_count:,}".replace(",", ".")
    exhaustive_evaluations = f"{exhaustive.evaluations:,}".replace(",", ".")
    grid_evaluations = f"{grid.evaluations:,}".replace(",", ".")
    text = f"""# Laufbericht: Schwellenwert-Optimierung

## Szenario

Der Generator erzeugt mit Seed `{DEFAULT_SEED}` insgesamt {record_count}
Klassifikationsbeispiele mit überlappenden Score-Verteilungen. Davon sind
{positive_count} positiv und {negative_count} negativ. Ein Score
ab dem Schwellenwert wird als positive Vorhersage gewertet.

## Nebenbedingung und Monotonie

Gesucht wird maximaler Recall unter der Nebenbedingung
`False Positive Rate ≤ {MAX_FALSE_POSITIVE_RATE:.0%}`. Mit steigendem Schwellenwert
kann die Menge positiver Vorhersagen nur kleiner werden. Daher können weder False
Positives noch True Positives zunehmen: FPR und Recall sind beide monoton nicht
steigend. Die **kleinste zulässige Schwelle** maximiert somit den Recall unter der
FPR-Grenze.

Precision wurde bewusst nicht als Suchprädikat verwendet. Sie ist auf endlichen
Datensätzen im Allgemeinen nicht monoton und würde die Korrektheitsvoraussetzung
der Binary Search verletzen.

## Exaktes Ergebnis

- Schwellenwert: `{metrics.threshold:.8f}`
- Recall: {metrics.recall:.4f}
- False Positive Rate: {metrics.false_positive_rate:.4f}
- Precision: {metrics.precision:.4f}
- True Positives: {metrics.true_positives}
- False Positives: {metrics.false_positives}
- Exakte Kandidatenschwellen: {exact_candidates}
- Metrikauswertungen der Binary Search: {binary.evaluations}

Der direkte erschöpfende Lauf über alle exakten Kandidaten liefert denselben
Schwellenwert und dieselben Confusion Counts. Er benötigt
{exhaustive_evaluations} Auswertungen und damit {evaluation_factor:.1f}-mal so
viele wie die Binary Search.

## Vergleich mit der Rastersuche

Die naive Rastersuche prüft {grid_evaluations} gleichmäßig verteilte
Schwellenwerte. Sie benötigt {grid_factor:.1f}-mal so viele Auswertungen wie die
Binary Search und ist nur auf ihre Rasterauflösung genau.

- Raster-Schwellenwert: `{grid.metrics.threshold:.8f}`
- Raster-Recall: {grid.metrics.recall:.4f}
- Raster-FPR: {grid.metrics.false_positive_rate:.4f}

Die Vorverarbeitung sortiert positive und negative Scores einmalig. Eine
Metrikauswertung zählt danach die Werte oberhalb einer Schwelle mit zwei
`bisect_left`-Suchen. Die Optimierung benötigt `O(log u)` solcher Auswertungen für
`u` exakte Kandidatenschwellen; eine vollständige Rastersuche benötigt dagegen
eine Auswertung pro Rasterpunkt.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    """Generate data, optimize the threshold, verify, and write artifacts."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_score_data(DATA_PATH, record_count=RECORD_COUNT, seed=DEFAULT_SEED)
    index = ThresholdIndex(read_score_csv(DATA_PATH))
    binary = optimize_threshold_binary(index, MAX_FALSE_POSITIVE_RATE)
    grid = optimize_threshold_grid(
        index, MAX_FALSE_POSITIVE_RATE, grid_intervals=GRID_INTERVALS
    )
    exhaustive = optimize_threshold_exhaustive(index, MAX_FALSE_POSITIVE_RATE)

    if binary.metrics != exhaustive.metrics:
        raise RuntimeError("binary search disagrees with exhaustive exact reference")
    if binary.metrics.false_positive_rate > MAX_FALSE_POSITIVE_RATE:
        raise RuntimeError("selected threshold violates the FPR constraint")
    if binary.metrics.recall < grid.metrics.recall:
        raise RuntimeError("exact search must not underperform the grid baseline")

    _write_trace(binary)
    _write_comparison((binary, grid, exhaustive))
    _write_plot(index, binary.metrics.threshold)
    _write_report(index, binary, grid, exhaustive)
    print(
        f"Optimized {index.record_count} scores: threshold={binary.metrics.threshold:.8f}, "
        f"recall={binary.metrics.recall:.4f}, FPR={binary.metrics.false_positive_rate:.4f}, "
        f"evaluations={binary.evaluations}/{exhaustive.evaluations}"
    )


if __name__ == "__main__":
    main()
