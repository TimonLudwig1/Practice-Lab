"""Run both queue policies and create tables, a report and a visualization."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile


# Matplotlib needs a writable cache even in restricted learning environments.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dsa_job_queue_matplotlib")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.generate_data import DEFAULT_JOB_COUNT, DEFAULT_SEED, write_csv  # noqa: E402
from simulation import (  # noqa: E402
    JobResult,
    Policy,
    QueueMetrics,
    read_jobs_csv,
    simulate,
    summarize,
    write_results_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "jobs.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"
PRIORITY_NAMES = {1: "urgent", 2: "standard", 3: "batch"}


def segmented_metrics(results: list[JobResult]) -> dict[str, QueueMetrics]:
    """Return overall metrics and one segment per priority level."""
    segments = {"all": summarize(results)}
    for priority, name in PRIORITY_NAMES.items():
        group = [row for row in results if row.priority == priority]
        if group:
            segments[name] = summarize(group)
    return segments


def write_summary_csv(
    path: Path, metrics: dict[Policy, dict[str, QueueMetrics]]
) -> Path:
    """Write all overall and priority-segment metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "segment",
        "job_count",
        "mean_wait",
        "median_wait",
        "p95_wait",
        "max_wait",
        "mean_turnaround",
        "makespan",
        "throughput",
        "utilization",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for policy, segments in metrics.items():
            for segment, values in segments.items():
                row = {"policy": policy, "segment": segment}
                row.update(
                    {
                        "job_count": values.job_count,
                        "mean_wait": f"{values.mean_wait:.4f}",
                        "median_wait": f"{values.median_wait:.4f}",
                        "p95_wait": f"{values.p95_wait:.4f}",
                        "max_wait": f"{values.max_wait:.4f}",
                        "mean_turnaround": f"{values.mean_turnaround:.4f}",
                        "makespan": f"{values.makespan:.4f}",
                        "throughput": f"{values.throughput:.4f}",
                        "utilization": f"{values.utilization:.4f}",
                    }
                )
                writer.writerow(row)
    return path


def create_plot(
    path: Path, results: dict[Policy, list[JobResult]], metrics: dict[Policy, dict[str, QueueMetrics]]
) -> Path:
    """Plot waiting-time distributions and priority-group means."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (distribution_axis, priority_axis) = plt.subplots(
        1, 2, figsize=(12, 4.8), constrained_layout=True
    )
    colors = {"fifo": "#2878B5", "priority": "#D95319"}

    for policy, rows in results.items():
        waits = sorted(row.waiting_time for row in rows)
        cumulative = [(index + 1) / len(waits) for index in range(len(waits))]
        distribution_axis.step(
            waits,
            cumulative,
            where="post",
            label=policy.upper(),
            color=colors[policy],
            linewidth=2,
        )
    distribution_axis.set(
        title="Empirische Verteilung der Wartezeiten",
        xlabel="Wartezeit",
        ylabel="Anteil abgeschlossener Jobs",
        ylim=(0, 1.02),
    )
    distribution_axis.grid(alpha=0.25)
    distribution_axis.legend()

    segment_names = ["urgent", "standard", "batch"]
    positions = list(range(len(segment_names)))
    width = 0.36
    for offset, policy in ((-width / 2, "fifo"), (width / 2, "priority")):
        means = [metrics[policy][name].mean_wait for name in segment_names]
        priority_axis.bar(
            [position + offset for position in positions],
            means,
            width=width,
            label=policy.upper(),
            color=colors[policy],
        )
    priority_axis.set(
        title="Mittlere Wartezeit nach Priorität",
        xlabel="Job-Typ",
        ylabel="Mittlere Wartezeit",
        xticks=positions,
        xticklabels=segment_names,
    )
    priority_axis.grid(axis="y", alpha=0.25)
    priority_axis.legend()

    figure.suptitle("FIFO versus stabile Priority Queue")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def write_report(
    path: Path, metrics: dict[Policy, dict[str, QueueMetrics]]
) -> Path:
    """Write a compact interpretation of the deterministic experiment."""
    fifo = metrics["fifo"]
    priority = metrics["priority"]
    urgent_delta = priority["urgent"].mean_wait - fifo["urgent"].mean_wait
    batch_delta = priority["batch"].mean_wait - fifo["batch"].mean_wait

    content = f"""# Ergebnisbericht: FIFO versus Priority Queue

Der Vergleich verwendet denselben synthetischen Strom aus {fifo['all'].job_count}
Jobs für beide Strategien. Die Verarbeitung ist nicht-präemptiv: Ein gestarteter
Job wird nie von einem später eintreffenden dringenden Job unterbrochen.

| Strategie | Mittelwert Warten | Median | P95 | Maximum | Auslastung |
|---|---:|---:|---:|---:|---:|
| FIFO | {fifo['all'].mean_wait:.2f} | {fifo['all'].median_wait:.2f} | {fifo['all'].p95_wait:.2f} | {fifo['all'].max_wait:.2f} | {fifo['all'].utilization:.1%} |
| Priorität | {priority['all'].mean_wait:.2f} | {priority['all'].median_wait:.2f} | {priority['all'].p95_wait:.2f} | {priority['all'].max_wait:.2f} | {priority['all'].utilization:.1%} |

## Interpretation

Für dringende Jobs verändert die Prioritätsregel die mittlere Wartezeit um
{urgent_delta:+.2f} Zeiteinheiten gegenüber FIFO. Für Batch-Jobs beträgt die
Änderung {batch_delta:+.2f}. Die Strategie verteilt Wartezeit somit gezielt
zwischen Klassen um; ein einzelner Gesamtmittelwert reicht zur Bewertung nicht
aus. Insbesondere Maximum und P95 machen sichtbar, ob niedrig priorisierte Jobs
sehr lange warten.

Durchsatz und Auslastung sind nahezu identisch, weil beide Strategien dieselben
Jobs auf demselben einzelnen Worker ohne Leerlaufentscheidung verarbeiten. Die
Queue Policy verändert primär die Reihenfolge und damit die Fairness, nicht die
vorhandene Rechenkapazität.
"""
    path.write_text(content, encoding="utf-8")
    return path


def run_analysis(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[Policy, dict[str, QueueMetrics]]:
    """Generate missing data, simulate both policies and write all artifacts."""
    if not data_path.exists():
        write_csv(data_path, DEFAULT_JOB_COUNT, seed=DEFAULT_SEED)

    jobs = read_jobs_csv(data_path)
    results: dict[Policy, list[JobResult]] = {
        "fifo": simulate(jobs, "fifo"),
        "priority": simulate(jobs, "priority"),
    }
    metrics: dict[Policy, dict[str, QueueMetrics]] = {
        policy: segmented_metrics(rows) for policy, rows in results.items()
    }

    for policy, rows in results.items():
        write_results_csv(output_dir / f"{policy}_results.csv", rows)
    write_summary_csv(output_dir / "summary.csv", metrics)
    create_plot(output_dir / "wait_time_comparison.png", results, metrics)
    write_report(output_dir / "REPORT.md", metrics)
    return metrics


def main() -> None:
    """Run the default experiment and print its central metrics."""
    metrics = run_analysis()
    print("Policy    Mean wait   P95 wait   Max wait   Utilization")
    print("-" * 59)
    for policy in ("fifo", "priority"):
        values = metrics[policy]["all"]
        print(
            f"{policy:<10}{values.mean_wait:>9.2f}"
            f"{values.p95_wait:>11.2f}{values.max_wait:>11.2f}"
            f"{values.utilization:>13.1%}"
        )
    print(f"\nArtifacts written to {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
