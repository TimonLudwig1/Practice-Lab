"""Run FIFO and priority scheduling, write reports, and create a comparison plot."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODULE_DIR = PROJECT_DIR.parents[1]
DATA_PATH = MODULE_DIR / "data" / "scheduler_jobs.csv"
OUTPUT_DIR = PROJECT_DIR / "output"
PRIORITY_NAMES = {1: "critical", 2: "standard", 3: "batch"}

# Matplotlib needs a writable cache even in a sandboxed, headless environment.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dsa-module-11-matplotlib"),
)
import matplotlib  # noqa: E402


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scheduler import Job, Policy, PriorityMetrics, simulate, summarize_by_priority  # noqa: E402


def load_jobs(path: Path = DATA_PATH) -> list[Job]:
    """Load generated jobs from CSV."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = ["job_id", "arrival_time", "duration", "priority", "priority_name"]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected job CSV header: {reader.fieldnames}")
        jobs = [
            Job(
                job_id=row["job_id"],
                arrival_time=int(row["arrival_time"]),
                duration=int(row["duration"]),
                priority=int(row["priority"]),
            )
            for row in reader
        ]
    if not jobs:
        raise ValueError("job dataset is empty")
    return jobs


def write_schedule(
    path: Path,
    schedules: dict[Policy, list],
) -> None:
    """Write every completed job for both policies."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "policy",
                "job_id",
                "priority",
                "priority_name",
                "arrival_time",
                "duration",
                "start_time",
                "finish_time",
                "waiting_time",
                "turnaround_time",
            )
        )
        for policy, records in schedules.items():
            for record in records:
                writer.writerow(
                    (
                        policy.value,
                        record.job_id,
                        record.priority,
                        PRIORITY_NAMES.get(record.priority, str(record.priority)),
                        record.arrival_time,
                        record.duration,
                        record.start_time,
                        record.finish_time,
                        record.waiting_time,
                        record.turnaround_time,
                    )
                )


def write_summary(
    path: Path,
    summaries: dict[Policy, list[PriorityMetrics]],
) -> None:
    """Write waiting-time summaries by policy and priority."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "policy",
                "priority",
                "priority_name",
                "count",
                "mean_wait",
                "median_wait",
                "p95_wait",
                "max_wait",
                "mean_turnaround",
            )
        )
        for policy, metrics_list in summaries.items():
            for metrics in metrics_list:
                writer.writerow(
                    (
                        policy.value,
                        metrics.priority,
                        PRIORITY_NAMES.get(metrics.priority, str(metrics.priority)),
                        metrics.count,
                        f"{metrics.mean_wait:.4f}",
                        f"{metrics.median_wait:.4f}",
                        f"{metrics.p95_wait:.4f}",
                        metrics.max_wait,
                        f"{metrics.mean_turnaround:.4f}",
                    )
                )


def create_plot(
    path: Path,
    summaries: dict[Policy, list[PriorityMetrics]],
) -> None:
    """Visualize mean and p95 waiting time by priority class."""

    priorities = sorted({item.priority for values in summaries.values() for item in values})
    names = [PRIORITY_NAMES.get(priority, str(priority)) for priority in priorities]
    positions = list(range(len(priorities)))
    width = 0.36
    colors = {Policy.FIFO: "#4C78A8", Policy.PRIORITY: "#F58518"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for policy_index, policy in enumerate((Policy.FIFO, Policy.PRIORITY)):
        by_priority = {item.priority: item for item in summaries[policy]}
        offsets = [position + (policy_index - 0.5) * width for position in positions]
        axes[0].bar(
            offsets,
            [by_priority[priority].mean_wait for priority in priorities],
            width,
            label=policy.value.upper(),
            color=colors[policy],
        )
        axes[1].bar(
            offsets,
            [by_priority[priority].p95_wait for priority in priorities],
            width,
            label=policy.value.upper(),
            color=colors[policy],
        )

    for axis, title, ylabel in (
        (axes[0], "Mean waiting time", "Mean time units"),
        (axes[1], "95th-percentile waiting time", "P95 time units"),
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, names)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("FIFO vs. priority-queue scheduling by job class")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_simulation() -> dict[Policy, list[PriorityMetrics]]:
    """Execute the full comparison and create all output artifacts."""

    jobs = load_jobs()
    schedules = {
        Policy.FIFO: simulate(jobs, Policy.FIFO),
        Policy.PRIORITY: simulate(jobs, Policy.PRIORITY),
    }
    summaries = {
        policy: summarize_by_priority(records)
        for policy, records in schedules.items()
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_schedule(OUTPUT_DIR / "schedule_results.csv", schedules)
    write_summary(OUTPUT_DIR / "waiting_time_summary.csv", summaries)
    create_plot(OUTPUT_DIR / "waiting_time_comparison.png", summaries)
    return summaries


def main() -> None:
    summaries = run_simulation()
    print("SCHEDULER COMPARISON")
    for priority in sorted(PRIORITY_NAMES):
        fifo = next(item for item in summaries[Policy.FIFO] if item.priority == priority)
        priority_result = next(
            item for item in summaries[Policy.PRIORITY] if item.priority == priority
        )
        print(
            f"{PRIORITY_NAMES[priority]:>8}: "
            f"FIFO mean={fifo.mean_wait:7.2f}, "
            f"priority mean={priority_result.mean_wait:7.2f}, "
            f"change={priority_result.mean_wait - fifo.mean_wait:+7.2f}"
        )
    print(f"artifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
