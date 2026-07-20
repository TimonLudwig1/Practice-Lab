"""Generate a reproducible workload for the scheduler comparison."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


# The fixed seed keeps arrivals, durations, and priority classes reproducible.
SEED = 20260720
JOB_COUNT = 300
OUTPUT_PATH = Path(__file__).resolve().parent / "scheduler_jobs.csv"
PRIORITY_NAMES = {1: "critical", 2: "standard", 3: "batch"}


@dataclass(frozen=True)
class GeneratedJob:
    job_id: str
    arrival_time: int
    duration: int
    priority: int


def generate_jobs(count: int = JOB_COUNT) -> list[GeneratedJob]:
    """Create a bursty single-server workload with three priority classes."""

    if count < 1:
        raise ValueError("count must be positive")

    rng = random.Random(SEED)
    base_priorities = [1] * 20 + [2] * 50 + [3] * 30
    priorities = [base_priorities[index % 100] for index in range(count)]
    rng.shuffle(priorities)

    jobs: list[GeneratedJob] = []
    arrival = 0
    for index, priority in enumerate(priorities):
        if index:
            # Many zero/one-step gaps create contention while occasional gaps
            # produce idle periods near the beginning of smaller test samples.
            arrival += rng.choices(
                population=[0, 1, 2, 3, 6],
                weights=[18, 38, 25, 14, 5],
                k=1,
            )[0]
        duration = rng.randint(1, 10)
        jobs.append(
            GeneratedJob(
                job_id=f"JOB-{index:04d}",
                arrival_time=arrival,
                duration=duration,
                priority=priority,
            )
        )
    return jobs


def write_jobs(path: Path, jobs: list[GeneratedJob]) -> None:
    """Write generated jobs with stable LF line endings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ("job_id", "arrival_time", "duration", "priority", "priority_name")
        )
        for job in jobs:
            writer.writerow(
                (
                    job.job_id,
                    job.arrival_time,
                    job.duration,
                    job.priority,
                    PRIORITY_NAMES[job.priority],
                )
            )


def main() -> None:
    jobs = generate_jobs()
    write_jobs(OUTPUT_PATH, jobs)
    counts = {
        priority: sum(job.priority == priority for job in jobs)
        for priority in PRIORITY_NAMES
    }
    print(f"wrote {len(jobs)} jobs to {OUTPUT_PATH}")
    print(
        "classes: "
        + ", ".join(
            f"{PRIORITY_NAMES[priority]}={counts[priority]}"
            for priority in sorted(counts)
        )
    )


if __name__ == "__main__":
    main()
