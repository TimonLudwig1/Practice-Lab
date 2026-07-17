"""Generate a reproducible stream of arriving analysis jobs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random


# A fixed seed is intentional: both scheduling policies must see the exact same
# workload, and every learner must be able to reproduce the reported metrics.
DEFAULT_SEED = 20260717
DEFAULT_JOB_COUNT = 250
PRIORITY_LEVELS = (
    (1, "urgent", 0.15),
    (2, "standard", 0.60),
    (3, "batch", 0.25),
)


def generate_rows(
    job_count: int = DEFAULT_JOB_COUNT, *, seed: int = DEFAULT_SEED
) -> list[dict[str, str]]:
    """Return synthetic jobs ordered by arrival time.

    Inter-arrival times follow an exponential distribution. Service times use a
    bounded log-normal distribution, a simple model for mostly short analyses
    with a few substantially longer jobs.
    """
    if isinstance(job_count, bool) or not isinstance(job_count, int):
        raise TypeError("job_count must be an integer")
    if job_count <= 0:
        raise ValueError("job_count must be greater than zero")

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    arrival_time = 0.0
    priorities = [level for level, _, _ in PRIORITY_LEVELS]
    weights = [weight for _, _, weight in PRIORITY_LEVELS]
    names = {level: name for level, name, _ in PRIORITY_LEVELS}

    for index in range(job_count):
        if index:
            arrival_time += rng.expovariate(1 / 1.15)
        priority = rng.choices(priorities, weights=weights, k=1)[0]
        service_time = min(9.0, max(0.2, rng.lognormvariate(-0.05, 0.72)))
        rows.append(
            {
                "job_id": f"J{index + 1:04d}",
                "arrival_time": f"{arrival_time:.4f}",
                "service_time": f"{service_time:.4f}",
                "priority": str(priority),
                "job_type": names[priority],
            }
        )
    return rows


def write_csv(
    destination: Path,
    job_count: int = DEFAULT_JOB_COUNT,
    *,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Generate jobs and write them to ``destination``."""
    rows = generate_rows(job_count, seed=seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main() -> None:
    """Generate the default or a user-configured CSV data set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("jobs.csv"),
        help="destination CSV path",
    )
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOB_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()

    path = write_csv(arguments.output, arguments.jobs, seed=arguments.seed)
    print(f"Generated {arguments.jobs} jobs in {path}")


if __name__ == "__main__":
    main()
