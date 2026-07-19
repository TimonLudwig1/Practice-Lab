"""Generate a reproducible chronological server-log stream."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random


# The fixed seed makes duplicate locations, IP skew, paths and statuses stable,
# so hash- and sort-based engines always process the exact same workload.
DEFAULT_SEED = 60603
DEFAULT_EVENT_COUNT = 4_000
DEFAULT_DUPLICATE_RATE = 0.04


def _ip_pool(size: int = 200) -> list[str]:
    return [f"10.0.{index // 254}.{index % 254 + 1}" for index in range(size)]


def generate_rows(
    event_count: int = DEFAULT_EVENT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    duplicate_rate: float = DEFAULT_DUPLICATE_RATE,
) -> list[dict[str, str]]:
    """Return chronological log rows containing deliberate duplicate IDs."""
    if isinstance(event_count, bool) or not isinstance(event_count, int):
        raise TypeError("event_count must be an integer")
    if event_count <= 0:
        raise ValueError("event_count must be greater than zero")
    if not 0 <= duplicate_rate < 1:
        raise ValueError("duplicate_rate must be in the interval [0, 1)")

    rng = random.Random(seed)
    ips = _ip_pool()
    ip_weights = [1 / ((index + 1) ** 1.08) for index in range(len(ips))]
    paths = ["/", "/api/data", "/api/model", "/health", "/login"]
    path_weights = [0.12, 0.38, 0.22, 0.18, 0.10]
    statuses = [200, 201, 400, 404, 500]
    status_weights = [0.72, 0.08, 0.07, 0.08, 0.05]
    rows: list[dict[str, str]] = []
    unique_rows: list[dict[str, str]] = []
    timestamp = 0.0
    next_unique_id = 1

    for row_index in range(event_count):
        if row_index:
            timestamp += rng.expovariate(1 / 0.55)

        should_duplicate = bool(unique_rows) and rng.random() < duplicate_rate
        if should_duplicate:
            source = rng.choice(unique_rows[max(0, len(unique_rows) - 300) :])
            row = source.copy()
            row["timestamp"] = f"{timestamp:.4f}"
        else:
            row = {
                "event_id": f"evt-{next_unique_id:06d}",
                "timestamp": f"{timestamp:.4f}",
                "ip": rng.choices(ips, weights=ip_weights, k=1)[0],
                "path": rng.choices(paths, weights=path_weights, k=1)[0],
                "status": str(rng.choices(statuses, weights=status_weights, k=1)[0]),
            }
            unique_rows.append(row.copy())
            next_unique_id += 1
        rows.append(row)
    return rows


def write_csv(
    destination: Path,
    event_count: int = DEFAULT_EVENT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    duplicate_rate: float = DEFAULT_DUPLICATE_RATE,
) -> Path:
    """Generate server logs and write them to CSV."""
    rows = generate_rows(
        event_count, seed=seed, duplicate_rate=duplicate_rate
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("server_logs.csv"),
    )
    parser.add_argument("--events", type=int, default=DEFAULT_EVENT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--duplicate-rate", type=float, default=DEFAULT_DUPLICATE_RATE)
    arguments = parser.parse_args()

    path = write_csv(
        arguments.output,
        arguments.events,
        seed=arguments.seed,
        duplicate_rate=arguments.duplicate_rate,
    )
    print(f"Generated {arguments.events} log rows in {path}")


if __name__ == "__main__":
    main()
