"""Generate a reproducible, deliberately unsorted sensor-event CSV."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


FIELDNAMES = (
    "event_id",
    "timestamp",
    "sensor_id",
    "temperature_c",
    "quality",
    "payload",
)
DEFAULT_SEED = 70703


def generate_events(
    path: str | Path, *, record_count: int = 5_000, seed: int = DEFAULT_SEED
) -> Path:
    """Write events one by one so generation itself does not need the dataset in RAM."""

    if record_count < 0:
        raise ValueError("record_count must not be negative")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        for index in range(record_count):
            timestamp = base + timedelta(seconds=rng.randrange(900))
            writer.writerow(
                {
                    "event_id": f"evt-{index:08d}",
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "sensor_id": f"sensor-{rng.randrange(1, 13):03d}",
                    "temperature_c": f"{rng.uniform(-12.0, 38.0):.2f}",
                    "quality": rng.choice(("good", "good", "good", "suspect")),
                    "payload": f"zone={rng.randrange(1, 5)}, batch={index // 250}",
                }
            )
    return destination


if __name__ == "__main__":
    generated = generate_events(Path("data") / "unsorted_events.csv")
    print(f"Generated {generated}")
