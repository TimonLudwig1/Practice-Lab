"""Generate a reproducible metric stream with labeled injected anomalies."""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_SEED = 90903


def generate_metric_stream(
    path: str | Path,
    *,
    record_count: int = 12_000,
    anomaly_count: int = 20,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Write a smooth noisy signal with sparse, well-separated spikes."""

    if not isinstance(record_count, int) or isinstance(record_count, bool) or record_count < 500:
        raise ValueError("record_count must be an integer of at least 500")
    if (
        not isinstance(anomaly_count, int)
        or isinstance(anomaly_count, bool)
        or not 0 <= anomaly_count <= (record_count - 400) // 300
    ):
        raise ValueError("anomaly_count is too large for safe spacing")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    available = list(range(300, record_count - 100, 300))
    anomaly_indices = set(rng.sample(available, anomaly_count))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("timestamp", "value", "injected_anomaly"))
        for index in range(record_count):
            baseline = 100.0 + 2.0 * math.sin(index / 350.0) + 0.0002 * index
            value = baseline + rng.gauss(0.0, 1.0)
            injected = index in anomaly_indices
            if injected:
                direction = -1.0 if (index // 300) % 2 else 1.0
                value += direction * rng.uniform(9.0, 12.0)
            timestamp = (start + timedelta(minutes=index)).isoformat().replace(
                "+00:00", "Z"
            )
            writer.writerow((timestamp, f"{value:.8f}", injected))
    return destination


if __name__ == "__main__":
    generated = generate_metric_stream(Path("data") / "metric_stream.csv")
    print(f"Generated {generated}")
