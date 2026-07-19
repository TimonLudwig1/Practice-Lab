"""Generate reproducible binary labels and probabilistic classifier scores."""

from __future__ import annotations

import csv
import random
from pathlib import Path


DEFAULT_SEED = 80803


def generate_score_data(
    path: str | Path,
    *,
    record_count: int = 5_000,
    positive_rate: float = 0.30,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Write a seeded score dataset with overlapping class distributions."""

    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count < 1
    ):
        raise ValueError("record_count must be a positive integer")
    if not 0.0 < positive_rate < 1.0:
        raise ValueError("positive_rate must be strictly between zero and one")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("record_id", "label", "score"))
        for index in range(record_count):
            label = int(rng.random() < positive_rate)
            if label:
                score = rng.betavariate(5.0, 2.0)
            else:
                score = rng.betavariate(2.0, 5.0)
            writer.writerow((f"row-{index:08d}", label, f"{score:.8f}"))
    return destination


if __name__ == "__main__":
    generated = generate_score_data(Path("data") / "scores.csv")
    print(f"Generated {generated}")
