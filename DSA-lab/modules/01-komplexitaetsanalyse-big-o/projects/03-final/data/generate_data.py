"""Generate deterministic synthetic e-commerce events for the audit."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_SEED = 20_260_716
DEFAULT_ROWS = 8_000
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "events.csv"

FIELDNAMES = (
    "event_id",
    "customer_id",
    "category",
    "amount_cents",
    "discount_cents",
    "status",
    "event_timestamp",
)

CATEGORIES = (
    "books",
    "electronics",
    "fashion",
    "garden",
    "grocery",
    "sports",
    "toys",
    "travel",
)


def generate_events(
    destination: Path,
    rows: int = DEFAULT_ROWS,
    seed: int = DEFAULT_SEED,
) -> None:
    """Write a deterministic synthetic event CSV."""
    if rows < 1:
        raise ValueError("rows must be positive")

    # A fixed local RNG makes benchmark inputs reproducible without changing
    # process-wide random state. Reproducibility is essential when comparing
    # timing changes across refactorings.
    random_generator = random.Random(seed)
    customer_count = max(25, rows // 4)
    start_time = datetime(2025, 1, 1, 0, 0)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for index in range(rows):
            status_draw = random_generator.random()
            if status_draw < 0.72:
                status = "completed"
            elif status_draw < 0.88:
                status = "pending"
            else:
                status = "cancelled"

            amount_cents = random_generator.randint(500, 75_000)
            maximum_discount = min(amount_cents // 3, 5_000)
            discount_cents = random_generator.randint(0, maximum_discount)
            timestamp = start_time + timedelta(
                minutes=random_generator.randrange(365 * 24 * 60)
            )

            writer.writerow(
                {
                    "event_id": f"EVT-{index:07d}",
                    "customer_id": (
                        f"CUST-{random_generator.randrange(customer_count):05d}"
                    ),
                    "category": random_generator.choice(CATEGORIES),
                    "amount_cents": amount_cents,
                    "discount_cents": discount_cents,
                    "status": status,
                    "event_timestamp": timestamp.isoformat(timespec="minutes"),
                }
            )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic e-commerce events."
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Generate a CSV from command-line settings."""
    arguments = parse_arguments()
    try:
        generate_events(arguments.output, arguments.rows, arguments.seed)
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from error

    print(
        f"Generated {arguments.rows} events at {arguments.output} "
        f"with seed {arguments.seed}."
    )


if __name__ == "__main__":
    main()
