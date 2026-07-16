"""Baseline and optimized customer aggregation pipelines."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Event:
    """One immutable event loaded from the source CSV."""

    event_id: str
    customer_id: str
    category: str
    amount_cents: int
    discount_cents: int
    status: str
    event_timestamp: str

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> "Event":
        """Build and validate an event from a CSV mapping."""
        amount_cents = int(row["amount_cents"])
        discount_cents = int(row["discount_cents"])
        if amount_cents < 0:
            raise ValueError("amount_cents must not be negative")
        if not 0 <= discount_cents <= amount_cents:
            raise ValueError("discount_cents must be between zero and amount")

        return cls(
            event_id=row["event_id"],
            customer_id=row["customer_id"],
            category=row["category"],
            amount_cents=amount_cents,
            discount_cents=discount_cents,
            status=row["status"],
            event_timestamp=row["event_timestamp"],
        )

    @property
    def net_cents(self) -> int:
        """Return revenue after discount."""
        return self.amount_cents - self.discount_cents


@dataclass(frozen=True)
class CustomerSummary:
    """Aggregated metrics for one customer."""

    customer_id: str
    completed_events: int
    net_revenue_cents: int
    average_net_cents: int
    unique_categories: int
    latest_event_timestamp: str


@dataclass
class _Accumulator:
    """Mutable aggregation state used by the optimized pipeline."""

    completed_events: int = 0
    net_revenue_cents: int = 0
    categories: set[str] = field(default_factory=set)
    latest_event_timestamp: str = ""

    def update(self, event: Event) -> None:
        """Consume one completed event."""
        self.completed_events += 1
        self.net_revenue_cents += event.net_cents
        self.categories.add(event.category)
        if event.event_timestamp > self.latest_event_timestamp:
            self.latest_event_timestamp = event.event_timestamp

    def to_summary(self, customer_id: str) -> CustomerSummary:
        """Freeze accumulated values into an output row."""
        return CustomerSummary(
            customer_id=customer_id,
            completed_events=self.completed_events,
            net_revenue_cents=self.net_revenue_cents,
            average_net_cents=(
                self.net_revenue_cents // self.completed_events
            ),
            unique_categories=len(self.categories),
            latest_event_timestamp=self.latest_event_timestamp,
        )


def load_events(source: Path) -> list[Event]:
    """Load all events from a CSV file."""
    with source.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [Event.from_csv_row(row) for row in reader]


def inefficient_pipeline(events: Iterable[Event]) -> list[CustomerSummary]:
    """Aggregate through list membership checks and repeated full scans."""
    event_list = list(events)
    customer_ids: list[str] = []

    for event in event_list:
        if (
            event.status == "completed"
            and event.customer_id not in customer_ids
        ):
            customer_ids.append(event.customer_id)

    summaries = []
    for customer_id in customer_ids:
        customer_events = [
            event
            for event in event_list
            if (
                event.status == "completed"
                and event.customer_id == customer_id
            )
        ]
        net_revenue_cents = sum(event.net_cents for event in customer_events)
        categories = {event.category for event in customer_events}
        latest_timestamp = max(
            event.event_timestamp for event in customer_events
        )

        summaries.append(
            CustomerSummary(
                customer_id=customer_id,
                completed_events=len(customer_events),
                net_revenue_cents=net_revenue_cents,
                average_net_cents=(
                    net_revenue_cents // len(customer_events)
                ),
                unique_categories=len(categories),
                latest_event_timestamp=latest_timestamp,
            )
        )

    return sorted(summaries, key=lambda summary: summary.customer_id)


def optimized_pipeline(events: Iterable[Event]) -> list[CustomerSummary]:
    """Aggregate all completed events in one pass with hash lookups."""
    accumulators: dict[str, _Accumulator] = {}

    for event in events:
        if event.status != "completed":
            continue

        accumulator = accumulators.get(event.customer_id)
        if accumulator is None:
            accumulator = _Accumulator()
            accumulators[event.customer_id] = accumulator
        accumulator.update(event)

    return [
        accumulators[customer_id].to_summary(customer_id)
        for customer_id in sorted(accumulators)
    ]


def write_summaries(
    summaries: Iterable[CustomerSummary],
    destination: Path,
) -> None:
    """Write customer summaries to CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "customer_id",
        "completed_events",
        "net_revenue_cents",
        "average_net_cents",
        "unique_categories",
        "latest_event_timestamp",
    )

    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    fieldname: getattr(summary, fieldname)
                    for fieldname in fieldnames
                }
            )
