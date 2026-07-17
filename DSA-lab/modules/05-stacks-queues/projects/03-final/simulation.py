"""Event-driven simulation of FIFO and priority-based job queues."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
import csv
from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Literal, Protocol


Policy = Literal["fifo", "priority"]


@dataclass(frozen=True)
class Job:
    """One analysis job arriving at a single-worker system."""

    job_id: str
    arrival_time: float
    service_time: float
    priority: int
    job_type: str

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must not be negative")
        if self.service_time <= 0:
            raise ValueError("service_time must be greater than zero")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("priority must be an integer")
        if self.priority <= 0:
            raise ValueError("priority must be greater than zero")
        if not self.job_type:
            raise ValueError("job_type must not be empty")


@dataclass(frozen=True)
class JobResult:
    """Timing data for one completed job."""

    policy: Policy
    job_id: str
    priority: int
    job_type: str
    arrival_time: float
    service_time: float
    start_time: float
    finish_time: float

    @property
    def waiting_time(self) -> float:
        """Return time spent waiting before service starts."""
        return self.start_time - self.arrival_time

    @property
    def turnaround_time(self) -> float:
        """Return total time from arrival through completion."""
        return self.finish_time - self.arrival_time


@dataclass(frozen=True)
class QueueMetrics:
    """Aggregate waiting and throughput metrics for result rows."""

    job_count: int
    mean_wait: float
    median_wait: float
    p95_wait: float
    max_wait: float
    mean_turnaround: float
    makespan: float
    throughput: float
    utilization: float


class _WaitingQueue(Protocol):
    """Minimal interface shared by the two scheduling queues."""

    def push(self, sequence: int, job: Job) -> None: ...

    def pop(self) -> Job: ...

    def __bool__(self) -> bool: ...


class _FifoQueue:
    """Arrival-ordered waiting queue backed by ``collections.deque``."""

    def __init__(self) -> None:
        self._items: deque[Job] = deque()

    def push(self, sequence: int, job: Job) -> None:
        del sequence
        self._items.append(job)

    def pop(self) -> Job:
        return self._items.popleft()

    def __bool__(self) -> bool:
        return bool(self._items)


class _StablePriorityQueue:
    """Priority queue preserving FIFO order among equal priorities."""

    def __init__(self) -> None:
        self._items: list[tuple[int, int, Job]] = []

    def push(self, sequence: int, job: Job) -> None:
        heapq.heappush(self._items, (job.priority, sequence, job))

    def pop(self) -> Job:
        return heapq.heappop(self._items)[2]

    def __bool__(self) -> bool:
        return bool(self._items)


def simulate(jobs: Iterable[Job], policy: Policy) -> list[JobResult]:
    """Run a non-preemptive, single-worker queue simulation.

    Jobs that have arrived while another job was running enter the waiting queue
    before the next job is selected. A running job is never interrupted.
    """
    if policy not in {"fifo", "priority"}:
        raise ValueError("policy must be 'fifo' or 'priority'")

    indexed_jobs = list(enumerate(jobs))
    identifiers = [job.job_id for _, job in indexed_jobs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("job_id values must be unique")
    if not indexed_jobs:
        return []

    pending = sorted(
        indexed_jobs,
        key=lambda pair: (pair[1].arrival_time, pair[0]),
    )
    waiting: _WaitingQueue = _FifoQueue() if policy == "fifo" else _StablePriorityQueue()
    results: list[JobResult] = []
    next_pending = 0
    current_time = 0.0

    while next_pending < len(pending) or waiting:
        if not waiting:
            current_time = max(current_time, pending[next_pending][1].arrival_time)

        while (
            next_pending < len(pending)
            and pending[next_pending][1].arrival_time <= current_time
        ):
            sequence, job = pending[next_pending]
            waiting.push(sequence, job)
            next_pending += 1

        job = waiting.pop()
        start_time = current_time
        finish_time = start_time + job.service_time
        results.append(
            JobResult(
                policy=policy,
                job_id=job.job_id,
                priority=job.priority,
                job_type=job.job_type,
                arrival_time=job.arrival_time,
                service_time=job.service_time,
                start_time=start_time,
                finish_time=finish_time,
            )
        )
        current_time = finish_time

    return results


def percentile(values: Iterable[float], percentage: float) -> float:
    """Return a linearly interpolated percentile in the closed range 0..100."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")

    position = (len(ordered) - 1) * percentage / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def summarize(results: Iterable[JobResult]) -> QueueMetrics:
    """Aggregate waiting-time and system metrics for completed jobs."""
    rows = list(results)
    if not rows:
        raise ValueError("cannot summarize an empty result set")

    waits = sorted(row.waiting_time for row in rows)
    turnarounds = [row.turnaround_time for row in rows]
    first_arrival = min(row.arrival_time for row in rows)
    final_finish = max(row.finish_time for row in rows)
    makespan = final_finish - first_arrival
    busy_time = sum(row.service_time for row in rows)
    middle = len(waits) // 2
    median = (
        waits[middle]
        if len(waits) % 2
        else (waits[middle - 1] + waits[middle]) / 2
    )

    return QueueMetrics(
        job_count=len(rows),
        mean_wait=sum(waits) / len(waits),
        median_wait=median,
        p95_wait=percentile(waits, 95),
        max_wait=max(waits),
        mean_turnaround=sum(turnarounds) / len(turnarounds),
        makespan=makespan,
        throughput=len(rows) / makespan,
        utilization=busy_time / makespan,
    )


def read_jobs_csv(path: Path) -> list[Job]:
    """Read and validate jobs from a CSV file."""
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"job_id", "arrival_time", "service_time", "priority", "job_type"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")

        jobs: list[Job] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                jobs.append(
                    Job(
                        job_id=row["job_id"],
                        arrival_time=float(row["arrival_time"]),
                        service_time=float(row["service_time"]),
                        priority=int(row["priority"]),
                        job_type=row["job_type"],
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid job at CSV line {line_number}: {error}") from error
    return jobs


def write_results_csv(path: Path, results: Iterable[JobResult]) -> Path:
    """Write detailed completion rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "job_id",
        "priority",
        "job_type",
        "arrival_time",
        "service_time",
        "start_time",
        "finish_time",
        "waiting_time",
        "turnaround_time",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "policy": row.policy,
                    "job_id": row.job_id,
                    "priority": row.priority,
                    "job_type": row.job_type,
                    "arrival_time": f"{row.arrival_time:.4f}",
                    "service_time": f"{row.service_time:.4f}",
                    "start_time": f"{row.start_time:.4f}",
                    "finish_time": f"{row.finish_time:.4f}",
                    "waiting_time": f"{row.waiting_time:.4f}",
                    "turnaround_time": f"{row.turnaround_time:.4f}",
                }
            )
    return path
