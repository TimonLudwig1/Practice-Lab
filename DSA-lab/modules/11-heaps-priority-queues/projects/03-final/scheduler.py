"""Non-preemptive FIFO and priority-queue scheduler simulation."""

from __future__ import annotations

import heapq
import statistics
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Policy(str, Enum):
    FIFO = "fifo"
    PRIORITY = "priority"


@dataclass(frozen=True)
class Job:
    """An immutable job arriving at a single-server scheduler."""

    job_id: str
    arrival_time: int
    duration: int
    priority: int


@dataclass(frozen=True)
class ScheduledJob:
    """One completed job plus its derived timing metrics."""

    job_id: str
    arrival_time: int
    duration: int
    priority: int
    start_time: int
    finish_time: int
    waiting_time: int
    turnaround_time: int


@dataclass(frozen=True)
class PriorityMetrics:
    """Waiting-time summary for one priority class."""

    priority: int
    count: int
    mean_wait: float
    median_wait: float
    p95_wait: float
    max_wait: int
    mean_turnaround: float


def simulate(jobs: Iterable[Job], policy: Policy | str) -> list[ScheduledJob]:
    """Run a deterministic, non-preemptive single-server simulation."""

    try:
        selected_policy = Policy(policy)
    except ValueError as error:
        raise ValueError(f"unknown scheduling policy: {policy!r}") from error

    ordered = sorted(jobs, key=lambda job: (job.arrival_time, job.job_id))
    _validate_jobs(ordered)
    if not ordered:
        return []

    fifo_ready: deque[Job] = deque()
    priority_ready: list[tuple[int, int, str, Job]] = []
    current_time = ordered[0].arrival_time
    next_arrival = 0
    completed: list[ScheduledJob] = []

    def ready_is_empty() -> bool:
        return not fifo_ready if selected_policy is Policy.FIFO else not priority_ready

    def enqueue(job: Job) -> None:
        if selected_policy is Policy.FIFO:
            fifo_ready.append(job)
        else:
            heapq.heappush(
                priority_ready,
                (job.priority, job.arrival_time, job.job_id, job),
            )

    def choose_next() -> Job:
        if selected_policy is Policy.FIFO:
            return fifo_ready.popleft()
        return heapq.heappop(priority_ready)[3]

    while next_arrival < len(ordered) or not ready_is_empty():
        if ready_is_empty() and next_arrival < len(ordered):
            current_time = max(current_time, ordered[next_arrival].arrival_time)

        while (
            next_arrival < len(ordered)
            and ordered[next_arrival].arrival_time <= current_time
        ):
            enqueue(ordered[next_arrival])
            next_arrival += 1

        job = choose_next()
        start = current_time
        finish = start + job.duration
        completed.append(
            ScheduledJob(
                job_id=job.job_id,
                arrival_time=job.arrival_time,
                duration=job.duration,
                priority=job.priority,
                start_time=start,
                finish_time=finish,
                waiting_time=start - job.arrival_time,
                turnaround_time=finish - job.arrival_time,
            )
        )
        current_time = finish

    assert_valid_schedule(ordered, completed)
    return completed


def summarize_by_priority(records: Iterable[ScheduledJob]) -> list[PriorityMetrics]:
    """Aggregate waiting and turnaround times by priority class."""

    groups: dict[int, list[ScheduledJob]] = {}
    for record in records:
        groups.setdefault(record.priority, []).append(record)

    summaries: list[PriorityMetrics] = []
    for priority in sorted(groups):
        group = groups[priority]
        waits = [record.waiting_time for record in group]
        turnarounds = [record.turnaround_time for record in group]
        summaries.append(
            PriorityMetrics(
                priority=priority,
                count=len(group),
                mean_wait=statistics.fmean(waits),
                median_wait=float(statistics.median(waits)),
                p95_wait=percentile(waits, 0.95),
                max_wait=max(waits),
                mean_turnaround=statistics.fmean(turnarounds),
            )
        )
    return summaries


def percentile(values: Iterable[int], probability: float) -> float:
    """Return a linearly interpolated percentile for non-empty values."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def assert_valid_schedule(jobs: Iterable[Job], records: Iterable[ScheduledJob]) -> None:
    """Raise ``AssertionError`` if a schedule loses, duplicates, or overlaps jobs."""

    expected = list(jobs)
    actual = list(records)
    assert len(actual) == len(expected), "schedule has the wrong job count"
    expected_ids = {job.job_id for job in expected}
    actual_ids = [record.job_id for record in actual]
    assert len(expected_ids) == len(expected), "input job ids are not unique"
    assert set(actual_ids) == expected_ids, "schedule changed the set of jobs"
    assert len(set(actual_ids)) == len(actual_ids), "schedule duplicated a job"

    previous_finish: int | None = None
    for record in actual:
        assert record.start_time >= record.arrival_time, "job starts before arrival"
        assert record.finish_time == record.start_time + record.duration
        assert record.waiting_time == record.start_time - record.arrival_time
        assert record.turnaround_time == record.finish_time - record.arrival_time
        if previous_finish is not None:
            assert record.start_time >= previous_finish, "jobs overlap"
        previous_finish = record.finish_time


def _validate_jobs(jobs: list[Job]) -> None:
    ids: set[str] = set()
    for job in jobs:
        if not job.job_id:
            raise ValueError("job_id must not be empty")
        if job.job_id in ids:
            raise ValueError(f"duplicate job_id: {job.job_id}")
        ids.add(job.job_id)
        if job.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")
        if job.duration <= 0:
            raise ValueError("duration must be positive")
        if job.priority <= 0:
            raise ValueError("priority must be positive")
