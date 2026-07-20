"""Tests for scheduler semantics, metrics, data, and full output generation."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from run_simulation import (
    DATA_PATH,
    OUTPUT_DIR,
    load_jobs,
    run_simulation,
)
from scheduler import (
    Job,
    Policy,
    ScheduledJob,
    assert_valid_schedule,
    percentile,
    simulate,
    summarize_by_priority,
)


MODULE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = MODULE_DIR / "data"
sys.path.insert(0, str(DATA_DIR))
from generate_data import JOB_COUNT, SEED, generate_jobs  # noqa: E402


def sample_jobs() -> list[Job]:
    return [
        Job("A", arrival_time=0, duration=5, priority=2),
        Job("B", arrival_time=1, duration=4, priority=3),
        Job("C", arrival_time=1, duration=1, priority=1),
        Job("D", arrival_time=2, duration=2, priority=2),
    ]


def by_id(records: list[ScheduledJob]) -> dict[str, ScheduledJob]:
    return {record.job_id: record for record in records}


def test_empty_schedule() -> None:
    assert simulate([], Policy.FIFO) == []
    assert simulate([], Policy.PRIORITY) == []


def test_fifo_order_and_metrics() -> None:
    records = simulate(sample_jobs(), Policy.FIFO)

    assert [record.job_id for record in records] == ["A", "B", "C", "D"]
    assert [record.start_time for record in records] == [0, 5, 9, 10]
    assert [record.waiting_time for record in records] == [0, 4, 8, 8]
    assert [record.turnaround_time for record in records] == [5, 8, 9, 10]


def test_priority_order_and_metrics() -> None:
    records = simulate(sample_jobs(), Policy.PRIORITY)

    assert [record.job_id for record in records] == ["A", "C", "D", "B"]
    assert [record.start_time for record in records] == [0, 5, 6, 8]
    assert [record.waiting_time for record in records] == [0, 4, 4, 7]


def test_string_policy_is_accepted() -> None:
    assert simulate(sample_jobs(), "fifo") == simulate(sample_jobs(), Policy.FIFO)
    assert simulate(sample_jobs(), "priority") == simulate(
        sample_jobs(), Policy.PRIORITY
    )


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        simulate(sample_jobs(), "shortest-job")


def test_priority_scheduler_is_non_preemptive() -> None:
    jobs = [
        Job("low", 0, 10, 3),
        Job("critical", 1, 1, 1),
    ]

    records = simulate(jobs, Policy.PRIORITY)

    assert [record.job_id for record in records] == ["low", "critical"]
    assert records[1].start_time == 10
    assert records[1].waiting_time == 9


def test_idle_time_jumps_to_next_arrival() -> None:
    jobs = [Job("A", 3, 2, 1), Job("B", 10, 1, 1)]

    records = simulate(jobs, Policy.FIFO)

    assert [(record.start_time, record.finish_time) for record in records] == [
        (3, 5),
        (10, 11),
    ]
    assert [record.waiting_time for record in records] == [0, 0]


def test_fifo_tie_breaks_by_job_id() -> None:
    jobs = [Job("Z", 0, 1, 2), Job("A", 0, 1, 2), Job("M", 0, 1, 2)]

    assert [record.job_id for record in simulate(jobs, Policy.FIFO)] == ["A", "M", "Z"]


def test_priority_tie_breaks_by_arrival_then_job_id() -> None:
    jobs = [
        Job("root", 0, 5, 1),
        Job("later", 2, 1, 2),
        Job("Z", 1, 1, 2),
        Job("A", 1, 1, 2),
    ]

    assert [record.job_id for record in simulate(jobs, Policy.PRIORITY)] == [
        "root",
        "A",
        "Z",
        "later",
    ]


@pytest.mark.parametrize(
    ("jobs", "message"),
    [
        ([Job("", 0, 1, 1)], "job_id"),
        ([Job("A", -1, 1, 1)], "arrival_time"),
        ([Job("A", 0, 0, 1)], "duration"),
        ([Job("A", 0, -2, 1)], "duration"),
        ([Job("A", 0, 1, 0)], "priority"),
        ([Job("A", 0, 1, 1), Job("A", 2, 1, 2)], "duplicate"),
    ],
)
def test_invalid_jobs_are_rejected(jobs: list[Job], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        simulate(jobs, Policy.FIFO)


def test_assert_valid_schedule_accepts_both_policies() -> None:
    jobs = sample_jobs()

    assert_valid_schedule(jobs, simulate(jobs, Policy.FIFO))
    assert_valid_schedule(jobs, simulate(jobs, Policy.PRIORITY))


def test_assert_valid_schedule_detects_missing_job() -> None:
    jobs = sample_jobs()
    records = simulate(jobs, Policy.FIFO)

    with pytest.raises(AssertionError, match="count"):
        assert_valid_schedule(jobs, records[:-1])


def test_assert_valid_schedule_detects_overlap() -> None:
    jobs = [Job("A", 0, 2, 1), Job("B", 0, 2, 1)]
    records = [
        ScheduledJob("A", 0, 2, 1, 0, 2, 0, 2),
        ScheduledJob("B", 0, 2, 1, 1, 3, 1, 3),
    ]

    with pytest.raises(AssertionError, match="overlap"):
        assert_valid_schedule(jobs, records)


@pytest.mark.parametrize(
    ("values", "probability", "expected"),
    [
        ([5], 0.95, 5.0),
        ([0, 10], 0.0, 0.0),
        ([0, 10], 0.5, 5.0),
        ([0, 10], 0.95, 9.5),
        ([0, 10], 1.0, 10.0),
        ([0, 10, 20, 30, 40], 0.25, 10.0),
    ],
)
def test_percentile(values: list[int], probability: float, expected: float) -> None:
    assert percentile(values, probability) == pytest.approx(expected)


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_percentile_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="between"):
        percentile([1], probability)


def test_percentile_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)


def test_summary_by_priority() -> None:
    records = simulate(sample_jobs(), Policy.FIFO)

    metrics = {item.priority: item for item in summarize_by_priority(records)}

    assert list(metrics) == [1, 2, 3]
    assert metrics[1].count == 1
    assert metrics[1].mean_wait == 8.0
    assert metrics[2].count == 2
    assert metrics[2].mean_wait == 4.0
    assert metrics[2].median_wait == 4.0
    assert metrics[2].max_wait == 8
    assert metrics[3].mean_turnaround == 8.0


def test_summary_of_empty_schedule() -> None:
    assert summarize_by_priority([]) == []


def test_generation_is_reproducible_with_expected_classes() -> None:
    first = generate_jobs()
    second = generate_jobs()

    assert first == second
    assert len(first) == JOB_COUNT == 300
    assert sum(job.priority == 1 for job in first) == 60
    assert sum(job.priority == 2 for job in first) == 150
    assert sum(job.priority == 3 for job in first) == 90
    assert all(a.arrival_time <= b.arrival_time for a, b in zip(first, first[1:]))
    assert all(1 <= job.duration <= 10 for job in first)
    assert SEED == 20260720


def test_generated_csv_loads_all_jobs() -> None:
    jobs = load_jobs(DATA_PATH)

    assert len(jobs) == 300
    assert len({job.job_id for job in jobs}) == 300
    assert {job.priority for job in jobs} == {1, 2, 3}


def test_full_schedules_are_valid_and_show_priority_tradeoff() -> None:
    jobs = load_jobs(DATA_PATH)
    fifo_records = simulate(jobs, Policy.FIFO)
    priority_records = simulate(jobs, Policy.PRIORITY)
    fifo = {item.priority: item for item in summarize_by_priority(fifo_records)}
    priority = {
        item.priority: item for item in summarize_by_priority(priority_records)
    }

    assert_valid_schedule(jobs, fifo_records)
    assert_valid_schedule(jobs, priority_records)
    assert priority[1].mean_wait < fifo[1].mean_wait
    assert priority[3].mean_wait > fifo[3].mean_wait
    assert fifo_records[-1].finish_time == priority_records[-1].finish_time


def test_full_run_creates_all_artifacts() -> None:
    summaries = run_simulation()

    assert set(summaries) == {Policy.FIFO, Policy.PRIORITY}
    schedule_path = OUTPUT_DIR / "schedule_results.csv"
    summary_path = OUTPUT_DIR / "waiting_time_summary.csv"
    plot_path = OUTPUT_DIR / "waiting_time_comparison.png"
    assert schedule_path.exists() and schedule_path.stat().st_size > 10_000
    assert summary_path.exists() and summary_path.stat().st_size > 200
    assert plot_path.exists() and plot_path.stat().st_size > 10_000

    with schedule_path.open(newline="", encoding="utf-8") as handle:
        assert sum(1 for _ in csv.reader(handle)) == 601
    with summary_path.open(newline="", encoding="utf-8") as handle:
        assert sum(1 for _ in csv.reader(handle)) == 7
