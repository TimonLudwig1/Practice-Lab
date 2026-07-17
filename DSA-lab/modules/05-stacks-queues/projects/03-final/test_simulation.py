"""Tests for workload generation, scheduling and result reporting."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from data.generate_data import generate_rows, write_csv
from run_analysis import run_analysis, segmented_metrics
from simulation import (
    Job,
    JobResult,
    percentile,
    read_jobs_csv,
    simulate,
    summarize,
    write_results_csv,
)


def make_job(
    job_id: str,
    arrival: float,
    service: float,
    priority: int = 2,
    job_type: str = "standard",
) -> Job:
    """Create a concise test job."""
    return Job(job_id, arrival, service, priority, job_type)


@pytest.mark.parametrize(
    "arguments",
    [
        ("", 0.0, 1.0, 1, "urgent"),
        ("A", -1.0, 1.0, 1, "urgent"),
        ("A", 0.0, 0.0, 1, "urgent"),
        ("A", 0.0, -1.0, 1, "urgent"),
        ("A", 0.0, 1.0, 0, "urgent"),
        ("A", 0.0, 1.0, 1, ""),
    ],
)
def test_job_rejects_invalid_values(arguments: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        Job(*arguments)  # type: ignore[arg-type]


def test_fifo_uses_arrival_order() -> None:
    jobs = [
        make_job("A", 0, 4, 2),
        make_job("B", 1, 3, 3, "batch"),
        make_job("C", 1, 1, 1, "urgent"),
    ]

    results = simulate(jobs, "fifo")

    assert [row.job_id for row in results] == ["A", "B", "C"]
    assert [row.waiting_time for row in results] == [0, 3, 6]


def test_priority_selects_urgent_waiting_job_first() -> None:
    jobs = [
        make_job("A", 0, 4, 2),
        make_job("B", 1, 3, 3, "batch"),
        make_job("C", 1, 1, 1, "urgent"),
    ]

    results = simulate(jobs, "priority")

    assert [row.job_id for row in results] == ["A", "C", "B"]
    assert [row.waiting_time for row in results] == [0, 3, 4]


def test_priority_is_stable_for_equal_priority_and_arrival() -> None:
    jobs = [make_job("A", 0, 2), make_job("B", 0, 1), make_job("C", 0, 1)]

    assert [row.job_id for row in simulate(jobs, "priority")] == ["A", "B", "C"]


def test_running_job_is_not_preempted() -> None:
    jobs = [
        make_job("batch", 0, 10, 3, "batch"),
        make_job("urgent", 1, 1, 1, "urgent"),
    ]

    results = simulate(jobs, "priority")

    assert [row.job_id for row in results] == ["batch", "urgent"]
    assert results[1].start_time == 10


def test_worker_jumps_over_idle_period() -> None:
    jobs = [make_job("A", 5, 2), make_job("B", 10, 1)]

    results = simulate(jobs, "fifo")

    assert [(row.start_time, row.finish_time) for row in results] == [(5, 7), (10, 11)]
    assert [row.waiting_time for row in results] == [0, 0]


def test_simulation_accepts_unsorted_input_without_mutating_it() -> None:
    jobs = [make_job("later", 2, 1), make_job("first", 0, 1)]
    original = jobs.copy()

    results = simulate(jobs, "fifo")

    assert jobs == original
    assert [row.job_id for row in results] == ["first", "later"]


def test_simulation_handles_empty_input() -> None:
    assert simulate([], "fifo") == []


def test_simulation_rejects_unknown_policy_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="policy"):
        simulate([], "lifo")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        simulate([make_job("A", 0, 1), make_job("A", 1, 1)], "fifo")


@pytest.mark.parametrize(
    ("values", "percentage", "expected"),
    [
        ([10], 95, 10),
        ([0, 10], 50, 5),
        ([0, 10, 20, 30, 40], 25, 10),
        ([40, 10, 30, 20], 100, 40),
    ],
)
def test_percentile(values: list[float], percentage: float, expected: float) -> None:
    assert percentile(values, percentage) == pytest.approx(expected)


def test_percentile_rejects_empty_values_and_invalid_percentage() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_summarize_calculates_waiting_and_system_metrics() -> None:
    results = simulate([make_job("A", 0, 2), make_job("B", 1, 2)], "fifo")

    metrics = summarize(results)

    assert metrics.job_count == 2
    assert metrics.mean_wait == pytest.approx(0.5)
    assert metrics.median_wait == pytest.approx(0.5)
    assert metrics.mean_turnaround == pytest.approx(2.5)
    assert metrics.makespan == pytest.approx(4)
    assert metrics.throughput == pytest.approx(0.5)
    assert metrics.utilization == pytest.approx(1)


def test_summarize_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize([])


def test_generator_is_reproducible_and_arrivals_are_ordered() -> None:
    first = generate_rows(20, seed=7)
    second = generate_rows(20, seed=7)
    different = generate_rows(20, seed=8)

    assert first == second
    assert first != different
    arrivals = [float(row["arrival_time"]) for row in first]
    assert arrivals == sorted(arrivals)
    assert {int(row["priority"]) for row in first} <= {1, 2, 3}


@pytest.mark.parametrize("job_count", [0, -1])
def test_generator_rejects_non_positive_job_count(job_count: int) -> None:
    with pytest.raises(ValueError):
        generate_rows(job_count)


def test_job_csv_round_trip(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "jobs.csv", 12, seed=10)

    jobs = read_jobs_csv(path)

    assert len(jobs) == 12
    assert jobs[0].job_id == "J0001"
    assert jobs[0].arrival_time == 0


def test_read_jobs_csv_reports_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text("job_id,arrival_time\nA,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="columns"):
        read_jobs_csv(path)


def test_write_results_csv_includes_derived_times(tmp_path: Path) -> None:
    results = simulate([make_job("A", 0, 2), make_job("B", 1, 1)], "fifo")
    path = write_results_csv(tmp_path / "results.csv", results)

    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert rows[1]["waiting_time"] == "1.0000"
    assert rows[1]["turnaround_time"] == "2.0000"


def test_segmented_metrics_separates_priority_groups() -> None:
    jobs = [
        make_job("U", 0, 1, 1, "urgent"),
        make_job("S", 0, 1, 2, "standard"),
        make_job("B", 0, 1, 3, "batch"),
    ]

    segments = segmented_metrics(simulate(jobs, "fifo"))

    assert set(segments) == {"all", "urgent", "standard", "batch"}
    assert all(segments[name].job_count == 1 for name in segments if name != "all")


def test_full_analysis_writes_all_artifacts(tmp_path: Path) -> None:
    data_path = write_csv(tmp_path / "data" / "jobs.csv", 40, seed=99)
    output_dir = tmp_path / "output"

    metrics = run_analysis(data_path, output_dir)

    assert set(metrics) == {"fifo", "priority"}
    assert metrics["fifo"]["all"].job_count == 40
    for filename in (
        "fifo_results.csv",
        "priority_results.csv",
        "summary.csv",
        "wait_time_comparison.png",
        "REPORT.md",
    ):
        artifact = output_dir / filename
        assert artifact.exists()
        assert artifact.stat().st_size > 0
