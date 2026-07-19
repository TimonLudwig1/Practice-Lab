"""Tests for data generation, both analysis engines and the full pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from data.generate_data import generate_rows, write_csv
from log_engine import (
    LogEvent,
    analyze_hash,
    analyze_sort,
    equivalent_results,
    read_logs_csv,
    write_sessionized_csv,
)
from run_analysis import run_analysis, run_benchmark


def event(
    event_id: str,
    timestamp: float,
    ip: str = "10.0.0.1",
    path: str = "/",
    status: int = 200,
) -> LogEvent:
    return LogEvent(event_id, timestamp, ip, path, status)


@pytest.mark.parametrize(
    "arguments",
    [
        ("", 0, "10.0.0.1", "/", 200),
        ("A", -1, "10.0.0.1", "/", 200),
        ("A", 0, "", "/", 200),
        ("A", 0, "10.0.0.1", "", 200),
        ("A", 0, "10.0.0.1", "/", 99),
        ("A", 0, "10.0.0.1", "/", 600),
    ],
)
def test_event_validation(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        LogEvent(*arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("engine", [analyze_hash, analyze_sort])
def test_empty_input(engine) -> None:
    result = engine([], top_k=3)

    assert result.input_count == 0
    assert result.unique_count == 0
    assert result.duplicate_count == 0
    assert result.session_count == 0
    assert result.top_ips == ()
    assert result.events == ()


@pytest.mark.parametrize("engine", [analyze_hash, analyze_sort])
def test_duplicate_filter_keeps_first_occurrence(engine) -> None:
    logs = [
        event("A", 0, path="/first"),
        event("B", 1),
        event("A", 2, path="/duplicate"),
    ]

    result = engine(logs)

    assert result.duplicate_count == 1
    assert [row.event.path for row in result.events] == ["/first", "/"]


@pytest.mark.parametrize("engine", [analyze_hash, analyze_sort])
def test_top_ips_use_counts_and_lexical_tie_break(engine) -> None:
    logs = [
        event("A", 0, "10.0.0.2"),
        event("B", 1, "10.0.0.1"),
        event("C", 2, "10.0.0.2"),
        event("D", 3, "10.0.0.1"),
        event("E", 4, "10.0.0.3"),
    ]

    result = engine(logs, top_k=2)

    assert result.top_ips == (("10.0.0.1", 2), ("10.0.0.2", 2))


@pytest.mark.parametrize("engine", [analyze_hash, analyze_sort])
def test_session_timeout_boundary_and_new_session(engine) -> None:
    logs = [
        event("A", 0),
        event("B", 30),
        event("C", 60.01),
    ]

    result = engine(logs, session_timeout=30)

    assert [row.session_id for row in result.events] == [
        "10.0.0.1-s0001",
        "10.0.0.1-s0001",
        "10.0.0.1-s0002",
    ]
    assert result.session_count == 2


@pytest.mark.parametrize("engine", [analyze_hash, analyze_sort])
def test_interleaved_ips_have_independent_session_state(engine) -> None:
    logs = [
        event("A", 0, "ip-A"),
        event("B", 5, "ip-B"),
        event("C", 10, "ip-A"),
        event("D", 50, "ip-B"),
    ]

    result = engine(logs, session_timeout=30)

    assert [row.session_id for row in result.events] == [
        "ip-A-s0001",
        "ip-B-s0001",
        "ip-A-s0001",
        "ip-B-s0002",
    ]
    assert result.session_count == 3


def test_engines_are_equivalent_on_mixed_workload() -> None:
    logs = [
        event("A", 0, "ip-2"),
        event("B", 1, "ip-1"),
        event("A", 2, "ip-2"),
        event("C", 40, "ip-2", status=500),
        event("D", 50, "ip-1"),
    ]

    hash_result = analyze_hash(logs, top_k=2, session_timeout=30)
    sort_result = analyze_sort(logs, top_k=2, session_timeout=30)

    assert equivalent_results(hash_result, sort_result)
    assert hash_result.method == "hash"
    assert sort_result.method == "sort"


@pytest.mark.parametrize("engine", [analyze_hash, analyze_sort])
def test_rejects_unsorted_events(engine) -> None:
    with pytest.raises(ValueError, match="ordered"):
        engine([event("A", 2), event("B", 1)])


@pytest.mark.parametrize("engine", [analyze_hash, analyze_sort])
def test_rejects_invalid_configuration(engine) -> None:
    with pytest.raises(ValueError, match="top_k"):
        engine([], top_k=0)
    with pytest.raises(ValueError, match="session_timeout"):
        engine([], session_timeout=-1)


def test_equivalence_detects_different_results() -> None:
    left = analyze_hash([event("A", 0)])
    right = analyze_sort([event("A", 0), event("B", 1)])

    assert not equivalent_results(left, right)


def test_generator_is_reproducible_and_chronological() -> None:
    first = generate_rows(200, seed=7, duplicate_rate=0.2)
    second = generate_rows(200, seed=7, duplicate_rate=0.2)
    different = generate_rows(200, seed=8, duplicate_rate=0.2)

    assert first == second
    assert first != different
    timestamps = [float(row["timestamp"]) for row in first]
    assert timestamps == sorted(timestamps)
    assert len({row["event_id"] for row in first}) < len(first)


@pytest.mark.parametrize("event_count", [0, -1])
def test_generator_rejects_non_positive_count(event_count: int) -> None:
    with pytest.raises(ValueError):
        generate_rows(event_count)


@pytest.mark.parametrize("duplicate_rate", [-0.1, 1.0, 2.0])
def test_generator_rejects_invalid_duplicate_rate(duplicate_rate: float) -> None:
    with pytest.raises(ValueError):
        generate_rows(10, duplicate_rate=duplicate_rate)


def test_csv_round_trip(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "logs.csv", 50, seed=4)

    logs = read_logs_csv(path)

    assert len(logs) == 50
    assert logs[0].event_id == "evt-000001"
    assert logs[0].timestamp == 0


def test_csv_requires_all_columns(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text("event_id,timestamp\nA,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="columns"):
        read_logs_csv(path)


def test_write_sessionized_csv(tmp_path: Path) -> None:
    result = analyze_hash([event("A", 0), event("B", 1)])

    path = write_sessionized_csv(tmp_path / "sessions.csv", result)

    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert rows[0]["session_id"] == "10.0.0.1-s0001"


def test_small_benchmark_checks_both_engines() -> None:
    path_rows = generate_rows(80, seed=5, duplicate_rate=0.1)
    temp_path = Path("unused")
    del temp_path
    logs = [
        LogEvent(
            row["event_id"],
            float(row["timestamp"]),
            row["ip"],
            row["path"],
            int(row["status"]),
        )
        for row in path_rows
    ]

    rows = run_benchmark(logs, (20, 80), top_k=5)

    assert len(rows) == 4
    assert {row.method for row in rows} == {"hash", "sort"}
    assert all(row.elapsed_ms >= 0 for row in rows)


def test_benchmark_rejects_invalid_sizes() -> None:
    logs = [event("A", 0)]
    with pytest.raises(ValueError):
        run_benchmark(logs, ())
    with pytest.raises(ValueError):
        run_benchmark(logs, (2,))


def test_full_pipeline_writes_artifacts(tmp_path: Path) -> None:
    data_path = write_csv(tmp_path / "data" / "logs.csv", 120, seed=11)
    output_dir = tmp_path / "output"

    result, benchmark = run_analysis(data_path, output_dir, top_k=5)

    assert result.input_count == 120
    assert benchmark
    for filename in (
        "sessionized_events.csv",
        "top_ips.csv",
        "benchmark.csv",
        "REPORT.md",
    ):
        artifact = output_dir / filename
        assert artifact.exists()
        assert artifact.stat().st_size > 0
