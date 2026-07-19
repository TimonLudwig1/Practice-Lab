"""Tests for the external sorting pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from external_sort import external_sort_csv, verify_sorted_output
from generate_data import FIELDNAMES, generate_events


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def event(index: int, timestamp: str, sensor: str = "sensor-001") -> dict[str, str]:
    return {
        "event_id": f"evt-{index:08d}",
        "timestamp": timestamp,
        "sensor_id": sensor,
        "temperature_c": "20.00",
        "quality": "good",
        "payload": f"source, {index}",
    }


@pytest.mark.parametrize("memory_limit", [1, 2, 3, 8, 50])
@pytest.mark.parametrize("fan_in", [2, 3, 8])
def test_matches_stable_builtin_sort(
    tmp_path: Path, memory_limit: int, fan_in: int
) -> None:
    source = generate_events(tmp_path / "input.csv", record_count=47, seed=123)
    destination = tmp_path / "output.csv"
    original = read_csv(source)

    metrics = external_sort_csv(
        source,
        destination,
        key_fields=("timestamp", "sensor_id"),
        memory_limit_records=memory_limit,
        merge_fan_in=fan_in,
    )

    assert read_csv(destination) == sorted(
        original, key=lambda row: (row["timestamp"], row["sensor_id"])
    )
    assert metrics.record_count == 47
    assert metrics.max_chunk_records <= memory_limit
    assert metrics.max_heap_entries <= fan_in


def test_stability_across_chunk_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    destination = tmp_path / "output.csv"
    rows = [event(index, "2026-01-01T00:00:00Z") for index in range(12)]
    write_csv(source, rows)

    external_sort_csv(
        source,
        destination,
        key_fields=("timestamp", "sensor_id"),
        memory_limit_records=2,
        merge_fan_in=3,
    )

    assert [row["event_id"] for row in read_csv(destination)] == [
        row["event_id"] for row in rows
    ]


def test_multiple_key_fields_are_applied_in_order(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    destination = tmp_path / "output.csv"
    rows = [
        event(0, "2026-01-02T00:00:00Z", "sensor-001"),
        event(1, "2026-01-01T00:00:00Z", "sensor-002"),
        event(2, "2026-01-01T00:00:00Z", "sensor-001"),
    ]
    write_csv(source, rows)

    external_sort_csv(
        source,
        destination,
        key_fields=("timestamp", "sensor_id"),
        memory_limit_records=1,
        merge_fan_in=2,
    )

    assert [row["event_id"] for row in read_csv(destination)] == [
        "evt-00000002",
        "evt-00000001",
        "evt-00000000",
    ]


def test_empty_csv_keeps_header(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    destination = tmp_path / "output.csv"
    write_csv(source, [])

    metrics = external_sort_csv(
        source,
        destination,
        key_fields=("timestamp",),
        memory_limit_records=5,
    )

    assert read_csv(destination) == []
    assert destination.read_text(encoding="utf-8").startswith("event_id,")
    assert metrics.record_count == 0
    assert metrics.initial_runs == 0
    assert metrics.runs_by_pass == (0,)


def test_single_run_needs_no_merge(tmp_path: Path) -> None:
    source = generate_events(tmp_path / "input.csv", record_count=7, seed=1)
    destination = tmp_path / "output.csv"
    metrics = external_sort_csv(
        source,
        destination,
        key_fields=("timestamp",),
        memory_limit_records=10,
        merge_fan_in=2,
    )
    assert metrics.initial_runs == 1
    assert metrics.merge_passes == 0
    assert metrics.max_heap_entries == 0


def test_expected_number_of_runs_and_passes(tmp_path: Path) -> None:
    source = generate_events(tmp_path / "input.csv", record_count=25, seed=2)
    destination = tmp_path / "output.csv"
    metrics = external_sort_csv(
        source,
        destination,
        key_fields=("timestamp",),
        memory_limit_records=4,
        merge_fan_in=3,
    )
    assert metrics.initial_runs == 7
    assert metrics.runs_by_pass == (7, 3, 1)
    assert metrics.merge_passes == 2


def test_csv_quoting_and_unicode_survive(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    destination = tmp_path / "output.csv"
    rows = [event(0, "2026-01-02T00:00:00Z"), event(1, "2026-01-01T00:00:00Z")]
    rows[0]["payload"] = 'München, "Nord"'
    write_csv(source, rows)
    external_sort_csv(
        source,
        destination,
        key_fields=("timestamp",),
        memory_limit_records=1,
        merge_fan_in=2,
    )
    assert read_csv(destination)[1]["payload"] == 'München, "Nord"'


@pytest.mark.parametrize(
    ("key_fields", "memory_limit", "fan_in"),
    [
        ((), 2, 2),
        (("timestamp", "timestamp"), 2, 2),
        (("timestamp",), 0, 2),
        (("timestamp",), -1, 2),
        (("timestamp",), 2, 1),
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path,
    key_fields: tuple[str, ...],
    memory_limit: int,
    fan_in: int,
) -> None:
    source = generate_events(tmp_path / "input.csv", record_count=1)
    with pytest.raises(ValueError):
        external_sort_csv(
            source,
            tmp_path / "output.csv",
            key_fields=key_fields,
            memory_limit_records=memory_limit,
            merge_fan_in=fan_in,
        )


def test_missing_sort_column_is_rejected(tmp_path: Path) -> None:
    source = generate_events(tmp_path / "input.csv", record_count=1)
    with pytest.raises(ValueError, match="missing"):
        external_sort_csv(
            source,
            tmp_path / "output.csv",
            key_fields=("unknown",),
            memory_limit_records=2,
        )


@pytest.mark.parametrize(
    "content",
    [
        "",
        "event_id,timestamp,timestamp\n1,a,b\n",
        "event_id,timestamp\n1\n",
        "event_id,timestamp\n1,a,extra\n",
    ],
)
def test_malformed_csv_is_rejected(tmp_path: Path, content: str) -> None:
    source = tmp_path / "input.csv"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        external_sort_csv(
            source,
            tmp_path / "output.csv",
            key_fields=("timestamp",),
            memory_limit_records=2,
        )


def test_input_can_safely_be_replaced(tmp_path: Path) -> None:
    path = generate_events(tmp_path / "events.csv", record_count=20, seed=9)
    expected = sorted(read_csv(path), key=lambda row: row["timestamp"])
    external_sort_csv(
        path,
        path,
        key_fields=("timestamp",),
        memory_limit_records=3,
        merge_fan_in=2,
    )
    assert read_csv(path) == expected


def test_verifier_accepts_correct_output(tmp_path: Path) -> None:
    source = generate_events(tmp_path / "input.csv", record_count=31, seed=5)
    destination = tmp_path / "output.csv"
    external_sort_csv(
        source,
        destination,
        key_fields=("timestamp", "sensor_id"),
        memory_limit_records=4,
    )
    result = verify_sorted_output(
        source, destination, key_fields=("timestamp", "sensor_id")
    )
    assert result.ok
    assert result.record_count == 31


def test_verifier_detects_wrong_order(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    rows = [event(0, "2026-01-02T00:00:00Z"), event(1, "2026-01-01T00:00:00Z")]
    write_csv(source, rows)
    write_csv(output, rows)
    result = verify_sorted_output(source, output, key_fields=("timestamp",))
    assert not result.is_sorted
    assert result.same_records
    assert not result.ok


def test_verifier_detects_changed_record(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    rows = [event(0, "2026-01-01T00:00:00Z")]
    write_csv(source, rows)
    changed = [dict(rows[0])]
    changed[0]["quality"] = "suspect"
    write_csv(output, changed)
    result = verify_sorted_output(source, output, key_fields=("timestamp",))
    assert result.is_sorted
    assert not result.same_records


def test_generator_is_reproducible_and_stream_friendly(tmp_path: Path) -> None:
    first = generate_events(tmp_path / "first.csv", record_count=50, seed=42)
    second = generate_events(tmp_path / "second.csv", record_count=50, seed=42)
    third = generate_events(tmp_path / "third.csv", record_count=50, seed=43)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() != third.read_bytes()
    assert len(read_csv(first)) == 50


def test_generator_rejects_negative_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_events(tmp_path / "input.csv", record_count=-1)
