"""Stable external sorting for CSV files with explicit resource limits."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, TextIO


@dataclass(frozen=True)
class SortMetrics:
    """Observable resource and pipeline metrics."""

    record_count: int
    initial_runs: int
    merge_passes: int
    runs_by_pass: tuple[int, ...]
    max_chunk_records: int
    max_heap_entries: int


@dataclass(frozen=True)
class VerificationResult:
    """Result of a streaming output verification."""

    record_count: int
    is_sorted: bool
    same_records: bool

    @property
    def ok(self) -> bool:
        """Return whether all verification conditions hold."""

        return self.is_sorted and self.same_records


DecoratedRecord = tuple[int, dict[str, str]]


def _validate_configuration(
    key_fields: Sequence[str], memory_limit_records: int, merge_fan_in: int
) -> tuple[str, ...]:
    fields = tuple(key_fields)
    if not fields or any(not field for field in fields):
        raise ValueError("key_fields must contain at least one non-empty field")
    if len(set(fields)) != len(fields):
        raise ValueError("key_fields must not contain duplicates")
    if memory_limit_records < 1:
        raise ValueError("memory_limit_records must be at least 1")
    if merge_fan_in < 2:
        raise ValueError("merge_fan_in must be at least 2")
    return fields


def _record_key(
    record: DecoratedRecord, key_fields: Sequence[str]
) -> tuple[str | int, ...]:
    ordinal, row = record
    return (*[row[field] for field in key_fields], ordinal)


def _write_run(path: Path, records: Sequence[DecoratedRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for ordinal, row in records:
            handle.write(
                json.dumps(
                    {"ordinal": ordinal, "row": row},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def _read_run(path: Path) -> Iterator[DecoratedRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
                yield int(payload["ordinal"]), dict(payload["row"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid temporary run {path.name} at line {line_number}"
                ) from error


def _merge_group(
    input_paths: Sequence[Path], output_path: Path, key_fields: Sequence[str]
) -> int:
    iterators = [iter(_read_run(path)) for path in input_paths]
    heap: list[tuple[tuple[str | int, ...], int, DecoratedRecord]] = []

    for run_index, iterator in enumerate(iterators):
        record = next(iterator, None)
        if record is not None:
            heapq.heappush(
                heap, (_record_key(record, key_fields), run_index, record)
            )

    max_heap_entries = len(heap)
    with output_path.open("w", encoding="utf-8") as output:
        while heap:
            _, run_index, record = heapq.heappop(heap)
            ordinal, row = record
            output.write(
                json.dumps(
                    {"ordinal": ordinal, "row": row},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            output.write("\n")

            successor = next(iterators[run_index], None)
            if successor is not None:
                heapq.heappush(
                    heap,
                    (_record_key(successor, key_fields), run_index, successor),
                )
                max_heap_entries = max(max_heap_entries, len(heap))

    return max_heap_entries


def _validated_header(reader: csv.DictReader[str], key_fields: Sequence[str]) -> list[str]:
    if reader.fieldnames is None:
        raise ValueError("input CSV must contain a header")
    fieldnames = list(reader.fieldnames)
    if any(not field for field in fieldnames):
        raise ValueError("input CSV contains an empty column name")
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError("input CSV contains duplicate column names")
    missing = [field for field in key_fields if field not in fieldnames]
    if missing:
        raise ValueError(f"sort key columns missing from input: {', '.join(missing)}")
    return fieldnames


def _validated_row(
    row: dict[str | None, str | list[str] | None],
    fieldnames: Sequence[str],
    line_number: int,
) -> dict[str, str]:
    if None in row or any(row.get(field) is None for field in fieldnames):
        raise ValueError(f"malformed CSV row at line {line_number}")
    return {field: str(row[field]) for field in fieldnames}


def external_sort_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    key_fields: Sequence[str],
    memory_limit_records: int,
    merge_fan_in: int = 8,
    temporary_directory: str | Path | None = None,
) -> SortMetrics:
    """Sort a CSV file stably while bounding chunk and merge memory.

    The original zero-based record position is used as a final hidden key. This
    preserves input order for equal public keys across chunk boundaries.
    """

    keys = _validate_configuration(
        key_fields, memory_limit_records, merge_fan_in
    )
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(temporary_directory) if temporary_directory else None
    if temp_parent is not None:
        temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="external-sort-", dir=temp_parent
    ) as workspace_name:
        workspace = Path(workspace_name)
        runs: list[Path] = []
        chunk: list[DecoratedRecord] = []
        record_count = 0
        max_chunk_records = 0

        with source.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            fieldnames = _validated_header(reader, keys)
            for ordinal, raw_row in enumerate(reader):
                row = _validated_row(raw_row, fieldnames, ordinal + 2)
                chunk.append((ordinal, row))
                record_count += 1
                max_chunk_records = max(max_chunk_records, len(chunk))
                if len(chunk) == memory_limit_records:
                    chunk.sort(key=lambda record: _record_key(record, keys))
                    run_path = workspace / f"initial_{len(runs):06d}.jsonl"
                    _write_run(run_path, chunk)
                    runs.append(run_path)
                    chunk = []

        if chunk:
            chunk.sort(key=lambda record: _record_key(record, keys))
            run_path = workspace / f"initial_{len(runs):06d}.jsonl"
            _write_run(run_path, chunk)
            runs.append(run_path)

        initial_runs = len(runs)
        runs_by_pass = [initial_runs]
        merge_passes = 0
        max_heap_entries = 0

        while len(runs) > 1:
            merge_passes += 1
            next_runs: list[Path] = []
            for group_start in range(0, len(runs), merge_fan_in):
                group = runs[group_start : group_start + merge_fan_in]
                merged_path = workspace / (
                    f"pass_{merge_passes:03d}_{len(next_runs):06d}.jsonl"
                )
                heap_entries = _merge_group(group, merged_path, keys)
                max_heap_entries = max(max_heap_entries, heap_entries)
                next_runs.append(merged_path)
            for old_run in runs:
                old_run.unlink()
            runs = next_runs
            runs_by_pass.append(len(runs))

        with destination.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            if runs:
                for _, row in _read_run(runs[0]):
                    writer.writerow(row)

    return SortMetrics(
        record_count=record_count,
        initial_runs=initial_runs,
        merge_passes=merge_passes,
        runs_by_pass=tuple(runs_by_pass),
        max_chunk_records=max_chunk_records,
        max_heap_entries=max_heap_entries,
    )


def _row_digest(row: dict[str, str], fieldnames: Sequence[str]) -> int:
    payload = json.dumps(
        [row[field] for field in fieldnames],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _stream_fingerprint(
    path: Path,
) -> tuple[list[str], int, int, int]:
    modulus = 1 << 256
    count = 0
    digest_sum = 0
    digest_xor = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = _validated_header(reader, ())
        for line_number, raw_row in enumerate(reader, start=2):
            row = _validated_row(raw_row, fieldnames, line_number)
            digest = _row_digest(row, fieldnames)
            count += 1
            digest_sum = (digest_sum + digest) % modulus
            digest_xor ^= digest
    return fieldnames, count, digest_sum, digest_xor


def verify_sorted_output(
    input_path: str | Path,
    output_path: str | Path,
    *,
    key_fields: Sequence[str],
) -> VerificationResult:
    """Verify order and record preservation without loading either CSV fully."""

    keys = tuple(key_fields)
    source_fingerprint = _stream_fingerprint(Path(input_path))
    output_fingerprint = _stream_fingerprint(Path(output_path))
    source_header, source_count, source_sum, source_xor = source_fingerprint
    output_header, output_count, output_sum, output_xor = output_fingerprint

    missing = [field for field in keys if field not in output_header]
    if not keys or missing:
        raise ValueError("all key_fields must exist in the output CSV")

    is_sorted = True
    previous_key: tuple[str, ...] | None = None
    with Path(output_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            current_key = tuple(str(row[field]) for field in keys)
            if previous_key is not None and current_key < previous_key:
                is_sorted = False
                break
            previous_key = current_key

    same_records = (
        source_header == output_header
        and source_count == output_count
        and source_sum == output_sum
        and source_xor == output_xor
    )
    return VerificationResult(output_count, is_sorted, same_records)
