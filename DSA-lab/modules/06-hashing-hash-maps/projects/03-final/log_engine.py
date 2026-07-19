"""Hash- and sort-based engines for server-log analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LogEvent:
    event_id: str
    timestamp: float
    ip: str
    path: str
    status: int

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if self.timestamp < 0:
            raise ValueError("timestamp must not be negative")
        if not self.ip:
            raise ValueError("ip must not be empty")
        if not self.path:
            raise ValueError("path must not be empty")
        if not 100 <= self.status <= 599:
            raise ValueError("status must be between 100 and 599")


@dataclass(frozen=True)
class SessionizedEvent:
    event: LogEvent
    session_id: str


@dataclass(frozen=True)
class AnalysisResult:
    method: str
    input_count: int
    duplicate_count: int
    unique_count: int
    session_count: int
    top_ips: tuple[tuple[str, int], ...]
    events: tuple[SessionizedEvent, ...]


def _validate_configuration(
    events: list[LogEvent], top_k: int, session_timeout: float
) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if session_timeout < 0:
        raise ValueError("session_timeout must not be negative")
    for previous, current in zip(events, events[1:]):
        if current.timestamp < previous.timestamp:
            raise ValueError("events must be ordered by timestamp")


def _rank_counts(counts: list[tuple[str, int]], top_k: int) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counts, key=lambda item: (-item[1], item[0]))[:top_k])


def analyze_hash(
    events: list[LogEvent], *, top_k: int = 10, session_timeout: float = 30.0
) -> AnalysisResult:
    """Analyze logs in one pass with sets and dictionaries."""
    _validate_configuration(events, top_k, session_timeout)
    seen_ids: set[str] = set()
    count_by_ip: dict[str, int] = {}
    session_state: dict[str, tuple[float, int]] = {}
    sessionized: list[SessionizedEvent] = []
    duplicate_count = 0
    session_count = 0

    for event in events:
        if event.event_id in seen_ids:
            duplicate_count += 1
            continue
        seen_ids.add(event.event_id)
        count_by_ip[event.ip] = count_by_ip.get(event.ip, 0) + 1

        previous = session_state.get(event.ip)
        if previous is None or event.timestamp - previous[0] > session_timeout:
            ordinal = 1 if previous is None else previous[1] + 1
            session_count += 1
        else:
            ordinal = previous[1]
        session_state[event.ip] = (event.timestamp, ordinal)
        sessionized.append(
            SessionizedEvent(event, f"{event.ip}-s{ordinal:04d}")
        )

    top_ips = _rank_counts(list(count_by_ip.items()), top_k)
    return AnalysisResult(
        method="hash",
        input_count=len(events),
        duplicate_count=duplicate_count,
        unique_count=len(sessionized),
        session_count=session_count,
        top_ips=top_ips,
        events=tuple(sessionized),
    )


def analyze_sort(
    events: list[LogEvent], *, top_k: int = 10, session_timeout: float = 30.0
) -> AnalysisResult:
    """Analyze the same workload through sorting and run-length grouping."""
    _validate_configuration(events, top_k, session_timeout)

    indexed = list(enumerate(events))
    by_id = sorted(indexed, key=lambda pair: (pair[1].event_id, pair[0]))
    unique_indexed: list[tuple[int, LogEvent]] = []
    previous_id: str | None = None
    for original_index, event in by_id:
        if event.event_id != previous_id:
            unique_indexed.append((original_index, event))
            previous_id = event.event_id
    unique_indexed.sort(key=lambda pair: pair[0])

    sorted_ips = sorted(event.ip for _, event in unique_indexed)
    counts: list[tuple[str, int]] = []
    for ip in sorted_ips:
        if counts and counts[-1][0] == ip:
            previous_ip, count = counts[-1]
            counts[-1] = (previous_ip, count + 1)
        else:
            counts.append((ip, 1))

    grouped = sorted(
        unique_indexed,
        key=lambda pair: (pair[1].ip, pair[1].timestamp, pair[0]),
    )
    sessionized_indexed: list[tuple[int, SessionizedEvent]] = []
    previous_ip: str | None = None
    previous_timestamp = 0.0
    ordinal = 0
    session_count = 0
    for original_index, event in grouped:
        if event.ip != previous_ip:
            ordinal = 1
            session_count += 1
        elif event.timestamp - previous_timestamp > session_timeout:
            ordinal += 1
            session_count += 1
        sessionized_indexed.append(
            (original_index, SessionizedEvent(event, f"{event.ip}-s{ordinal:04d}"))
        )
        previous_ip = event.ip
        previous_timestamp = event.timestamp
    sessionized_indexed.sort(key=lambda pair: pair[0])
    sessionized = tuple(row for _, row in sessionized_indexed)

    return AnalysisResult(
        method="sort",
        input_count=len(events),
        duplicate_count=len(events) - len(unique_indexed),
        unique_count=len(unique_indexed),
        session_count=session_count,
        top_ips=_rank_counts(counts, top_k),
        events=sessionized,
    )


def equivalent_results(left: AnalysisResult, right: AnalysisResult) -> bool:
    """Return whether two engines produced the same domain result."""
    return (
        left.input_count == right.input_count
        and left.duplicate_count == right.duplicate_count
        and left.unique_count == right.unique_count
        and left.session_count == right.session_count
        and left.top_ips == right.top_ips
        and left.events == right.events
    )


def read_logs_csv(path: Path) -> list[LogEvent]:
    """Read validated log events from CSV."""
    required = {"event_id", "timestamp", "ip", "path", "status"}
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        events: list[LogEvent] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                events.append(
                    LogEvent(
                        event_id=row["event_id"],
                        timestamp=float(row["timestamp"]),
                        ip=row["ip"],
                        path=row["path"],
                        status=int(row["status"]),
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid log at CSV line {line_number}: {error}") from error
    return events


def write_sessionized_csv(path: Path, result: AnalysisResult) -> Path:
    """Write unique events with their assigned sessions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["event_id", "timestamp", "ip", "path", "status", "session_id"]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in result.events:
            writer.writerow(
                {
                    "event_id": row.event.event_id,
                    "timestamp": f"{row.event.timestamp:.4f}",
                    "ip": row.event.ip,
                    "path": row.event.path,
                    "status": row.event.status,
                    "session_id": row.session_id,
                }
            )
    return path
