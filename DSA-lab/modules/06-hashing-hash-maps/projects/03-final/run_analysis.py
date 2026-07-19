"""Generate logs, validate both engines and write analysis artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

from data.generate_data import DEFAULT_EVENT_COUNT, DEFAULT_SEED, write_csv
from log_engine import (
    AnalysisResult,
    LogEvent,
    analyze_hash,
    analyze_sort,
    equivalent_results,
    read_logs_csv,
    write_sessionized_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "server_logs.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"


@dataclass(frozen=True)
class BenchmarkRow:
    method: str
    event_count: int
    elapsed_ms: float
    duplicate_count: int
    session_count: int


def _measure(
    method: str,
    function: Callable[..., AnalysisResult],
    events: list[LogEvent],
    *,
    top_k: int,
    session_timeout: float,
) -> tuple[AnalysisResult, BenchmarkRow]:
    start = perf_counter_ns()
    result = function(events, top_k=top_k, session_timeout=session_timeout)
    elapsed_ms = (perf_counter_ns() - start) / 1_000_000
    return result, BenchmarkRow(
        method=method,
        event_count=len(events),
        elapsed_ms=elapsed_ms,
        duplicate_count=result.duplicate_count,
        session_count=result.session_count,
    )


def run_benchmark(
    events: list[LogEvent],
    sizes: tuple[int, ...] = (500, 1_000, 2_000, 4_000),
    *,
    top_k: int = 10,
    session_timeout: float = 30.0,
) -> list[BenchmarkRow]:
    """Measure both engines on growing prefixes and assert equivalence."""
    if not sizes or any(size <= 0 or size > len(events) for size in sizes):
        raise ValueError("sizes must be positive and no larger than the input")
    rows: list[BenchmarkRow] = []
    for size in sizes:
        sample = events[:size]
        hash_result, hash_row = _measure(
            "hash",
            analyze_hash,
            sample,
            top_k=top_k,
            session_timeout=session_timeout,
        )
        sort_result, sort_row = _measure(
            "sort",
            analyze_sort,
            sample,
            top_k=top_k,
            session_timeout=session_timeout,
        )
        assert equivalent_results(hash_result, sort_result)
        rows.extend((hash_row, sort_row))
    return rows


def write_benchmark_csv(path: Path, rows: list[BenchmarkRow]) -> Path:
    if not rows:
        raise ValueError("rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(asdict(rows[0])),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def write_top_ips_csv(path: Path, result: AnalysisResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["rank", "ip", "event_count"])
        for rank, (ip, count) in enumerate(result.top_ips, start=1):
            writer.writerow([rank, ip, count])
    return path


def write_report(
    path: Path,
    result: AnalysisResult,
    benchmark: list[BenchmarkRow],
    *,
    session_timeout: float,
) -> Path:
    largest_size = max(row.event_count for row in benchmark)
    final_rows = [row for row in benchmark if row.event_count == largest_size]
    time_by_method = {row.method: row.elapsed_ms for row in final_rows}
    speedup = time_by_method["sort"] / time_by_method["hash"]
    top_ip, top_count = result.top_ips[0]
    content = f"""# Ergebnisbericht: Hash-basierte Log-Analyse

Die Engine verarbeitet {result.input_count} chronologisch sortierte Logzeilen.
Nach Event-ID-Deduplizierung bleiben {result.unique_count} eindeutige Ereignisse;
{result.duplicate_count} Zeilen wurden als Wiederholung entfernt.

Bei einem Inaktivitätsfenster von {session_timeout:.0f} Sekunden entstehen
{result.session_count} Sessions. Die häufigste IP ist `{top_ip}` mit {top_count}
eindeutigen Ereignissen.

## Vergleich mit der sortierbasierten Referenz

Beide Engines liefern für jeden Benchmark-Präfix exakt dieselben Top-K-Werte,
Duplikate und Session-Zuordnungen. Für {largest_size} Eingabezeilen benötigt die
Hash-Engine {time_by_method['hash']:.3f} ms, die sortierbasierte Referenz
{time_by_method['sort']:.3f} ms. Das entspricht in diesem Lauf einem Faktor von
{speedup:.2f} zugunsten der Hash-Variante.

Absolute Zeiten sind systemabhängig. Der strukturelle Unterschied ist stabil:
Die Hash-Engine aggregiert und sessionisiert in einem erwarteten O(n)-Durchlauf;
die Referenz sortiert mehrfach und benötigt O(n log n). Die abschließende
Rangfolge der unterschiedlichen IPs wird in beiden Fällen sortiert.
"""
    path.write_text(content, encoding="utf-8")
    return path


def run_analysis(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    event_count: int = DEFAULT_EVENT_COUNT,
    seed: int = DEFAULT_SEED,
    top_k: int = 10,
    session_timeout: float = 30.0,
) -> tuple[AnalysisResult, list[BenchmarkRow]]:
    """Run the deterministic end-to-end analysis pipeline."""
    if not data_path.exists():
        write_csv(data_path, event_count, seed=seed)
    events = read_logs_csv(data_path)
    hash_result = analyze_hash(
        events, top_k=top_k, session_timeout=session_timeout
    )
    sort_result = analyze_sort(
        events, top_k=top_k, session_timeout=session_timeout
    )
    if not equivalent_results(hash_result, sort_result):
        raise AssertionError("hash and sort engines produced different results")

    default_sizes = tuple(
        size for size in (500, 1_000, 2_000, 4_000) if size <= len(events)
    )
    if not default_sizes or default_sizes[-1] != len(events):
        default_sizes = (*default_sizes, len(events))
    benchmark = run_benchmark(
        events,
        default_sizes,
        top_k=top_k,
        session_timeout=session_timeout,
    )

    write_sessionized_csv(output_dir / "sessionized_events.csv", hash_result)
    write_top_ips_csv(output_dir / "top_ips.csv", hash_result)
    write_benchmark_csv(output_dir / "benchmark.csv", benchmark)
    write_report(
        output_dir / "REPORT.md",
        hash_result,
        benchmark,
        session_timeout=session_timeout,
    )
    return hash_result, benchmark


def main() -> None:
    result, benchmark = run_analysis()
    print(
        f"Input={result.input_count}, unique={result.unique_count}, "
        f"duplicates={result.duplicate_count}, sessions={result.session_count}"
    )
    print("\nTop IPs")
    for rank, (ip, count) in enumerate(result.top_ips, start=1):
        print(f"{rank:>2}. {ip:<16} {count:>5}")
    print("\nBenchmark")
    for row in benchmark:
        print(f"{row.method:<5} n={row.event_count:>4}: {row.elapsed_ms:>8.3f} ms")
    print(f"\nArtifacts written to {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
