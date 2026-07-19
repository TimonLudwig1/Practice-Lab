"""Compare a resizing hash map with an intentionally overfilled fixed table."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns

from hash_map import ChainedHashMap


@dataclass(frozen=True)
class BenchmarkRow:
    """One build-and-lookup measurement."""

    strategy: str
    item_count: int
    capacity: int
    load_factor: float
    max_chain_length: int
    build_ns_per_item: float
    lookup_ns_per_item: float


def _measure_strategy(
    item_count: int, *, enable_rehash: bool, repetitions: int, initial_capacity: int
) -> BenchmarkRow:
    mapping: ChainedHashMap[int, int] = ChainedHashMap(
        initial_capacity=initial_capacity,
        enable_rehash=enable_rehash,
    )
    start = perf_counter_ns()
    for key in range(item_count):
        mapping.put(key, key * 2)
    build_elapsed = perf_counter_ns() - start

    checksum = 0
    start = perf_counter_ns()
    for _ in range(repetitions):
        for key in range(item_count):
            checksum += mapping.get(key)
    lookup_elapsed = perf_counter_ns() - start
    assert checksum >= 0

    stats = mapping.stats()
    return BenchmarkRow(
        strategy="rehashing" if enable_rehash else "fixed_capacity",
        item_count=item_count,
        capacity=stats.capacity,
        load_factor=stats.load_factor,
        max_chain_length=stats.max_chain_length,
        build_ns_per_item=build_elapsed / item_count,
        lookup_ns_per_item=lookup_elapsed / (item_count * repetitions),
    )


def run_benchmark(
    sizes: tuple[int, ...] = (100, 500, 1_000, 2_500),
    *,
    repetitions: int = 5,
    initial_capacity: int = 16,
) -> list[BenchmarkRow]:
    """Measure both strategies for every requested table size."""
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("sizes must contain positive integers")
    if repetitions <= 0:
        raise ValueError("repetitions must be greater than zero")

    rows: list[BenchmarkRow] = []
    for size in sizes:
        rows.append(
            _measure_strategy(
                size,
                enable_rehash=True,
                repetitions=repetitions,
                initial_capacity=initial_capacity,
            )
        )
        rows.append(
            _measure_strategy(
                size,
                enable_rehash=False,
                repetitions=repetitions,
                initial_capacity=initial_capacity,
            )
        )
    return rows


def write_csv(path: Path, rows: list[BenchmarkRow]) -> Path:
    """Write benchmark rows with stable LF line endings."""
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


def main() -> None:
    """Run the default benchmark and print a compact comparison."""
    rows = run_benchmark()
    output_path = write_csv(Path(__file__).parent / "output" / "benchmark.csv", rows)
    print(
        "strategy       items capacity load_factor max_chain "
        "build_ns/item lookup_ns/item"
    )
    print("-" * 85)
    for row in rows:
        print(
            f"{row.strategy:<15}{row.item_count:>6}{row.capacity:>9}"
            f"{row.load_factor:>12.2f}{row.max_chain_length:>10}"
            f"{row.build_ns_per_item:>14.1f}{row.lookup_ns_per_item:>15.1f}"
        )
    print(f"\nCSV written to {output_path}")


if __name__ == "__main__":
    main()
