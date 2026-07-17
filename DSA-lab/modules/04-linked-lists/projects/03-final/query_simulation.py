"""Reproducible cache-aside simulation for an expensive data query."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from lru_cache import LRUCache


DEFAULT_SEED = 20_260_717


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Measurements from equivalent uncached and cached workloads."""

    requests: int
    key_space: int
    hot_key_count: int
    hot_probability: float
    capacity: int
    rounds_per_query: int
    seed: int
    hits: int
    misses: int
    hit_rate: float
    source_calls: int
    uncached_seconds: float
    cached_seconds: float
    speedup: float
    final_lru_to_mru: tuple[int, ...]


class ExpensiveDataSource:
    """Deterministic CPU-bound stand-in for a slow database or API query."""

    def __init__(self, rounds: int = 4_000) -> None:
        if isinstance(rounds, bool) or not isinstance(rounds, int):
            raise TypeError("rounds must be an integer")
        if rounds <= 0:
            raise ValueError("rounds must be greater than zero")
        self.rounds = rounds
        self.calls = 0

    def query(self, key: int) -> int:
        """Return a deterministic checksum after deliberately expensive work."""

        self.calls += 1
        state = ((key + 1) * 2_654_435_761) & 0xFFFFFFFF
        for index in range(self.rounds):
            state = (state * 1_664_525 + 1_013_904_223 + index) & 0xFFFFFFFF
        return state


def generate_workload(
    request_count: int = 1_000,
    key_space: int = 100,
    hot_key_count: int = 10,
    hot_probability: float = 0.8,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """Generate a seeded workload with a configurable hot-key cluster."""

    for name, value in (("request_count", request_count), ("key_space", key_space), ("hot_key_count", hot_key_count)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if request_count <= 0:
        raise ValueError("request_count must be greater than zero")
    if key_space <= 0:
        raise ValueError("key_space must be greater than zero")
    if not 1 <= hot_key_count <= key_space:
        raise ValueError("hot_key_count must be between one and key_space")
    if isinstance(hot_probability, bool) or not isinstance(hot_probability, (int, float)):
        raise TypeError("hot_probability must be numeric")
    if not 0.0 <= hot_probability <= 1.0:
        raise ValueError("hot_probability must be between zero and one")

    rng = random.Random(seed)
    hot_keys = range(hot_key_count)
    cold_keys = range(hot_key_count, key_space)
    only_hot_keys = hot_key_count == key_space
    workload: list[int] = []
    for _ in range(request_count):
        use_hot_key = only_hot_keys or rng.random() < hot_probability
        workload.append(rng.choice(hot_keys if use_hot_key else cold_keys))
    return workload


def run_simulation(
    *,
    request_count: int = 1_000,
    key_space: int = 100,
    hot_key_count: int = 10,
    hot_probability: float = 0.8,
    capacity: int = 20,
    rounds: int = 4_000,
    seed: int = DEFAULT_SEED,
) -> SimulationResult:
    """Run identical work once uncached and once through an LRU cache."""

    workload = generate_workload(request_count, key_space, hot_key_count, hot_probability, seed)

    uncached_source = ExpensiveDataSource(rounds)
    started = perf_counter()
    uncached_values = [uncached_source.query(key) for key in workload]
    uncached_seconds = perf_counter() - started

    cached_source = ExpensiveDataSource(rounds)
    cache: LRUCache[int, int] = LRUCache(capacity)
    cached_values: list[int] = []
    started = perf_counter()
    for key in workload:
        result = cache.get_or_compute(key, lambda key=key: cached_source.query(key))
        cached_values.append(result.value)
    cached_seconds = perf_counter() - started

    if cached_values != uncached_values:
        raise AssertionError("cached and uncached query results differ")
    cache.check_invariants()
    stats = cache.stats
    return SimulationResult(
        requests=request_count,
        key_space=key_space,
        hot_key_count=hot_key_count,
        hot_probability=float(hot_probability),
        capacity=capacity,
        rounds_per_query=rounds,
        seed=seed,
        hits=stats.hits,
        misses=stats.misses,
        hit_rate=stats.hit_rate,
        source_calls=cached_source.calls,
        uncached_seconds=uncached_seconds,
        cached_seconds=cached_seconds,
        speedup=uncached_seconds / cached_seconds if cached_seconds else float("inf"),
        final_lru_to_mru=cache.keys_lru_to_mru(),
    )


def write_result_csv(result: SimulationResult, path: str | Path) -> Path:
    """Write one simulation result as a machine-readable CSV row."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    row["final_lru_to_mru"] = " ".join(map(str, result.final_lru_to_mru))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1_000)
    parser.add_argument("--key-space", type=int, default=100)
    parser.add_argument("--hot-keys", type=int, default=10)
    parser.add_argument("--hot-probability", type=float, default=0.8)
    parser.add_argument("--capacity", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("results/cache_simulation.csv"))
    return parser


def main() -> None:
    """Run the configured experiment and print its central measurements."""

    args = build_parser().parse_args()
    result = run_simulation(
        request_count=args.requests,
        key_space=args.key_space,
        hot_key_count=args.hot_keys,
        hot_probability=args.hot_probability,
        capacity=args.capacity,
        rounds=args.rounds,
        seed=args.seed,
    )
    destination = write_result_csv(result, args.output)
    print(f"Requests:       {result.requests}")
    print(f"Hits / misses:  {result.hits} / {result.misses}")
    print(f"Hit rate:       {result.hit_rate:.2%}")
    print(f"Source calls:   {result.source_calls}")
    print(f"Uncached:       {result.uncached_seconds:.4f} s")
    print(f"Cached:         {result.cached_seconds:.4f} s")
    print(f"Speedup:        {result.speedup:.2f}x")
    print(f"CSV:            {destination}")


if __name__ == "__main__":
    main()
