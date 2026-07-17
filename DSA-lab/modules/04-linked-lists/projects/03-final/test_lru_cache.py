"""Tests for the LRU cache and its reproducible query simulation."""

from __future__ import annotations

import csv
import random
from collections import OrderedDict

import pytest

from lru_cache import CacheStats, Evicted, LRUCache
from query_simulation import (
    DEFAULT_SEED,
    ExpensiveDataSource,
    generate_workload,
    run_simulation,
    write_result_csv,
)


@pytest.mark.parametrize("capacity", [0, -1, -20])
def test_capacity_must_be_positive(capacity: int) -> None:
    with pytest.raises(ValueError):
        LRUCache(capacity)


@pytest.mark.parametrize("capacity", [True, 2.5, "3", None])
def test_capacity_must_be_an_integer(capacity: object) -> None:
    with pytest.raises(TypeError):
        LRUCache(capacity)  # type: ignore[arg-type]


def test_new_cache_is_empty_and_valid() -> None:
    cache: LRUCache[str, int] = LRUCache(3)

    assert len(cache) == 0
    assert cache.keys_lru_to_mru() == ()
    assert cache.items_lru_to_mru() == ()
    assert cache.stats == CacheStats(0, 0)
    assert cache.check_invariants()


def test_put_and_get_value() -> None:
    cache: LRUCache[str, int] = LRUCache(2)

    assert cache.put("a", 10) is None
    assert cache.get("a") == 10
    assert cache.stats == CacheStats(hits=1, misses=0)


def test_get_missing_key_records_a_miss() -> None:
    cache: LRUCache[str, int] = LRUCache(2)

    with pytest.raises(KeyError, match="missing"):
        cache.get("missing")

    assert cache.stats == CacheStats(hits=0, misses=1)


def test_get_moves_entry_to_mru() -> None:
    cache: LRUCache[str, int] = LRUCache(3)
    for key in "abc":
        cache.put(key, ord(key))

    cache.get("a")

    assert cache.keys_lru_to_mru() == ("b", "c", "a")
    assert cache.check_invariants()


def test_get_of_current_mru_preserves_order() -> None:
    cache: LRUCache[str, int] = LRUCache(3)
    for key in "abc":
        cache.put(key, ord(key))

    cache.get("c")

    assert cache.keys_lru_to_mru() == ("a", "b", "c")


def test_put_evicts_least_recent_entry() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)

    evicted = cache.put("c", 3)

    assert evicted == Evicted("a", 1)
    assert "a" not in cache
    assert cache.items_lru_to_mru() == (("b", 2), ("c", 3))


def test_access_changes_which_entry_is_evicted() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")

    evicted = cache.put("c", 3)

    assert evicted == Evicted("b", 2)
    assert cache.keys_lru_to_mru() == ("a", "c")


def test_update_changes_value_and_moves_entry_to_mru() -> None:
    cache: LRUCache[str, int] = LRUCache(3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    assert cache.put("a", 100) is None

    assert len(cache) == 3
    assert cache.peek("a") == 100
    assert cache.keys_lru_to_mru() == ("b", "c", "a")


def test_capacity_one_repeatedly_evicts_previous_entry() -> None:
    cache: LRUCache[int, str] = LRUCache(1)

    assert cache.put(1, "one") is None
    assert cache.put(2, "two") == Evicted(1, "one")
    assert cache.put(3, "three") == Evicted(2, "two")
    assert cache.items_lru_to_mru() == ((3, "three"),)


def test_none_can_be_a_real_key_and_value() -> None:
    cache: LRUCache[object, object] = LRUCache(2)
    cache.put(None, None)
    cache.put("key", "value")

    assert cache.get(None) is None
    assert cache.keys_lru_to_mru() == ("key", None)
    assert cache.check_invariants()


def test_peek_does_not_change_order_or_statistics() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.peek("a") == 1
    assert cache.keys_lru_to_mru() == ("a", "b")
    assert cache.stats == CacheStats(0, 0)


def test_missing_peek_does_not_record_miss() -> None:
    cache: LRUCache[str, int] = LRUCache(2)

    with pytest.raises(KeyError):
        cache.peek("absent")

    assert cache.stats == CacheStats(0, 0)


def test_delete_returns_value_and_unlinks_node() -> None:
    cache: LRUCache[str, int] = LRUCache(3)
    for key in "abc":
        cache.put(key, ord(key))

    assert cache.delete("b") == ord("b")
    assert cache.keys_lru_to_mru() == ("a", "c")
    assert cache.check_invariants()


def test_delete_missing_key_raises_key_error() -> None:
    cache: LRUCache[str, int] = LRUCache(2)

    with pytest.raises(KeyError):
        cache.delete("absent")


def test_clear_removes_all_entries_and_resets_stats() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    cache.put("a", 1)
    cache.get("a")
    with pytest.raises(KeyError):
        cache.get("b")

    cache.clear()

    assert len(cache) == 0
    assert cache.stats == CacheStats(0, 0)
    assert cache.check_invariants()


def test_clear_can_preserve_stats() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    with pytest.raises(KeyError):
        cache.get("a")

    cache.clear(reset_stats=False)

    assert cache.stats == CacheStats(0, 1)


def test_stats_snapshot_is_immutable_and_stable() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    snapshot = cache.stats
    cache.put("a", 1)
    cache.get("a")

    assert snapshot == CacheStats(0, 0)
    with pytest.raises(AttributeError):
        snapshot.hits = 20  # type: ignore[misc]


def test_hit_rate_before_and_after_requests() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    assert cache.stats.hit_rate == 0.0
    cache.put("a", 1)
    cache.get("a")
    with pytest.raises(KeyError):
        cache.get("b")

    assert cache.stats.requests == 2
    assert cache.stats.hit_rate == pytest.approx(0.5)


def test_get_or_compute_loads_once_then_hits() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    calls = 0

    def loader() -> int:
        nonlocal calls
        calls += 1
        return 42

    first = cache.get_or_compute("answer", loader)
    second = cache.get_or_compute("answer", loader)

    assert (first.value, first.hit, first.evicted) == (42, False, None)
    assert (second.value, second.hit, second.evicted) == (42, True, None)
    assert calls == 1
    assert cache.stats == CacheStats(1, 1)


def test_get_or_compute_reports_eviction() -> None:
    cache: LRUCache[str, int] = LRUCache(1)
    cache.put("old", 1)

    result = cache.get_or_compute("new", lambda: 2)

    assert result.evicted == Evicted("old", 1)
    assert result.hit is False


@pytest.mark.parametrize("value", [0, False, "", None])
def test_get_or_compute_caches_falsey_values(value: object) -> None:
    cache: LRUCache[str, object] = LRUCache(1)

    assert cache.get_or_compute("key", lambda: value).value is value
    assert cache.get_or_compute("key", lambda: object()).value is value


def test_loader_exception_does_not_insert_partial_entry() -> None:
    cache: LRUCache[str, int] = LRUCache(2)

    def failing_loader() -> int:
        raise RuntimeError("source unavailable")

    with pytest.raises(RuntimeError, match="source unavailable"):
        cache.get_or_compute("a", failing_loader)

    assert "a" not in cache
    assert cache.stats == CacheStats(0, 1)
    assert cache.check_invariants()


def test_iteration_repr_and_items_follow_recency_order() -> None:
    cache: LRUCache[str, int] = LRUCache(3)
    cache.put("x", 1)
    cache.put("y", 2)

    assert list(cache) == ["x", "y"]
    assert cache.items_lru_to_mru() == (("x", 1), ("y", 2))
    assert repr(cache) == "LRUCache(capacity=3, lru_to_mru={'x': 1, 'y': 2})"


def test_random_operations_match_ordered_dictionary_reference() -> None:
    rng = random.Random(12_345)
    cache: LRUCache[int, int] = LRUCache(7)
    reference: OrderedDict[int, int] = OrderedDict()

    for _ in range(1_000):
        key = rng.randrange(15)
        if rng.random() < 0.6:
            value = rng.randrange(10_000)
            cache.put(key, value)
            reference[key] = value
            reference.move_to_end(key)
            if len(reference) > cache.capacity:
                reference.popitem(last=False)
        elif key in reference:
            assert cache.get(key) == reference[key]
            reference.move_to_end(key)
        else:
            with pytest.raises(KeyError):
                cache.get(key)

        assert cache.items_lru_to_mru() == tuple(reference.items())
        assert cache.check_invariants()


def test_invariant_check_detects_broken_bidirectional_link() -> None:
    cache: LRUCache[str, int] = LRUCache(2)
    cache.put("a", 1)
    node = cache._nodes["a"]
    node.previous = node

    with pytest.raises(AssertionError):
        cache.check_invariants()


def test_expensive_source_is_deterministic_and_counts_calls() -> None:
    source = ExpensiveDataSource(rounds=5)

    assert source.query(7) == source.query(7)
    assert source.query(7) != source.query(8)
    assert source.calls == 4


@pytest.mark.parametrize("rounds", [0, -5])
def test_expensive_source_rejects_non_positive_rounds(rounds: int) -> None:
    with pytest.raises(ValueError):
        ExpensiveDataSource(rounds)


def test_workload_is_reproducible_and_seed_sensitive() -> None:
    first = generate_workload(100, 20, 4, 0.8, DEFAULT_SEED)
    second = generate_workload(100, 20, 4, 0.8, DEFAULT_SEED)
    different = generate_workload(100, 20, 4, 0.8, DEFAULT_SEED + 1)

    assert first == second
    assert first != different


def test_hot_only_workload_uses_hot_key_range() -> None:
    workload = generate_workload(100, 20, 4, 1.0, DEFAULT_SEED)

    assert set(workload) <= set(range(4))


def test_all_keys_hot_is_supported_at_any_probability() -> None:
    workload = generate_workload(100, 5, 5, 0.0, DEFAULT_SEED)

    assert len(workload) == 100
    assert set(workload) <= set(range(5))


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"request_count": 0}, ValueError),
        ({"key_space": 0}, ValueError),
        ({"hot_key_count": 0}, ValueError),
        ({"key_space": 5, "hot_key_count": 6}, ValueError),
        ({"hot_probability": -0.01}, ValueError),
        ({"hot_probability": 1.01}, ValueError),
        ({"request_count": True}, TypeError),
        ({"hot_probability": "high"}, TypeError),
    ],
)
def test_workload_validation(kwargs: dict[str, object], exception: type[Exception]) -> None:
    with pytest.raises(exception):
        generate_workload(**kwargs)  # type: ignore[arg-type]


def test_simulation_outputs_are_equivalent_and_metrics_consistent() -> None:
    result = run_simulation(
        request_count=200,
        key_space=20,
        hot_key_count=4,
        hot_probability=0.85,
        capacity=8,
        rounds=20,
        seed=123,
    )

    assert result.hits + result.misses == result.requests
    assert result.source_calls == result.misses
    assert result.hit_rate == pytest.approx(result.hits / result.requests)
    assert 0 < result.misses <= result.requests
    assert len(result.final_lru_to_mru) <= result.capacity
    assert result.uncached_seconds >= 0
    assert result.cached_seconds >= 0
    assert result.speedup > 0


def test_hot_workload_produces_cache_hits() -> None:
    result = run_simulation(
        request_count=300,
        key_space=30,
        hot_key_count=3,
        hot_probability=0.9,
        capacity=6,
        rounds=5,
        seed=456,
    )

    assert result.hits > result.misses
    assert result.hit_rate > 0.5


def test_csv_contains_all_measurements(tmp_path) -> None:
    result = run_simulation(request_count=20, key_space=5, hot_key_count=2, capacity=3, rounds=2)

    destination = write_result_csv(result, tmp_path / "nested" / "result.csv")

    with destination.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert int(row["requests"]) == 20
    assert int(row["hits"]) == result.hits
    assert int(row["misses"]) == result.misses
    assert float(row["hit_rate"]) == pytest.approx(result.hit_rate)
    assert row["final_lru_to_mru"] == " ".join(map(str, result.final_lru_to_mru))
