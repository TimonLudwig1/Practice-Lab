"""Tests for the chained hash map and its benchmark."""

from pathlib import Path

import pytest

from benchmark import run_benchmark, write_csv
from hash_map import ChainedHashMap, HashMapStats


def constant_hash(key: object) -> int:
    """Force collisions for deterministic tests."""
    del key
    return 1


def test_new_map_is_empty() -> None:
    mapping: ChainedHashMap[str, int] = ChainedHashMap()

    assert len(mapping) == 0
    assert not mapping
    assert mapping.load_factor == 0
    assert list(mapping.items()) == []


@pytest.mark.parametrize("capacity", [0, -1, -20])
def test_rejects_non_positive_capacity(capacity: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        ChainedHashMap(initial_capacity=capacity)


@pytest.mark.parametrize("capacity", [True, 2.5, "8", None])
def test_rejects_non_integer_capacity(capacity: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        ChainedHashMap(initial_capacity=capacity)  # type: ignore[arg-type]


@pytest.mark.parametrize("factor", [0, -0.1, 1.1])
def test_rejects_invalid_load_factor(factor: float) -> None:
    with pytest.raises(ValueError, match="interval"):
        ChainedHashMap(max_load_factor=factor)


def test_put_and_get_multiple_types() -> None:
    mapping: ChainedHashMap[object, object] = ChainedHashMap()
    mapping.put("name", "Ada")
    mapping.put(42, [1, 2])
    mapping.put((1, 2), None)

    assert mapping.get("name") == "Ada"
    assert mapping.get(42) == [1, 2]
    assert mapping.get((1, 2)) is None


def test_update_replaces_value_without_changing_size_or_capacity() -> None:
    mapping: ChainedHashMap[str, int] = ChainedHashMap(initial_capacity=2)
    mapping.put("key", 1)
    capacity = mapping.capacity

    mapping.put("key", 2)

    assert mapping.get("key") == 2
    assert len(mapping) == 1
    assert mapping.capacity == capacity


def test_missing_get_raises_key_error() -> None:
    mapping: ChainedHashMap[str, int] = ChainedHashMap()

    with pytest.raises(KeyError) as error:
        mapping.get("missing")

    assert error.value.args == ("missing",)


def test_delete_returns_value_and_removes_key() -> None:
    mapping: ChainedHashMap[str, int] = ChainedHashMap()
    mapping.put("A", 10)

    assert mapping.delete("A") == 10
    assert "A" not in mapping
    assert len(mapping) == 0


def test_missing_delete_preserves_state() -> None:
    mapping: ChainedHashMap[str, int] = ChainedHashMap()
    mapping.put("A", 1)

    with pytest.raises(KeyError):
        mapping.delete("B")

    assert list(mapping.items()) == [("A", 1)]


def test_chaining_preserves_all_colliding_keys() -> None:
    mapping: ChainedHashMap[str, int] = ChainedHashMap(
        initial_capacity=8, hash_function=constant_hash
    )
    for index, key in enumerate(("A", "B", "C", "D")):
        mapping.put(key, index)

    assert [mapping.get(key) for key in ("A", "B", "C", "D")] == [0, 1, 2, 3]
    assert mapping.stats().collision_count == 3
    assert mapping.stats().max_chain_length == 4


def test_delete_removes_only_matching_collision() -> None:
    mapping: ChainedHashMap[str, int] = ChainedHashMap(
        hash_function=constant_hash
    )
    for key in ("A", "B", "C"):
        mapping.put(key, ord(key))

    mapping.delete("B")

    assert mapping.get("A") == ord("A")
    assert mapping.get("C") == ord("C")
    assert "B" not in mapping


def test_rehash_occurs_only_above_threshold() -> None:
    mapping: ChainedHashMap[int, int] = ChainedHashMap(initial_capacity=4)
    for key in range(3):
        mapping.put(key, key)

    assert mapping.capacity == 4
    assert mapping.load_factor == pytest.approx(0.75)

    mapping.put(3, 3)

    assert mapping.capacity == 8
    assert mapping.load_factor == pytest.approx(0.5)
    assert mapping.rehash_count == 1


def test_multiple_rehashes_preserve_every_entry() -> None:
    mapping: ChainedHashMap[int, int] = ChainedHashMap(initial_capacity=2)
    for key in range(200):
        mapping.put(key, key * key)

    assert len(mapping) == 200
    assert mapping.capacity >= 267
    assert [mapping.get(key) for key in range(200)] == [key * key for key in range(200)]
    mapping.check_invariants()


def test_rehash_redistributes_negative_hash_values() -> None:
    mapping: ChainedHashMap[int, int] = ChainedHashMap(
        initial_capacity=2, hash_function=lambda key: -key
    )
    for key in range(20):
        mapping.put(key, key)

    assert all(mapping.get(key) == key for key in range(20))
    mapping.check_invariants()


def test_disabled_rehash_keeps_fixed_capacity() -> None:
    mapping: ChainedHashMap[int, int] = ChainedHashMap(
        initial_capacity=4, enable_rehash=False
    )
    for key in range(40):
        mapping.put(key, key)

    assert mapping.capacity == 4
    assert mapping.load_factor == 10
    assert mapping.stats().max_chain_length == 10


def test_mapping_protocol_methods() -> None:
    mapping: ChainedHashMap[str, int] = ChainedHashMap()
    mapping["A"] = 1
    mapping["B"] = 2

    assert mapping["A"] == 1
    assert "B" in mapping
    assert "C" not in mapping
    del mapping["A"]
    assert "A" not in mapping


def test_iterators_expose_every_entry() -> None:
    mapping: ChainedHashMap[int, str] = ChainedHashMap()
    for key in range(5):
        mapping.put(key, str(key))

    assert sorted(mapping.keys()) == list(range(5))
    assert sorted(mapping) == list(range(5))
    assert sorted(mapping.values()) == ["0", "1", "2", "3", "4"]
    assert sorted(mapping.items()) == [(key, str(key)) for key in range(5)]


def test_clear_retains_capacity_and_resets_size() -> None:
    mapping: ChainedHashMap[int, int] = ChainedHashMap(initial_capacity=2)
    for key in range(10):
        mapping.put(key, key)
    capacity = mapping.capacity

    mapping.clear()

    assert mapping.capacity == capacity
    assert len(mapping) == 0
    assert mapping.stats().max_chain_length == 0


def test_stats_return_distribution_snapshot() -> None:
    mapping: ChainedHashMap[str, int] = ChainedHashMap(
        initial_capacity=4,
        enable_rehash=False,
        hash_function=constant_hash,
    )
    mapping.put("A", 1)
    mapping.put("B", 2)

    assert mapping.stats() == HashMapStats(
        size=2,
        capacity=4,
        load_factor=0.5,
        non_empty_buckets=1,
        collision_count=1,
        max_chain_length=2,
        rehash_count=0,
    )


def test_unhashable_key_is_rejected_by_default_hash() -> None:
    mapping: ChainedHashMap[object, int] = ChainedHashMap()

    with pytest.raises(TypeError):
        mapping.put([1, 2], 3)


def test_custom_hash_function_must_return_integer() -> None:
    mapping: ChainedHashMap[str, int] = ChainedHashMap(
        hash_function=lambda key: "invalid"  # type: ignore[return-value]
    )

    with pytest.raises(TypeError, match="integer"):
        mapping.put("A", 1)


def test_repr_contains_capacity_and_items() -> None:
    mapping: ChainedHashMap[int, str] = ChainedHashMap(initial_capacity=4)
    mapping.put(1, "one")

    representation = repr(mapping)

    assert "ChainedHashMap" in representation
    assert "(1, 'one')" in representation
    assert "capacity=4" in representation


def test_small_benchmark_compares_both_strategies() -> None:
    rows = run_benchmark((20, 50), repetitions=1, initial_capacity=4)

    assert len(rows) == 4
    assert {row.strategy for row in rows} == {"rehashing", "fixed_capacity"}
    for size in (20, 50):
        resizing, fixed = [row for row in rows if row.item_count == size]
        if resizing.strategy == "fixed_capacity":
            resizing, fixed = fixed, resizing
        assert resizing.load_factor <= 0.75
        assert fixed.capacity == 4
        assert fixed.max_chain_length >= resizing.max_chain_length


def test_benchmark_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        run_benchmark(())
    with pytest.raises(ValueError):
        run_benchmark((10,), repetitions=0)


def test_benchmark_csv_contains_all_rows(tmp_path: Path) -> None:
    rows = run_benchmark((10,), repetitions=1, initial_capacity=4)

    path = write_csv(tmp_path / "benchmark.csv", rows)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("strategy,item_count,capacity")
