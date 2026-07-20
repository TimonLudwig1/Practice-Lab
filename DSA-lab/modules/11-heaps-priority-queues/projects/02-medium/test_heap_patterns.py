"""Property and edge-case tests for all three heap patterns."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import pytest

from heap_patterns import RunningMedian, merge_sorted, top_k_frequent
from run_benchmarks import all_prefix_medians, naive_prefix_medians, top_k_reference


def top_k_reference_generic(values: list[object], k: int) -> list[tuple[object, int]]:
    counts = Counter(values)
    first: dict[object, int] = {}
    for index, value in enumerate(values):
        first.setdefault(value, index)
    ordered = sorted(counts, key=lambda value: (-counts[value], first[value]))
    return [(value, counts[value]) for value in ordered[:k]]


def median_reference(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def test_top_k_basic_example() -> None:
    values = ["api", "db", "api", "cache", "db", "api"]

    assert top_k_frequent(values, 2) == [("api", 3), ("db", 2)]


def test_top_k_ties_keep_first_appearance() -> None:
    values = ["late", "first", "late", "second", "first", "second"]

    assert top_k_frequent(values, 3) == [
        ("late", 2),
        ("first", 2),
        ("second", 2),
    ]
    assert top_k_frequent(values, 2) == [("late", 2), ("first", 2)]


def test_top_k_zero_and_larger_than_unique_count() -> None:
    assert top_k_frequent([1, 1, 2], 0) == []
    assert top_k_frequent([1, 1, 2], 99) == [(1, 2), (2, 1)]
    assert top_k_frequent([], 5) == []


def test_top_k_negative_k_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        top_k_frequent([1, 2], -1)


def test_top_k_accepts_generator() -> None:
    assert top_k_frequent((value % 3 for value in range(10)), 2) == [(0, 4), (1, 3)]


@dataclass(frozen=True)
class Token:
    name: str


def test_top_k_payloads_need_not_be_orderable() -> None:
    first = Token("first")
    second = Token("second")

    assert top_k_frequent([first, second, second, first], 2) == [
        (first, 2),
        (second, 2),
    ]


@pytest.mark.parametrize("seed", range(15))
def test_top_k_seeded_property(seed: int) -> None:
    rng = random.Random(20260720 + seed)
    values = [rng.randrange(30) for _ in range(rng.randrange(0, 500))]
    k = rng.randrange(0, 40)

    assert top_k_frequent(values, k) == top_k_reference(values, k)


def test_merge_basic_example() -> None:
    sequences = [[1, 7, 10], [2, 3, 11], [4, 8]]

    assert merge_sorted(sequences) == [1, 2, 3, 4, 7, 8, 10, 11]


@pytest.mark.parametrize(
    ("sequences", "expected"),
    [
        ([], []),
        ([[]], []),
        ([[], [1], []], [1]),
        ([[1, 1], [1, 1]], [1, 1, 1, 1]),
        ([[-5, 0], [-4, -1], [3]], [-5, -4, -1, 0, 3]),
    ],
)
def test_merge_edge_cases(sequences: list[list[int]], expected: list[int]) -> None:
    assert merge_sorted(sequences) == expected


def test_merge_accepts_generators() -> None:
    sequences = ((value for value in sequence) for sequence in ([1, 4], [2, 3]))

    assert merge_sorted(sequences) == [1, 2, 3, 4]


def test_merge_strings_and_tuples() -> None:
    assert merge_sorted([["a", "d"], ["b", "c"]]) == ["a", "b", "c", "d"]
    assert merge_sorted([[(1, "a"), (3, "c")], [(1, "b"), (2, "x")]]) == [
        (1, "a"),
        (1, "b"),
        (2, "x"),
        (3, "c"),
    ]


@pytest.mark.parametrize("seed", range(15))
def test_merge_seeded_property(seed: int) -> None:
    rng = random.Random(110200 + seed)
    sequences = [
        sorted(rng.randrange(-100, 101) for _ in range(rng.randrange(0, 80)))
        for _ in range(rng.randrange(0, 25))
    ]
    expected = sorted(value for sequence in sequences for value in sequence)

    assert merge_sorted(sequences) == expected


def test_empty_running_median() -> None:
    tracker = RunningMedian()

    assert len(tracker) == 0
    assert tracker.halves() == ((), ())
    assert tracker.is_valid()
    with pytest.raises(ValueError, match="empty"):
        tracker.median()


def test_running_median_known_sequence() -> None:
    tracker = RunningMedian()
    medians = []

    for value in [5, 2, 10, 4]:
        tracker.add(value)
        medians.append(tracker.median())
        assert tracker.is_valid()

    assert medians == [5.0, 3.5, 5.0, 4.5]
    assert tracker.halves() == ((2.0, 4.0), (5.0, 10.0))


def test_running_median_constructor_and_extend() -> None:
    tracker = RunningMedian([3, 1, 2])

    assert len(tracker) == 3
    assert tracker.median() == 2.0
    tracker.extend([10, -5])
    assert tracker.median() == 2.0
    assert tracker.is_valid()


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_running_median_rejects_nonfinite_values(value: float) -> None:
    tracker = RunningMedian([1.0])

    with pytest.raises(ValueError, match="finite"):
        tracker.add(value)
    assert len(tracker) == 1
    assert tracker.median() == 1.0


def test_running_median_duplicates_and_negatives() -> None:
    values = [-3, -3, 0, 4, 4, 4]
    tracker = RunningMedian(values)

    assert tracker.median() == 2.0
    assert tracker.is_valid()


@pytest.mark.parametrize("seed", range(12))
def test_running_median_seeded_prefix_property(seed: int) -> None:
    rng = random.Random(330000 + seed)
    tracker = RunningMedian()
    seen: list[float] = []

    for _ in range(300):
        value = rng.uniform(-1000.0, 1000.0)
        seen.append(value)
        tracker.add(value)
        assert tracker.median() == median_reference(seen)
        assert tracker.is_valid()


def test_prefix_median_implementations_match() -> None:
    rng = random.Random(99)
    values = [rng.uniform(-10, 10) for _ in range(200)]

    assert all_prefix_medians(values) == naive_prefix_medians(values)
