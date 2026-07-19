"""Tests for exact and boundary binary-search implementations."""

from __future__ import annotations

import random
from bisect import bisect_left, bisect_right
from dataclasses import FrozenInstanceError

import pytest

from binary_search import (
    SearchStep,
    binary_search,
    count_occurrences,
    equal_range,
    first_occurrence,
    insert_position,
    last_occurrence,
    lower_bound,
    upper_bound,
)


@pytest.mark.parametrize(
    ("values", "target", "expected"),
    [
        ([], 5, -1),
        ([5], 5, 0),
        ([1, 5], 1, 0),
        ([1, 5], 5, 1),
        ([1, 3, 5], 3, 1),
        ([-10, -5, 0, 8, 11], -10, 0),
        ([-10, -5, 0, 8, 11], 11, 4),
        ((2, 4, 6, 8), 6, 2),
    ],
)
def test_binary_search_finds_unique_values(values, target, expected) -> None:
    assert binary_search(values, target) == expected


@pytest.mark.parametrize(
    ("values", "target"),
    [
        ([], 0),
        ([5], 4),
        ([5], 6),
        ([1, 3], 2),
        ([1, 3, 5], 0),
        ([1, 3, 5], 6),
        ([-9, -4, 0, 7], -5),
    ],
)
def test_binary_search_reports_absent_values(values, target) -> None:
    assert binary_search(values, target) == -1


def test_binary_search_may_return_any_duplicate() -> None:
    values = [1, 4, 4, 4, 4, 9]
    index = binary_search(values, 4)
    assert 1 <= index <= 4
    assert values[index] == 4


@pytest.mark.parametrize(
    ("values", "target", "expected"),
    [
        ([], 4, 0),
        ([4], 3, 0),
        ([4], 4, 0),
        ([4], 5, 1),
        ([2, 4, 4, 4, 7, 9], 1, 0),
        ([2, 4, 4, 4, 7, 9], 2, 0),
        ([2, 4, 4, 4, 7, 9], 4, 1),
        ([2, 4, 4, 4, 7, 9], 5, 4),
        ([2, 4, 4, 4, 7, 9], 9, 5),
        ([2, 4, 4, 4, 7, 9], 10, 6),
    ],
)
def test_lower_bound_and_insert_position(values, target, expected) -> None:
    assert lower_bound(values, target) == expected
    assert insert_position(values, target) == expected


@pytest.mark.parametrize(
    ("values", "target", "expected"),
    [
        ([], 4, 0),
        ([4], 3, 0),
        ([4], 4, 1),
        ([4], 5, 1),
        ([2, 4, 4, 4, 7, 9], 1, 0),
        ([2, 4, 4, 4, 7, 9], 2, 1),
        ([2, 4, 4, 4, 7, 9], 4, 4),
        ([2, 4, 4, 4, 7, 9], 5, 4),
        ([2, 4, 4, 4, 7, 9], 9, 6),
        ([2, 4, 4, 4, 7, 9], 10, 6),
    ],
)
def test_upper_bound(values, target, expected) -> None:
    assert upper_bound(values, target) == expected


@pytest.mark.parametrize(
    ("target", "first", "last", "bounds", "count"),
    [
        (1, -1, -1, (0, 0), 0),
        (2, 0, 0, (0, 1), 1),
        (4, 1, 3, (1, 4), 3),
        (5, -1, -1, (4, 4), 0),
        (7, 4, 4, (4, 5), 1),
        (9, 5, 5, (5, 6), 1),
        (10, -1, -1, (6, 6), 0),
    ],
)
def test_occurrence_helpers(target, first, last, bounds, count) -> None:
    values = [2, 4, 4, 4, 7, 9]
    assert first_occurrence(values, target) == first
    assert last_occurrence(values, target) == last
    assert equal_range(values, target) == bounds
    assert count_occurrences(values, target) == count


def test_all_equal_values() -> None:
    values = [8] * 100
    assert first_occurrence(values, 8) == 0
    assert last_occurrence(values, 8) == 99
    assert equal_range(values, 8) == (0, 100)
    assert count_occurrences(values, 8) == 100
    assert lower_bound(values, 7) == 0
    assert upper_bound(values, 9) == 100


def test_property_cases_match_standard_library() -> None:
    rng = random.Random(80801)  # Fixed seed makes failures reproducible.
    for _ in range(500):
        values = sorted(rng.randrange(-30, 31) for _ in range(rng.randrange(80)))
        target = rng.randrange(-35, 36)
        assert lower_bound(values, target) == bisect_left(values, target)
        assert upper_bound(values, target) == bisect_right(values, target)
        assert insert_position(values, target) == bisect_left(values, target)
        assert equal_range(values, target) == (
            bisect_left(values, target),
            bisect_right(values, target),
        )
        index = binary_search(values, target)
        assert (index != -1) == (target in values)
        if index != -1:
            assert values[index] == target


def test_exact_trace_matches_hand_simulation() -> None:
    values = [3, 7, 11, 15, 18, 23, 29, 31, 42]
    trace: list[SearchStep[int]] = []
    assert binary_search(values, 23, trace=trace) == 5
    assert [
        (
            step.left,
            step.right,
            step.middle,
            step.value,
            step.decision,
            step.next_left,
            step.next_right,
        )
        for step in trace
    ] == [
        (0, 8, 4, 18, "discard_left", 5, 8),
        (5, 8, 6, 29, "discard_right", 5, 5),
        (5, 5, 5, 23, "found", 5, 5),
    ]


def test_unsuccessful_trace_ends_with_empty_closed_interval() -> None:
    trace: list[SearchStep[int]] = []
    assert binary_search([3, 7, 11, 15, 18, 23, 29, 31, 42], 16, trace=trace) == -1
    assert trace[-1].next_left > trace[-1].next_right
    assert all(step.interval == "closed" for step in trace)
    assert all(step.next_size < step.size for step in trace)


def test_exact_trace_preserves_target_invariant() -> None:
    values = list(range(0, 200, 2))
    target = 134
    trace: list[SearchStep[int]] = []
    found = binary_search(values, target, trace=trace)
    target_index = values.index(target)
    assert found == target_index
    for step in trace:
        assert step.left <= target_index <= step.right
        if step.decision != "found":
            assert step.next_left <= target_index <= step.next_right


def test_lower_bound_trace_matches_hand_simulation() -> None:
    trace: list[SearchStep[int]] = []
    assert lower_bound([2, 4, 4, 4, 7, 9], 4, trace=trace) == 1
    assert [
        (step.left, step.right, step.middle, step.decision, step.next_left, step.next_right)
        for step in trace
    ] == [
        (0, 6, 3, "keep_left", 0, 3),
        (0, 3, 1, "keep_left", 0, 1),
        (0, 1, 0, "discard_left", 1, 1),
    ]


@pytest.mark.parametrize("target", [-10, 0, 4, 5, 9, 20])
def test_lower_bound_trace_preserves_partition_invariant(target: int) -> None:
    values = [-5, 0, 0, 4, 4, 9, 12]
    trace: list[SearchStep[int]] = []
    result = lower_bound(values, target, trace=trace)
    for step in trace:
        assert all(value < target for value in values[: step.left])
        assert all(value >= target for value in values[step.right :])
        assert step.next_size < step.size
    assert all(value < target for value in values[:result])
    assert all(value >= target for value in values[result:])


@pytest.mark.parametrize("target", [-10, 0, 4, 5, 9, 20])
def test_upper_bound_trace_preserves_partition_invariant(target: int) -> None:
    values = [-5, 0, 0, 4, 4, 9, 12]
    trace: list[SearchStep[int]] = []
    result = upper_bound(values, target, trace=trace)
    for step in trace:
        assert all(value <= target for value in values[: step.left])
        assert all(value > target for value in values[step.right :])
        assert step.next_size < step.size
    assert all(value <= target for value in values[:result])
    assert all(value > target for value in values[result:])


def test_empty_searches_produce_no_trace_steps() -> None:
    exact_trace: list[SearchStep[int]] = []
    lower_trace: list[SearchStep[int]] = []
    upper_trace: list[SearchStep[int]] = []
    assert binary_search([], 1, trace=exact_trace) == -1
    assert lower_bound([], 1, trace=lower_trace) == 0
    assert upper_bound([], 1, trace=upper_trace) == 0
    assert exact_trace == lower_trace == upper_trace == []


def test_search_does_not_mutate_input() -> None:
    values = [1, 2, 2, 5, 8]
    snapshot = list(values)
    binary_search(values, 2)
    lower_bound(values, 2)
    upper_bound(values, 2)
    assert values == snapshot


def test_search_step_is_immutable() -> None:
    trace: list[SearchStep[int]] = []
    binary_search([1, 2, 3], 2, trace=trace)
    with pytest.raises(FrozenInstanceError):
        trace[0].left = 99  # type: ignore[misc]


def test_strings_and_tuples_are_supported() -> None:
    words = ["alpha", "beta", "beta", "delta"]
    assert binary_search(words, "delta") == 3
    assert first_occurrence(words, "beta") == 1
    assert last_occurrence(words, "beta") == 2
    assert insert_position(words, "charlie") == 3
