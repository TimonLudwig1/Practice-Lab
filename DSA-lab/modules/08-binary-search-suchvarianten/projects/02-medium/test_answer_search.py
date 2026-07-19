"""Tests for reusable and problem-specific answer-space searches."""

from __future__ import annotations

import math
import random
from dataclasses import FrozenInstanceError

import pytest

from answer_search import AnswerStep, first_true
from problems import (
    count_less_equal,
    integer_square_root,
    kth_smallest,
    minimum_shipping_capacity,
    required_shipping_days,
)


@pytest.mark.parametrize(
    ("low", "high", "boundary"),
    [
        (0, 0, 0),
        (0, 1, 0),
        (0, 1, 1),
        (-10, 10, -10),
        (-10, 10, 0),
        (-10, 10, 10),
        (100, 10_000, 7_777),
    ],
)
def test_first_true_finds_boundary(low: int, high: int, boundary: int) -> None:
    assert first_true(low, high, lambda value: value >= boundary) == boundary


def test_first_true_trace_has_strict_progress() -> None:
    trace: list[AnswerStep] = []
    assert first_true(0, 100, lambda value: value >= 63, trace=trace) == 63
    assert trace
    assert all(step.next_size < step.size for step in trace)
    assert all(step.low <= 63 <= step.high for step in trace)
    assert all(step.next_low <= 63 <= step.next_high for step in trace)
    assert all(step.feasible == (step.middle >= 63) for step in trace)


def test_first_true_rejects_invalid_range_or_infeasible_high() -> None:
    with pytest.raises(ValueError, match="low"):
        first_true(5, 4, lambda value: True)
    with pytest.raises(ValueError, match="feasible"):
        first_true(0, 5, lambda value: False)


def test_answer_step_is_immutable() -> None:
    trace: list[AnswerStep] = []
    first_true(0, 3, lambda value: value >= 2, trace=trace)
    with pytest.raises(FrozenInstanceError):
        trace[0].low = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("weights", "capacity", "expected_days"),
    [
        ([], 5, 0),
        ([3], 3, 1),
        ([3], 2, math.inf),
        ([3, 2, 2, 4, 1, 4], 5, 4),
        ([3, 2, 2, 4, 1, 4], 6, 3),
        ([3, 2, 2, 4, 1, 4], 10, 2),
        ([3, 2, 2, 4, 1, 4], 16, 1),
        ([1, 1, 1, 1], 1, 4),
    ],
)
def test_required_shipping_days(weights, capacity, expected_days) -> None:
    assert required_shipping_days(weights, capacity) == expected_days


@pytest.mark.parametrize(
    ("weights", "capacity"),
    [
        ([1], 0),
        ([1], -1),
        ([0], 1),
        ([-1], 1),
        ([1.5], 2),
        ([True], 2),
    ],
)
def test_required_shipping_days_rejects_invalid_input(weights, capacity) -> None:
    with pytest.raises(ValueError):
        required_shipping_days(weights, capacity)


@pytest.mark.parametrize(
    ("weights", "day_limit", "expected"),
    [
        ([3, 2, 2, 4, 1, 4], 3, 6),
        ([1, 2, 3, 1, 1], 4, 3),
        ([1, 2, 3, 1, 1], 1, 8),
        ([1, 2, 3, 1, 1], 20, 3),
        ([10], 1, 10),
        ([5, 5, 5], 2, 10),
        ([5, 5, 5], 3, 5),
    ],
)
def test_minimum_shipping_capacity(weights, day_limit, expected) -> None:
    assert minimum_shipping_capacity(weights, day_limit) == expected


def brute_shipping_capacity(weights: list[int], day_limit: int) -> int:
    for capacity in range(max(weights), sum(weights) + 1):
        if required_shipping_days(weights, capacity) <= day_limit:
            return capacity
    raise AssertionError("sum(weights) must always be feasible")


def test_shipping_property_cases_match_brute_force() -> None:
    rng = random.Random(80802)  # Fixed seed makes failures reproducible.
    for _ in range(300):
        weights = [rng.randrange(1, 15) for _ in range(rng.randrange(1, 20))]
        day_limit = rng.randrange(1, len(weights) + 4)
        assert minimum_shipping_capacity(weights, day_limit) == brute_shipping_capacity(
            weights, day_limit
        )


def test_shipping_trace_and_minimality() -> None:
    weights = [3, 2, 2, 4, 1, 4]
    trace: list[AnswerStep] = []
    result = minimum_shipping_capacity(weights, 3, trace=trace)
    assert result == 6
    assert required_shipping_days(weights, result) <= 3
    assert required_shipping_days(weights, result - 1) > 3
    assert all(step.problem == "minimum_shipping_capacity" for step in trace)
    assert all(step.next_size < step.size for step in trace)


@pytest.mark.parametrize(
    ("weights", "day_limit"),
    [
        ([], 1),
        ([1, 2], 0),
        ([1, 2], -1),
        ([1, 2], True),
        ([1, 0], 2),
        ([1, -2], 2),
        ([1, 2.5], 2),
        ([True, 2], 2),
    ],
)
def test_invalid_shipping_input_is_rejected(weights, day_limit) -> None:
    with pytest.raises(ValueError):
        minimum_shipping_capacity(weights, day_limit)


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 2),
        (8, 2),
        (9, 3),
        (15, 3),
        (16, 4),
        (17, 4),
        (10**12, 10**6),
        (10**12 - 1, 10**6 - 1),
    ],
)
def test_integer_square_root_known_values(number: int, expected: int) -> None:
    assert integer_square_root(number) == expected


def test_integer_square_root_matches_math_isqrt() -> None:
    rng = random.Random(80803)  # Fixed seed makes failures reproducible.
    cases = [rng.randrange(0, 10**30) for _ in range(500)]
    for number in cases:
        assert integer_square_root(number) == math.isqrt(number)


def test_integer_square_root_trace_and_floor_contract() -> None:
    trace: list[AnswerStep] = []
    root = integer_square_root(200, trace=trace)
    assert root == 14
    assert root * root <= 200 < (root + 1) * (root + 1)
    assert all(step.problem == "integer_square_root" for step in trace)
    assert all(step.feasible == (step.middle * step.middle > 200) for step in trace)


@pytest.mark.parametrize("number", [-1, -10, 2.5, "9", True])
def test_integer_square_root_rejects_invalid_values(number) -> None:
    with pytest.raises(ValueError):
        integer_square_root(number)


@pytest.mark.parametrize(
    ("matrix", "k", "expected"),
    [
        ([[1]], 1, 1),
        ([[1, 2, 3]], 2, 2),
        ([[1], [2], [3]], 3, 3),
        ([[1, 5, 9], [10, 11, 13], [12, 13, 15]], 8, 13),
        ([[-5, -4], [-3, -1]], 1, -5),
        ([[-5, -4], [-3, -1]], 4, -1),
        ([[1, 1, 1], [1, 1, 2]], 5, 1),
        ([[1, 2], [1, 3], [2, 4]], 4, 2),
    ],
)
def test_kth_smallest_known_cases(matrix, k, expected) -> None:
    assert kth_smallest(matrix, k) == expected


def make_monotone_matrix(rng: random.Random, rows: int, columns: int) -> list[list[int]]:
    row_offsets = sorted(rng.randrange(-10, 11) for _ in range(rows))
    column_offsets = sorted(rng.randrange(-10, 11) for _ in range(columns))
    return [
        [row_offsets[row] + column_offsets[column] for column in range(columns)]
        for row in range(rows)
    ]


def test_matrix_property_cases_match_flattened_sort() -> None:
    rng = random.Random(80804)  # Fixed seed makes failures reproducible.
    for _ in range(250):
        rows = rng.randrange(1, 8)
        columns = rng.randrange(1, 8)
        matrix = make_monotone_matrix(rng, rows, columns)
        ordered = sorted(value for row in matrix for value in row)
        k = rng.randrange(1, len(ordered) + 1)
        assert kth_smallest(matrix, k) == ordered[k - 1]


def test_count_less_equal_matches_flat_count() -> None:
    matrix = [[1, 4, 7], [2, 5, 9], [3, 6, 12], [8, 10, 15]]
    for candidate in range(-1, 18):
        expected = sum(value <= candidate for row in matrix for value in row)
        assert count_less_equal(matrix, candidate) == expected


def test_kth_smallest_trace_and_rank_boundary() -> None:
    matrix = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]
    trace: list[AnswerStep] = []
    result = kth_smallest(matrix, 8, trace=trace)
    assert result == 13
    assert count_less_equal(matrix, result) >= 8
    assert count_less_equal(matrix, result - 1) < 8
    assert all(step.problem == "kth_smallest" for step in trace)
    assert all(step.feasible == (count_less_equal(matrix, step.middle) >= 8) for step in trace)


@pytest.mark.parametrize(
    ("matrix", "k"),
    [
        ([], 1),
        ([[]], 1),
        ([[1, 2], [3]], 1),
        ([[2, 1], [3, 4]], 1),
        ([[1, 4], [0, 5]], 1),
        ([[1, 2], [3, 4]], 0),
        ([[1, 2], [3, 4]], 5),
        ([[1, 2], [3, 4]], True),
        ([[1, 2.5], [3, 4]], 1),
        ([[False, 1], [2, 3]], 1),
    ],
)
def test_kth_smallest_rejects_invalid_input(matrix, k) -> None:
    with pytest.raises(ValueError):
        kth_smallest(matrix, k)
