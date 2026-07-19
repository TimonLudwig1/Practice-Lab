"""Tests for six basic pointer and window exercises."""

from __future__ import annotations

import random

import pytest

from patterns import (
    filter_copy,
    filter_in_place,
    longest_unique_substring,
    longest_unique_substring_brute,
    max_container,
    max_container_brute,
    minimum_length,
    minimum_length_brute,
    pair_sum_brute,
    pair_sum_sorted,
    rolling_sums,
    rolling_sums_brute,
)


def valid_pair(values, target, pair) -> bool:
    return pair is not None and pair[0] < pair[1] and sum(values[index] for index in pair) == target


@pytest.mark.parametrize(
    ("values", "target", "exists"),
    [
        ([], 1, False),
        ([1], 2, False),
        ([1, 2], 3, True),
        ([1, 2], 4, False),
        ([-5, -2, 0, 3, 9], 4, True),
        ([1, 1, 1, 1], 2, True),
        ([1, 2, 4, 8, 16], 12, True),
        ([1, 2, 4, 8, 16], 7, False),
    ],
)
def test_pair_sum_known_cases(values, target, exists) -> None:
    brute = pair_sum_brute(values, target)
    optimized = pair_sum_sorted(values, target)
    assert (brute is not None) == exists
    assert (optimized is not None) == exists
    if exists:
        assert valid_pair(values, target, brute)
        assert valid_pair(values, target, optimized)


def test_pair_sum_property_cases() -> None:
    rng = random.Random(90911)
    for _ in range(500):
        values = sorted(rng.randrange(-20, 21) for _ in range(rng.randrange(30)))
        target = rng.randrange(-45, 46)
        brute = pair_sum_brute(values, target)
        optimized = pair_sum_sorted(values, target)
        assert (brute is None) == (optimized is None)
        if optimized is not None:
            assert valid_pair(values, target, optimized)


@pytest.mark.parametrize(
    ("heights", "expected"),
    [
        ([], 0),
        ([5], 0),
        ([1, 1], 1),
        ([1, 8, 6, 2, 5, 4, 8, 3, 7], 49),
        ([4, 3, 2, 1, 4], 16),
        ([0, 0, 0], 0),
    ],
)
def test_container_known_cases(heights, expected) -> None:
    assert max_container_brute(heights) == expected
    assert max_container(heights) == expected


def test_container_property_cases() -> None:
    rng = random.Random(90912)
    for _ in range(400):
        heights = [rng.randrange(20) for _ in range(rng.randrange(25))]
        assert max_container(heights) == max_container_brute(heights)


@pytest.mark.parametrize("heights", [[-1], [1, -2], [1.5], [True]])
def test_container_rejects_invalid_heights(heights) -> None:
    with pytest.raises(ValueError):
        max_container(heights)
    with pytest.raises(ValueError):
        max_container_brute(heights)


@pytest.mark.parametrize(
    ("values", "threshold", "expected"),
    [
        ([], 0, []),
        ([1], 0, [1]),
        ([-1], 0, []),
        ([3, -1, 4, -2, 0, 5], 0, [3, 4, 0, 5]),
        ([1, 2, 3], 10, []),
    ],
)
def test_filter_variants(values, threshold, expected) -> None:
    keep = lambda value: value >= threshold
    mutable = list(values)
    assert filter_copy(values, keep) == expected
    assert filter_in_place(mutable, keep) == len(expected)
    assert mutable == expected


def test_filter_is_stable_and_reuses_list() -> None:
    values = [(2, "a"), (1, "b"), (2, "c"), (1, "d")]
    identity = id(values)
    filter_in_place(values, lambda item: item[0] == 2)
    assert id(values) == identity
    assert values == [(2, "a"), (2, "c")]


@pytest.mark.parametrize(
    ("values", "width", "expected"),
    [
        ([], 1, []),
        ([5], 1, [5]),
        ([1, 2], 3, []),
        ([2, 1, 5, 1, 3, 2], 3, [8, 7, 9, 6]),
        ([-2, 3, -1, 4], 2, [1, 2, 3]),
    ],
)
def test_rolling_sums_known_cases(values, width, expected) -> None:
    assert rolling_sums_brute(values, width) == expected
    assert rolling_sums(values, width) == expected


def test_rolling_sum_property_cases() -> None:
    rng = random.Random(90913)
    for _ in range(500):
        values = [rng.randrange(-50, 51) for _ in range(rng.randrange(40))]
        width = rng.randrange(1, 45)
        assert rolling_sums(values, width) == rolling_sums_brute(values, width)


@pytest.mark.parametrize("width", [0, -1, 1.5, True])
def test_rolling_sums_reject_invalid_width(width) -> None:
    with pytest.raises(ValueError):
        rolling_sums([1, 2], width)
    with pytest.raises(ValueError):
        rolling_sums_brute([1, 2], width)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("a", "a"),
        ("bbbbb", "b"),
        ("abcabcbb", "abc"),
        ("abcaeb", "bcae"),
        ("pwwkew", "wke"),
        ("åßåç", "ßåç"),
    ],
)
def test_longest_unique_known_cases(text, expected) -> None:
    assert longest_unique_substring_brute(text) == expected
    assert longest_unique_substring(text) == expected


def test_longest_unique_property_cases() -> None:
    rng = random.Random(90914)
    alphabet = "abcde"
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(30)))
        assert longest_unique_substring(text) == longest_unique_substring_brute(text)


@pytest.mark.parametrize(
    ("values", "target", "expected"),
    [
        ([], 1, 0),
        ([1], 1, 1),
        ([1], 2, 0),
        ([2, 3, 1, 2, 4, 3], 7, 2),
        ([1, 4, 4], 4, 1),
        ([1, 1, 1, 1], 3, 3),
    ],
)
def test_minimum_length_known_cases(values, target, expected) -> None:
    assert minimum_length_brute(values, target) == expected
    assert minimum_length(values, target) == expected


def test_minimum_length_property_cases() -> None:
    rng = random.Random(90915)
    for _ in range(500):
        values = [rng.randrange(1, 11) for _ in range(rng.randrange(30))]
        target = rng.randrange(1, 100)
        assert minimum_length(values, target) == minimum_length_brute(values, target)


@pytest.mark.parametrize(
    ("values", "target"),
    [([1], 0), ([1], -1), ([0], 1), ([-1], 1), ([1.5], 1), ([True], 1)],
)
def test_minimum_length_rejects_invalid_input(values, target) -> None:
    with pytest.raises(ValueError):
        minimum_length(values, target)
    with pytest.raises(ValueError):
        minimum_length_brute(values, target)
