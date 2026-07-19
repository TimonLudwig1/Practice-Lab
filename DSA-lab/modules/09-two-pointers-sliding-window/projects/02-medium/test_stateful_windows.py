"""Tests for stateful sliding-window algorithms and their invariants."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from stateful_windows import (
    WindowStep,
    anagram_starts,
    anagram_starts_brute,
    longest_ones_with_flips,
    longest_ones_with_flips_brute,
    minimum_covering_window,
    minimum_covering_window_brute,
)


@pytest.mark.parametrize(
    ("text", "target", "expected"),
    [
        ("", "", ""),
        ("", "A", ""),
        ("A", "", ""),
        ("A", "A", "A"),
        ("A", "AA", ""),
        ("ADOBECODEBANC", "ABC", "BANC"),
        ("aa", "aa", "aa"),
        ("aaflslflsldkalskaaa", "aaa", "aaa"),
        ("xyzab", "ba", "ab"),
        ("äöüß", "ßä", "äöüß"),
    ],
)
def test_minimum_cover_known_cases(text, target, expected) -> None:
    assert minimum_covering_window_brute(text, target) == expected
    assert minimum_covering_window(text, target) == expected


def test_minimum_cover_property_cases() -> None:
    rng = random.Random(90921)
    alphabet = "abcd"
    for _ in range(600):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(18)))
        target = "".join(rng.choice(alphabet) for _ in range(rng.randrange(7)))
        assert minimum_covering_window(text, target) == minimum_covering_window_brute(
            text, target
        )


def test_minimum_cover_trace_missing_invariant() -> None:
    text = "ADOBECODEBANC"
    target = "ABC"
    trace: list[WindowStep] = []
    assert minimum_covering_window(text, target, trace=trace) == "BANC"
    assert trace
    assert all(step.problem == "minimum_covering_window" for step in trace)
    assert all(step.valid == (step.value("missing") == 0) for step in trace)
    assert all(0 <= step.value("missing") <= len(target) for step in trace)


@pytest.mark.parametrize(
    ("values", "flips", "expected"),
    [
        ([], 0, 0),
        ([1], 0, 1),
        ([0], 0, 0),
        ([0], 1, 1),
        ([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2, 6),
        ([0, 0, 1, 1, 0], 0, 2),
        ([0, 0, 0], 3, 3),
        ([1, 1, 1], 100, 3),
    ],
)
def test_longest_ones_known_cases(values, flips, expected) -> None:
    assert longest_ones_with_flips_brute(values, flips) == expected
    assert longest_ones_with_flips(values, flips) == expected


def test_longest_ones_property_cases() -> None:
    rng = random.Random(90922)
    for _ in range(800):
        values = [rng.randrange(2) for _ in range(rng.randrange(35))]
        flips = rng.randrange(0, 8)
        assert longest_ones_with_flips(values, flips) == longest_ones_with_flips_brute(
            values, flips
        )


@pytest.mark.parametrize(
    ("values", "flips"),
    [([0, 1], -1), ([0, 1], 1.5), ([0, 1], True), ([0, 2], 1), ([-1], 1), ([False], 1)],
)
def test_longest_ones_rejects_invalid_input(values, flips) -> None:
    with pytest.raises(ValueError):
        longest_ones_with_flips(values, flips)
    with pytest.raises(ValueError):
        longest_ones_with_flips_brute(values, flips)


def test_longest_ones_trace_invariant() -> None:
    values = [1, 0, 0, 1, 1, 0, 1]
    trace: list[WindowStep] = []
    longest_ones_with_flips(values, 1, trace=trace)
    assert len(trace) == len(values)
    for step in trace:
        actual_zeros = values[step.left : step.right + 1].count(0)
        assert step.valid
        assert step.value("zeros") == actual_zeros <= step.value("flips")


@pytest.mark.parametrize(
    ("text", "pattern", "expected"),
    [
        ("", "", []),
        ("abc", "", []),
        ("", "a", []),
        ("a", "a", [0]),
        ("cbaebabacd", "abc", [0, 6]),
        ("abab", "ab", [0, 1, 2]),
        ("aaaa", "aa", [0, 1, 2]),
        ("abc", "abcd", []),
        ("äöä", "äö", [0, 1]),
    ],
)
def test_anagram_known_cases(text, pattern, expected) -> None:
    assert anagram_starts_brute(text, pattern) == expected
    assert anagram_starts(text, pattern) == expected


def test_anagram_property_cases() -> None:
    rng = random.Random(90923)
    alphabet = "abcde"
    for _ in range(800):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(35)))
        pattern = "".join(rng.choice(alphabet) for _ in range(rng.randrange(8)))
        assert anagram_starts(text, pattern) == anagram_starts_brute(text, pattern)


def test_anagram_trace_matches_window_counters() -> None:
    text = "cbaebabacd"
    pattern = "abc"
    trace: list[WindowStep] = []
    assert anagram_starts(text, pattern, trace=trace) == [0, 6]
    assert len(trace) == len(text) - len(pattern) + 1
    for step in trace:
        window = text[step.left : step.right + 1]
        assert step.length == len(pattern)
        assert step.valid == (Counter(window) == Counter(pattern))
        assert step.valid == (step.value("nonzero") == 0)


def test_window_step_is_immutable() -> None:
    trace: list[WindowStep] = []
    anagram_starts("abc", "ab", trace=trace)
    with pytest.raises(FrozenInstanceError):
        trace[0].left = 5  # type: ignore[misc]
