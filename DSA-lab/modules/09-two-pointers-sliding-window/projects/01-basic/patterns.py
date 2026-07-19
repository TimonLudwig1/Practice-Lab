"""Reference and optimized implementations of six pointer/window exercises."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar


T = TypeVar("T")


def pair_sum_brute(values: Sequence[int], target: int) -> tuple[int, int] | None:
    """Return one target-sum index pair by checking every pair."""

    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] + values[right] == target:
                return left, right
    return None


def pair_sum_sorted(values: Sequence[int], target: int) -> tuple[int, int] | None:
    """Return one target-sum index pair from a non-decreasing sequence."""

    left, right = 0, len(values) - 1
    while left < right:
        current = values[left] + values[right]
        if current == target:
            return left, right
        if current < target:
            left += 1
        else:
            right -= 1
    return None


def max_container_brute(heights: Sequence[int]) -> int:
    """Return the maximum container area after checking every line pair."""

    _validate_heights(heights)
    best = 0
    for left in range(len(heights)):
        for right in range(left + 1, len(heights)):
            best = max(
                best,
                min(heights[left], heights[right]) * (right - left),
            )
    return best


def max_container(heights: Sequence[int]) -> int:
    """Return the maximum container area with opposing pointers."""

    _validate_heights(heights)
    left, right = 0, len(heights) - 1
    best = 0
    while left < right:
        best = max(best, min(heights[left], heights[right]) * (right - left))
        if heights[left] <= heights[right]:
            left += 1
        else:
            right -= 1
    return best


def _validate_heights(heights: Sequence[int]) -> None:
    if any(
        not isinstance(height, int) or isinstance(height, bool) or height < 0
        for height in heights
    ):
        raise ValueError("heights must contain non-negative integers")


def filter_copy(values: Sequence[T], keep: Callable[[T], bool]) -> list[T]:
    """Return a stable filtered copy as the extra-space reference."""

    return [value for value in values if keep(value)]


def filter_in_place(values: list[T], keep: Callable[[T], bool]) -> int:
    """Stably filter a list in place and return its resulting length."""

    write = 0
    for value in values:
        if keep(value):
            values[write] = value
            write += 1
    del values[write:]
    return write


def rolling_sums_brute(values: Sequence[float], width: int) -> list[float]:
    """Return fixed-window sums by recomputing every window."""

    _validate_width(width)
    return [
        sum(values[start : start + width])
        for start in range(len(values) - width + 1)
    ]


def rolling_sums(values: Sequence[float], width: int) -> list[float]:
    """Return fixed-window sums with one entering and one leaving update."""

    _validate_width(width)
    if width > len(values):
        return []
    current = sum(values[:width])
    result = [current]
    for right in range(width, len(values)):
        current += values[right] - values[right - width]
        result.append(current)
    return result


def _validate_width(width: int) -> None:
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("width must be a positive integer")


def longest_unique_substring_brute(text: str) -> str:
    """Return the earliest longest unique substring by enumerating starts."""

    best_start = 0
    best_length = 0
    for start in range(len(text)):
        seen: set[str] = set()
        for end in range(start, len(text)):
            if text[end] in seen:
                break
            seen.add(text[end])
            length = end - start + 1
            if length > best_length:
                best_start = start
                best_length = length
    return text[best_start : best_start + best_length]


def longest_unique_substring(text: str) -> str:
    """Return the earliest longest unique substring with a variable window."""

    last_seen: dict[str, int] = {}
    left = 0
    best_start = 0
    best_length = 0
    for right, character in enumerate(text):
        previous = last_seen.get(character)
        if previous is not None and previous >= left:
            left = previous + 1
        last_seen[character] = right
        length = right - left + 1
        if length > best_length:
            best_start = left
            best_length = length
    return text[best_start : best_start + best_length]


def minimum_length_brute(values: Sequence[int], target: int) -> int:
    """Return the shortest positive-value window reaching target by enumeration."""

    _validate_positive_window_input(values, target)
    best = len(values) + 1
    for left in range(len(values)):
        current = 0
        for right in range(left, len(values)):
            current += values[right]
            if current >= target:
                best = min(best, right - left + 1)
                break
    return 0 if best == len(values) + 1 else best


def minimum_length(values: Sequence[int], target: int) -> int:
    """Return the shortest positive-value window reaching target in linear time."""

    _validate_positive_window_input(values, target)
    left = 0
    current = 0
    best = len(values) + 1
    for right, value in enumerate(values):
        current += value
        while current >= target:
            best = min(best, right - left + 1)
            current -= values[left]
            left += 1
    return 0 if best == len(values) + 1 else best


def _validate_positive_window_input(values: Sequence[int], target: int) -> None:
    if not isinstance(target, int) or isinstance(target, bool) or target < 1:
        raise ValueError("target must be a positive integer")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in values
    ):
        raise ValueError("values must contain positive integers")
