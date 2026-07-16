"""A traced divide-and-conquer toolkit for three classic problems."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Number, Real
from typing import TypeVar


T = TypeVar("T")


def _record(trace: list[str] | None, depth: int, message: str) -> None:
    """Append one indented line when tracing is enabled."""
    if trace is not None:
        trace.append(f"{'  ' * depth}{message}")


def binary_power(
    base: Number, exponent: int, *, trace: list[str] | None = None
) -> Number:
    """Return base**exponent using recursive exponentiation by squaring."""
    if isinstance(base, bool) or not isinstance(base, Number):
        raise TypeError("base must be a number")
    if not isinstance(exponent, int) or isinstance(exponent, bool):
        raise TypeError("exponent must be an integer")
    if base == 0 and exponent < 0:
        raise ZeroDivisionError("zero cannot be raised to a negative exponent")

    def solve(current_exponent: int, depth: int) -> Number:
        _record(trace, depth, f"power(e={current_exponent})")
        if current_exponent == 0:
            _record(trace, depth, "return 1")
            return 1

        half = solve(current_exponent // 2, depth + 1)
        result = half * half
        if current_exponent % 2 == 1:
            result *= base
        _record(trace, depth, f"return {result!r}")
        return result

    magnitude = solve(abs(exponent), 0)
    if exponent < 0:
        result = 1 / magnitude
        _record(trace, 0, f"reciprocal = {result!r}")
        return result
    return magnitude


@dataclass(frozen=True)
class SubarrayResult:
    """Best half-open subarray range and its sum."""

    start: int
    end: int
    total: Real

    @property
    def length(self) -> int:
        """Return the number of values in the selected range."""
        return self.end - self.start


def _result_key(result: SubarrayResult) -> tuple[Real, int, int]:
    """Rank by sum, then earliest start, then shortest length."""
    return result.total, -result.start, -result.length


def _crossing_subarray(
    values: Sequence[Real], left: int, middle: int, right: int
) -> SubarrayResult:
    """Return the best subarray crossing the split at middle."""
    running: Real = 0
    best_left_sum: Real | None = None
    best_start = middle - 1
    for index in range(middle - 1, left - 1, -1):
        running += values[index]
        if best_left_sum is None or running > best_left_sum or (
            running == best_left_sum and index < best_start
        ):
            best_left_sum = running
            best_start = index

    running = 0
    best_right_sum: Real | None = None
    best_end = middle
    for index in range(middle, right):
        running += values[index]
        if best_right_sum is None or running > best_right_sum:
            best_right_sum = running
            best_end = index + 1

    assert best_left_sum is not None and best_right_sum is not None
    return SubarrayResult(best_start, best_end, best_left_sum + best_right_sum)


def maximum_subarray(
    values: Sequence[Real], *, trace: list[str] | None = None
) -> SubarrayResult:
    """Find a maximum-sum non-empty subarray via divide and conquer."""
    if not values:
        raise ValueError("values must not be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("values must contain real numbers")
        if not math.isfinite(float(value)):
            raise ValueError("values must be finite")

    def solve(left: int, right: int, depth: int) -> SubarrayResult:
        _record(trace, depth, f"segment[{left}:{right})")
        if right - left == 1:
            result = SubarrayResult(left, right, values[left])
            _record(trace, depth, f"leaf -> {result}")
            return result

        middle = (left + right) // 2
        left_result = solve(left, middle, depth + 1)
        right_result = solve(middle, right, depth + 1)
        crossing_result = _crossing_subarray(values, left, middle, right)
        result = max(
            (left_result, right_result, crossing_result), key=_result_key
        )
        _record(
            trace,
            depth,
            f"choose [{result.start}:{result.end}) sum={result.total!r}",
        )
        return result

    return solve(0, len(values), 0)


def count_inversions(
    values: Sequence[T], *, trace: list[str] | None = None
) -> tuple[list[T], int]:
    """Return a sorted copy and the number of index inversions in O(n log n)."""

    def solve(items: list[T], depth: int) -> tuple[list[T], int]:
        _record(trace, depth, f"sort {items!r}")
        if len(items) < 2:
            _record(trace, depth, f"return {items!r}, inversions=0")
            return items, 0

        middle = len(items) // 2
        left, left_count = solve(items[:middle], depth + 1)
        right, right_count = solve(items[middle:], depth + 1)

        merged: list[T] = []
        left_index = 0
        right_index = 0
        split_count = 0

        while left_index < len(left) and right_index < len(right):
            if left[left_index] <= right[right_index]:
                merged.append(left[left_index])
                left_index += 1
            else:
                merged.append(right[right_index])
                right_index += 1
                split_count += len(left) - left_index

        merged.extend(left[left_index:])
        merged.extend(right[right_index:])
        total = left_count + right_count + split_count
        _record(trace, depth, f"merge -> {merged!r}, inversions={total}")
        return merged, total

    return solve(list(values), 0)
