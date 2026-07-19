"""Five comparison sorting algorithms with metrics and optional traces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar


T = TypeVar("T")
KeyFunction = Callable[[T], Any]


@dataclass
class SortMetrics:
    """Mutable counters collected during one sorting run."""

    comparisons: int = 0
    swaps: int = 0
    writes: int = 0
    recursive_calls: int = 0
    max_recursion_depth: int = 0


@dataclass(frozen=True)
class TraceStep:
    """One observable mutation or merge during a sorting run."""

    algorithm: str
    action: str
    indices: tuple[int, ...]
    state: tuple[Any, ...]


def _identity(value: T) -> T:
    return value


def _prepare(
    key: KeyFunction[T] | None, metrics: SortMetrics | None
) -> tuple[KeyFunction[T], SortMetrics]:
    return (key or _identity), (metrics or SortMetrics())


def _trace(
    trace: list[TraceStep] | None,
    algorithm: str,
    action: str,
    indices: tuple[int, ...],
    state: list[T],
) -> None:
    if trace is not None:
        trace.append(TraceStep(algorithm, action, indices, tuple(state)))


def bubble_sort(
    values: list[T],
    *,
    key: KeyFunction[T] | None = None,
    metrics: SortMetrics | None = None,
    trace: list[TraceStep] | None = None,
) -> list[T]:
    """Return a stable bubble-sorted copy with an early-exit optimization."""
    key_function, counters = _prepare(key, metrics)
    result = values.copy()
    for end in range(len(result) - 1, 0, -1):
        swapped = False
        for index in range(end):
            counters.comparisons += 1
            if key_function(result[index + 1]) < key_function(result[index]):
                result[index], result[index + 1] = result[index + 1], result[index]
                counters.swaps += 1
                counters.writes += 2
                swapped = True
                _trace(trace, "bubble", "swap", (index, index + 1), result)
        if not swapped:
            _trace(trace, "bubble", "early_exit", (end,), result)
            break
    return result


def selection_sort(
    values: list[T],
    *,
    key: KeyFunction[T] | None = None,
    metrics: SortMetrics | None = None,
    trace: list[TraceStep] | None = None,
) -> list[T]:
    """Return a selection-sorted copy."""
    key_function, counters = _prepare(key, metrics)
    result = values.copy()
    for start in range(len(result) - 1):
        minimum = start
        for index in range(start + 1, len(result)):
            counters.comparisons += 1
            if key_function(result[index]) < key_function(result[minimum]):
                minimum = index
        if minimum != start:
            result[start], result[minimum] = result[minimum], result[start]
            counters.swaps += 1
            counters.writes += 2
            _trace(trace, "selection", "place_minimum", (start, minimum), result)
    return result


def insertion_sort(
    values: list[T],
    *,
    key: KeyFunction[T] | None = None,
    metrics: SortMetrics | None = None,
    trace: list[TraceStep] | None = None,
) -> list[T]:
    """Return a stable insertion-sorted copy."""
    key_function, counters = _prepare(key, metrics)
    result = values.copy()
    for index in range(1, len(result)):
        item = result[index]
        item_key = key_function(item)
        position = index - 1
        while position >= 0:
            counters.comparisons += 1
            if not item_key < key_function(result[position]):
                break
            result[position + 1] = result[position]
            counters.writes += 1
            _trace(trace, "insertion", "shift", (position, position + 1), result)
            position -= 1
        result[position + 1] = item
        counters.writes += 1
        _trace(trace, "insertion", "insert", (position + 1,), result)
    return result


def merge_sort(
    values: list[T],
    *,
    key: KeyFunction[T] | None = None,
    metrics: SortMetrics | None = None,
    trace: list[TraceStep] | None = None,
) -> list[T]:
    """Return a stable merge-sorted copy."""
    key_function, counters = _prepare(key, metrics)

    def sort_range(items: list[T], depth: int) -> list[T]:
        counters.recursive_calls += 1
        counters.max_recursion_depth = max(counters.max_recursion_depth, depth)
        if len(items) <= 1:
            return items.copy()
        middle = len(items) // 2
        left = sort_range(items[:middle], depth + 1)
        right = sort_range(items[middle:], depth + 1)
        merged: list[T] = []
        left_index = 0
        right_index = 0
        while left_index < len(left) and right_index < len(right):
            counters.comparisons += 1
            if key_function(right[right_index]) < key_function(left[left_index]):
                merged.append(right[right_index])
                right_index += 1
            else:
                merged.append(left[left_index])
                left_index += 1
            counters.writes += 1
        merged.extend(left[left_index:])
        merged.extend(right[right_index:])
        counters.writes += len(left) - left_index + len(right) - right_index
        _trace(trace, "merge", f"merge_depth_{depth}", (), merged)
        return merged

    return sort_range(values, 1)


def quick_sort(
    values: list[T],
    *,
    key: KeyFunction[T] | None = None,
    metrics: SortMetrics | None = None,
    trace: list[TraceStep] | None = None,
) -> list[T]:
    """Return a Lomuto-partitioned quick-sorted copy using the last pivot."""
    key_function, counters = _prepare(key, metrics)
    result = values.copy()

    def partition(low: int, high: int) -> int:
        pivot_key = key_function(result[high])
        boundary = low
        for index in range(low, high):
            counters.comparisons += 1
            if not pivot_key < key_function(result[index]):
                if boundary != index:
                    result[boundary], result[index] = result[index], result[boundary]
                    counters.swaps += 1
                    counters.writes += 2
                    _trace(trace, "quick", "partition_swap", (boundary, index), result)
                boundary += 1
        if boundary != high:
            result[boundary], result[high] = result[high], result[boundary]
            counters.swaps += 1
            counters.writes += 2
        _trace(trace, "quick", "place_pivot", (boundary, high), result)
        return boundary

    def sort_range(low: int, high: int, depth: int) -> None:
        counters.recursive_calls += 1
        counters.max_recursion_depth = max(counters.max_recursion_depth, depth)
        if low >= high:
            return
        pivot = partition(low, high)
        sort_range(low, pivot - 1, depth + 1)
        sort_range(pivot + 1, high, depth + 1)

    sort_range(0, len(result) - 1, 1)
    return result


ALGORITHMS = {
    "bubble": bubble_sort,
    "selection": selection_sort,
    "insertion": insertion_sort,
    "merge": merge_sort,
    "quick": quick_sort,
}
