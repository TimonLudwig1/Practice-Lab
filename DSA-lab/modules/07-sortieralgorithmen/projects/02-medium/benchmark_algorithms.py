"""Self-contained instrumented algorithms for systematic benchmarking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ComparisonCounter:
    comparisons: int = 0


SortAlgorithm = Callable[[list[int], ComparisonCounter], list[int]]


def bubble_sort(values: list[int], counter: ComparisonCounter) -> list[int]:
    result = values.copy()
    for end in range(len(result) - 1, 0, -1):
        swapped = False
        for index in range(end):
            counter.comparisons += 1
            if result[index + 1] < result[index]:
                result[index], result[index + 1] = result[index + 1], result[index]
                swapped = True
        if not swapped:
            break
    return result


def selection_sort(values: list[int], counter: ComparisonCounter) -> list[int]:
    result = values.copy()
    for start in range(len(result) - 1):
        minimum = start
        for index in range(start + 1, len(result)):
            counter.comparisons += 1
            if result[index] < result[minimum]:
                minimum = index
        result[start], result[minimum] = result[minimum], result[start]
    return result


def insertion_sort(values: list[int], counter: ComparisonCounter) -> list[int]:
    result = values.copy()
    for index in range(1, len(result)):
        item = result[index]
        position = index - 1
        while position >= 0:
            counter.comparisons += 1
            if result[position] <= item:
                break
            result[position + 1] = result[position]
            position -= 1
        result[position + 1] = item
    return result


def merge_sort(values: list[int], counter: ComparisonCounter) -> list[int]:
    if len(values) <= 1:
        return values.copy()
    middle = len(values) // 2
    left = merge_sort(values[:middle], counter)
    right = merge_sort(values[middle:], counter)
    result: list[int] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        counter.comparisons += 1
        if left[left_index] <= right[right_index]:
            result.append(left[left_index])
            left_index += 1
        else:
            result.append(right[right_index])
            right_index += 1
    result.extend(left[left_index:])
    result.extend(right[right_index:])
    return result


def quick_sort_three_way(values: list[int], counter: ComparisonCounter) -> list[int]:
    """Use three-way partitioning to avoid duplicate-heavy degeneration."""
    if len(values) <= 1:
        return values.copy()
    pivot = values[len(values) // 2]
    lower: list[int] = []
    equal: list[int] = []
    higher: list[int] = []
    for value in values:
        counter.comparisons += 1
        if value < pivot:
            lower.append(value)
            continue
        counter.comparisons += 1
        if pivot < value:
            higher.append(value)
        else:
            equal.append(value)
    return (
        quick_sort_three_way(lower, counter)
        + equal
        + quick_sort_three_way(higher, counter)
    )


class _TrackedInt:
    """Integer wrapper that counts comparisons performed by Timsort."""

    __slots__ = ("value", "counter")

    def __init__(self, value: int, counter: ComparisonCounter) -> None:
        self.value = value
        self.counter = counter

    def __lt__(self, other: _TrackedInt) -> bool:
        self.counter.comparisons += 1
        return self.value < other.value


def python_timsort(values: list[int], counter: ComparisonCounter) -> list[int]:
    tracked = [_TrackedInt(value, counter) for value in values]
    return [item.value for item in sorted(tracked)]


ALGORITHMS: dict[str, SortAlgorithm] = {
    "bubble": bubble_sort,
    "selection": selection_sort,
    "insertion": insertion_sort,
    "merge": merge_sort,
    "quick_3way": quick_sort_three_way,
    "python_timsort": python_timsort,
}
