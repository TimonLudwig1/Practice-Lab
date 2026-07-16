"""Classic array and string patterns with explicit complexity trade-offs."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from typing import TypeVar


T = TypeVar("T")
Number = int | float


def _reverse_range(values: MutableSequence[T], left: int, right: int) -> None:
    """Reverse an inclusive range without allocating another sequence."""
    while left < right:
        values[left], values[right] = values[right], values[left]
        left += 1
        right -= 1


def rotate_right_in_place(values: MutableSequence[T], steps: int) -> None:
    """Rotate values to the right using three reversals and O(1) extra space.

    Negative steps rotate to the left. The sequence is modified in place.
    """
    if not isinstance(steps, int):
        raise TypeError("steps must be an integer")

    size = len(values)
    if size < 2:
        return

    steps %= size
    if steps == 0:
        return

    _reverse_range(values, 0, size - 1)
    _reverse_range(values, 0, steps - 1)
    _reverse_range(values, steps, size - 1)


def _is_non_decreasing(values: Sequence[T]) -> bool:
    """Return whether every adjacent pair is ordered non-decreasingly."""
    return all(values[index - 1] <= values[index] for index in range(1, len(values)))


def merge_sorted_in_place(
    target: MutableSequence[T], valid_count: int, other: Sequence[T]
) -> None:
    """Merge two sorted ranges into target from right to left.

    ``target[:valid_count]`` and ``other`` contain the sorted inputs. The
    remaining target positions are buffer slots. The total target length must
    equal ``valid_count + len(other)``.
    """
    if not isinstance(valid_count, int):
        raise TypeError("valid_count must be an integer")
    if valid_count < 0 or valid_count > len(target):
        raise ValueError("valid_count is outside target")
    if len(target) != valid_count + len(other):
        raise ValueError("target must contain exactly enough buffer slots")
    if not _is_non_decreasing(target[:valid_count]):
        raise ValueError("the valid target range must be sorted")
    if not _is_non_decreasing(other):
        raise ValueError("other must be sorted")

    left = valid_count - 1
    right = len(other) - 1
    write = len(target) - 1

    while right >= 0:
        if left >= 0 and target[left] > other[right]:
            target[write] = target[left]
            left -= 1
        else:
            target[write] = other[right]
            right -= 1
        write -= 1


@dataclass(frozen=True)
class PrefixSum:
    """Immutable prefix-sum index for repeated half-open range queries."""

    _prefix: tuple[Number, ...]

    @classmethod
    def from_values(cls, values: Sequence[Number]) -> PrefixSum:
        """Build an index in O(n) time."""
        prefix: list[Number] = [0]
        for value in values:
            prefix.append(prefix[-1] + value)
        return cls(tuple(prefix))

    def __len__(self) -> int:
        return len(self._prefix) - 1

    def range_sum(self, start: int, end: int) -> Number:
        """Return the sum of values[start:end] in O(1) time."""
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("range boundaries must be integers")
        if start < 0 or end < start or end > len(self):
            raise IndexError("range must satisfy 0 <= start <= end <= length")
        return self._prefix[end] - self._prefix[start]


def are_anagrams(left: str, right: str) -> bool:
    """Return whether two strings contain identical Unicode code-point counts."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("both inputs must be strings")
    if len(left) != len(right):
        return False

    counts: dict[str, int] = {}
    for character in left:
        counts[character] = counts.get(character, 0) + 1

    for character in right:
        remaining = counts.get(character, 0) - 1
        if remaining < 0:
            return False
        if remaining == 0:
            counts.pop(character)
        else:
            counts[character] = remaining
    return not counts


def remove_duplicates_sorted(values: MutableSequence[T]) -> int:
    """Remove duplicates from a sorted sequence and return its new length."""
    if not _is_non_decreasing(values):
        raise ValueError("values must be sorted")
    if not values:
        return 0

    write = 1
    for read in range(1, len(values)):
        if values[read] != values[write - 1]:
            values[write] = values[read]
            write += 1

    del values[write:]
    return write


def move_zeros_to_end(values: MutableSequence[Number]) -> int:
    """Move numeric zeros to the end while preserving non-zero order.

    Return the number of non-zero elements.
    """
    write = 0
    for read in range(len(values)):
        if values[read] != 0:
            values[write], values[read] = values[read], values[write]
            write += 1
    return write


def product_except_self(values: Sequence[Number]) -> list[Number]:
    """Return all products except the current value without using division."""
    result: list[Number] = [1] * len(values)

    prefix: Number = 1
    for index, value in enumerate(values):
        result[index] = prefix
        prefix *= value

    suffix: Number = 1
    for index in range(len(values) - 1, -1, -1):
        result[index] *= suffix
        suffix *= values[index]
    return result


def longest_unique_substring(text: str) -> str:
    """Return the earliest longest substring with no repeated character."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    last_seen: dict[str, int] = {}
    window_start = 0
    best_start = 0
    best_length = 0

    for index, character in enumerate(text):
        previous = last_seen.get(character)
        if previous is not None and previous >= window_start:
            window_start = previous + 1
        last_seen[character] = index

        window_length = index - window_start + 1
        if window_length > best_length:
            best_start = window_start
            best_length = window_length

    return text[best_start : best_start + best_length]


def spiral_order(matrix: Sequence[Sequence[T]]) -> list[T]:
    """Return a rectangular matrix in clockwise spiral order."""
    if not matrix:
        return []

    column_count = len(matrix[0])
    if any(len(row) != column_count for row in matrix):
        raise ValueError("matrix must be rectangular")
    if column_count == 0:
        return []

    result: list[T] = []
    top = 0
    bottom = len(matrix) - 1
    left = 0
    right = column_count - 1

    while top <= bottom and left <= right:
        for column in range(left, right + 1):
            result.append(matrix[top][column])
        top += 1

        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        if top <= bottom:
            for column in range(right, left - 1, -1):
                result.append(matrix[bottom][column])
            bottom -= 1

        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result


def compress_runs_in_place(characters: MutableSequence[str]) -> int:
    """Run-length encode a character sequence in place and return its length."""
    if any(not isinstance(character, str) or len(character) != 1 for character in characters):
        raise ValueError("each element must be a single character")

    write = 0
    read = 0
    while read < len(characters):
        run_start = read
        current = characters[read]
        while read < len(characters) and characters[read] == current:
            read += 1

        characters[write] = current
        write += 1
        run_length = read - run_start
        if run_length > 1:
            for digit in str(run_length):
                characters[write] = digit
                write += 1

    del characters[write:]
    return write
