"""Binary-search variants with explicit interval contracts and optional traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, Sequence, TypeVar


T = TypeVar("T")
IntervalKind = Literal["closed", "half_open"]
Decision = Literal["found", "discard_left", "discard_right", "keep_left"]


@dataclass(frozen=True)
class SearchStep(Generic[T]):
    """One observable transition of a binary-search interval."""

    algorithm: str
    interval: IntervalKind
    left: int
    right: int
    middle: int
    value: T
    decision: Decision
    next_left: int
    next_right: int

    @property
    def size(self) -> int:
        """Return the number of candidates before this transition."""

        if self.interval == "closed":
            return self.right - self.left + 1
        return self.right - self.left

    @property
    def next_size(self) -> int:
        """Return the number of candidates after this transition."""

        if self.decision == "found":
            return self.size
        if self.interval == "closed":
            return max(0, self.next_right - self.next_left + 1)
        return self.next_right - self.next_left


def _record(
    trace: list[SearchStep[T]] | None,
    *,
    algorithm: str,
    interval: IntervalKind,
    left: int,
    right: int,
    middle: int,
    value: T,
    decision: Decision,
    next_left: int,
    next_right: int,
) -> None:
    if trace is not None:
        trace.append(
            SearchStep(
                algorithm=algorithm,
                interval=interval,
                left=left,
                right=right,
                middle=middle,
                value=value,
                decision=decision,
                next_left=next_left,
                next_right=next_right,
            )
        )


def binary_search(
    values: Sequence[T], target: T, *, trace: list[SearchStep[T]] | None = None
) -> int:
    """Return any target index, or -1 when target is absent.

    Precondition: ``values`` is sorted in non-decreasing order.

    Interval contract: ``[left, right]`` is closed.
    Invariant: if target occurs and has not been returned, at least one
    occurrence remains inside ``[left, right]``.
    """

    left, right = 0, len(values) - 1
    while left <= right:
        middle = left + (right - left) // 2
        value = values[middle]
        if value == target:
            _record(
                trace,
                algorithm="binary_search",
                interval="closed",
                left=left,
                right=right,
                middle=middle,
                value=value,
                decision="found",
                next_left=left,
                next_right=right,
            )
            return middle
        if value < target:
            next_left, next_right = middle + 1, right
            decision: Decision = "discard_left"
        else:
            next_left, next_right = left, middle - 1
            decision = "discard_right"
        _record(
            trace,
            algorithm="binary_search",
            interval="closed",
            left=left,
            right=right,
            middle=middle,
            value=value,
            decision=decision,
            next_left=next_left,
            next_right=next_right,
        )
        left, right = next_left, next_right
    return -1


def lower_bound(
    values: Sequence[T], target: T, *, trace: list[SearchStep[T]] | None = None
) -> int:
    """Return the first index whose value is greater than or equal to target.

    Precondition: ``values`` is sorted in non-decreasing order.

    Interval contract: ``[left, right)`` is half-open.
    Invariant: indices before ``left`` contain values below target; indices at
    or after ``right`` contain values greater than or equal to target.
    """

    left, right = 0, len(values)
    while left < right:
        middle = left + (right - left) // 2
        value = values[middle]
        if value < target:
            next_left, next_right = middle + 1, right
            decision: Decision = "discard_left"
        else:
            next_left, next_right = left, middle
            decision = "keep_left"
        _record(
            trace,
            algorithm="lower_bound",
            interval="half_open",
            left=left,
            right=right,
            middle=middle,
            value=value,
            decision=decision,
            next_left=next_left,
            next_right=next_right,
        )
        left, right = next_left, next_right
    return left


def upper_bound(
    values: Sequence[T], target: T, *, trace: list[SearchStep[T]] | None = None
) -> int:
    """Return the first index whose value is strictly greater than target.

    Precondition: ``values`` is sorted in non-decreasing order.

    Interval contract: ``[left, right)`` is half-open.
    Invariant: indices before ``left`` contain values at most target; indices at
    or after ``right`` contain values strictly greater than target.
    """

    left, right = 0, len(values)
    while left < right:
        middle = left + (right - left) // 2
        value = values[middle]
        if value <= target:
            next_left, next_right = middle + 1, right
            decision: Decision = "discard_left"
        else:
            next_left, next_right = left, middle
            decision = "keep_left"
        _record(
            trace,
            algorithm="upper_bound",
            interval="half_open",
            left=left,
            right=right,
            middle=middle,
            value=value,
            decision=decision,
            next_left=next_left,
            next_right=next_right,
        )
        left, right = next_left, next_right
    return left


def insert_position(values: Sequence[T], target: T) -> int:
    """Return the stable left insertion position for target."""

    return lower_bound(values, target)


def first_occurrence(values: Sequence[T], target: T) -> int:
    """Return the first target index, or -1 when target is absent."""

    index = lower_bound(values, target)
    if index < len(values) and values[index] == target:
        return index
    return -1


def last_occurrence(values: Sequence[T], target: T) -> int:
    """Return the last target index, or -1 when target is absent."""

    index = upper_bound(values, target) - 1
    if index >= 0 and values[index] == target:
        return index
    return -1


def equal_range(values: Sequence[T], target: T) -> tuple[int, int]:
    """Return the half-open interval containing every target occurrence."""

    return lower_bound(values, target), upper_bound(values, target)


def count_occurrences(values: Sequence[T], target: T) -> int:
    """Return the number of target occurrences in logarithmic search time."""

    first, after_last = equal_range(values, target)
    return after_last - first
