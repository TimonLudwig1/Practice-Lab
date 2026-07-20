"""Reusable heap patterns: Top-K, k-way merge, and running median."""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator
from typing import Generic, TypeVar


H = TypeVar("H", bound=Hashable)
T = TypeVar("T")


def top_k_frequent(values: Iterable[H], k: int) -> list[tuple[H, int]]:
    """Return up to *k* most frequent values with deterministic ties.

    Results are ordered by frequency descending. Equal frequencies keep the
    order in which values first appeared. The heap contains at most *k* unique
    values; its root is the worst current winner.
    """

    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return []

    counts: Counter[H] = Counter()
    first_position: dict[H, int] = {}
    for position, value in enumerate(values):
        counts[value] += 1
        first_position.setdefault(value, position)

    # Lower frequency is worse. On a tie, the later first position is worse;
    # negation places that candidate nearer the min-heap root.
    heap: list[tuple[int, int, int, H]] = []
    for value, frequency in counts.items():
        first = first_position[value]
        candidate = (frequency, -first, first, value)
        if len(heap) < k:
            heapq.heappush(heap, candidate)
        elif candidate[:2] > heap[0][:2]:
            heapq.heapreplace(heap, candidate)

    ordered = sorted(heap, key=lambda entry: (-entry[0], entry[2]))
    return [(value, frequency) for frequency, _, _, value in ordered]


def merge_sorted(sequences: Iterable[Iterable[T]]) -> list[T]:
    """Merge sorted iterables using one current head per active iterable."""

    heap: list[tuple[T, int, Iterator[T]]] = []
    for sequence_index, sequence in enumerate(sequences):
        iterator = iter(sequence)
        try:
            first = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (first, sequence_index, iterator))

    result: list[T] = []
    while heap:
        value, sequence_index, iterator = heapq.heappop(heap)
        result.append(value)
        try:
            next_value = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (next_value, sequence_index, iterator))
    return result


class RunningMedian:
    """Maintain exact medians of a numeric stream with two heaps."""

    def __init__(self, values: Iterable[float] = ()) -> None:
        self._lower: list[float] = []  # Negated values form a max-heap.
        self._upper: list[float] = []
        self.extend(values)

    def __len__(self) -> int:
        return len(self._lower) + len(self._upper)

    def add(self, value: float) -> None:
        """Add one finite numeric value in O(log n) time."""

        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("running median accepts only finite values")

        if not self._lower or numeric <= -self._lower[0]:
            heapq.heappush(self._lower, -numeric)
        else:
            heapq.heappush(self._upper, numeric)
        self._rebalance()

    def extend(self, values: Iterable[float]) -> None:
        """Add all values in iteration order."""

        for value in values:
            self.add(value)

    def median(self) -> float:
        """Return the current median in O(1) time."""

        if not self._lower:
            raise ValueError("median is undefined for an empty stream")
        if len(self._lower) == len(self._upper):
            return (-self._lower[0] + self._upper[0]) / 2.0
        return -self._lower[0]

    def halves(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return sorted snapshots of the lower and upper halves."""

        lower = tuple(sorted(-value for value in self._lower))
        upper = tuple(sorted(self._upper))
        return lower, upper

    def is_valid(self) -> bool:
        """Check heap, ordering, and size invariants."""

        size_valid = len(self._lower) in {
            len(self._upper),
            len(self._upper) + 1,
        }
        lower_heap_valid = all(
            self._lower[(child - 1) // 2] <= self._lower[child]
            for child in range(1, len(self._lower))
        )
        upper_heap_valid = all(
            self._upper[(child - 1) // 2] <= self._upper[child]
            for child in range(1, len(self._upper))
        )
        order_valid = (
            not self._lower
            or not self._upper
            or -self._lower[0] <= self._upper[0]
        )
        return size_valid and lower_heap_valid and upper_heap_valid and order_valid

    def _rebalance(self) -> None:
        if len(self._lower) > len(self._upper) + 1:
            heapq.heappush(self._upper, -heapq.heappop(self._lower))
        elif len(self._upper) > len(self._lower):
            heapq.heappush(self._lower, -heapq.heappop(self._upper))
