"""An array-based binary min-heap implemented without ``heapq``."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar


T = TypeVar("T")


class MinHeap(Generic[T]):
    """A binary min-heap for comparable values.

    Construction copies the input and applies bottom-up heapify. Duplicate
    values are allowed. Mutating a value after insertion is unsupported because
    it may silently invalidate the ordering invariant.
    """

    def __init__(self, values: Iterable[T] = ()) -> None:
        self._items: list[T] = []
        self.heapify(values)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return f"MinHeap({self._items!r})"

    def __iter__(self) -> Iterator[T]:
        """Yield values in ascending order without changing this heap."""

        copy = MinHeap(self._items)
        while copy:
            yield copy.pop()

    @property
    def size(self) -> int:
        """Return the number of stored values."""

        return len(self._items)

    def to_list(self) -> list[T]:
        """Return a shallow copy of the internal level-order array."""

        return self._items.copy()

    def clear(self) -> None:
        """Remove all values."""

        self._items.clear()

    def peek(self) -> T:
        """Return the minimum in O(1) time without removing it."""

        if not self._items:
            raise IndexError("peek from empty heap")
        return self._items[0]

    def push(self, value: T) -> None:
        """Insert *value* in O(log n) worst-case time."""

        self._items.append(value)
        self._sift_up(len(self._items) - 1)

    def pop(self) -> T:
        """Remove and return the minimum in O(log n) worst-case time."""

        if not self._items:
            raise IndexError("pop from empty heap")

        minimum = self._items[0]
        last = self._items.pop()
        if self._items:
            self._items[0] = last
            self._sift_down(0)
        return minimum

    def heapify(self, values: Iterable[T]) -> None:
        """Replace the contents and build a heap bottom-up in O(n) time."""

        self._items = list(values)
        last_parent = (len(self._items) - 2) // 2
        for parent in range(last_parent, -1, -1):
            self._sift_down(parent)

    def is_valid(self) -> bool:
        """Return whether every parent is no greater than its children."""

        for child in range(1, len(self._items)):
            parent = self.parent_index(child)
            if self._items[parent] > self._items[child]:
                return False
        return True

    @staticmethod
    def parent_index(index: int) -> int:
        """Return the parent index of a non-root node."""

        if index <= 0:
            raise ValueError("root has no parent")
        return (index - 1) // 2

    @staticmethod
    def left_child_index(index: int) -> int:
        """Return the potential left-child index."""

        if index < 0:
            raise ValueError("index must be non-negative")
        return 2 * index + 1

    @staticmethod
    def right_child_index(index: int) -> int:
        """Return the potential right-child index."""

        if index < 0:
            raise ValueError("index must be non-negative")
        return 2 * index + 2

    def _sift_up(self, child: int) -> None:
        while child > 0:
            parent = self.parent_index(child)
            if self._items[parent] <= self._items[child]:
                return
            self._items[parent], self._items[child] = (
                self._items[child],
                self._items[parent],
            )
            child = parent

    def _sift_down(self, parent: int) -> None:
        size = len(self._items)
        while True:
            left = self.left_child_index(parent)
            if left >= size:
                return

            right = self.right_child_index(parent)
            smaller = left
            if right < size and self._items[right] < self._items[left]:
                smaller = right

            if self._items[parent] <= self._items[smaller]:
                return
            self._items[parent], self._items[smaller] = (
                self._items[smaller],
                self._items[parent],
            )
            parent = smaller


def heap_sort(values: Iterable[T]) -> list[T]:
    """Return values ascending using the custom min-heap.

    This teaching version uses O(n) additional space so that sorting is composed
    entirely from the public heap operations: O(n) heapify followed by n pops.
    """

    heap = MinHeap(values)
    return [heap.pop() for _ in range(len(heap))]
