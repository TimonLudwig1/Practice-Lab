"""A dynamic array backed by a fixed-size ctypes buffer."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar, cast

T = TypeVar("T")


@dataclass(frozen=True)
class GrowthEvent:
    """One buffer replacement caused by capacity exhaustion."""

    length_before: int
    old_capacity: int
    new_capacity: int
    copied_elements: int


class DynamicArray(Generic[T]):
    """Store Python objects in a manually resized contiguous reference buffer.

    The Python list used for growth events is instrumentation only. User
    elements are stored exclusively inside a fixed-size ctypes.py_object array.
    """

    def __init__(self, initial_capacity: int = 1) -> None:
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be at least 1")

        self._size = 0
        self._capacity = initial_capacity
        self._buffer = self._make_buffer(initial_capacity)
        self._growth_events: list[GrowthEvent] = []
        self._total_copied_elements = 0

    @staticmethod
    def _make_buffer(capacity: int):
        """Allocate a fixed-size contiguous array of Python object references."""
        buffer_type = ctypes.py_object * capacity
        return buffer_type()

    def __len__(self) -> int:
        """Return the number of active elements."""
        return self._size

    @property
    def capacity(self) -> int:
        """Return the number of positions in the current fixed buffer."""
        return self._capacity

    @property
    def growth_events(self) -> tuple[GrowthEvent, ...]:
        """Return an immutable snapshot of all automatic resizes."""
        return tuple(self._growth_events)

    @property
    def total_copied_elements(self) -> int:
        """Return how many element references all resizes copied."""
        return self._total_copied_elements

    def __getitem__(self, index: int) -> T:
        """Return the element at index, supporting negative indices."""
        normalized_index = self._normalize_index(index)
        return cast(T, self._buffer[normalized_index])

    def __setitem__(self, index: int, value: T) -> None:
        """Replace the element at index."""
        normalized_index = self._normalize_index(index)
        self._buffer[normalized_index] = value

    def __iter__(self) -> Iterator[T]:
        """Yield active elements in logical order."""
        for index in range(self._size):
            yield cast(T, self._buffer[index])

    def __repr__(self) -> str:
        """Return a concise representation with length and capacity."""
        return (
            f"DynamicArray({self.to_list()!r}, "
            f"capacity={self._capacity})"
        )

    def append(self, value: T) -> None:
        """Append one value, doubling capacity when the buffer is full."""
        self._ensure_capacity()
        self._buffer[self._size] = value
        self._size += 1

    def insert(self, index: int, value: T) -> None:
        """Insert value before a nonnegative logical index."""
        if not 0 <= index <= self._size:
            raise IndexError("insert index out of range")

        self._ensure_capacity()
        for position in range(self._size, index, -1):
            self._buffer[position] = self._buffer[position - 1]

        self._buffer[index] = value
        self._size += 1

    def delete(self, index: int) -> T:
        """Delete and return the element at index."""
        normalized_index = self._normalize_index(index)
        removed = cast(T, self._buffer[normalized_index])

        for position in range(normalized_index, self._size - 1):
            self._buffer[position] = self._buffer[position + 1]

        self._size -= 1
        self._buffer[self._size] = None
        return removed

    def to_list(self) -> list[T]:
        """Return a Python-list snapshot for inspection and interoperability."""
        return [cast(T, self._buffer[index]) for index in range(self._size)]

    def _ensure_capacity(self) -> None:
        """Double the fixed buffer if no free position remains."""
        if self._size == self._capacity:
            self._resize(self._capacity * 2)

    def _resize(self, new_capacity: int) -> None:
        """Replace the current buffer and copy all active references."""
        if new_capacity <= self._capacity:
            raise ValueError("new_capacity must exceed current capacity")

        old_capacity = self._capacity
        new_buffer = self._make_buffer(new_capacity)
        for index in range(self._size):
            new_buffer[index] = self._buffer[index]

        event = GrowthEvent(
            length_before=self._size,
            old_capacity=old_capacity,
            new_capacity=new_capacity,
            copied_elements=self._size,
        )
        self._buffer = new_buffer
        self._capacity = new_capacity
        self._growth_events.append(event)
        self._total_copied_elements += event.copied_elements

    def _normalize_index(self, index: int) -> int:
        """Normalize a negative index and enforce active bounds."""
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError("array index out of range")
        return index
