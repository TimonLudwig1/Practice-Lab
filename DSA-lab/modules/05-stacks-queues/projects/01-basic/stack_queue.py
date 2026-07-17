"""Array-backed stack and fixed-capacity circular queue implementations."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar, cast


T = TypeVar("T")
_EMPTY = object()


class StackUnderflowError(IndexError):
    """Raised when reading or removing an item from an empty stack."""


class QueueUnderflowError(IndexError):
    """Raised when reading or removing an item from an empty queue."""


class QueueOverflowError(OverflowError):
    """Raised when adding an item to a full fixed-capacity queue."""


class Stack(Generic[T]):
    """A LIFO stack backed by a dynamic Python array.

    ``push``, ``pop`` and ``peek`` operate at the end of the backing list. This
    makes ``push`` amortized O(1) and the other two operations worst-case O(1).
    """

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        """Place ``item`` on top of the stack."""
        self._items.append(item)

    def pop(self) -> T:
        """Remove and return the top item.

        Raises:
            StackUnderflowError: If the stack is empty.
        """
        if not self._items:
            raise StackUnderflowError("cannot pop from an empty stack")
        return self._items.pop()

    def peek(self) -> T:
        """Return the top item without removing it."""
        if not self._items:
            raise StackUnderflowError("cannot peek at an empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        """Return whether the stack contains no items."""
        return not self._items

    def to_list(self) -> list[T]:
        """Return a copy ordered from the bottom to the top."""
        return self._items.copy()

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return f"Stack(bottom_to_top={self._items!r})"


class FixedCapacityQueue(Generic[T]):
    """A FIFO queue implemented as a fixed-capacity circular buffer.

    The queue tracks only the head index and its current size. The next free
    slot is therefore ``(head + size) % capacity``. No elements are shifted,
    so enqueue, dequeue and peek are all worst-case O(1).
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")

        self._capacity = capacity
        self._buffer: list[object] = [_EMPTY] * capacity
        self._head = 0
        self._size = 0

    @property
    def capacity(self) -> int:
        """Return the queue's immutable maximum number of items."""
        return self._capacity

    def enqueue(self, item: T) -> None:
        """Add ``item`` at the back of the queue."""
        if self.is_full():
            raise QueueOverflowError(
                f"cannot enqueue into a full queue (capacity={self._capacity})"
            )

        tail = (self._head + self._size) % self._capacity
        self._buffer[tail] = item
        self._size += 1

    def dequeue(self) -> T:
        """Remove and return the oldest item."""
        if self.is_empty():
            raise QueueUnderflowError("cannot dequeue from an empty queue")

        item = self._buffer[self._head]
        self._buffer[self._head] = _EMPTY
        self._head = (self._head + 1) % self._capacity
        self._size -= 1

        # A canonical empty state makes debugging and invariant checks easier.
        if self._size == 0:
            self._head = 0

        return cast(T, item)

    def peek(self) -> T:
        """Return the oldest item without removing it."""
        if self.is_empty():
            raise QueueUnderflowError("cannot peek at an empty queue")
        return cast(T, self._buffer[self._head])

    def is_empty(self) -> bool:
        """Return whether the queue contains no items."""
        return self._size == 0

    def is_full(self) -> bool:
        """Return whether the queue has reached its fixed capacity."""
        return self._size == self._capacity

    def to_list(self) -> list[T]:
        """Return the logical FIFO order without modifying the queue."""
        return [
            cast(T, self._buffer[(self._head + offset) % self._capacity])
            for offset in range(self._size)
        ]

    def __iter__(self) -> Iterator[T]:
        return iter(self.to_list())

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __repr__(self) -> str:
        return (
            "FixedCapacityQueue("
            f"front_to_back={self.to_list()!r}, capacity={self._capacity})"
        )
