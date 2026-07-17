"""A complete singly linked list implemented without container shortcuts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(slots=True, eq=False)
class Node(Generic[T]):
    """One identity-based node in a singly linked list."""

    value: T
    next: Node[T] | None = None


class SinglyLinkedList(Generic[T]):
    """A singly linked list with head, tail, and constant-time length."""

    __slots__ = ("_head", "_tail", "_size")

    def __init__(self, values: Iterable[T] = ()) -> None:
        self._head: Node[T] | None = None
        self._tail: Node[T] | None = None
        self._size = 0
        for value in values:
            self.append(value)

    @property
    def head(self) -> Node[T] | None:
        """Return the first node, or None for an empty list."""
        return self._head

    @property
    def tail(self) -> Node[T] | None:
        """Return the last node, or None for an empty list."""
        return self._tail

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __iter__(self) -> Iterator[T]:
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    def __contains__(self, value: object) -> bool:
        return self.find(value) is not None

    def __repr__(self) -> str:
        return f"SinglyLinkedList({self.to_list()!r})"

    def _validate_integer_index(self, index: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("index must be an integer")

    def _normalize_access_index(self, index: int) -> int:
        self._validate_integer_index(index)
        if index < 0:
            index += self._size
        if index < 0 or index >= self._size:
            raise IndexError("linked-list index out of range")
        return index

    def _node_at_normalized(self, index: int) -> Node[T]:
        current = self._head
        for _ in range(index):
            assert current is not None
            current = current.next
        assert current is not None
        return current

    def node_at(self, index: int) -> Node[T]:
        """Return the node at index after Python-style negative normalization."""
        return self._node_at_normalized(self._normalize_access_index(index))

    def __getitem__(self, index: int) -> T:
        return self.node_at(index).value

    def append(self, value: T) -> Node[T]:
        """Append value in O(1) time and return the new tail node."""
        node = Node(value)
        if self._tail is None:
            self._head = node
        else:
            self._tail.next = node
        self._tail = node
        self._size += 1
        return node

    def prepend(self, value: T) -> Node[T]:
        """Prepend value in O(1) time and return the new head node."""
        node = Node(value, self._head)
        self._head = node
        if self._tail is None:
            self._tail = node
        self._size += 1
        return node

    def insert(self, index: int, value: T) -> Node[T]:
        """Insert before non-negative index in [0, len] and return the node."""
        self._validate_integer_index(index)
        if index < 0 or index > self._size:
            raise IndexError("insert index must satisfy 0 <= index <= length")
        if index == 0:
            return self.prepend(value)
        if index == self._size:
            return self.append(value)

        previous = self._node_at_normalized(index - 1)
        node = Node(value, previous.next)
        previous.next = node
        self._size += 1
        return node

    def delete(self, index: int) -> T:
        """Delete at index, detach the node, and return its value."""
        normalized = self._normalize_access_index(index)

        if normalized == 0:
            target = self._head
            assert target is not None
            self._head = target.next
            if self._size == 1:
                self._tail = None
        else:
            previous = self._node_at_normalized(normalized - 1)
            target = previous.next
            assert target is not None
            previous.next = target.next
            if target is self._tail:
                self._tail = previous

        target.next = None
        self._size -= 1
        return target.value

    def remove(self, value: T) -> T:
        """Delete and return the first matching value, or raise ValueError."""
        previous: Node[T] | None = None
        current = self._head

        while current is not None and current.value != value:
            previous = current
            current = current.next
        if current is None:
            raise ValueError(f"{value!r} is not in linked list")

        if previous is None:
            self._head = current.next
        else:
            previous.next = current.next
        if current is self._tail:
            self._tail = previous

        current.next = None
        self._size -= 1
        return current.value

    def find(self, value: object) -> int | None:
        """Return the first index containing value, or None."""
        current = self._head
        index = 0
        while current is not None:
            if current.value == value:
                return index
            current = current.next
            index += 1
        return None

    def clear(self) -> None:
        """Detach every node and reset the list to its empty invariants."""
        current = self._head
        while current is not None:
            next_node = current.next
            current.next = None
            current = next_node
        self._head = None
        self._tail = None
        self._size = 0

    def to_list(self) -> list[T]:
        """Return a Python-list snapshot of all values."""
        return list(self)

    def check_invariants(self) -> None:
        """Raise AssertionError if head, tail, size, reachability, or links disagree."""
        assert self._size >= 0, "size must be non-negative"
        if self._size == 0:
            assert self._head is None, "empty list must not have a head"
            assert self._tail is None, "empty list must not have a tail"
            return

        assert self._head is not None, "non-empty list must have a head"
        assert self._tail is not None, "non-empty list must have a tail"
        assert self._tail.next is None, "tail.next must be None"

        seen: set[int] = set()
        count = 0
        current: Node[T] | None = self._head
        last: Node[T] | None = None
        while current is not None:
            identity = id(current)
            assert identity not in seen, "list must not contain a cycle"
            seen.add(identity)
            count += 1
            last = current
            current = current.next

        assert count == self._size, "reachable node count must equal size"
        assert last is self._tail, "last reachable node must be tail"
