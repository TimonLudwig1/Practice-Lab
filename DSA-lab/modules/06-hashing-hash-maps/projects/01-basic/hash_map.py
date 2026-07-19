"""A generic hash map implemented with separate chaining."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots=True)
class _Entry(Generic[K, V]):
    key: K
    value: V


@dataclass(frozen=True)
class HashMapStats:
    """A snapshot of the map's bucket distribution."""

    size: int
    capacity: int
    load_factor: float
    non_empty_buckets: int
    collision_count: int
    max_chain_length: int
    rehash_count: int


class ChainedHashMap(Generic[K, V]):
    """Hash map with separate chaining and optional geometric rehashing.

    The implementation deliberately avoids Python's ``dict`` and ``set``. A
    custom hash function can be injected to make collision scenarios testable.
    """

    def __init__(
        self,
        initial_capacity: int = 8,
        *,
        max_load_factor: float = 0.75,
        enable_rehash: bool = True,
        hash_function: Callable[[K], int] = hash,
    ) -> None:
        if isinstance(initial_capacity, bool) or not isinstance(initial_capacity, int):
            raise TypeError("initial_capacity must be an integer")
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be greater than zero")
        if not 0 < max_load_factor <= 1:
            raise ValueError("max_load_factor must be in the interval (0, 1]")
        if not callable(hash_function):
            raise TypeError("hash_function must be callable")

        self._buckets: list[list[_Entry[K, V]]] = [
            [] for _ in range(initial_capacity)
        ]
        self._size = 0
        self._max_load_factor = max_load_factor
        self._enable_rehash = enable_rehash
        self._hash_function = hash_function
        self._rehash_count = 0

    @property
    def capacity(self) -> int:
        """Return the current number of buckets."""
        return len(self._buckets)

    @property
    def load_factor(self) -> float:
        """Return entries divided by buckets."""
        return self._size / self.capacity

    @property
    def rehash_count(self) -> int:
        """Return how often the bucket array has grown."""
        return self._rehash_count

    def _index(self, key: K, capacity: int | None = None) -> int:
        bucket_count = self.capacity if capacity is None else capacity
        hash_value = self._hash_function(key)
        if isinstance(hash_value, bool) or not isinstance(hash_value, int):
            raise TypeError("hash_function must return an integer")
        return hash_value % bucket_count

    def _entry_position(self, bucket: list[_Entry[K, V]], key: K) -> int | None:
        for position, entry in enumerate(bucket):
            if entry.key == key:
                return position
        return None

    def put(self, key: K, value: V) -> None:
        """Insert a new key or replace the value of an existing key."""
        bucket = self._buckets[self._index(key)]
        position = self._entry_position(bucket, key)
        if position is not None:
            bucket[position].value = value
            return

        projected_load = (self._size + 1) / self.capacity
        if self._enable_rehash and projected_load > self._max_load_factor:
            self._resize(self.capacity * 2)
            bucket = self._buckets[self._index(key)]

        bucket.append(_Entry(key, value))
        self._size += 1

    def get(self, key: K) -> V:
        """Return the value for ``key`` or raise ``KeyError``."""
        bucket = self._buckets[self._index(key)]
        position = self._entry_position(bucket, key)
        if position is None:
            raise KeyError(key)
        return bucket[position].value

    def delete(self, key: K) -> V:
        """Remove ``key`` and return its previous value."""
        bucket = self._buckets[self._index(key)]
        position = self._entry_position(bucket, key)
        if position is None:
            raise KeyError(key)
        entry = bucket.pop(position)
        self._size -= 1
        return entry.value

    def _resize(self, new_capacity: int) -> None:
        old_buckets = self._buckets
        self._buckets = [[] for _ in range(new_capacity)]
        for bucket in old_buckets:
            for entry in bucket:
                self._buckets[self._index(entry.key)].append(entry)
        self._rehash_count += 1

    def clear(self) -> None:
        """Remove all entries while retaining the current capacity."""
        self._buckets = [[] for _ in range(self.capacity)]
        self._size = 0

    def keys(self) -> Iterator[K]:
        """Yield every stored key in bucket order."""
        for bucket in self._buckets:
            for entry in bucket:
                yield entry.key

    def values(self) -> Iterator[V]:
        """Yield every stored value in bucket order."""
        for bucket in self._buckets:
            for entry in bucket:
                yield entry.value

    def items(self) -> Iterator[tuple[K, V]]:
        """Yield key-value pairs in bucket order."""
        for bucket in self._buckets:
            for entry in bucket:
                yield entry.key, entry.value

    def stats(self) -> HashMapStats:
        """Return collision and bucket-distribution metrics."""
        chain_lengths = [len(bucket) for bucket in self._buckets]
        non_empty = sum(length > 0 for length in chain_lengths)
        return HashMapStats(
            size=self._size,
            capacity=self.capacity,
            load_factor=self.load_factor,
            non_empty_buckets=non_empty,
            collision_count=self._size - non_empty,
            max_chain_length=max(chain_lengths, default=0),
            rehash_count=self._rehash_count,
        )

    def check_invariants(self) -> None:
        """Raise ``AssertionError`` if an internal map invariant is broken."""
        entries: list[_Entry[K, V]] = []
        for bucket_index, bucket in enumerate(self._buckets):
            for entry in bucket:
                assert self._index(entry.key) == bucket_index
                assert not any(existing.key == entry.key for existing in entries)
                entries.append(entry)
        assert len(entries) == self._size

    def __getitem__(self, key: K) -> V:
        return self.get(key)

    def __setitem__(self, key: K, value: V) -> None:
        self.put(key, value)

    def __delitem__(self, key: K) -> None:
        self.delete(key)

    def __contains__(self, key: object) -> bool:
        try:
            self.get(key)  # type: ignore[arg-type]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[K]:
        return self.keys()

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __repr__(self) -> str:
        return (
            f"ChainedHashMap({list(self.items())!r}, "
            f"capacity={self.capacity}, load_factor={self.load_factor:.3f})"
        )
