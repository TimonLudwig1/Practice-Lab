"""An O(1) LRU cache built from a hash map and a doubly linked list."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterator, TypeVar


K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots=True, eq=False)
class _Node(Generic[K, V]):
    """A list node. Identity equality is essential for pointer invariants."""

    key: K | None = None
    value: V | None = None
    previous: _Node[K, V] | None = None
    next: _Node[K, V] | None = None


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Immutable snapshot of cache lookup statistics."""

    hits: int
    misses: int

    @property
    def requests(self) -> int:
        """Return the number of recorded lookups."""

        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Return hits divided by requests, or zero before the first lookup."""

        return self.hits / self.requests if self.requests else 0.0


@dataclass(frozen=True, slots=True)
class Evicted(Generic[K, V]):
    """The key-value pair removed by an insertion."""

    key: K
    value: V


@dataclass(frozen=True, slots=True)
class LookupResult(Generic[K, V]):
    """Result of a cache-aside lookup."""

    value: V
    hit: bool
    evicted: Evicted[K, V] | None = None


class LRUCache(Generic[K, V]):
    """Least-recently-used cache with constant-time get and put operations.

    A dictionary maps keys directly to list nodes. A circular doubly linked
    list stores the recency order, from least recent after the sentinel to most
    recent before it.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")

        self.capacity = capacity
        self._nodes: dict[K, _Node[K, V]] = {}
        self._root: _Node[K, V] = _Node()
        self._root.previous = self._root
        self._root.next = self._root
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, key: object) -> bool:
        return key in self._nodes

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys from least to most recently used."""

        node = self._root.next
        while node is not None and node is not self._root:
            yield node.key  # type: ignore[misc]
            node = node.next

    def __repr__(self) -> str:
        entries = ", ".join(f"{key!r}: {value!r}" for key, value in self.items_lru_to_mru())
        return f"LRUCache(capacity={self.capacity}, lru_to_mru={{{entries}}})"

    @property
    def stats(self) -> CacheStats:
        """Return a snapshot; later lookups do not mutate the snapshot."""

        return CacheStats(hits=self._hits, misses=self._misses)

    def get(self, key: K) -> V:
        """Return a value and mark it most recent; raise KeyError on a miss."""

        node = self._nodes.get(key)
        if node is None:
            self._misses += 1
            raise KeyError(key)

        self._hits += 1
        self._touch(node)
        return node.value  # type: ignore[return-value]

    def peek(self, key: K) -> V:
        """Return a value without changing recency or lookup statistics."""

        node = self._nodes.get(key)
        if node is None:
            raise KeyError(key)
        return node.value  # type: ignore[return-value]

    def put(self, key: K, value: V) -> Evicted[K, V] | None:
        """Insert or update a value and return an evicted pair if necessary."""

        node = self._nodes.get(key)
        if node is not None:
            node.value = value
            self._touch(node)
            return None

        node = _Node(key=key, value=value)
        self._nodes[key] = node
        self._append_mru(node)
        if len(self._nodes) <= self.capacity:
            return None
        return self._evict_lru()

    def get_or_compute(self, key: K, loader: Callable[[], V]) -> LookupResult[K, V]:
        """Return a cached value or compute, cache, and return a missing value."""

        try:
            return LookupResult(value=self.get(key), hit=True)
        except KeyError:
            value = loader()
            evicted = self.put(key, value)
            return LookupResult(value=value, hit=False, evicted=evicted)

    def delete(self, key: K) -> V:
        """Delete a key without affecting lookup statistics."""

        node = self._nodes.pop(key)
        self._unlink(node)
        return node.value  # type: ignore[return-value]

    def clear(self, *, reset_stats: bool = True) -> None:
        """Remove all entries and optionally preserve lookup statistics."""

        self._nodes.clear()
        self._root.previous = self._root
        self._root.next = self._root
        if reset_stats:
            self._hits = 0
            self._misses = 0

    def keys_lru_to_mru(self) -> tuple[K, ...]:
        """Return a stable snapshot of the recency order."""

        return tuple(self)

    def items_lru_to_mru(self) -> tuple[tuple[K, V], ...]:
        """Return key-value pairs from least to most recently used."""

        items: list[tuple[K, V]] = []
        node = self._root.next
        while node is not None and node is not self._root:
            items.append((node.key, node.value))  # type: ignore[arg-type]
            node = node.next
        return tuple(items)

    def check_invariants(self) -> bool:
        """Validate list links and the one-to-one map-to-node relationship."""

        root = self._root
        if root.next is None or root.previous is None:
            raise AssertionError("sentinel links must not be None")
        if root.next.previous is not root or root.previous.next is not root:
            raise AssertionError("sentinel links are not paired")

        visited: set[int] = set()
        node = root.next
        count = 0
        while node is not root:
            if node is None or node.previous is None or node.next is None:
                raise AssertionError("entry links must not be None")
            identity = id(node)
            if identity in visited:
                raise AssertionError("cycle does not close at the sentinel")
            visited.add(identity)
            if node.previous.next is not node or node.next.previous is not node:
                raise AssertionError("bidirectional links are inconsistent")
            if node.key not in self._nodes or self._nodes[node.key] is not node:
                raise AssertionError("dictionary does not reference this node")
            count += 1
            node = node.next

        if count != len(self._nodes):
            raise AssertionError("dictionary and list sizes differ")
        if count > self.capacity:
            raise AssertionError("cache exceeds its capacity")
        if {id(node) for node in self._nodes.values()} != visited:
            raise AssertionError("dictionary contains a node outside the list")
        return True

    def _unlink(self, node: _Node[K, V]) -> None:
        previous = node.previous
        following = node.next
        if previous is None or following is None:
            raise AssertionError("cannot unlink a detached node")
        previous.next = following
        following.previous = previous
        node.previous = None
        node.next = None

    def _append_mru(self, node: _Node[K, V]) -> None:
        previous = self._root.previous
        if previous is None:
            raise AssertionError("sentinel is corrupt")
        node.previous = previous
        node.next = self._root
        previous.next = node
        self._root.previous = node

    def _touch(self, node: _Node[K, V]) -> None:
        if node.next is self._root:
            return
        self._unlink(node)
        self._append_mru(node)

    def _evict_lru(self) -> Evicted[K, V]:
        node = self._root.next
        if node is None or node is self._root:
            raise AssertionError("cannot evict from an empty cache")
        self._unlink(node)
        del self._nodes[node.key]  # type: ignore[arg-type]
        return Evicted(key=node.key, value=node.value)  # type: ignore[arg-type]
