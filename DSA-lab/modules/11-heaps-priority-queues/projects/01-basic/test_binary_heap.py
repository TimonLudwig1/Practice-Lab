"""Unit and property tests for the custom binary min-heap."""

from __future__ import annotations

import heapq
import random

import pytest

from binary_heap import MinHeap, heap_sort


def assert_matches_reference(heap: MinHeap[int], reference: list[int]) -> None:
    """Check size, minimum, multiset, and the heap-order invariant."""

    assert len(heap) == len(reference)
    assert heap.size == len(reference)
    assert bool(heap) == bool(reference)
    assert heap.is_valid()
    assert sorted(heap.to_list()) == sorted(reference)
    if reference:
        assert heap.peek() == reference[0]


def test_empty_heap() -> None:
    heap = MinHeap[int]()

    assert len(heap) == 0
    assert not heap
    assert heap.size == 0
    assert heap.to_list() == []
    assert heap.is_valid()
    assert list(heap) == []
    assert repr(heap) == "MinHeap([])"


@pytest.mark.parametrize(("operation", "message"), [("peek", "peek"), ("pop", "pop")])
def test_empty_operation_raises(operation: str, message: str) -> None:
    heap = MinHeap[int]()

    with pytest.raises(IndexError, match=message):
        getattr(heap, operation)()


@pytest.mark.parametrize(
    ("index", "parent", "left", "right"),
    [
        (1, 0, 3, 4),
        (2, 0, 5, 6),
        (5, 2, 11, 12),
        (10, 4, 21, 22),
    ],
)
def test_index_arithmetic(index: int, parent: int, left: int, right: int) -> None:
    assert MinHeap.parent_index(index) == parent
    assert MinHeap.left_child_index(index) == left
    assert MinHeap.right_child_index(index) == right


@pytest.mark.parametrize("index", [0, -1])
def test_root_or_negative_index_has_no_parent(index: int) -> None:
    with pytest.raises(ValueError, match="no parent"):
        MinHeap.parent_index(index)


@pytest.mark.parametrize("method", [MinHeap.left_child_index, MinHeap.right_child_index])
def test_negative_child_index_is_rejected(method: object) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        method(-1)  # type: ignore[operator]


def test_bottom_up_heapify_known_layout() -> None:
    source = [9, 4, 7, 1, -2, 6, 5]

    heap = MinHeap(source)

    assert heap.to_list() == [-2, 1, 5, 9, 4, 6, 7]
    assert source == [9, 4, 7, 1, -2, 6, 5]
    assert heap.is_valid()


def test_heapify_replaces_existing_contents() -> None:
    heap = MinHeap([1, 2, 3])

    heap.heapify([8, 0, 4, -1])

    assert_matches_reference(heap, [-1, 0, 4, 8])


def test_heapify_accepts_generator() -> None:
    heap = MinHeap(value for value in range(10, -1, -1))

    assert list(heap) == list(range(11))


def test_push_sifts_to_root() -> None:
    heap = MinHeap([4, 7, 9, 10, 12, 15])

    heap.push(3)

    assert heap.to_list() == [3, 7, 4, 10, 12, 15, 9]
    assert heap.peek() == 3
    assert heap.is_valid()


@pytest.mark.parametrize(
    "values",
    [
        list(range(30)),
        list(range(29, -1, -1)),
        [5] * 30,
        [-5, 0, -10, 3, 3, -2],
    ],
)
def test_push_sequences(values: list[int]) -> None:
    heap = MinHeap[int]()
    reference: list[int] = []

    for value in values:
        heap.push(value)
        heapq.heappush(reference, value)
        assert_matches_reference(heap, reference)


def test_peek_does_not_mutate() -> None:
    heap = MinHeap([4, 1, 7])
    before = heap.to_list()

    assert heap.peek() == 1
    assert heap.to_list() == before


def test_pop_single_element() -> None:
    heap = MinHeap([42])

    assert heap.pop() == 42
    assert not heap
    assert heap.to_list() == []


def test_pop_sifts_down_through_multiple_levels() -> None:
    heap = MinHeap([2, 5, 4, 12, 9, 8, 7])

    assert heap.pop() == 2
    assert heap.to_list() == [4, 5, 7, 12, 9, 8]
    assert heap.is_valid()


@pytest.mark.parametrize(
    "values",
    [
        [8, 3, 10, 1, 6, 14, 4, 7, 13],
        [2, 2, 2, 1, 1],
        list(range(50)),
        list(range(49, -1, -1)),
    ],
)
def test_repeated_pop_is_sorted(values: list[int]) -> None:
    heap = MinHeap(values)

    popped = [heap.pop() for _ in range(len(heap))]

    assert popped == sorted(values)
    assert not heap


def test_clear() -> None:
    heap = MinHeap([3, 1, 2])

    heap.clear()

    assert not heap
    assert heap.to_list() == []
    heap.push(9)
    assert heap.peek() == 9


def test_to_list_is_a_defensive_copy() -> None:
    heap = MinHeap([3, 1, 2])
    snapshot = heap.to_list()

    snapshot[0] = 999

    assert heap.peek() == 1
    assert heap.is_valid()


def test_iteration_is_sorted_and_non_destructive() -> None:
    heap = MinHeap([5, 1, 4, 1, 3])
    before = heap.to_list()

    assert list(heap) == [1, 1, 3, 4, 5]
    assert heap.to_list() == before


def test_comparable_tuples_are_supported() -> None:
    values = [(2, "report"), (1, "hotfix"), (1, "alarm")]
    heap = MinHeap(values)

    assert list(heap) == sorted(values)


def test_strings_are_supported() -> None:
    heap = MinHeap(["pear", "apple", "plum", "banana"])

    assert list(heap) == ["apple", "banana", "pear", "plum"]


@pytest.mark.parametrize("seed", range(12))
def test_random_heapify_matches_heapq(seed: int) -> None:
    rng = random.Random(20260720 + seed)
    values = [rng.randrange(-1000, 1001) for _ in range(rng.randrange(0, 300))]
    heap = MinHeap(values)
    reference = values.copy()
    heapq.heapify(reference)

    assert_matches_reference(heap, reference)
    assert list(heap) == [heapq.heappop(reference) for _ in range(len(reference))]


def test_seeded_mixed_operations_match_heapq() -> None:
    rng = random.Random(20260720)
    heap = MinHeap[int]()
    reference: list[int] = []

    for _ in range(2_500):
        if not reference or rng.random() < 0.62:
            value = rng.randrange(-500, 501)
            heap.push(value)
            heapq.heappush(reference, value)
        else:
            assert heap.pop() == heapq.heappop(reference)
        assert_matches_reference(heap, reference)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [1],
        [2, 1],
        [7, 2, 9, 2, -1, 5],
        list(range(20)),
        list(range(19, -1, -1)),
    ],
)
def test_heap_sort_examples(values: list[int]) -> None:
    before = values.copy()

    assert heap_sort(values) == sorted(values)
    assert values == before


def test_heap_sort_seeded_property() -> None:
    rng = random.Random(1101)

    for _ in range(300):
        values = [rng.randrange(-50, 51) for _ in range(rng.randrange(0, 100))]
        assert heap_sort(values) == sorted(values)
