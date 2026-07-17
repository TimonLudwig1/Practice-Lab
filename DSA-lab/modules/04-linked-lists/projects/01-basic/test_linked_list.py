"""Pytest suite for all singly linked list operations and invariants."""

from __future__ import annotations

import pytest

from linked_list import Node, SinglyLinkedList


def assert_valid(linked: SinglyLinkedList[object], expected: list[object]) -> None:
    """Assert values, length, endpoints, terminal link, and internal invariants."""
    linked.check_invariants()
    assert linked.to_list() == expected
    assert list(linked) == expected
    assert len(linked) == len(expected)
    assert bool(linked) is bool(expected)
    if expected:
        assert linked.head is not None and linked.head.value == expected[0]
        assert linked.tail is not None and linked.tail.value == expected[-1]
        assert linked.tail.next is None
    else:
        assert linked.head is None
        assert linked.tail is None


class TestConstructionAndRepresentation:
    def test_empty_initial_state(self) -> None:
        linked: SinglyLinkedList[int] = SinglyLinkedList()
        assert_valid(linked, [])

    def test_constructs_from_iterable_in_order(self) -> None:
        linked = SinglyLinkedList(value * 2 for value in range(4))
        assert_valid(linked, [0, 2, 4, 6])

    def test_repr_is_unambiguous(self) -> None:
        assert repr(SinglyLinkedList([1, "x"])) == "SinglyLinkedList([1, 'x'])"
        assert repr(SinglyLinkedList()) == "SinglyLinkedList([])"

    def test_node_equality_uses_identity(self) -> None:
        first = Node(1)
        second = Node(1)
        assert first != second
        assert first is not second


class TestAppendAndPrepend:
    def test_append_to_empty_sets_same_head_and_tail(self) -> None:
        linked: SinglyLinkedList[int] = SinglyLinkedList()
        node = linked.append(5)
        assert linked.head is node
        assert linked.tail is node
        assert_valid(linked, [5])

    def test_append_reuses_old_tail_and_updates_link(self) -> None:
        linked = SinglyLinkedList([1])
        old_tail = linked.tail
        new_tail = linked.append(2)
        assert old_tail is not None and old_tail.next is new_tail
        assert linked.tail is new_tail
        assert_valid(linked, [1, 2])

    def test_multiple_appends_preserve_order(self) -> None:
        linked: SinglyLinkedList[int] = SinglyLinkedList()
        for value in range(20):
            linked.append(value)
        assert_valid(linked, list(range(20)))

    def test_prepend_to_empty_sets_same_head_and_tail(self) -> None:
        linked: SinglyLinkedList[int] = SinglyLinkedList()
        node = linked.prepend(5)
        assert linked.head is node
        assert linked.tail is node
        assert_valid(linked, [5])

    def test_prepend_links_to_old_head_without_changing_tail(self) -> None:
        linked = SinglyLinkedList([2, 3])
        old_head = linked.head
        old_tail = linked.tail
        new_head = linked.prepend(1)
        assert new_head.next is old_head
        assert linked.tail is old_tail
        assert_valid(linked, [1, 2, 3])


class TestInsert:
    @pytest.mark.parametrize(
        ("index", "value", "expected"),
        [
            (0, 0, [0, 1, 2, 3]),
            (2, 9, [1, 2, 9, 3]),
            (3, 4, [1, 2, 3, 4]),
        ],
    )
    def test_inserts_at_start_middle_and_end(
        self, index: int, value: int, expected: list[int]
    ) -> None:
        linked = SinglyLinkedList([1, 2, 3])
        node = linked.insert(index, value)
        assert node.value == value
        assert_valid(linked, expected)

    def test_insert_zero_into_empty_list(self) -> None:
        linked: SinglyLinkedList[str] = SinglyLinkedList()
        node = linked.insert(0, "only")
        assert linked.head is node is linked.tail
        assert_valid(linked, ["only"])

    @pytest.mark.parametrize("index", [-1, 4])
    def test_rejects_out_of_range_insert_index(self, index: int) -> None:
        linked = SinglyLinkedList([1, 2, 3])
        with pytest.raises(IndexError):
            linked.insert(index, 9)
        assert_valid(linked, [1, 2, 3])

    @pytest.mark.parametrize("index", [1.0, "1", True])
    def test_rejects_non_integer_insert_index(self, index: object) -> None:
        linked = SinglyLinkedList([1, 2])
        with pytest.raises(TypeError):
            linked.insert(index, 9)  # type: ignore[arg-type]


class TestDelete:
    def test_delete_only_element_resets_both_endpoints(self) -> None:
        linked = SinglyLinkedList([7])
        node = linked.head
        assert linked.delete(0) == 7
        assert node is not None and node.next is None
        assert_valid(linked, [])

    def test_delete_head_detaches_node(self) -> None:
        linked = SinglyLinkedList([1, 2, 3])
        old_head = linked.head
        assert linked.delete(0) == 1
        assert old_head is not None and old_head.next is None
        assert_valid(linked, [2, 3])

    def test_delete_middle_preserves_neighbor_link(self) -> None:
        linked = SinglyLinkedList([1, 2, 3, 4])
        previous = linked.node_at(0)
        target = linked.node_at(1)
        successor = linked.node_at(2)
        assert linked.delete(1) == 2
        assert previous.next is successor
        assert target.next is None
        assert_valid(linked, [1, 3, 4])

    def test_delete_tail_updates_tail(self) -> None:
        linked = SinglyLinkedList([1, 2, 3])
        old_tail = linked.tail
        assert linked.delete(2) == 3
        assert linked.tail is linked.node_at(1)
        assert old_tail is not None and old_tail.next is None
        assert_valid(linked, [1, 2])

    def test_negative_index_deletes_from_end(self) -> None:
        linked = SinglyLinkedList([1, 2, 3])
        assert linked.delete(-1) == 3
        assert linked.delete(-2) == 1
        assert_valid(linked, [2])

    @pytest.mark.parametrize("index", [-4, 3])
    def test_rejects_out_of_range_delete_index(self, index: int) -> None:
        linked = SinglyLinkedList([1, 2, 3])
        with pytest.raises(IndexError):
            linked.delete(index)
        assert_valid(linked, [1, 2, 3])

    def test_delete_from_empty_list_raises(self) -> None:
        linked: SinglyLinkedList[int] = SinglyLinkedList()
        with pytest.raises(IndexError):
            linked.delete(0)


class TestSearchAndRemove:
    def test_find_returns_first_duplicate_index(self) -> None:
        linked = SinglyLinkedList([4, 8, 4, 9])
        assert linked.find(4) == 0
        assert linked.find(8) == 1

    def test_find_returns_none_for_missing_or_empty(self) -> None:
        assert SinglyLinkedList([1, 2]).find(9) is None
        assert SinglyLinkedList().find(9) is None

    def test_contains_delegates_to_search_semantics(self) -> None:
        linked = SinglyLinkedList([None, "x"])
        assert None in linked
        assert "x" in linked
        assert "missing" not in linked

    def test_remove_first_duplicate(self) -> None:
        linked = SinglyLinkedList([1, 2, 1, 3])
        assert linked.remove(1) == 1
        assert_valid(linked, [2, 1, 3])

    def test_remove_head_tail_and_only_element(self) -> None:
        linked = SinglyLinkedList([1, 2, 3])
        assert linked.remove(1) == 1
        assert linked.remove(3) == 3
        assert linked.remove(2) == 2
        assert_valid(linked, [])

    def test_remove_missing_value_raises_without_mutation(self) -> None:
        linked = SinglyLinkedList([1, 2])
        with pytest.raises(ValueError, match="not in linked list"):
            linked.remove(9)
        assert_valid(linked, [1, 2])


class TestAccessAndLifecycle:
    def test_getitem_and_node_at_support_negative_indices(self) -> None:
        linked = SinglyLinkedList([10, 20, 30])
        assert linked[0] == 10
        assert linked[-1] == 30
        assert linked.node_at(-2).value == 20

    @pytest.mark.parametrize("index", [-4, 3])
    def test_access_bounds_are_enforced(self, index: int) -> None:
        linked = SinglyLinkedList([1, 2, 3])
        with pytest.raises(IndexError):
            _ = linked[index]

    @pytest.mark.parametrize("index", [0.0, "0", False])
    def test_access_rejects_non_integer_indices(self, index: object) -> None:
        linked = SinglyLinkedList([1])
        with pytest.raises(TypeError):
            linked.node_at(index)  # type: ignore[arg-type]

    def test_clear_detaches_all_nodes(self) -> None:
        linked = SinglyLinkedList([1, 2, 3])
        nodes = [linked.node_at(index) for index in range(3)]
        linked.clear()
        assert all(node.next is None for node in nodes)
        assert_valid(linked, [])

    def test_clear_is_idempotent(self) -> None:
        linked: SinglyLinkedList[int] = SinglyLinkedList()
        linked.clear()
        linked.clear()
        assert_valid(linked, [])

    def test_mixed_mutation_sequence_preserves_invariants(self) -> None:
        linked: SinglyLinkedList[object] = SinglyLinkedList()
        linked.append("b")
        linked.prepend("a")
        linked.insert(2, "d")
        linked.insert(2, "c")
        assert linked.delete(1) == "b"
        linked.append(None)
        assert linked.remove(None) is None
        assert_valid(linked, ["a", "c", "d"])


class TestInvariantChecker:
    def test_detects_cycle(self) -> None:
        linked = SinglyLinkedList([1, 2, 3])
        assert linked.tail is not None and linked.head is not None
        linked.tail.next = linked.head
        with pytest.raises(AssertionError, match="tail.next|cycle"):
            linked.check_invariants()

    def test_detects_reachable_count_mismatch(self) -> None:
        linked = SinglyLinkedList([1, 2])
        linked._size = 3  # type: ignore[attr-defined]
        with pytest.raises(AssertionError, match="count"):
            linked.check_invariants()
