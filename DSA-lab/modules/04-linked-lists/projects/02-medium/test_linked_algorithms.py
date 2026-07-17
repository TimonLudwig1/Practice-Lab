"""Pytest suite for four classic singly linked list pointer algorithms."""

from __future__ import annotations

import pytest

from linked_algorithms import (
    CycleInfo,
    Node,
    build_chain,
    chain_nodes,
    chain_values,
    detect_cycle,
    has_cycle,
    merge_sorted,
    middle_node,
    reverse_in_place,
)


class TestHelpers:
    def test_build_chain_preserves_values(self) -> None:
        assert chain_values(build_chain(value * 2 for value in range(4))) == [0, 2, 4, 6]

    def test_empty_chain_helpers(self) -> None:
        assert build_chain([]) is None
        assert chain_nodes(None) == []
        assert chain_values(None) == []

    def test_materializer_rejects_cycle(self) -> None:
        node = Node(1)
        node.next = node
        with pytest.raises(ValueError, match="cycle"):
            chain_nodes(node)


class TestReverseInPlace:
    def test_empty_chain(self) -> None:
        assert reverse_in_place(None) is None

    def test_single_node_keeps_identity(self) -> None:
        node = Node("only")
        assert reverse_in_place(node) is node
        assert node.next is None

    def test_reverses_values_and_node_identities(self) -> None:
        head = build_chain([1, 2, 3, 4])
        original_nodes = chain_nodes(head)
        reversed_head = reverse_in_place(head)
        assert chain_values(reversed_head) == [4, 3, 2, 1]
        assert chain_nodes(reversed_head) == list(reversed(original_nodes))

    def test_old_head_becomes_terminal_node(self) -> None:
        head = build_chain([1, 2, 3])
        assert head is not None
        old_head = head
        reverse_in_place(head)
        assert old_head.next is None

    def test_reversing_twice_restores_original_chain(self) -> None:
        head = build_chain([1, 2, 3])
        original_nodes = chain_nodes(head)
        restored = reverse_in_place(reverse_in_place(head))
        assert chain_nodes(restored) == original_nodes

    def test_trace_records_saved_successor_before_mutation(self) -> None:
        trace: list[str] = []
        reverse_in_place(build_chain([1, 2]), trace=trace)
        assert trace[0] == "step=0: previous=None current=1 saved_next=2"
        assert trace[-1] == "result: head=2"


class TestMiddleNode:
    def test_empty_chain_has_no_middle(self) -> None:
        assert middle_node(None) is None

    def test_single_node_is_middle(self) -> None:
        node = Node(1)
        assert middle_node(node) is node

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([1, 2, 3], 2),
            ([1, 2, 3, 4, 5], 3),
            ([1, 2, 3, 4, 5, 6, 7], 4),
        ],
    )
    def test_odd_lengths(self, values: list[int], expected: int) -> None:
        middle = middle_node(build_chain(values))
        assert middle is not None and middle.value == expected

    @pytest.mark.parametrize(
        ("values", "expected"),
        [([1, 2], 2), ([1, 2, 3, 4], 3), ([1, 2, 3, 4, 5, 6], 4)],
    )
    def test_even_lengths_choose_second_middle(
        self, values: list[int], expected: int
    ) -> None:
        middle = middle_node(build_chain(values))
        assert middle is not None and middle.value == expected

    def test_does_not_modify_links(self) -> None:
        head = build_chain([1, 2, 3, 4])
        before = chain_nodes(head)
        middle_node(head)
        assert chain_nodes(head) == before

    def test_trace_shows_fast_pointer_reaching_end(self) -> None:
        trace: list[str] = []
        middle_node(build_chain([1, 2, 3, 4]), trace=trace)
        assert trace[0] == "step=0: slow=1 fast=1"
        assert "fast=None" in trace[-2]
        assert trace[-1] == "result: middle=3"


class TestFloydCycleDetection:
    def test_empty_and_single_acyclic_chains(self) -> None:
        assert detect_cycle(None) is None
        assert detect_cycle(Node(1)) is None
        assert not has_cycle(None)

    def test_acyclic_chain(self) -> None:
        head = build_chain([1, 2, 3, 4])
        assert detect_cycle(head) is None
        assert not has_cycle(head)

    def test_self_loop(self) -> None:
        node = Node("loop")
        node.next = node
        info = detect_cycle(node)
        assert info == CycleInfo(node, cycle_length=1, prefix_length=0)

    def test_cycle_entering_at_head(self) -> None:
        head = build_chain([1, 2, 3, 4])
        nodes = chain_nodes(head)
        nodes[-1].next = nodes[0]
        info = detect_cycle(head)
        assert info is not None
        assert info.entry is nodes[0]
        assert info.cycle_length == 4
        assert info.prefix_length == 0

    def test_cycle_entering_in_middle(self) -> None:
        head = build_chain([1, 2, 3, 4, 5, 6])
        nodes = chain_nodes(head)
        nodes[-1].next = nodes[2]
        info = detect_cycle(head)
        assert info is not None
        assert info.entry is nodes[2]
        assert info.cycle_length == 4
        assert info.prefix_length == 2

    def test_duplicate_values_do_not_fake_cycle(self) -> None:
        assert detect_cycle(build_chain([1, 1, 1, 1])) is None

    def test_duplicate_values_still_use_entry_identity(self) -> None:
        head = build_chain([1, 1, 1, 1])
        nodes = chain_nodes(head)
        nodes[-1].next = nodes[1]
        info = detect_cycle(head)
        assert info is not None and info.entry is nodes[1]

    def test_detection_does_not_modify_cycle(self) -> None:
        head = build_chain([1, 2, 3, 4])
        nodes = chain_nodes(head)
        nodes[-1].next = nodes[1]
        detect_cycle(head)
        assert nodes[-1].next is nodes[1]

    def test_trace_reports_metadata(self) -> None:
        head = build_chain(["A", "B", "C", "D"])
        nodes = chain_nodes(head)
        nodes[-1].next = nodes[1]
        trace: list[str] = []
        detect_cycle(head, trace=trace)
        assert trace[0].startswith("meet-step=1")
        assert trace[-1] == "result: entry='B' cycle_length=3 prefix_length=1"


class TestMergeSorted:
    def test_two_empty_chains(self) -> None:
        assert merge_sorted(None, None) is None

    def test_one_empty_chain_returns_other_identity(self) -> None:
        right = build_chain([1, 2])
        assert merge_sorted(None, right) is right
        left = build_chain([3, 4])
        assert merge_sorted(left, None) is left

    def test_interleaved_values(self) -> None:
        merged = merge_sorted(build_chain([1, 4, 7]), build_chain([2, 3, 8]))
        assert chain_values(merged) == [1, 2, 3, 4, 7, 8]

    def test_negative_and_unequal_lengths(self) -> None:
        merged = merge_sorted(build_chain([-5, 0, 9, 12]), build_chain([-2]))
        assert chain_values(merged) == [-5, -2, 0, 9, 12]

    def test_duplicates_are_preserved(self) -> None:
        merged = merge_sorted(build_chain([1, 2, 2]), build_chain([2, 2, 3]))
        assert chain_values(merged) == [1, 2, 2, 2, 2, 3]

    def test_equal_values_are_stable_to_left_chain(self) -> None:
        left = build_chain([1, 2])
        right = build_chain([1, 3])
        assert left is not None and right is not None
        merged = merge_sorted(left, right)
        nodes = chain_nodes(merged)
        assert nodes[0] is left
        assert nodes[1] is right

    def test_reuses_every_input_node_without_new_data_nodes(self) -> None:
        left = build_chain([1, 3, 5])
        right = build_chain([2, 4, 6])
        input_ids = {id(node) for node in chain_nodes(left) + chain_nodes(right)}
        merged = merge_sorted(left, right)
        output_nodes = chain_nodes(merged)
        assert {id(node) for node in output_nodes} == input_ids
        assert len(output_nodes) == len(input_ids)

    def test_trace_records_choices_and_remainder(self) -> None:
        trace: list[str] = []
        merge_sorted(build_chain([1, 4]), build_chain([2, 3]), trace=trace)
        assert trace[0].startswith("step=0: take left value=1")
        assert trace[-2] == "attach remainder=4"
        assert trace[-1] == "result: head=1"
