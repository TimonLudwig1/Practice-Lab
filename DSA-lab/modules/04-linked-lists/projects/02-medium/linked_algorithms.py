"""Classic pointer algorithms on identity-based singly linked nodes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(slots=True, eq=False)
class Node(Generic[T]):
    """One identity-based node in a singly linked chain."""

    value: T
    next: Node[T] | None = None


def build_chain(values: Iterable[T]) -> Node[T] | None:
    """Build an acyclic chain in input order."""
    head: Node[T] | None = None
    tail: Node[T] | None = None
    for value in values:
        node = Node(value)
        if head is None:
            head = node
        else:
            assert tail is not None
            tail.next = node
        tail = node
    return head


def chain_nodes(head: Node[T] | None) -> list[Node[T]]:
    """Return nodes in order, rejecting cycles instead of looping forever."""
    nodes: list[Node[T]] = []
    seen: set[int] = set()
    current = head
    while current is not None:
        identity = id(current)
        if identity in seen:
            raise ValueError("cycle detected while materializing chain")
        seen.add(identity)
        nodes.append(current)
        current = current.next
    return nodes


def chain_values(head: Node[T] | None) -> list[T]:
    """Return an acyclic chain's values in order."""
    return [node.value for node in chain_nodes(head)]


def _label(node: Node[object] | None) -> str:
    """Return a compact trace label for a node reference."""
    return "None" if node is None else repr(node.value)


def reverse_in_place(
    head: Node[T] | None, *, trace: list[str] | None = None
) -> Node[T] | None:
    """Reverse a chain in place with three references and return its new head."""
    previous: Node[T] | None = None
    current = head
    step = 0

    while current is not None:
        next_node = current.next
        if trace is not None:
            trace.append(
                f"step={step}: previous={_label(previous)} "
                f"current={_label(current)} saved_next={_label(next_node)}"
            )
        current.next = previous
        previous = current
        current = next_node
        step += 1

    if trace is not None:
        trace.append(f"result: head={_label(previous)}")
    return previous


def middle_node(
    head: Node[T] | None, *, trace: list[str] | None = None
) -> Node[T] | None:
    """Return the middle node, choosing the second middle for even lengths."""
    slow = head
    fast = head
    step = 0
    if trace is not None:
        trace.append(f"step=0: slow={_label(slow)} fast={_label(fast)}")

    while fast is not None and fast.next is not None:
        assert slow is not None
        slow = slow.next
        fast = fast.next.next
        step += 1
        if trace is not None:
            trace.append(f"step={step}: slow={_label(slow)} fast={_label(fast)}")

    if trace is not None:
        trace.append(f"result: middle={_label(slow)}")
    return slow


@dataclass(frozen=True)
class CycleInfo(Generic[T]):
    """Cycle entry plus cycle and acyclic-prefix lengths."""

    entry: Node[T]
    cycle_length: int
    prefix_length: int


def detect_cycle(
    head: Node[T] | None, *, trace: list[str] | None = None
) -> CycleInfo[T] | None:
    """Return Floyd cycle metadata, or None for an acyclic chain."""
    slow = head
    fast = head
    meeting: Node[T] | None = None
    step = 0

    while fast is not None and fast.next is not None:
        assert slow is not None
        slow = slow.next
        fast = fast.next.next
        step += 1
        if trace is not None:
            trace.append(
                f"meet-step={step}: slow={_label(slow)} fast={_label(fast)}"
            )
        if slow is fast:
            meeting = slow
            break

    if meeting is None:
        if trace is not None:
            trace.append("result: no cycle")
        return None

    cycle_length = 1
    cursor = meeting.next
    while cursor is not meeting:
        assert cursor is not None
        cycle_length += 1
        cursor = cursor.next

    entry_cursor = head
    meeting_cursor: Node[T] | None = meeting
    prefix_length = 0
    while entry_cursor is not meeting_cursor:
        assert entry_cursor is not None and meeting_cursor is not None
        entry_cursor = entry_cursor.next
        meeting_cursor = meeting_cursor.next
        prefix_length += 1
        if trace is not None:
            trace.append(
                f"entry-step={prefix_length}: from-head={_label(entry_cursor)} "
                f"from-meeting={_label(meeting_cursor)}"
            )

    assert entry_cursor is not None
    result = CycleInfo(entry_cursor, cycle_length, prefix_length)
    if trace is not None:
        trace.append(
            f"result: entry={_label(result.entry)} "
            f"cycle_length={cycle_length} prefix_length={prefix_length}"
        )
    return result


def has_cycle(head: Node[T] | None) -> bool:
    """Return whether Floyd's algorithm finds a cycle."""
    return detect_cycle(head) is not None


def merge_sorted(
    left: Node[T] | None,
    right: Node[T] | None,
    *,
    trace: list[str] | None = None,
) -> Node[T] | None:
    """Stably relink two disjoint sorted chains into one sorted chain."""
    dummy: Node[T | None] = Node(None)
    tail = dummy
    step = 0

    while left is not None and right is not None:
        if left.value <= right.value:
            selected = left
            left = left.next
            source = "left"
        else:
            selected = right
            right = right.next
            source = "right"

        tail.next = selected
        tail = selected
        if trace is not None:
            trace.append(
                f"step={step}: take {source} value={selected.value!r}; "
                f"next-left={_label(left)} next-right={_label(right)}"
            )
        step += 1

    remainder = left if left is not None else right
    tail.next = remainder
    if trace is not None:
        trace.append(f"attach remainder={_label(remainder)}")
        trace.append(f"result: head={_label(dummy.next)}")
    return dummy.next
