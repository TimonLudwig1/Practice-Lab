"""Tests for the array-backed stack and circular queue."""

import pytest

from stack_queue import (
    FixedCapacityQueue,
    QueueOverflowError,
    QueueUnderflowError,
    Stack,
    StackUnderflowError,
)


def test_new_stack_is_empty() -> None:
    stack: Stack[int] = Stack()

    assert stack.is_empty()
    assert not stack
    assert len(stack) == 0
    assert stack.to_list() == []


def test_stack_follows_lifo_order() -> None:
    stack: Stack[int] = Stack()

    for item in (10, 20, 30):
        stack.push(item)

    assert stack.to_list() == [10, 20, 30]
    assert [stack.pop(), stack.pop(), stack.pop()] == [30, 20, 10]
    assert stack.is_empty()


def test_stack_peek_does_not_remove_item() -> None:
    stack: Stack[str] = Stack()
    stack.push("first")
    stack.push("top")

    assert stack.peek() == "top"
    assert stack.peek() == "top"
    assert len(stack) == 2


@pytest.mark.parametrize("operation", ["pop", "peek"])
def test_empty_stack_operations_raise_underflow(operation: str) -> None:
    stack: Stack[int] = Stack()

    with pytest.raises(StackUnderflowError, match="empty stack"):
        getattr(stack, operation)()


def test_stack_accepts_none_as_a_regular_value() -> None:
    stack: Stack[object | None] = Stack()
    stack.push(None)

    assert stack.peek() is None
    assert stack.pop() is None


@pytest.mark.parametrize("capacity", [0, -1, -20])
def test_queue_rejects_non_positive_capacity(capacity: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        FixedCapacityQueue[int](capacity)


@pytest.mark.parametrize("capacity", [True, 2.5, "3", None])
def test_queue_rejects_non_integer_capacity(capacity: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        FixedCapacityQueue(capacity)  # type: ignore[arg-type]


def test_new_queue_is_empty_and_exposes_capacity() -> None:
    queue: FixedCapacityQueue[int] = FixedCapacityQueue(capacity=3)

    assert queue.capacity == 3
    assert queue.is_empty()
    assert not queue.is_full()
    assert not queue
    assert len(queue) == 0
    assert queue.to_list() == []


def test_queue_follows_fifo_order() -> None:
    queue: FixedCapacityQueue[str] = FixedCapacityQueue(capacity=3)

    for item in ("A", "B", "C"):
        queue.enqueue(item)

    assert queue.is_full()
    assert list(queue) == ["A", "B", "C"]
    assert [queue.dequeue(), queue.dequeue(), queue.dequeue()] == ["A", "B", "C"]
    assert queue.is_empty()


def test_queue_peek_does_not_remove_item() -> None:
    queue: FixedCapacityQueue[int] = FixedCapacityQueue(capacity=2)
    queue.enqueue(10)
    queue.enqueue(20)

    assert queue.peek() == 10
    assert queue.peek() == 10
    assert len(queue) == 2


@pytest.mark.parametrize("operation", ["dequeue", "peek"])
def test_empty_queue_operations_raise_underflow(operation: str) -> None:
    queue: FixedCapacityQueue[int] = FixedCapacityQueue(capacity=1)

    with pytest.raises(QueueUnderflowError, match="empty queue"):
        getattr(queue, operation)()


def test_full_queue_raises_overflow_without_changing_state() -> None:
    queue: FixedCapacityQueue[int] = FixedCapacityQueue(capacity=2)
    queue.enqueue(1)
    queue.enqueue(2)

    with pytest.raises(QueueOverflowError, match="capacity=2"):
        queue.enqueue(3)

    assert queue.to_list() == [1, 2]
    assert len(queue) == 2


def test_capacity_one_queue_can_reuse_its_only_slot() -> None:
    queue: FixedCapacityQueue[str] = FixedCapacityQueue(capacity=1)

    queue.enqueue("first")
    assert queue.dequeue() == "first"
    queue.enqueue("second")

    assert queue.peek() == "second"
    assert queue.is_full()


def test_queue_wraps_around_and_preserves_logical_order() -> None:
    queue: FixedCapacityQueue[int] = FixedCapacityQueue(capacity=4)
    for item in (1, 2, 3, 4):
        queue.enqueue(item)

    assert queue.dequeue() == 1
    assert queue.dequeue() == 2
    queue.enqueue(5)
    queue.enqueue(6)

    assert queue.to_list() == [3, 4, 5, 6]
    assert list(queue) == [3, 4, 5, 6]
    assert [queue.dequeue() for _ in range(4)] == [3, 4, 5, 6]


def test_queue_survives_multiple_wrap_around_cycles() -> None:
    queue: FixedCapacityQueue[int] = FixedCapacityQueue(capacity=3)

    for cycle in range(10):
        values = [cycle * 3 + offset for offset in range(3)]
        for value in values:
            queue.enqueue(value)
        assert [queue.dequeue() for _ in values] == values

    assert queue.is_empty()
    queue.enqueue(99)
    assert queue.dequeue() == 99


def test_queue_accepts_none_as_a_regular_value() -> None:
    queue: FixedCapacityQueue[object | None] = FixedCapacityQueue(capacity=2)
    queue.enqueue(None)
    queue.enqueue("value")

    assert queue.to_list() == [None, "value"]
    assert queue.dequeue() is None


def test_representations_show_logical_order() -> None:
    stack: Stack[int] = Stack()
    stack.push(1)
    queue: FixedCapacityQueue[int] = FixedCapacityQueue(capacity=2)
    queue.enqueue(7)

    assert repr(stack) == "Stack(bottom_to_top=[1])"
    assert repr(queue) == "FixedCapacityQueue(front_to_back=[7], capacity=2)"
