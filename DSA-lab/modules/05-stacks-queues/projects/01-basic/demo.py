"""Print a compact operation trace for the stack and circular queue."""

from stack_queue import (
    FixedCapacityQueue,
    QueueOverflowError,
    QueueUnderflowError,
    Stack,
    StackUnderflowError,
)


def show_stack_demo() -> None:
    """Demonstrate that the most recently pushed item leaves first."""
    stack: Stack[str] = Stack()
    print("STACK (bottom -> top)")

    for item in ("load", "clean", "analyze"):
        stack.push(item)
        print(f"push({item!r:9}) -> {stack.to_list()}")

    print(f"peek()          -> {stack.peek()!r}; state={stack.to_list()}")
    while stack:
        print(f"pop()           -> {stack.pop()!r}; state={stack.to_list()}")

    try:
        stack.pop()
    except StackUnderflowError as error:
        print(f"empty pop       -> {type(error).__name__}: {error}")


def show_queue_demo() -> None:
    """Demonstrate FIFO order, overflow and circular reuse of slots."""
    queue: FixedCapacityQueue[str] = FixedCapacityQueue(capacity=3)
    print("\nQUEUE (front -> back)")

    for item in ("job-A", "job-B", "job-C"):
        queue.enqueue(item)
        print(f"enqueue({item!r}) -> {queue.to_list()}")

    try:
        queue.enqueue("job-D")
    except QueueOverflowError as error:
        print(f"full enqueue    -> {type(error).__name__}: {error}")

    print(f"dequeue()       -> {queue.dequeue()!r}; state={queue.to_list()}")
    queue.enqueue("job-D")
    print(f"enqueue('job-D')-> {queue.to_list()}  (wrap-around)")

    while queue:
        print(f"dequeue()       -> {queue.dequeue()!r}; state={queue.to_list()}")

    try:
        queue.dequeue()
    except QueueUnderflowError as error:
        print(f"empty dequeue   -> {type(error).__name__}: {error}")


if __name__ == "__main__":
    show_stack_demo()
    show_queue_demo()
