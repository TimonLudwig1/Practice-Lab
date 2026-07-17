"""Demonstrate every public singly linked list operation."""

from linked_list import SinglyLinkedList


def show(label: str, linked: SinglyLinkedList[object]) -> None:
    """Print values and structural metadata after one operation."""
    linked.check_invariants()
    head = None if linked.head is None else linked.head.value
    tail = None if linked.tail is None else linked.tail.value
    print(f"{label:22} {linked!r:42} len={len(linked)} head={head!r} tail={tail!r}")


def main() -> None:
    """Run a complete mutation sequence including return values."""
    linked: SinglyLinkedList[object] = SinglyLinkedList()
    show("Start", linked)

    linked.append(20)
    show("append(20)", linked)
    linked.prepend(10)
    show("prepend(10)", linked)
    linked.append(40)
    show("append(40)", linked)
    linked.insert(2, 30)
    show("insert(2, 30)", linked)

    print(f"find(30)               index={linked.find(30)}")
    print(f"delete(-1)             value={linked.delete(-1)!r}")
    show("nach delete(-1)", linked)
    print(f"remove(20)             value={linked.remove(20)!r}")
    show("nach remove(20)", linked)
    print(f"linked[1]              value={linked[1]!r}")

    linked.clear()
    show("clear()", linked)


if __name__ == "__main__":
    main()
