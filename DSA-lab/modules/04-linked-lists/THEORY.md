# Module 04 — Linked Lists

## Learning objectives

After this module you can implement a Singly Linked List with all basic operations and
understand each pointer change on paper beforehand. You understand the difference
between " already possess the insertion point" and " first have to search the insertion
point", and can therefore justify runtime information precisely instead of blanket. You
can also control in-place reverse, runner technique and Floyd's cycle detection and
explain why Doubly Linked Lists with Sentinel nodes in LRU caches and deques are useful.

Linked lists are less ubiquitous than Python lists, but they sharpen a central
algorithmic ability: A small change in references can transform an entire structure. The
order of assignments counts. A lost link means a lost subtree; a wrong link can create a
cycle.

---

## 1. Motivation: Array or Linked List?

### 1.1 Intuition: Block of houses and treasure hunt

An array resembles a numbered block of houses. From the address `i` you can calculate
directly where the ith element lies. For this the places must be organized logically
coherently. If a house is inserted in the middle, all the following residents must move
on to a place.

A Linked List resembles a treasure hunt. Each station contains the value and the
reference to the next station. A station can lie somewhere; it just has to call the
right successor. If you already hold the current station in your hand, you can plug in a
new station right behind it. But the 500. Find a station, you have to follow 499 clues.

This is the basic exchange:

- Arrays provide direct index access and good storage locality.
- Linked lists provide local structure changes without moving subsequent values.

### 1.2 What "related memory" means practically

A classic array stores similar elements side by side. The address of element `i` is the
base address plus `i * element_width`. Therefore, access is by index Θ(1), and
consecutive reading uses CPU caches well.

A Linked List node stores at least two things:

```text
┌──────────────┬──────────────┐
│ value        │ next         │
└──────────────┴──────┬───────┘
                      │ Reference to other nodes
```

Nodes can be distributed in memory. Each step requires a further reference resolution.
Even if array and linked list pass through both Θ(n) elements, the array is often faster
in practice: less object overhead, better cache locality and more compact display.

### 1.3 Python speciality

Pythons `list` is a dynamic array of references, not a linked list. The referenced
Python objects may be distributed, but the references themselves are in a coherent
buffer. A linked list node is also a Python object and requires the `next` attribute.
For normal data collections, a Python list is usually smaller and faster.

Linked lists are interesting if the algorithm often needs to hang known nodes locally or
requires stable node identities.

---

## 2. Singly Linked List: node, Head and End

### 2.1 Intuition: The head is the only entrance

A Singly Linked List consists of nodes with exactly one forward reference. The variable
`head` points to the first node. From there, each achievable node is accessible via
repeated `.next`.

```text
head
 │
 ▼
┌─────┬──────┐   ┌─────┬──────┐   ┌─────┬──────┐
│  7  │   ●──┼──▶│ 12  │   ●──┼──▶│ 19  │ None │
└─────┴──────┘   └─────┴──────┘   └─────┴──────┘
```

The empty list is `head is None`. A single node has `head.next is None`. These two forms
are not exotic special cases, but the most important test cases of each operation.

### 2.2 A minimum node model

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class SinglyNode(Generic[T]):
    """One node in a singly linked list."""

    value: T
    next: SinglyNode[T] | None = None


def build_chain(values: Iterable[T]) -> SinglyNode[T] | None:
    """Build a chain while keeping a tail reference."""
    head: SinglyNode[T] | None = None
    tail: SinglyNode[T] | None = None
    for value in values:
        node = SinglyNode(value)
        if head is None:
            head = node
        else:
            assert tail is not None
            tail.next = node
        tail = node
    return head


def iterate_chain(head: SinglyNode[T] | None) -> Iterator[T]:
    """Yield values from head to the terminal None reference."""
    current = head
    while current is not None:
        yield current.value
        current = current.next


assert list(iterate_chain(build_chain([]))) == []
assert list(iterate_chain(build_chain([7, 12, 19]))) == [7, 12, 19]
```

`head` has no magic connection to all nodes. He's just a reference to the first. If
`head` is overwritten without backing up the old chain in advance, access to the entire
list can be lost.

---

## 3. Basic operations with pointer diagrams

### 3.1 Prepend: Paste in front

Output:

```text
head ──▶ [7 | ●] ──▶ [12 ] None
new  ──▶ [3 ] None
```

Two assignments in exactly this logical order:

```text
1. new.next = head
2. head = new

head ──▶ [3 | ●] ──▶ [7 | ●] ──▶ [12 ] None
```

```python
def prepend(
    head: SinglyNode[T] | None, value: T
) -> SinglyNode[T]:
    """Return a new head containing value."""
    return SinglyNode(value, head)


prepend_example = prepend(build_chain([7, 12]), 3)
assert list(iterate_chain(prepend_example)) == [3, 7, 12]
```

Time and additional memory are Θ(1). This also applies to the empty list, because the
new node then gets `next = None`.

### 3.2 Append without and with Tail

Without saved end must run from head to last node:

```python
def append_without_tail(
    head: SinglyNode[T] | None, value: T
) -> SinglyNode[T]:
    """Append by traversing to the end and return the possibly new head."""
    node = SinglyNode(value)
    if head is None:
        return node

    current = head
    while current.next is not None:
        current = current.next
    current.next = node
    return head


append_example = append_without_tail(build_chain([2, 4]), 6)
assert list(iterate_chain(append_example)) == [2, 4, 6]
```

Searching for the end costs Θ(s). A list class can also save `tail`. Then Append is
Θ(1), but must receive the invariant for each mutation:

```text
empty list:      head is None and tail is None
Non-empty:        tail is accessible and tail.next is None
one element:      head is tail
```

### 3.3 Paste after a known node

Output:

```text
Current ──▶ [7 | ●] ──▶ [12 | ●] ──▶ [19 ] None
new     ──▶ [10 ] None
```

If `current.next = new` were set first, the reference would be overwritten to `12`.
Therefore, the old successor is first secured in the new node:

```text
1. new.next = current.next
2. current.next = new

[7 | ●] ──▶ [10 | ●] ──▶ [12 | ●] ──▶ [19 ] None
```

```python
def insert_after(node: SinglyNode[T], value: T) -> SinglyNode[T]:
    """Insert after a known node and return the new node."""
    new_node = SinglyNode(value, node.next)
    node.next = new_node
    return new_node


insert_example = build_chain([7, 12, 19])
assert insert_example is not None
insert_after(insert_example, 10)
assert list(iterate_chain(insert_example)) == [7, 10, 12, 19]
```

The local mutation is Θ(1). However, "insert to index `i`" remains Θ(i) because the
predecessor has to be found first.

### 3.4 Delete Head

```text
head ──▶ [7 | ●] ──▶ [12 | ●] ──▶ [19 ] None

head = head.next

head ───────────────▶ [12 | ●] ──▶ [19 ] None
```

The old head can no longer be reached from the list. If there is no other reference to
him, Python can release him. Clearing the head is Θ(1).

### 3.5 Delete after a known node

To remove the successor of `previous`:

```text
previous ──▶ [7 | ●] ──▶ [12 | ●] ──▶ [19 ] None
                          target

previous.next = target.next

previous ──▶ [7 | ●] ─────────────────▶ [19 ] None
```

```python
def delete_after(node: SinglyNode[T]) -> T:
    """Delete and return the value after node."""
    target = node.next
    if target is None:
        raise IndexError("no node exists after the given node")
    node.next = target.next
    target.next = None
    return target.value


delete_example = build_chain([7, 12, 19])
assert delete_example is not None
assert delete_after(delete_example) == 12
assert list(iterate_chain(delete_example)) == [7, 19]
```

`target.next = None` is not mandatory for the structure, but makes the separation
explicit. Again, if the predecessor is known, Θ(1); you have to search for a value or
index, Θ(n).

### 3.6 Search and index access

```python
def find_first(
    head: SinglyNode[T] | None, target: T
) -> SinglyNode[T] | None:
    """Return the first node equal to target, or None."""
    current = head
    while current is not None:
        if current.value == target:
            return current
        current = current.next
    return None


search_example = build_chain([5, 8, 13])
found = find_first(search_example, 8)
assert found is not None and found.value == 8
assert find_first(search_example, 99) is None
```

In the worst case, all nodes are visited: Θ(n). Access to index `i` costs Θ(i) and thus
in the worst case Θ(n). Linked lists do not support an arithmetically calculated jump to
the ithn node.

---

## 4. Invariants of a complete list class

A robust Singly-Linked-List class often stores `head`, `tail` and `size`. After each
public operation:

1. `size >= 0`.
2. `size == 0` exactly when `head is None` and `tail is None`.
3. For `size > 0`, `head` and `tail` are real nodes.
4. `tail.next is None`.
5. From the head, all nodes to the tail can be reached by exactly `size` steps.
6. There is no cycle within the list.

Edge cases can be derived from these invariants:

| Operation | Empty list | One element | Multiple elements |
|---|---|---|---|
| Prepend | Head and Tail become new nodes | Head changes | Head changes |
| Append | Head and Tail become new nodes | Tail changes | Tail changes |
| Delete Head | Error or defined blank | Head and tail become `None` | Head becomes successor |
| Delete Tail | Error or defined blank | Head and tail become `None` | Predecessor must be searched |

For a Singly Linked List, even with `tail`, the last node Θ(s) is deleted because the
reverse reference to its predecessor is missing.

---

## 5. Doubly Linked List

### 5.1 Intuition: clues in both directions

A double-chained node stores `next` and `prev`:

```text
None── [A] [B] [C] ──▶ None
```

This allows you to navigate from a known node in both directions and remove a known node
itself in Θ(1). The price:

- an additional reference per node;
- more pointer changes per mutation,
- more invariants that must be right at the same time.

For adjacent nodes `a` and `b`, the following applies in pairs:

```text
a.next is b exactly when b.prev is a
```

### 5.2 Remove a known node

```text
before 简sthe ▶ target 简sTheafter

1. before.next = after
2. after.prev = before
```

Without sentinels, `before is None` and `after is None` must be treated separately for
head and end. These case distinctions are a common source of errors.

---

## 6. Sentinel nodes

### 6.1 Intuition: Artificial Limits

Sentinels are permanent nodes without useful value. A `root` sentinel may limit a
circular Doubly Linked List:

```text
          ┌────────────────────────────────┐
          ▼                                │
[root] 简┘
```

For the empty list, `root.next` and `root.prev` will point to `root` itself. Every real
node has a predecessor and successor. Inserting and removing do not require special
cases for head, end or a single element.

### 6.2 Minimum Sentinel Example

```python
@dataclass(slots=True)
class DoublyNode(Generic[T]):
    """One node in a doubly linked list."""

    value: T | None
    prev: DoublyNode[T] | None = None
    next: DoublyNode[T] | None = None


class SentinelList(Generic[T]):
    """A circular doubly linked list with one permanent sentinel."""

    def __init__(self) -> None:
        self.root: DoublyNode[T] = DoublyNode(None)
        self.root.prev = self.root
        self.root.next = self.root
        self.size = 0

    def append(self, value: T) -> DoublyNode[T]:
        """Append value before the root sentinel."""
        tail = self.root.prev
        assert tail is not None
        node = DoublyNode(value, prev=tail, next=self.root)
        tail.next = node
        self.root.prev = node
        self.size += 1
        return node

    def remove(self, node: DoublyNode[T]) -> T:
        """Unlink a known non-sentinel node in constant time."""
        if node is self.root or node.prev is None or node.next is None:
            raise ValueError("node is not a linked data node")
        before = node.prev
        after = node.next
        before.next = after
        after.prev = before
        node.prev = None
        node.next = None
        self.size -= 1
        assert node.value is not None
        return node.value

    def values(self) -> list[T]:
        """Return all data values from front to back."""
        result: list[T] = []
        current = self.root.next
        while current is not self.root:
            assert current is not None and current.value is not None
            result.append(current.value)
            current = current.next
        return result


sentinel_example: SentinelList[int] = SentinelList()
first_node = sentinel_example.append(4)
second_node = sentinel_example.append(8)
assert sentinel_example.values() == [4, 8]
assert sentinel_example.remove(first_node) == 4
assert sentinel_example.values() == [8]
assert sentinel_example.remove(second_node) == 8
assert sentinel_example.values() == []
```

Sentinels move complexity from each mutation into a unique initialization. The permanent
circle structure is internal; outwardly, the class can continue to act like a normal
finite sequence.

---

## 7. Comparison of costs

The table deliberately distinguishes between operations at the known node and operations
in which first it is necessary to search.

| Operation | Dynamic array | Singly Linked List | Doubly Linked List |
|---|---:|---:|---:|
| Access Index `i` | Θ(1) | Θ(i), worst Θ(s) | Θ(min(i, n-i))* |
| Search for value | Θ(s) | Θ(s) | Θ(s) |
| Prepend | Θ(s) | Θ(1) | Θ(1) |
| Append | amortised Θ(1) | Θ(1) with tail, otherwise Θ(s) | Θ(1) with tail/sentinel |
| Insert by known node | Move Θ(s) | Θ(1) | Θ(1) |
| Delete Head | Move Θ(s) | Θ(1) | Θ(1) |
| Delete known nodes | Move Θ(s) | Θ(1) only with predecessor | Θ(1) |
| Delete Tail | Θ(1) | Θ(s) | Θ(1) |
| Extra references per element | no link reference | `next` | `prev` and `next` |

`*` If head and tail are present and traversed from the closer side.

"Having linked lists O(1) insert" is misleading without context. Correct:

> Local hanging is O(1) if the required node references are already
> is meant. Finding this place can cost O(n).

---

## 8. In-place reverse

### 8.1 Intuition: Turn a street sign for sign

The remaining list, which has not yet been processed, must not be lost when it is
reversed. Three references suffice:

- `previous`: already reversed prefix,
- `current`: currently hanging node,
- `next_node`: previously secured rest.

Output:

```text
previous = None
Current ──▶ [A] ──▶ [B] ──▶ [C] ──▶ None
```

One step:

```text
1. next_node = current.next     # B secure
2. current.next = previous      # A points backwards to None
3. previous = current           # reverse prefix ends with A
4. current = next_node          # continue with B
```

### 8.2 Implementation and invariant

```python
def reverse_in_place(
    head: SinglyNode[T] | None,
) -> SinglyNode[T] | None:
    """Reverse a chain and return its new head."""
    previous: SinglyNode[T] | None = None
    current = head
    while current is not None:
        next_node = current.next
        current.next = previous
        previous = current
        current = next_node
    return previous


reverse_example = build_chain([1, 2, 3, 4])
reverse_example = reverse_in_place(reverse_example)
assert list(iterate_chain(reverse_example)) == [4, 3, 2, 1]
```

Grinding invariant:

```text
previous is the head of the correctly reversed processed prefix.
Current is the head of the unchanged suffix.
Both subchains together contain exactly all original nodes.
```

Time Θ(s), additional memory Θ(1). No new data node is created.

---

## 9. Runner technology: Faster and slower hands

### 9.1 Find the Center

`slow` goes one step per round, `fast` two. When `fast` reaches the end, `slow` has
reached about half.

```text
Start:     slow,fast
              ▼
           [1] → [2] → [3] → [4] → [5]

Round 1:         slow almost
                   ▼     ▼
           [1] → [2] → [3] → [4] → [5]

Round 2:               slow        almost=none after two steps
                         ▼
           [1] → [2] → [3] → [4] → [5]
```

```python
def middle_node(head: SinglyNode[T] | None) -> SinglyNode[T] | None:
    """Return the middle node, choosing the second middle for even length."""
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        assert slow is not None
        slow = slow.next
        fast = fast.next.next
    return slow


odd_middle = middle_node(build_chain([1, 2, 3, 4, 5]))
even_middle = middle_node(build_chain([1, 2, 3, 4]))
assert odd_middle is not None and odd_middle.value == 3
assert even_middle is not None and even_middle.value == 3
```

The variant delivers the second of the two middle nodes at a straight length. A slightly
changed starting position or loop condition can deliver the first one instead. This
behaviour must be specified and tested.

Time Θ(s), additional memory Θ(1), only one run.

---

## 10. Floyd cycle detection

### 10.1 What is a cycle?

An incorrect or deliberately cyclical list does not end at `None`:

```text
[A] → [B] → [C] → [D]
            ▲           │
            └───────────┘
```

A normal traversal would run endlessly. A lot of nodes already visited recognize
repetitions in Θ(n) time, but requires Θ(n). Floyd's algorithm uses two runners and Θ(1)
memory.

### 10.2 Intuition of the meeting point

Outside the cycle, both hands run the same way at different speeds. In the cycle, `fast`
wins exactly one node to `slow` per round. With finite cycle length, this relative
distance must eventually become zero modulo cycle length: the hands meet.

```python
def has_cycle(head: SinglyNode[T] | None) -> bool:
    """Return whether a chain contains a cycle using Floyd's runners."""
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        assert slow is not None
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


acyclic = build_chain([1, 2, 3])
assert not has_cycle(acyclic)

cycle_a = SinglyNode("A")
cycle_b = SinglyNode("B")
cycle_c = SinglyNode("C")
cycle_a.next = cycle_b
cycle_b.next = cycle_c
cycle_c.next = cycle_b
assert has_cycle(cycle_a)
```

Values must be repeated; node identity must be compared to `is`, not equal to `==`.

Floyd can be expanded to determine cycle input and cycle length. Meeting point or
reaching `None` are sufficient for the pure examination of existence.

---

## 11. Two sorted lists merge

The Combine idea from Merge Sort can be implemented without new data nodes. A dummy
sentinel keeps the beginning of the result, `tail` its end:

```python
def merge_sorted_chains(
    left: SinglyNode[T] | None,
    right: SinglyNode[T] | None,
) -> SinglyNode[T] | None:
    """Relink two sorted chains into one sorted chain."""
    dummy: SinglyNode[T | None] = SinglyNode(None)
    tail = dummy

    while left is not None and right is not None:
        if left.value <= right.value:
            tail.next = left
            left = left.next
        else:
            tail.next = right
            right = right.next
        tail = tail.next

    tail.next = left if left is not None else right
    return dummy.next


merged_chain = merge_sorted_chains(build_chain([1, 4, 7]), build_chain([2, 3, 8]))
assert list(iterate_chain(merged_chain)) == [1, 2, 3, 4, 7, 8]
```

The dummy is only a local aid limit and is not part of the result. The invariant reads:
Behind `dummy` to `tail` is always the sorted prefix from all already selected nodes.
Time Θ(n + m), additional node memory Θ(1).

---

## 12. Where Linked Lists Happen Real

### 12.1 LRU cache

A Least-Recent-Used cache combined:

- Hash Map: Key → Nodes expected in Θ(1)
- Doubly Linked List: Timeline,
- Sentinel boundaries: Move and remove without edge drop branches.

With `get(key)`, the map directly returns the node. This is moved to the "newest" list
end in Θ(1). If the cache is full at `put`, the oldest node at the other end is removed
in Θ(1) and its key is deleted from the map.

Without Hash Map, the key search would be Θ(n). Without Doubly Linked List, removing a
known middle node or moving would be more elaborate.

### 12.2 Deques

A deque requires insertion and removal at both ends. A Doubly Linked List with Head/Tail
or Sentinels fulfills all four operations in Θ(1).

In CPython, however, `collections.deque` is not implemented as a simple node-per-value
compound, but uses blocks to achieve better locality and less allocations. The abstract
operating table remains similar, the practical implementation is more optimized.

### 12.3 Memory management and free lists

Allocators often manage free storage blocks in chained structures. A free block can
carry the link information itself. Removing or merging known blocks is possible locally
without moving a large index structure.

### 12.4 Operating Systems and Intrusive Lists

For a **intrusive list**, the link fields are located directly in the managed object,
not in a separate wrapper node. Operating system kernels use such structures when
allocation control and constant local mutations are more important than a convenient
generic API.

---

## 13. Common Error Images

### Lose the rest of the list

In the Reversal, `current.next` is overwritten before the old successor is secured.
After that, the unprocessed suffix is no longer available.

### Don't update Tail

After deleting the single element, `tail` remains on a separated node. Or after append,
`tail` continues to point to the penultimate node. Head, tail and size must be
considered a common invariant.

### Compare value instead of identity

Two different nodes can save the same value. Object identity is crucial for cycle
recognition and node membership.

### Off-by-one on insert/delete

To delete index `i`, a Singly Linked List usually needs the node to `i - 1`. Index `0`
is therefore a separate structural case, if no head-sentinel is used.

### Keep remote nodes linked

The main list can be correct, while an externally held remote node still points to
neighbors. Explicit zeroing of its links makes possession and debugging clearer and
prevents accidental double removal.

### Endless loop in `__repr__`

A display function, which runs blindly to `None`, never stops at a cycle. Debug helpers
should optionally use a node limit or visited identities.

---

## 14. Procedure for pointer mutations

Before each implementation:

1. Draw only the nodes involved and their current links.
2. Mark references that need to be preserved after the operation.
3. Secure links that will be overwritten immediately.
4. Change pointers in a sequence that does not create an unattainable part.
5. Check empty list, an element, head, center and end.
6. Then check head, tail, size and neighborhood variants.

A useful local scheme for Doubly Lists:

```text
Remove node:
before = node.prev
after = node.next
before.next = after
after.prev = before

Insert node between before and after:
node.prev   = before
node.next   = after
before.next = node
after.prev = node
```

Sentinels guarantee that `before` and `after` are real node references.

---

## 15. Decision Guide

Choose an array or a python list if:

- direct index access is important;
- data are mainly attached or passed through;
- memory consumption and cache locality are important,
- mean insertions are rare,
- existing library operations already solve the problem well.

Rather choose a Linked List if:

- frequently known nodes are removed or moved locally,
- Node references must remain stable over a long period of time,
- Insert/Delete at both ends is central,
- the structure serves as part of a combination with hash map or other indices;
- the problem structure itself is a sequence of linked states.

Question always: **Is the node already known?** If no, the necessary search can
completely remove the supposed O(1) advantage.

---

## 16. Executive summary

- Singly Linked Lists consist of a head and forward references up to `None`.
- Local mutations are O(1), but finding a node is O(n).
- A tail speeds up append, not deleting the tail in a Singly List.
- Doubly Linked lists allow removal of known nodes in O(1), but require more memory and
  consistent links in both directions.
- Sentinel nodes eliminate edge trap branches through permanent artificial boundaries.
- In-place reverse requires `previous`, `current` and a secured successor.
- Runner technology finds the middle in a pass.
- Floyd detects cycles in O(n) time and O(1) memory via node identity.
- Linked lists are particularly strong as part of combined structures such as LRU
  caches, less than a standard replacement for arrays.

The central habit of this module is: first pointer diagram, then code. Who knows before
each assignment, which nodes must still be reachable after that, dominates Linked Lists
instead of changing them only by attempt and error.

---

# Deutsche Fassung

# Modul 04 — Linked Lists

## Lernziele

Nach diesem Modul kannst du eine Singly Linked List samt aller Grundoperationen
implementieren und jede Zeigeränderung vorher auf Papier nachvollziehen. Du
verstehst den Unterschied zwischen „den Einfügepunkt bereits besitzen“ und „den
Einfügepunkt erst suchen müssen“ und kannst deshalb Laufzeitangaben präzise statt
pauschal begründen. Außerdem beherrschst du In-place-Reversal, Runner-Technik und
Floyds Zyklenerkennung und kannst erklären, warum Doubly Linked Lists mit
Sentinel-Knoten in LRU-Caches und Deques nützlich sind.

Linked Lists sind weniger allgegenwärtig als Python-Listen, aber sie schärfen
eine zentrale algorithmische Fähigkeit: Eine kleine Änderung an Referenzen kann
eine ganze Struktur umformen. Dabei zählt die Reihenfolge der Zuweisungen. Ein
verlorener Link bedeutet einen verlorenen Teilbaum; ein falscher Link kann einen
Zyklus erzeugen.

---

## 1. Motivation: Array oder Linked List?

### 1.1 Intuition: Häuserblock und Schatzsuche

Ein Array ähnelt einem nummerierten Häuserblock. Aus der Adresse `i` lässt sich
direkt berechnen, wo das i-te Element liegt. Dafür müssen die Plätze logisch
zusammenhängend organisiert sein. Wird in der Mitte ein Haus eingefügt, müssen
alle folgenden Bewohner einen Platz weiterrücken.

Eine Linked List ähnelt einer Schatzsuche. Jede Station enthält den Wert und den
Hinweis auf die nächste Station. Eine Station kann irgendwo liegen; sie muss nur
den richtigen Nachfolger nennen. Wenn du die aktuelle Station bereits in der
Hand hältst, kannst du direkt dahinter eine neue Station einschleusen. Um aber
die 500. Station zu finden, musst du 499 Hinweisen folgen.

Das ist der grundlegende Tausch:

- Arrays bieten direkten Indexzugriff und gute Speicherlokalität.
- Linked Lists bieten lokale Strukturänderungen ohne Verschieben nachfolgender
  Werte.

### 1.2 Was „zusammenhängender Speicher“ praktisch bedeutet

Ein klassisches Array speichert gleichartige Elemente nebeneinander. Die Adresse
von Element `i` ist Basisadresse plus `i * element_width`. Deshalb ist Zugriff per
Index Θ(1), und aufeinanderfolgendes Lesen nutzt CPU-Caches gut.

Ein Linked-List-Knoten speichert mindestens zwei Dinge:

```text
┌──────────────┬──────────────┐
│ value        │ next         │
└──────────────┴──────┬───────┘
                      │ Reference to other nodes
```

Knoten können im Speicher verteilt sein. Jeder Schritt benötigt eine weitere
Referenzauflösung. Selbst wenn Array und Linked List beide Θ(n) Elemente
durchlaufen, ist das Array in der Praxis häufig schneller: weniger
Objekt-Overhead, bessere Cache-Lokalität und kompaktere Darstellung.

### 1.3 Python-Besonderheit

Pythons `list` ist ein dynamisches Array von Referenzen, keine Linked List. Die
referenzierten Python-Objekte können verteilt liegen, aber die Referenzen selbst
liegen in einem zusammenhängenden Puffer. Ein eigener Linked-List-Knoten ist
ebenfalls ein Python-Objekt und benötigt zusätzlich das `next`-Attribut. Für
normale Datensammlungen ist eine Python-Liste daher meist kleiner und schneller.

Linked Lists werden interessant, wenn der Algorithmus häufig bekannte Knoten
lokal umhängen muss oder stabile Knotenidentitäten benötigt.

---

## 2. Singly Linked List: Knoten, Kopf und Ende

### 2.1 Intuition: Der Kopf ist der einzige Eingang

Eine Singly Linked List besteht aus Knoten mit genau einer Vorwärtsreferenz. Die
Variable `head` zeigt auf den ersten Knoten. Von dort ist jeder erreichbare
Knoten über wiederholtes `.next` zugänglich.

```text
head
 │
 ▼
┌─────┬──────┐   ┌─────┬──────┐   ┌─────┬──────┐
│  7  │   ●──┼──▶│ 12  │   ●──┼──▶│ 19  │ None │
└─────┴──────┘   └─────┴──────┘   └─────┴──────┘
```

Für die leere Liste gilt `head is None`. Ein einzelner Knoten hat
`head.next is None`. Diese beiden Formen sind keine exotischen Sonderfälle,
sondern die wichtigsten Testfälle jeder Operation.

### 2.2 Ein minimales Knotenmodell

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class SinglyNode(Generic[T]):
    """One node in a singly linked list."""

    value: T
    next: SinglyNode[T] | None = None


def build_chain(values: Iterable[T]) -> SinglyNode[T] | None:
    """Build a chain while keeping a tail reference."""
    head: SinglyNode[T] | None = None
    tail: SinglyNode[T] | None = None
    for value in values:
        node = SinglyNode(value)
        if head is None:
            head = node
        else:
            assert tail is not None
            tail.next = node
        tail = node
    return head


def iterate_chain(head: SinglyNode[T] | None) -> Iterator[T]:
    """Yield values from head to the terminal None reference."""
    current = head
    while current is not None:
        yield current.value
        current = current.next


assert list(iterate_chain(build_chain([]))) == []
assert list(iterate_chain(build_chain([7, 12, 19]))) == [7, 12, 19]
```

`head` besitzt keine magische Verbindung zu allen Knoten. Er ist nur eine
Referenz auf den ersten. Wird `head` überschrieben, ohne die alte Kette vorher
anders zu sichern, kann der Zugang zur gesamten Liste verloren gehen.

---

## 3. Grundoperationen mit Zeigerdiagrammen

### 3.1 Prepend: Vorne einfügen

Ausgang:

```text
head ──▶ [7 | ●] ──▶ [12 ] None
new  ──▶ [3 ] None
```

Zwei Zuweisungen in genau dieser logischen Reihenfolge:

```text
1. new.next = head
2. head = new

head ──▶ [3 | ●] ──▶ [7 | ●] ──▶ [12 ] None
```

```python
def prepend(
    head: SinglyNode[T] | None, value: T
) -> SinglyNode[T]:
    """Return a new head containing value."""
    return SinglyNode(value, head)


prepend_example = prepend(build_chain([7, 12]), 3)
assert list(iterate_chain(prepend_example)) == [3, 7, 12]
```

Zeit und Zusatzspeicher sind Θ(1). Das gilt auch für die leere Liste, weil der
neue Knoten dann `next = None` erhält.

### 3.2 Append ohne und mit Tail

Ohne gespeichertes Ende muss vom Kopf bis zum letzten Knoten gelaufen werden:

```python
def append_without_tail(
    head: SinglyNode[T] | None, value: T
) -> SinglyNode[T]:
    """Append by traversing to the end and return the possibly new head."""
    node = SinglyNode(value)
    if head is None:
        return node

    current = head
    while current.next is not None:
        current = current.next
    current.next = node
    return head


append_example = append_without_tail(build_chain([2, 4]), 6)
assert list(iterate_chain(append_example)) == [2, 4, 6]
```

Die Suche nach dem Ende kostet Θ(n). Eine Listenklasse kann zusätzlich `tail`
speichern. Dann ist Append Θ(1), muss aber bei jeder Mutation die Invariante
erhalten:

```text
empty list:      head is None and tail is None
Non-empty:        tail is accessible and tail.next is None
one element:      head is tail
```

### 3.3 Nach einem bekannten Knoten einfügen

Ausgang:

```text
Current ──▶ [7 | ●] ──▶ [12 | ●] ──▶ [19 ] None
new     ──▶ [10 ] None
```

Wenn zuerst `current.next = new` gesetzt würde, wäre die Referenz auf `12`
überschrieben. Deshalb wird der alte Nachfolger zuerst im neuen Knoten gesichert:

```text
1. new.next = current.next
2. current.next = new

[7 | ●] ──▶ [10 | ●] ──▶ [12 | ●] ──▶ [19 ] None
```

```python
def insert_after(node: SinglyNode[T], value: T) -> SinglyNode[T]:
    """Insert after a known node and return the new node."""
    new_node = SinglyNode(value, node.next)
    node.next = new_node
    return new_node


insert_example = build_chain([7, 12, 19])
assert insert_example is not None
insert_after(insert_example, 10)
assert list(iterate_chain(insert_example)) == [7, 10, 12, 19]
```

Die lokale Mutation ist Θ(1). „An Index `i` einfügen“ bleibt jedoch Θ(i), weil
der Vorgänger zuerst gefunden werden muss.

### 3.4 Kopf löschen

```text
head ──▶ [7 | ●] ──▶ [12 | ●] ──▶ [19 ] None

head = head.next

head ───────────────▶ [12 | ●] ──▶ [19 ] None
```

Der alte Kopf ist danach aus der Liste nicht mehr erreichbar. Wenn keine andere
Referenz auf ihn existiert, kann Python ihn freigeben. Löschen am Kopf ist Θ(1).

### 3.5 Nach einem bekannten Knoten löschen

Um den Nachfolger von `previous` zu entfernen:

```text
previous ──▶ [7 | ●] ──▶ [12 | ●] ──▶ [19 ] None
                          target

previous.next = target.next

previous ──▶ [7 | ●] ─────────────────▶ [19 ] None
```

```python
def delete_after(node: SinglyNode[T]) -> T:
    """Delete and return the value after node."""
    target = node.next
    if target is None:
        raise IndexError("no node exists after the given node")
    node.next = target.next
    target.next = None
    return target.value


delete_example = build_chain([7, 12, 19])
assert delete_example is not None
assert delete_after(delete_example) == 12
assert list(iterate_chain(delete_example)) == [7, 19]
```

`target.next = None` ist für die Struktur nicht zwingend, macht aber die
Abtrennung explizit. Wieder gilt: Ist der Vorgänger bekannt, Θ(1); muss erst nach
einem Wert oder Index gesucht werden, Θ(n).

### 3.6 Suche und Indexzugriff

```python
def find_first(
    head: SinglyNode[T] | None, target: T
) -> SinglyNode[T] | None:
    """Return the first node equal to target, or None."""
    current = head
    while current is not None:
        if current.value == target:
            return current
        current = current.next
    return None


search_example = build_chain([5, 8, 13])
found = find_first(search_example, 8)
assert found is not None and found.value == 8
assert find_first(search_example, 99) is None
```

Im schlechtesten Fall werden alle Knoten besucht: Θ(n). Ein Zugriff auf Index
`i` kostet Θ(i) und damit im Worst Case Θ(n). Linked Lists unterstützen keinen
arithmetisch berechneten Sprung zum i-ten Knoten.

---

## 4. Invarianten einer vollständigen Listenklasse

Eine robuste Singly-Linked-List-Klasse speichert häufig `head`, `tail` und
`size`. Nach jeder öffentlichen Operation müssen gelten:

1. `size >= 0`.
2. `size == 0` genau dann, wenn `head is None` und `tail is None`.
3. Bei `size > 0` sind `head` und `tail` echte Knoten.
4. `tail.next is None`.
5. Vom Kopf sind durch genau `size` Schritte alle Knoten bis zum Tail erreichbar.
6. Innerhalb der Liste gibt es keinen Zyklus.

Kantenfälle lassen sich aus diesen Invarianten ableiten:

| Operation | Leere Liste | Ein Element | Mehrere Elemente |
|---|---|---|---|
| Prepend | Head und Tail werden neuer Knoten | Head ändert sich | Head ändert sich |
| Append | Head und Tail werden neuer Knoten | Tail ändert sich | Tail ändert sich |
| Delete Head | Fehler oder definierter Leerwert | Head und Tail werden `None` | Head wird Nachfolger |
| Delete Tail | Fehler oder definierter Leerwert | Head und Tail werden `None` | Vorgänger muss gesucht werden |

Bei einer Singly Linked List ist selbst mit `tail` das Löschen des letzten
Knotens Θ(n), weil die Rückwärtsreferenz auf seinen Vorgänger fehlt.

---

## 5. Doubly Linked List

### 5.1 Intuition: Hinweise in beide Richtungen

Ein doppelt verketteter Knoten speichert `next` und `prev`:

```text
None── [A] [B] [C] ──▶ None
```

Damit kann von einem bekannten Knoten in beide Richtungen navigiert und ein
bekannter Knoten selbst in Θ(1) entfernt werden. Der Preis:

- eine zusätzliche Referenz pro Knoten,
- mehr Zeigeränderungen pro Mutation,
- mehr Invarianten, die gleichzeitig stimmen müssen.

Für benachbarte Knoten `a` und `b` gilt immer paarweise:

```text
a.next is b exactly when b.prev is a
```

### 5.2 Einen bekannten Knoten entfernen

```text
before 简sthe ▶ target 简sTheafter

1. before.next = after
2. after.prev = before
```

Ohne Sentinels müssen `before is None` und `after is None` für Kopf und Ende
separat behandelt werden. Diese Fallunterscheidungen sind eine häufige Quelle für
Fehler.

---

## 6. Sentinel-Knoten

### 6.1 Intuition: Künstliche Grenzen

Sentinels sind permanente Knoten ohne Nutzwert. Ein `root`-Sentinel kann eine
zirkuläre Doubly Linked List begrenzen:

```text
          ┌────────────────────────────────┐
          ▼                                │
[root] 简┘
```

Für die leere Liste zeigt `root.next` und `root.prev` auf `root` selbst. Jeder
echte Knoten hat dadurch immer einen Vorgänger und Nachfolger. Einfügen und
Entfernen benötigen keine Sonderfälle für Kopf, Ende oder ein einzelnes Element.

### 6.2 Minimales Sentinel-Beispiel

```python
@dataclass(slots=True)
class DoublyNode(Generic[T]):
    """One node in a doubly linked list."""

    value: T | None
    prev: DoublyNode[T] | None = None
    next: DoublyNode[T] | None = None


class SentinelList(Generic[T]):
    """A circular doubly linked list with one permanent sentinel."""

    def __init__(self) -> None:
        self.root: DoublyNode[T] = DoublyNode(None)
        self.root.prev = self.root
        self.root.next = self.root
        self.size = 0

    def append(self, value: T) -> DoublyNode[T]:
        """Append value before the root sentinel."""
        tail = self.root.prev
        assert tail is not None
        node = DoublyNode(value, prev=tail, next=self.root)
        tail.next = node
        self.root.prev = node
        self.size += 1
        return node

    def remove(self, node: DoublyNode[T]) -> T:
        """Unlink a known non-sentinel node in constant time."""
        if node is self.root or node.prev is None or node.next is None:
            raise ValueError("node is not a linked data node")
        before = node.prev
        after = node.next
        before.next = after
        after.prev = before
        node.prev = None
        node.next = None
        self.size -= 1
        assert node.value is not None
        return node.value

    def values(self) -> list[T]:
        """Return all data values from front to back."""
        result: list[T] = []
        current = self.root.next
        while current is not self.root:
            assert current is not None and current.value is not None
            result.append(current.value)
            current = current.next
        return result


sentinel_example: SentinelList[int] = SentinelList()
first_node = sentinel_example.append(4)
second_node = sentinel_example.append(8)
assert sentinel_example.values() == [4, 8]
assert sentinel_example.remove(first_node) == 4
assert sentinel_example.values() == [8]
assert sentinel_example.remove(second_node) == 8
assert sentinel_example.values() == []
```

Sentinels verschieben Komplexität aus jeder Mutation in eine einmalige
Initialisierung. Die permanente Kreisstruktur ist intern; nach außen kann die
Klasse weiterhin wie eine normale endliche Sequenz wirken.

---

## 7. Kostenvergleich

Die Tabelle unterscheidet bewusst zwischen Operationen am bekannten Knoten und
Operationen, bei denen zuerst gesucht werden muss.

| Operation | Dynamisches Array | Singly Linked List | Doubly Linked List |
|---|---:|---:|---:|
| Zugriff Index `i` | Θ(1) | Θ(i), worst Θ(n) | Θ(min(i, n-i))* |
| Suche nach Wert | Θ(n) | Θ(n) | Θ(n) |
| Prepend | Θ(n) | Θ(1) | Θ(1) |
| Append | amortisiert Θ(1) | Θ(1) mit Tail, sonst Θ(n) | Θ(1) mit Tail/Sentinel |
| Insert nach bekanntem Knoten | Θ(n) Verschieben | Θ(1) | Θ(1) |
| Delete Head | Θ(n) Verschieben | Θ(1) | Θ(1) |
| Delete bekannten Knoten | Θ(n) Verschieben | Θ(1) nur mit Vorgänger | Θ(1) |
| Delete Tail | Θ(1) | Θ(n) | Θ(1) |
| Extra-Referenzen pro Element | keine Link-Referenz | `next` | `prev` und `next` |

`*` Wenn Kopf und Tail vorhanden sind und von der näheren Seite traversiert wird.

„Linked Lists haben O(1) Insert“ ist ohne Kontext irreführend. Korrekt ist:

> Das lokale Umhängen ist O(1), wenn die benötigten Knotenreferenzen bereits
> vorliegen. Das Finden dieser Stelle kann O(n) kosten.

---

## 8. In-place-Reversal

### 8.1 Intuition: Eine Straße Schild für Schild umdrehen

Beim Umkehren darf die noch nicht verarbeitete Restliste nicht verloren gehen.
Drei Referenzen reichen:

- `previous`: bereits umgedrehter Prefix,
- `current`: aktuell umzuhängender Knoten,
- `next_node`: vorher gesicherter Rest.

Ausgang:

```text
previous = None
Current ──▶ [A] ──▶ [B] ──▶ [C] ──▶ None
```

Ein Schritt:

```text
1. next_node = current.next     # B secure
2. current.next = previous      # A points backwards to None
3. previous = current           # reverse prefix ends with A
4. current = next_node          # continue with B
```

### 8.2 Implementierung und Invariante

```python
def reverse_in_place(
    head: SinglyNode[T] | None,
) -> SinglyNode[T] | None:
    """Reverse a chain and return its new head."""
    previous: SinglyNode[T] | None = None
    current = head
    while current is not None:
        next_node = current.next
        current.next = previous
        previous = current
        current = next_node
    return previous


reverse_example = build_chain([1, 2, 3, 4])
reverse_example = reverse_in_place(reverse_example)
assert list(iterate_chain(reverse_example)) == [4, 3, 2, 1]
```

Schleifeninvariante:

```text
previous is the head of the correctly reversed processed prefix.
Current is the head of the unchanged suffix.
Both subchains together contain exactly all original nodes.
```

Zeit Θ(n), Zusatzspeicher Θ(1). Kein neuer Datenknoten wird erzeugt.

---

## 9. Runner-Technik: Schneller und langsamer Zeiger

### 9.1 Die Mitte finden

`slow` geht pro Runde einen Schritt, `fast` zwei. Wenn `fast` das Ende erreicht,
hat `slow` ungefähr die Hälfte zurückgelegt.

```text
Start:     slow,fast
              ▼
           [1] → [2] → [3] → [4] → [5]

Round 1:         slow almost
                   ▼     ▼
           [1] → [2] → [3] → [4] → [5]

Round 2:               slow        almost=none after two steps
                         ▼
           [1] → [2] → [3] → [4] → [5]
```

```python
def middle_node(head: SinglyNode[T] | None) -> SinglyNode[T] | None:
    """Return the middle node, choosing the second middle for even length."""
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        assert slow is not None
        slow = slow.next
        fast = fast.next.next
    return slow


odd_middle = middle_node(build_chain([1, 2, 3, 4, 5]))
even_middle = middle_node(build_chain([1, 2, 3, 4]))
assert odd_middle is not None and odd_middle.value == 3
assert even_middle is not None and even_middle.value == 3
```

Die Variante liefert bei gerader Länge den zweiten der beiden mittleren Knoten.
Eine leicht veränderte Startposition oder Schleifenbedingung kann stattdessen den
ersten liefern. Dieses Verhalten muss spezifiziert und getestet werden.

Zeit Θ(n), Zusatzspeicher Θ(1), nur ein Durchlauf.

---

## 10. Zyklenerkennung nach Floyd

### 10.1 Was ist ein Zyklus?

Eine fehlerhafte oder absichtlich zyklische Liste endet nicht bei `None`:

```text
[A] → [B] → [C] → [D]
            ▲           │
            └───────────┘
```

Eine normale Traversierung würde endlos laufen. Eine Menge bereits besuchter
Knoten erkennt Wiederholungen in Θ(n) Zeit, benötigt aber Θ(n) Speicher. Floyds
Algorithmus verwendet zwei Runner und Θ(1) Speicher.

### 10.2 Intuition des Treffpunkts

Außerhalb des Zyklus laufen beide Zeiger denselben Weg mit unterschiedlicher
Geschwindigkeit. Im Zyklus gewinnt `fast` pro Runde genau einen Knoten auf
`slow`. Bei endlicher Zykluslänge muss dieser relative Abstand irgendwann null
modulo Zykluslänge werden: Die Zeiger treffen sich.

```python
def has_cycle(head: SinglyNode[T] | None) -> bool:
    """Return whether a chain contains a cycle using Floyd's runners."""
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        assert slow is not None
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


acyclic = build_chain([1, 2, 3])
assert not has_cycle(acyclic)

cycle_a = SinglyNode("A")
cycle_b = SinglyNode("B")
cycle_c = SinglyNode("C")
cycle_a.next = cycle_b
cycle_b.next = cycle_c
cycle_c.next = cycle_b
assert has_cycle(cycle_a)
```

Werte dürfen sich wiederholen; verglichen werden muss Knotenidentität mit `is`,
nicht Wertgleichheit mit `==`.

Floyd lässt sich erweitern, um Zykluseingang und Zykluslänge zu bestimmen. Für
die reine Existenzprüfung genügen Treffpunkt oder Erreichen von `None`.

---

## 11. Zwei sortierte Listen mergen

Die Combine-Idee aus Merge Sort lässt sich ohne neue Datenknoten umsetzen. Ein
Dummy-Sentinel hält den Anfang des Resultats, `tail` dessen Ende:

```python
def merge_sorted_chains(
    left: SinglyNode[T] | None,
    right: SinglyNode[T] | None,
) -> SinglyNode[T] | None:
    """Relink two sorted chains into one sorted chain."""
    dummy: SinglyNode[T | None] = SinglyNode(None)
    tail = dummy

    while left is not None and right is not None:
        if left.value <= right.value:
            tail.next = left
            left = left.next
        else:
            tail.next = right
            right = right.next
        tail = tail.next

    tail.next = left if left is not None else right
    return dummy.next


merged_chain = merge_sorted_chains(build_chain([1, 4, 7]), build_chain([2, 3, 8]))
assert list(iterate_chain(merged_chain)) == [1, 2, 3, 4, 7, 8]
```

Der Dummy ist nur eine lokale Hilfsgrenze und gehört nicht zum Ergebnis. Die
Invariante lautet: Hinter `dummy` bis `tail` steht stets der sortierte Prefix aus
allen bereits gewählten Knoten. Zeit Θ(n + m), zusätzlicher Knotenspeicher Θ(1).

---

## 12. Wo Linked Lists real vorkommen

### 12.1 LRU-Cache

Ein Least-Recently-Used-Cache kombiniert:

- Hash Map: Schlüssel → Knoten in Θ(1) erwartet,
- Doubly Linked List: Aktualitätsreihenfolge,
- Sentinel-Grenzen: Verschieben und Entfernen ohne Randfallzweige.

Bei `get(key)` liefert die Map direkt den Knoten. Dieser wird in Θ(1) ans
„neueste“ Listenende verschoben. Ist der Cache bei `put` voll, wird der älteste
Knoten am anderen Ende in Θ(1) entfernt und sein Schlüssel aus der Map gelöscht.

Ohne Hash Map wäre die Schlüsselsuche Θ(n). Ohne Doubly Linked List wäre das
Entfernen eines bekannten mittleren Knotens oder das Verschieben aufwendiger.

### 12.2 Deques

Eine Deque benötigt Einfügen und Entfernen an beiden Enden. Eine Doubly Linked
List mit Head/Tail oder Sentinels erfüllt alle vier Operationen in Θ(1).

In CPython ist `collections.deque` allerdings nicht als einfacher Knoten-pro-Wert-
Verbund implementiert, sondern nutzt Blöcke, um bessere Lokalität und weniger
Allokationen zu erreichen. Die abstrakte Operationstabelle bleibt ähnlich, die
praktische Umsetzung ist optimierter.

### 12.3 Speicherverwaltung und freie Listen

Allocatoren verwalten freie Speicherblöcke häufig in verketteten Strukturen. Ein
freier Block kann die Link-Information selbst tragen. Das Entfernen oder
Zusammenführen bekannter Blöcke ist lokal möglich, ohne eine große Indexstruktur
zu verschieben.

### 12.4 Betriebssysteme und intrusive Lists

Bei einer **intrusive list** liegen die Linkfelder direkt im verwalteten Objekt,
nicht in einem separaten Wrapper-Knoten. Betriebssystem-Kernel nutzen solche
Strukturen, wenn Allokationskontrolle und konstante lokale Mutationen wichtiger
sind als eine bequeme generische API.

---

## 13. Häufige Fehlerbilder

### Den Rest der Liste verlieren

Beim Reversal wird `current.next` überschrieben, bevor der alte Nachfolger
gesichert wurde. Danach ist der unverarbeitete Suffix nicht mehr erreichbar.

### Tail nicht aktualisieren

Nach Löschen des einzigen Elements bleibt `tail` auf einem abgetrennten Knoten.
Oder nach Append zeigt `tail` weiter auf den vorletzten Knoten. Head, Tail und
Size müssen als gemeinsame Invariante betrachtet werden.

### Wert statt Identität vergleichen

Zwei verschiedene Knoten können denselben Wert speichern. Für Zyklenerkennung
und Knotenmitgliedschaft ist Objektidentität entscheidend.

### Off-by-one bei Insert/Delete

Zum Löschen an Index `i` braucht eine Singly Linked List normalerweise den
Knoten an `i - 1`. Index `0` ist deshalb ein eigener struktureller Fall, sofern
kein Head-Sentinel verwendet wird.

### Entfernten Knoten verlinkt lassen

Die Hauptliste kann korrekt sein, während ein extern gehaltener entfernter
Knoten noch auf Nachbarn zeigt. Explizites Nullsetzen seiner Links macht Besitz
und Debugging klarer und verhindert versehentliche Doppelentfernung.

### Endlosschleife in `__repr__`

Eine Darstellungsfunktion, die blind bis `None` läuft, terminiert bei einem
Zyklus nie. Debug-Helfer sollten optional ein Knotenlimit oder besuchte
Identitäten verwenden.

---

## 14. Vorgehen bei Zeigermutationen

Vor jeder Implementierung:

1. Zeichne nur die beteiligten Knoten und ihre aktuellen Links.
2. Markiere Referenzen, die nach der Operation erhalten bleiben müssen.
3. Sichere Links, die gleich überschrieben werden.
4. Ändere Zeiger in einer Reihenfolge, die keinen unerreichbaren Teil erzeugt.
5. Prüfe leere Liste, ein Element, Kopf, Mitte und Ende.
6. Prüfe danach Head-, Tail-, Size- und Nachbarschaftsinvarianten.

Ein nützliches lokales Schema für Doubly Lists:

```text
Remove node:
before = node.prev
after = node.next
before.next = after
after.prev = before

Insert node between before and after:
node.prev   = before
node.next   = after
before.next = node
after.prev = node
```

Sentinels garantieren, dass `before` und `after` echte Knotenreferenzen sind.

---

## 15. Entscheidungsleitfaden

Wähle eher ein Array beziehungsweise eine Python-Liste, wenn:

- direkter Indexzugriff wichtig ist,
- Daten überwiegend angehängt oder durchlaufen werden,
- Speicherverbrauch und Cache-Lokalität wichtig sind,
- mittlere Einfügungen selten sind,
- vorhandene Bibliotheksoperationen das Problem bereits gut lösen.

Wähle eher eine Linked List, wenn:

- häufig bekannte Knoten lokal entfernt oder verschoben werden,
- Knotenreferenzen über längere Zeit stabil bleiben müssen,
- Einfügen/Löschen an beiden Enden zentral ist,
- die Struktur als Teil einer Kombination mit Hash Map oder anderen Indizes
  dient,
- die Problemstruktur selbst eine Folge verlinkter Zustände ist.

Frage immer: **Ist der Knoten bereits bekannt?** Wenn nein, kann die notwendige
Suche den vermeintlichen O(1)-Vorteil vollständig aufheben.

---

## 16. Zusammenfassung

- Singly Linked Lists bestehen aus einem Head und Vorwärtsreferenzen bis `None`.
- Lokale Mutationen sind O(1), aber das Finden eines Knotens ist O(n).
- Ein Tail beschleunigt Append, nicht das Löschen des Tails in einer Singly List.
- Doubly Linked Lists erlauben Entfernen bekannter Knoten in O(1), benötigen aber
  mehr Speicher und konsistente Links in beide Richtungen.
- Sentinel-Knoten eliminieren Randfallzweige durch permanente künstliche Grenzen.
- In-place-Reversal benötigt `previous`, `current` und einen gesicherten
  Nachfolger.
- Runner-Technik findet die Mitte in einem Durchlauf.
- Floyd erkennt Zyklen in O(n) Zeit und O(1) Speicher über Knotenidentität.
- Linked Lists sind besonders stark als Bestandteil kombinierter Strukturen wie
  LRU-Caches, weniger als pauschaler Ersatz für Arrays.

Die zentrale Gewohnheit dieses Moduls lautet: Erst Zeigerdiagramm, dann Code.
Wer vor jeder Zuweisung weiß, welche Knoten danach noch erreichbar sein müssen,
beherrscht Linked Lists statt sie nur durch Versuch und Irrtum zu verändern.
