# Module 03 — Recursion & Divide and Conquer

## Learning objectives

After this module, you can not only write recursive functions, but systematically assess
them. For a specific call, you can write down the active stack frames, read the runtime
from a recursion tree and explain why a function is guaranteed to terminate. In
addition, you can see when recursion naturally expresses the structure of a problem and
when an explicit stack or loop is the more robust choice.

Recursion is not one's own "magical" calculation mechanism. It is a form of function
execution in which every call that has not yet been completed is stored in the call
stack. Divide and Conquer frequently uses recursion, but is a separate design paradigm:
a problem is divided into smaller subproblems, these are solved and their results
combined.

---

## 1. Recursion as self-similarity

### 1.1 Intuition: Delegating a task of the same kind

Imagine a series of nested boxes. To find out how many boxes are included, open the
outer box and ask exactly the same question for the inner box. Eventually, you will
reach a box without another box. There is no need for a delegation.

A recursive solution always consists of three contracts:

1. **Base case:** Which smallest input can be answered directly?
2. **Recursion case:** How is the answer composed using a smaller instance of the same
   problem?
3. **Progress guarantee:** Why is every recursive call demonstrably approaching a basic
   case?

If the base case is missing, the delegation runs endlessly. If the base case is present,
but the input is not smaller, it may be unattainable. Therefore, "there is a base case"
is weaker than a real termination justification.

### 1.2 Simulation: factorial step by step

The factorial is defined as:

\[
n! = n \cdot (n-1) \cdot \ldots \cdot 1, \qquad 0! = 1.
\]

The definition already contains the same task for `n - 1`:

```python
def factorial(n: int) -> int:
    """Return n! for a non-negative integer."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1
    return n * factorial(n - 1)


assert factorial(0) == 1
assert factorial(5) == 120
```

For `factorial(4)` a chain of open invoices is created:

```text
factorial(4) needs 4 * factorial(3)
factorial(3) needs 3 * factorial(2)
factorial(2) needs 2 * factorial(1)
factorial(1) needs 1 * factorial(0)
factorial(0) delivers 1
```

Only after that results are returned in reverse order:

```text
factorial(0) = 1
factorial(1) = 1 * 1 = 1
factorial(2) = 2 * 1 = 2
factorial(3) = 3 * 2 = 6
factorial(4) = 4 * 6 = 24
```

The input `n` is a natural progress size. It sinks strictly by one at each call and
cannot remain infinitely positive. This safely achieves the base case `n == 0`.

### 1.3 Formalization: The recursion contract

For recursive function on a problem of size `n`, these claims should be substantiated:

- The base case is correct for the smallest valid size.
- Each recursion case calls the function only with strictly smaller size.
- Assuming that the smaller calls are correct, the current frame combines their answers
  to the correct answer for `n`.

This is the structure of an induction proof. Recursive implementation and proof of
correctness reflect one another:

```text
Base case of the function     ↔ Induction start
recursive call          ↔ Induction
Combination in frame       ↔ Induction step
```

---

## 2. The Call Stack

### 2.1 Intuition: Paused worksheets

A function call has local variables, parameters and a return point. If he calls for
another function, his work is not finished yet. Python places this paused state as
**Stack Frame**. The last started call must be terminated first: Last In, First Out.

A frame contains conceptually:

- the arguments of the call;
- its local variables,
- the place where it continues after the summons;
- the later return value.

### 2.2 Simulation: Frames of `factorial(3)`

After each descent, the call stack looks like this; the active frame is above:

```text
Step 1                  Step 2                  Step 3
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ factorial(3)    │       │ factorial(2)    │       │ factorial(1)    │
│ waiting for f(2) │       │ waiting for f(1) │       │ waiting for f(0) │
└─────────────────┘       ├─────────────────┤       ├─────────────────┤
                          │ factorial(3)    │       │ factorial(2)    │
                          │ waiting for f(2) │       │ waiting for f(1) │
                          └─────────────────┘       ├─────────────────┤
                                                    │ factorial(3)    │
                                                    │ waiting for f(2) │
                                                    └─────────────────┘
```

At the lowest point, `factorial(0)` is added and immediately returns `1`. Then the
frames are removed one after the other:

| Event | Active frame | Calculation | Return |
|---|---|---:|---:|
| Basic case | `factorial(0)` | Direct | 1 |
| Return | `factorial(1)` | `1 * 1` | 1 |
| Return | `factorial(2)` | `2 * 1` | 2 |
| Return | `factorial(3)` | `3 * 2` | 6 |

Important is the separation of **Descent** and **Ascent**. Code before recursive call
runs on descent. Code will not run until a lower frame returns.

### 2.3 A branched call stack: Fibonacci

The naive Fibonacci function calls up twice per inner node:

```python
def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number using the direct recurrence."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(10) == 55
```

Although the recursion tree branches, only one path is kept in the call stack at the
same time. Python first evaluates `fibonacci(n - 1)` completely, returns and then starts
`fibonacci(n - 2)`. Tree size and maximum stack depth are therefore different sizes.

---

## 3. Recursion trees and costs

### 3.1 Intuition: Work per level instead of per line code

A competition describes the cost of a problem over smaller problems. A recursion tree
makes visible:

- how many subproblems exist at each level,
- the size of these subproblems,
- how much non-recursive work each node does,
- how many levels to the basic case arise.

The total cost is the sum of the work of all nodes. It is often enough to determine the
work **per level** and the number of levels.

### 3.2 Linear recursion

For factorial only a multiplication occurs next to the sub-call:

\[
T(n) = T(n-1) + \Theta(1).
\]

The "tree" is a chain with `n + 1` nodes:

```text
T(n)       Θ(1)
 └─T(n-1) Θ(1)
    └─T(n-2) Θ(1)
       ...
          └─T(0) Θ(1)
```

There are Θ(s) levels with Θ(1) work: total Θ(s). At the same time, up to Θ(n) frames
are on the stack: Θ(n), additional memory.

### 3.3 Branched recursion without re-use

For naive Fibonacci, the following applies:

\[
T(n) = T(n-1) + T(n-2) + \Theta(1).
\]

A section for `fibonacci(5)`:

```text
f(5)
├─ f(4)
│  ├─ f(3)
│  │  ├─ f(2)
│  │  └─ f(1)
│  └─ f(2)
└─ f(3)
   ├─ f(2)
   └─ f(1)
```

`f(3)` and `f(2)` are calculated multiple times. The node number grows exponentially,
more precisely Θ(φn) with the golden cut φ, and is often given roughly as O(2n). The
depth, on the other hand, remains Θ(n). This is followed by exponential time, but only
linear stack memory.

### 3.4 Halving with linear plane work

Merge Sort splits an array into two halves and merges the sorted results linearly:

\[
T(n) = 2T(n/2) + \Theta(n).
\]

```text
Level 0:       1 Problem of size n       → Total work Θ(s)
Level 1:       2 Problems of size n/2     → Total work Θ(s)
Level 2:       4 Problems of size n/4     → Total work Θ(s)
...
Level log n:   n Problems of size 1       → Total work Θ(s)
```

There are Θ(log n) levels, each cost Θ(n): total Θ(n log n).

---

## 4. Recursion and Iteration

### 4.1 Intuition: Implicit or explicit state

Recursion stores the still open state implicitly in the call stack. An iterative
solution stores it in loop variables or in its own data structure. Both variants must
receive the same information; they only leave them in different places.

Recursion is often natural with:

- hierarchical structures such as trees and folders,
- Divide-and-Conquer procedures;
- problems whose definition itself is recursive,
- Backtracking with clear decision states.

Iteration is often cheaper for:

- very deep linear chains,
- easy accumulation over sequences,
- productive code with unknown or uncontrolled depth,
- Situations in which stack state should be explicitly inspected or limited.

### 4.2 The same sum in two forms

```python
def recursive_sum(values: list[int], index: int = 0) -> int:
    """Return the sum from index to the end recursively."""
    if index == len(values):
        return 0
    return values[index] + recursive_sum(values, index + 1)


def iterative_sum(values: list[int]) -> int:
    """Return the sum using constant auxiliary stack space."""
    total = 0
    for value in values:
        total += value
    return total


assert recursive_sum([2, 4, 6]) == 12
assert iterative_sum([2, 4, 6]) == 12
```

Both need Θ(n) time. The recursive variant holds Θ(n) frames, the iterative only
constant additional memory. For a flat list, the loop therefore expresses the problem
more directly and robustly.

### 4.3 Python and the recursion limit

CPython protects the native process stack by a recursion limit. It is typically about a
thousand frames, but is a runtime configuration and not a language contract. It can be
queried:

```python
import sys


recursion_limit = sys.getrecursionlimit()
assert recursion_limit > 0
```

Crossing leads to `RecursionError`. Raising the limit at a flat rate is not an
algorithmic solution and can endanger the process stack. At unknown depth, the algorithm
should be formulated iteratively with an explicit stack.

Python also does not perform tail call optimization. Even if the recursive call is the
last operation, a separate frame remains for each call. A tail-recursive linear function
therefore saves no stack memory in Python.

---

## 5. Divide and Conquer

### 5.1 Intuition: Sharing, Conquering, Combining

Divide and Conquer consists of three phases:

1. **Divide:** Split a problem into smaller, as balanced as possible parts.
2. **Conquer:** Solve the subproblems recursively.
3. **Combine:** Compose the partial results for the overall solution.

The basic case solves sufficiently small subproblems directly. The balance is important:
halving creates logarithmic depth; disassembly in sizes `1` and `n - 1` can become
linearly deep.

### 5.2 Simulation of the merge principle

Two already sorted sequences can be connected linearly with two pointers:

```text
left = [2, 5, 8]
right = [1, 4, 9]

Compare 2 and 1 → [1]
compare 2 and 4 → [1, 2]
Compare 5 and 4 → [1, 2, 4]
compare 5 and 9 → [1, 2, 4, 5]
compare 8 and 9 → [1, 2, 4, 5, 8]
Rest right        → [1, 2, 4, 5, 8, 9]
```

```python
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def merge(left: Sequence[T], right: Sequence[T]) -> list[T]:
    """Merge two non-decreasing sequences."""
    merged: list[T] = []
    left_index = 0
    right_index = 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1

    merged.extend(left[left_index:])
    merged.extend(right[right_index:])
    return merged


def merge_sort(values: Sequence[T]) -> list[T]:
    """Return a sorted copy using divide and conquer."""
    if len(values) < 2:
        return list(values)
    middle = len(values) // 2
    left = merge_sort(values[:middle])
    right = merge_sort(values[middle:])
    return merge(left, right)


assert merge([2, 5, 8], [1, 4, 9]) == [1, 2, 4, 5, 8, 9]
assert merge_sort([7, 2, 5, 2, 9, 1]) == [1, 2, 2, 5, 7, 9]
```

The correctness idea of the Combine phase is an invariant: Before each comparison,
`merged` contains exactly the smallest already processed elements in sorted order. The
smaller of the next two candidates is inevitably the next global element.

---

## 6. The master theorem intuitive

### 6.1 The Form

Many balanced Divide-and-Conquer competitions have the form

\[
T(n) = aT(n/b) + f(n).
\]

- `a`: number of recursive subproblems,
- `n/b`: size of each partial problem,
- `f(n)`: Work for sharing and combining in the current node.

The number of leaves of the recursion tree grows like \(n^{\log_b a}\). This size
roughly describes the entire work on the leaf plane. The master theorem compares it to
`f(n)`.

### 6.2 Case 1: The leaves dominate

If the work per node is significantly smaller than the growth of the subproblems, the
large number of leaves dominates.

```text
T(n) = 4T(n/2) + Θ(n)
Sheet size: n^(log_2 4) = n2
Result: Θ(n2)
```

The node number is quadrupled per level, while the node size is halved. The level work
grows geometrically and the last level dominates.

### 6.3 Case 2: All levels are equally expensive

If `f(n)` and the leaf size are equal, each of the logarithmically many planes
contributes the same order of magnitude.

```text
T(n) = 2T(n/2) + Θ(n)
Sheet size: n^(log_2 2) = n
Layers: Θ(log n)
Result: Θ(n log n)
```

This is the Merge Sort case.

### 6.4 Case 3: The root work dominates

If `f(n)` polynomial is greater than the blade size and the work decreases regularly
during descent, the upper plane dominates.

```text
T(n) = 2T(n/2) + Θ(n2)
Sheet size: n
Level work: n2, n2/2, n2/4,...
Result: Θ(n2)
```

The geometric sum remains in the order of its first term.

### 6.5 Boundaries

The master theorem does not match any competition unchanged. Examples:

- `T(n) = T(n - 1) + 1` does not divide by a constant factor.
- Naive Fibonacci creates subproblems of different sizes.
- Highly irregular additional work can violate the standard cases.

Then a direct recursion tree, substitution or other methods help. For this module it is
crucial to understand the three cases as a level comparison, not to memorize a formal
template of proof.

---

## 7. Memoization as Outlook

### 7.1 Intuition: Remember answers to repeated questions

Naive Fibonacci is slow because identical subproblems occur repeatedly. Memoization
stores the response on the first occurrence and returns it later in O(1) from a cache.

```python
def memoized_fibonacci(n: int, cache: dict[int, int] | None = None) -> int:
    """Return Fibonacci(n) while caching overlapping subproblems."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if cache is None:
        cache = {}
    if n < 2:
        return n
    if n not in cache:
        cache[n] = memoized_fibonacci(n - 1, cache) + memoized_fibonacci(
            n - 2, cache
        )
    return cache[n]


assert memoized_fibonacci(10) == 55
assert memoized_fibonacci(100) == 354224848179261915075
```

Now each `n` is calculated only once. The time drops from exponential to Θ(n), the cache
needs Θ(n) memory and the stack depth remains Θ(n). This is an anticipation of dynamic
programming: overlapping subproblems are not solved again.

Memoization only improves repetitions. At Merge Sort, the partial arrays are disjunct;
there are no identical subproblems that could save a cache.

---

## 8. Systematic trac

A table with one entry per call event is suitable for a manual trace:

| Step | Depth | Call | Base case? | Waiting for | Return |
|---:|---:|---|---|---|---|
| 1 | 0 | `factorial(3)` | yes | `factorial(2)` | Open |
| 2 | 1 | `factorial(2)` | yes | `factorial(1)` | Open |
| 3 | 2 | `factorial(1)` | yes | `factorial(0)` | Open |
| 4 | 3 | `factorial(0)` | Yes | — | 1 |
| 5 | 2 | Continued `factorial(1)` | — | — | 1 |
| 6 | 1 | Continued `factorial(2)` | — | — | 2 |
| 7 | 0 | Continued `factorial(3)` | — | — | 6 |

With branched calls you complete the order of the children. A depth-first run means: The
first child is completely processed before the second begins.

A good trace answers four questions:

1. Which arguments does each frame have?
2. What local work has already been done?
3. What sub-call is the frame waiting for?
4. What value is returned on Ascension?

---

## 9. Typical Errors

### Unattainable base case

`n` is increased, although the base case is `0`. The existence of the base case does not
help; the direction of progress is wrong.

### Lost return value

A recursive call is executed, but its result is not passed on with `return`. Then the
outer frame implicitly returns `None`.

### Covered additional costs

Slicing like `values[1:]` creates a new list for each linear recursion step. An apparent
competition `T(n)=T(n-1)+O(1)` thus becomes `T(n)=T(n-1)+O(n)` and total O(n2). An index
parameter avoids copies.

### Exponential repetition

Several recursive calls are not automatically wrong. They become problematic when the
same states are repeatedly calculated. A recursion tree makes this overlap visible.

### Confound tree size with stack depth

Runtime counts all nodes visited. The memory counts the longest simultaneously active
path. An exponential tree can have linear depth.

### Ignore Recursion Limit

A mathematically correct linear recursion in Python can still be practically unsuitable
for large inputs. The runtime complexity alone does not answer the question of robust
implementation.

---

## 10. Decision Guide

Before you use recursion, answer:

1. Is the problem, of course, dissolvable into similar smaller problems?
2. Is there a clear, directly detachable base case?
3. Which measurable size decreases with each call?
4. How deep can the longest path become realistic?
5. Are subproblems calculated several times?
6. Which work is done per node and per level?
7. Does it still have to be maintained after the summons?
8. Would a loop or an explicit stack be clearer or more robust?

For runtime analysis:

1. Formulate a competition.
2. Draw the first two to three levels.
3. Determine node number and work per node per level.
4. Multiply to level work.
5. Determine the depth.
6. Sum up the levels or assign the matching master case.

---

## 11. Executive summary

- Recursion needs a basic case, a recursion case and a guarantee of progress.
- Each open call occupies a stack frame; returns are made in reverse call order.
- Recursion trees separate total node number, layer work and maximum depth.
- Recursion and iteration store the same logical state implicitly or explicitly.
- Python limits recursion and does not optimize Tail Calls.
- Divide and Conquer means sharing, recursively dissolving and combining.
- The master theorem compares paperwork with non-recursive work per node.
- Memoization eliminates repeated calculation of overlapping subproblems, but is not a
  general accelerator for any recursion.

If you can simulate a recursive function frame for frame and evaluate your tree level
for level, you no longer need to treat recursion as a leap into the unknown.

---

# Deutsche Fassung

# Modul 03 — Rekursion & Divide and Conquer

## Lernziele

Nach diesem Modul kannst du rekursive Funktionen nicht nur schreiben, sondern
systematisch beurteilen. Du kannst für einen konkreten Aufruf die aktiven
Stack-Frames notieren, aus einem Rekursionsbaum die Laufzeit ablesen und erklären,
warum eine Funktion sicher terminiert. Außerdem erkennst du, wann Rekursion die
Struktur eines Problems natürlich ausdrückt und wann ein expliziter Stack oder
eine Schleife die robustere Wahl ist.

Rekursion ist kein eigener „magischer“ Rechenmechanismus. Sie ist eine Form der
Funktionsausführung, bei der jeder noch nicht beendete Aufruf im Call Stack
gespeichert bleibt. Divide and Conquer nutzt Rekursion häufig, ist aber ein
separates Entwurfsparadigma: Ein Problem wird in kleinere Teilprobleme zerlegt,
diese werden gelöst und ihre Ergebnisse kombiniert.

---

## 1. Rekursion als Selbstähnlichkeit

### 1.1 Intuition: Eine Aufgabe derselben Art delegieren

Stell dir eine Reihe verschachtelter Kisten vor. Um herauszufinden, wie viele
Kisten enthalten sind, öffnest du die äußere Kiste und stellst für die innere
Kiste exakt dieselbe Frage. Irgendwann erreichst du eine Kiste ohne weitere
Kiste. Dort ist keine Delegation mehr nötig.

Eine rekursive Lösung besteht immer aus drei Verträgen:

1. **Basisfall:** Welche kleinste Eingabe kann unmittelbar beantwortet werden?
2. **Rekursionsfall:** Wie wird die Antwort mithilfe einer kleineren Instanz
   desselben Problems zusammengesetzt?
3. **Fortschrittsgarantie:** Warum nähert sich jeder rekursive Aufruf nachweisbar
   einem Basisfall?

Fehlt der Basisfall, läuft die Delegation endlos. Ist der Basisfall vorhanden,
aber die Eingabe wird nicht kleiner, ist er möglicherweise unerreichbar. Deshalb
ist „es gibt einen Basisfall“ schwächer als eine echte Terminierungsbegründung.

### 1.2 Simulation: Fakultät Schritt für Schritt

Die Fakultät ist definiert als

\[
n! = n \cdot (n-1) \cdot \ldots \cdot 1, \qquad 0! = 1.
\]

Die Definition enthält bereits dieselbe Aufgabe für `n - 1`:

```python
def factorial(n: int) -> int:
    """Return n! for a non-negative integer."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1
    return n * factorial(n - 1)


assert factorial(0) == 1
assert factorial(5) == 120
```

Für `factorial(4)` entsteht zunächst eine Kette offener Rechnungen:

```text
factorial(4) needs 4 * factorial(3)
factorial(3) needs 3 * factorial(2)
factorial(2) needs 2 * factorial(1)
factorial(1) needs 1 * factorial(0)
factorial(0) delivers 1
```

Erst danach werden Ergebnisse in umgekehrter Reihenfolge zurückgegeben:

```text
factorial(0) = 1
factorial(1) = 1 * 1 = 1
factorial(2) = 2 * 1 = 2
factorial(3) = 3 * 2 = 6
factorial(4) = 4 * 6 = 24
```

Die Eingabe `n` ist eine natürliche Fortschrittsgröße. Sie sinkt bei jedem
Aufruf strikt um eins und kann nicht unendlich oft positiv bleiben. Damit wird
der Basisfall `n == 0` sicher erreicht.

### 1.3 Formalisierung: Der Rekursionsvertrag

Für eine rekursive Funktion auf einem Problem der Größe `n` sollten diese
Behauptungen belegbar sein:

- Der Basisfall ist für die kleinste gültige Größe korrekt.
- Jeder Rekursionsfall ruft die Funktion nur mit strikt kleinerer Größe auf.
- Unter der Annahme, dass die kleineren Aufrufe korrekt sind, kombiniert der
  aktuelle Frame ihre Antworten zur korrekten Antwort für `n`.

Das ist die Struktur eines Induktionsbeweises. Die rekursive Implementierung und
der Korrektheitsbeweis spiegeln einander:

```text
Base case of the function     ↔ Induction start
recursive call          ↔ Induction
Combination in frame       ↔ Induction step
```

---

## 2. Der Call Stack

### 2.1 Intuition: Pausierte Arbeitsblätter

Ein Funktionsaufruf besitzt lokale Variablen, Parameter und eine Rücksprungstelle.
Ruft er eine andere Funktion auf, ist seine Arbeit noch nicht fertig. Python legt
diesen pausierten Zustand als **Stack Frame** ab. Der zuletzt gestartete Aufruf
muss zuerst beendet werden: Last In, First Out.

Ein Frame enthält konzeptionell:

- die Argumente des Aufrufs,
- seine lokalen Variablen,
- die Stelle, an der nach dem Unteraufruf fortgesetzt wird,
- den späteren Rückgabewert.

### 2.2 Simulation: Frames von `factorial(3)`

Nach jedem Abstieg sieht der Call Stack so aus; der aktive Frame steht oben:

```text
Step 1                  Step 2                  Step 3
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ factorial(3)    │       │ factorial(2)    │       │ factorial(1)    │
│ waiting for f(2) │       │ waiting for f(1) │       │ waiting for f(0) │
└─────────────────┘       ├─────────────────┤       ├─────────────────┤
                          │ factorial(3)    │       │ factorial(2)    │
                          │ waiting for f(2) │       │ waiting for f(1) │
                          └─────────────────┘       ├─────────────────┤
                                                    │ factorial(3)    │
                                                    │ waiting for f(2) │
                                                    └─────────────────┘
```

Am tiefsten Punkt kommt `factorial(0)` hinzu und liefert sofort `1`. Danach
werden die Frames nacheinander entfernt:

| Ereignis | Aktiver Frame | Berechnung | Rückgabe |
|---|---|---:|---:|
| Basisfall | `factorial(0)` | direkt | 1 |
| Rückkehr | `factorial(1)` | `1 * 1` | 1 |
| Rückkehr | `factorial(2)` | `2 * 1` | 2 |
| Rückkehr | `factorial(3)` | `3 * 2` | 6 |

Wichtig ist die Trennung von **Abstieg** und **Aufstieg**. Code vor dem rekursiven
Aufruf läuft beim Abstieg. Code danach läuft erst, wenn ein tieferer Frame
zurückkehrt.

### 2.3 Ein verzweigter Call Stack: Fibonacci

Die naive Fibonacci-Funktion ruft sich pro innerem Knoten zweimal auf:

```python
def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number using the direct recurrence."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(10) == 55
```

Obwohl der Rekursionsbaum verzweigt, wird immer nur ein Pfad gleichzeitig im
Call Stack gehalten. Python wertet zuerst `fibonacci(n - 1)` vollständig aus,
kehrt zurück und beginnt danach `fibonacci(n - 2)`. Baumgröße und maximale
Stack-Tiefe sind deshalb verschiedene Größen.

---

## 3. Rekursionsbäume und Kosten

### 3.1 Intuition: Arbeit pro Ebene statt pro Zeile Code

Eine Rekurrenz beschreibt die Kosten eines Problems über kleinere Probleme. Ein
Rekursionsbaum macht sichtbar:

- wie viele Teilprobleme auf jeder Ebene existieren,
- wie groß diese Teilprobleme sind,
- wie viel nichtrekursive Arbeit jeder Knoten erledigt,
- wie viele Ebenen bis zum Basisfall entstehen.

Die Gesamtkosten sind die Summe der Arbeit aller Knoten. Häufig genügt es, die
Arbeit **pro Ebene** und die Anzahl der Ebenen zu bestimmen.

### 3.2 Lineare Rekursion

Für Fakultät fällt neben dem Unteraufruf nur eine Multiplikation an:

\[
T(n) = T(n-1) + \Theta(1).
\]

Der „Baum“ ist eine Kette mit `n + 1` Knoten:

```text
T(n)       Θ(1)
 └─T(n-1) Θ(1)
    └─T(n-2) Θ(1)
       ...
          └─T(0) Θ(1)
```

Es gibt Θ(n) Ebenen mit Θ(1) Arbeit: insgesamt Θ(n) Zeit. Gleichzeitig liegen
bis zu Θ(n) Frames auf dem Stack: Θ(n) zusätzlicher Speicher.

### 3.3 Verzweigte Rekursion ohne Wiederverwendung

Für naive Fibonacci gilt näherungsweise:

\[
T(n) = T(n-1) + T(n-2) + \Theta(1).
\]

Ein Ausschnitt für `fibonacci(5)`:

```text
f(5)
├─ f(4)
│  ├─ f(3)
│  │  ├─ f(2)
│  │  └─ f(1)
│  └─ f(2)
└─ f(3)
   ├─ f(2)
   └─ f(1)
```

`f(3)` und `f(2)` werden mehrfach berechnet. Die Knotenzahl wächst exponentiell,
genauer Θ(φⁿ) mit dem goldenen Schnitt φ, und wird oft grob als O(2ⁿ)
angegeben. Die Tiefe bleibt dagegen Θ(n). Daraus folgen exponentielle Zeit, aber
nur linearer Stack-Speicher.

### 3.4 Halbierung mit linearer Ebenenarbeit

Merge Sort zerlegt ein Array in zwei Hälften und führt die sortierten Ergebnisse
linear zusammen:

\[
T(n) = 2T(n/2) + \Theta(n).
\]

```text
Level 0:       1 Problem of size n       → Total work Θ(s)
Level 1:       2 Problems of size n/2     → Total work Θ(s)
Level 2:       4 Problems of size n/4     → Total work Θ(s)
...
Level log n:   n Problems of size 1       → Total work Θ(s)
```

Es gibt Θ(log n) Ebenen, jede kostet Θ(n): insgesamt Θ(n log n).

---

## 4. Rekursion und Iteration

### 4.1 Intuition: Impliziter oder expliziter Zustand

Rekursion speichert den noch offenen Zustand implizit im Call Stack. Eine
iterative Lösung speichert ihn in Schleifenvariablen oder in einer eigenen
Datenstruktur. Beide Varianten müssen dieselbe Information erhalten; sie legen
sie nur an unterschiedlichen Orten ab.

Rekursion ist oft natürlich bei:

- hierarchischen Strukturen wie Bäumen und Ordnern,
- Divide-and-Conquer-Verfahren,
- Problemen, deren Definition selbst rekursiv ist,
- Backtracking mit klaren Entscheidungszuständen.

Iteration ist oft günstiger bei:

- sehr tiefen linearen Ketten,
- einfachem Akkumulieren über Sequenzen,
- produktivem Code mit unbekannter oder unkontrollierter Tiefe,
- Situationen, in denen Stack-Zustand explizit inspiziert oder begrenzt werden
  soll.

### 4.2 Dieselbe Summe in zwei Formen

```python
def recursive_sum(values: list[int], index: int = 0) -> int:
    """Return the sum from index to the end recursively."""
    if index == len(values):
        return 0
    return values[index] + recursive_sum(values, index + 1)


def iterative_sum(values: list[int]) -> int:
    """Return the sum using constant auxiliary stack space."""
    total = 0
    for value in values:
        total += value
    return total


assert recursive_sum([2, 4, 6]) == 12
assert iterative_sum([2, 4, 6]) == 12
```

Beide brauchen Θ(n) Zeit. Die rekursive Variante hält Θ(n) Frames, die iterative
nur konstanten Zusatzspeicher. Für eine flache Liste drückt die Schleife das
Problem daher direkter und robuster aus.

### 4.3 Python und das Rekursionslimit

CPython schützt den nativen Prozessstack durch ein Rekursionslimit. Es liegt
typischerweise ungefähr bei tausend Frames, ist aber eine Laufzeitkonfiguration
und kein Sprachvertrag. Es kann abgefragt werden:

```python
import sys


recursion_limit = sys.getrecursionlimit()
assert recursion_limit > 0
```

Ein Überschreiten führt zu `RecursionError`. Das Limit pauschal stark zu erhöhen
ist keine algorithmische Lösung und kann den Prozessstack gefährden. Bei
unbekannter Tiefe sollte der Algorithmus iterativ mit einem expliziten Stack
formuliert werden.

Python führt außerdem keine Tail-Call-Optimierung durch. Auch wenn der rekursive
Aufruf die letzte Operation ist, bleibt für jeden Aufruf ein eigener Frame
erhalten. Eine tail-rekursive lineare Funktion spart daher in Python keinen
Stack-Speicher.

---

## 5. Divide and Conquer

### 5.1 Intuition: Teilen, Erobern, Kombinieren

Divide and Conquer besteht aus drei Phasen:

1. **Divide:** Zerlege ein Problem in kleinere, möglichst ausgewogene Teile.
2. **Conquer:** Löse die Teilprobleme rekursiv.
3. **Combine:** Setze die Teilergebnisse zur Gesamtlösung zusammen.

Der Basisfall löst ausreichend kleine Teilprobleme direkt. Die Balance ist
wichtig: Halbierung erzeugt logarithmische Tiefe; eine Zerlegung in Größen `1`
und `n - 1` kann linear tief werden.

### 5.2 Simulation des Merge-Prinzips

Zwei bereits sortierte Folgen lassen sich mit zwei Zeigern linear verbinden:

```text
left = [2, 5, 8]
right = [1, 4, 9]

Compare 2 and 1 → [1]
compare 2 and 4 → [1, 2]
Compare 5 and 4 → [1, 2, 4]
compare 5 and 9 → [1, 2, 4, 5]
compare 8 and 9 → [1, 2, 4, 5, 8]
Rest right        → [1, 2, 4, 5, 8, 9]
```

```python
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def merge(left: Sequence[T], right: Sequence[T]) -> list[T]:
    """Merge two non-decreasing sequences."""
    merged: list[T] = []
    left_index = 0
    right_index = 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1

    merged.extend(left[left_index:])
    merged.extend(right[right_index:])
    return merged


def merge_sort(values: Sequence[T]) -> list[T]:
    """Return a sorted copy using divide and conquer."""
    if len(values) < 2:
        return list(values)
    middle = len(values) // 2
    left = merge_sort(values[:middle])
    right = merge_sort(values[middle:])
    return merge(left, right)


assert merge([2, 5, 8], [1, 4, 9]) == [1, 2, 4, 5, 8, 9]
assert merge_sort([7, 2, 5, 2, 9, 1]) == [1, 2, 2, 5, 7, 9]
```

Die Korrektheitsidee der Combine-Phase ist eine Invariante: Vor jedem Vergleich
enthält `merged` genau die kleinsten bereits verarbeiteten Elemente in sortierter
Reihenfolge. Das kleinere der beiden nächsten Kandidaten ist zwangsläufig das
nächste globale Element.

---

## 6. Das Master-Theorem intuitiv

### 6.1 Die Form

Viele ausgewogene Divide-and-Conquer-Rekurrenzen haben die Form

\[
T(n) = aT(n/b) + f(n).
\]

- `a`: Zahl der rekursiven Teilprobleme,
- `n/b`: Größe jedes Teilproblems,
- `f(n)`: Arbeit für Teilen und Kombinieren im aktuellen Knoten.

Die Blätterzahl des Rekursionsbaums wächst wie
\(n^{\log_b a}\). Diese Größe beschreibt grob die gesamte Arbeit an der
Blattebene. Das Master-Theorem vergleicht sie mit `f(n)`.

### 6.2 Fall 1: Die Blätter dominieren

Ist die Arbeit pro Knoten deutlich kleiner als das Wachstum der Teilprobleme,
dominiert die große Zahl der Blätter.

```text
T(n) = 4T(n/2) + Θ(n)
Sheet size: n^(log_2 4) = n2
Result: Θ(n2)
```

Pro Ebene vervierfacht sich die Knotenzahl, während die Knotengröße halbiert
wird. Die Ebenenarbeit wächst geometrisch und die letzte Ebene dominiert.

### 6.3 Fall 2: Alle Ebenen sind gleich teuer

Sind `f(n)` und das Blattmaß gleich groß, trägt jede der logarithmisch vielen
Ebenen dieselbe Größenordnung bei.

```text
T(n) = 2T(n/2) + Θ(n)
Sheet size: n^(log_2 2) = n
Layers: Θ(log n)
Result: Θ(n log n)
```

Das ist der Merge-Sort-Fall.

### 6.4 Fall 3: Die Wurzelarbeit dominiert

Ist `f(n)` polynomial größer als das Blattmaß und nimmt die Arbeit beim Abstieg
regelmäßig ab, dominiert die obere Ebene.

```text
T(n) = 2T(n/2) + Θ(n2)
Sheet size: n
Level work: n2, n2/2, n2/4,...
Result: Θ(n2)
```

Die geometrische Summe bleibt in der Größenordnung ihres ersten Terms.

### 6.5 Grenzen

Das Master-Theorem passt nicht unverändert auf jede Rekurrenz. Beispiele:

- `T(n) = T(n - 1) + 1` teilt nicht durch einen konstanten Faktor.
- Naive Fibonacci erzeugt Teilprobleme unterschiedlicher Größen.
- Stark unregelmäßige Zusatzarbeit kann die Standardfälle verletzen.

Dann helfen ein direkter Rekursionsbaum, Substitution oder andere Verfahren.
Für dieses Modul ist entscheidend, die drei Fälle als Ebenenvergleich zu
verstehen, nicht eine formale Beweisschablone auswendig zu lernen.

---

## 7. Memoization als Ausblick

### 7.1 Intuition: Antworten auf wiederholte Fragen merken

Naive Fibonacci ist langsam, weil identische Teilprobleme wiederholt auftreten.
Memoization speichert die Antwort beim ersten Auftreten und liefert sie später in
O(1) aus einem Cache zurück.

```python
def memoized_fibonacci(n: int, cache: dict[int, int] | None = None) -> int:
    """Return Fibonacci(n) while caching overlapping subproblems."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if cache is None:
        cache = {}
    if n < 2:
        return n
    if n not in cache:
        cache[n] = memoized_fibonacci(n - 1, cache) + memoized_fibonacci(
            n - 2, cache
        )
    return cache[n]


assert memoized_fibonacci(10) == 55
assert memoized_fibonacci(100) == 354224848179261915075
```

Nun wird jedes `n` nur einmal vollständig berechnet. Die Zeit sinkt von
exponentiell auf Θ(n), der Cache benötigt Θ(n) Speicher und die Stack-Tiefe bleibt
Θ(n). Das ist ein Vorgriff auf Dynamic Programming: Überlappende Teilprobleme
werden nicht erneut gelöst.

Memoization verbessert nur Wiederholungen. Bei Merge Sort sind die Teilarrays
disjunkt; dort gibt es keine identischen Teilprobleme, die ein Cache einsparen
könnte.

---

## 8. Systematisches Tracen

Für einen manuellen Trace eignet sich eine Tabelle mit einem Eintrag pro
Aufrufereignis:

| Schritt | Tiefe | Aufruf | Basisfall? | Wartet auf | Rückgabe |
|---:|---:|---|---|---|---|
| 1 | 0 | `factorial(3)` | nein | `factorial(2)` | offen |
| 2 | 1 | `factorial(2)` | nein | `factorial(1)` | offen |
| 3 | 2 | `factorial(1)` | nein | `factorial(0)` | offen |
| 4 | 3 | `factorial(0)` | ja | — | 1 |
| 5 | 2 | Fortsetzung `factorial(1)` | — | — | 1 |
| 6 | 1 | Fortsetzung `factorial(2)` | — | — | 2 |
| 7 | 0 | Fortsetzung `factorial(3)` | — | — | 6 |

Bei verzweigten Aufrufen ergänzt du die Reihenfolge der Kinder. Ein
Depth-First-Ablauf bedeutet: Das erste Kind wird vollständig abgearbeitet, bevor
das zweite beginnt.

Ein guter Trace beantwortet vier Fragen:

1. Welche Argumente besitzt jeder Frame?
2. Welche lokale Arbeit ist schon erledigt?
3. Auf welchen Unteraufruf wartet der Frame?
4. Welcher Wert wird beim Aufstieg zurückgegeben?

---

## 9. Typische Fehler

### Unerreichbarer Basisfall

`n` wird erhöht, obwohl der Basisfall bei `0` liegt. Die Existenz des Basisfalls
hilft nicht; die Fortschrittsrichtung ist falsch.

### Verlorener Rückgabewert

Ein rekursiver Aufruf wird ausgeführt, aber sein Ergebnis nicht mit `return`
weitergegeben. Dann liefert der äußere Frame implizit `None`.

### Verdeckte Zusatzkosten

Slicing wie `values[1:]` erzeugt bei jedem linearen Rekursionsschritt eine neue
Liste. Eine scheinbare Rekurrenz `T(n)=T(n-1)+O(1)` wird dadurch zu
`T(n)=T(n-1)+O(n)` und insgesamt O(n²). Ein Indexparameter vermeidet die Kopien.

### Exponentielle Wiederholung

Mehrere rekursive Aufrufe sind nicht automatisch falsch. Sie werden problematisch,
wenn dieselben Zustände wiederholt berechnet werden. Ein Rekursionsbaum macht
diese Überlappung sichtbar.

### Baumgröße mit Stack-Tiefe verwechseln

Die Laufzeit zählt alle besuchten Knoten. Der Speicher zählt den längsten
gleichzeitig aktiven Pfad. Ein exponentieller Baum kann lineare Tiefe haben.

### Rekursionslimit ignorieren

Eine mathematisch korrekte lineare Rekursion kann in Python für große Eingaben
trotzdem praktisch ungeeignet sein. Die Laufzeitkomplexität allein beantwortet
nicht die Frage nach der robusten Implementierung.

---

## 10. Entscheidungsleitfaden

Bevor du Rekursion verwendest, beantworte:

1. Ist das Problem natürlich in gleichartige kleinere Probleme zerlegbar?
2. Gibt es einen klaren, direkt lösbaren Basisfall?
3. Welche messbare Größe sinkt bei jedem Aufruf?
4. Wie tief kann der längste Pfad realistisch werden?
5. Werden Teilprobleme mehrfach berechnet?
6. Welche Arbeit entsteht pro Knoten und pro Ebene?
7. Muss nach dem Unteraufruf noch Zustand erhalten bleiben?
8. Wäre eine Schleife oder ein expliziter Stack klarer oder robuster?

Für die Laufzeitanalyse:

1. Formuliere eine Rekurrenz.
2. Zeichne die ersten zwei bis drei Ebenen.
3. Bestimme Knotenzahl und Arbeit pro Knoten je Ebene.
4. Multipliziere zur Ebenenarbeit.
5. Bestimme die Tiefe.
6. Summiere die Ebenen oder ordne den passenden Master-Fall zu.

---

## 11. Zusammenfassung

- Rekursion braucht Basisfall, Rekursionsfall und Fortschrittsgarantie.
- Jeder offene Aufruf belegt einen Stack Frame; Rückgaben erfolgen in umgekehrter
  Aufrufreihenfolge.
- Rekursionsbäume trennen Gesamtknotenzahl, Ebenenarbeit und maximale Tiefe.
- Rekursion und Iteration speichern denselben logischen Zustand implizit oder
  explizit.
- Python begrenzt Rekursion und optimiert Tail Calls nicht.
- Divide and Conquer bedeutet Teilen, rekursiv Lösen und Kombinieren.
- Das Master-Theorem vergleicht Blattarbeit mit der nichtrekursiven Arbeit pro
  Knoten.
- Memoization beseitigt wiederholte Berechnung überlappender Teilprobleme, ist
  aber kein allgemeiner Beschleuniger für jede Rekursion.

Wer eine rekursive Funktion Frame für Frame simulieren und ihren Baum Ebene für
Ebene bewerten kann, muss Rekursion nicht mehr als Sprung ins Ungewisse behandeln.
