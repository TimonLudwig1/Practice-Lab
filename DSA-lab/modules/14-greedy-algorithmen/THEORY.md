# Module 14: Greedy Algorithms

A greedy algorithm makes the locally most attractive feasible decision at every
step and never revisits it. This sounds simple, which is exactly why greedy
algorithms are dangerous: a plausible local rule is not yet proof of global
optimality.

This module therefore treats greedy design as two connected tasks:

1. construct a local selection rule,
2. prove or disprove that it is always optimal.

## Learning objectives

After this module, you can:

- distinguish candidates, feasibility, the selection rule, and the objective,
- explain the greedy-choice property and optimal substructure,
- construct counterexamples systematically,
- implement and simulate earliest-finish-time interval scheduling,
- justify correctness with an exchange argument,
- run greedy coin change and demonstrate its limits,
- solve fractional knapsack optimally by value density,
- build a Huffman tree with a min-heap,
- encode and decode text with prefix-free Huffman codes,
- distinguish greedy algorithms from dynamic programming,
- assess whether a greedy strategy is suitable for a new problem.

---

# Part I — The greedy way of thinking

## 1. Four components

Almost every greedy problem can be described with four questions:

1. **Candidates:** Which elements could be selected?
2. **Feasibility:** Which choice preserves all constraints?
3. **Local priority:** Which feasible candidate currently looks best?
4. **Objective:** Which global quantity should be optimized?

Interval scheduling provides a useful example:

| Component | Meaning |
|---|---|
| candidates | time intervals |
| feasible | does not overlap any selected interval |
| local priority | earliest finish time |
| global objective | maximum number of compatible intervals |

The local priority must fit the global objective. “Shortest duration” sounds
reasonable for scheduling, but it is not always correct.

## 2. The common pattern

```text
sort or prioritize the candidates
solution = empty

for candidate in greedy order:
    if candidate is feasible:
        add candidate permanently

return solution
```

Greedy algorithms often use:

- sorting by a key,
- a min-heap or max-heap,
- a sweep line,
- Union-Find for fast feasibility checks,
- a set or counter for resources already in use.

## 3. Irrevocability

Backtracking tries a choice and reverses it if necessary. Dynamic programming
compares several partial decisions. Greedy reasoning says:

> This choice is safe. I will never need to reconsider it.

That claim requires justification.

## 4. Two structural properties

### Greedy-choice property

At least one globally optimal solution begins with the greedy choice. The local
choice therefore does not eliminate every optimal solution.

### Optimal substructure

After the first safe choice, a smaller problem remains, and an optimal solution
to that subproblem is part of an optimal solution to the whole problem.

Optimal substructure alone is insufficient. Many dynamic-programming problems
have it without having a safe greedy choice.

## 5. Runtime versus correctness

The analysis contains two independent questions:

- **Is the result optimal?** Answer with a proof or counterexample.
- **How expensive is the algorithm?** Usually sorting in `O(n log n)` followed
  by a linear scan.

An `O(n log n)` algorithm can compute the wrong result very quickly.

---

# Part II — How to prove a greedy algorithm

## 6. Exchange argument

The most common proof pattern is:

1. Consider any optimal solution `OPT`.
2. If `OPT` already begins with the greedy choice, there is nothing to change.
3. Otherwise, exchange the first choice in `OPT` for the greedy choice.
4. Show that the modified solution remains feasible and is no worse.
5. Repeat the argument for the remaining subproblem.

The exchanges gradually transform an optimal solution into the greedy solution
without reducing its value.

## 7. “Stays ahead”

Show after every step that greedy is at least as far ahead as every alternative.
For example, after selecting `k` intervals, the greedy schedule finishes no
later than any other compatible schedule with `k` intervals.

## 8. Cut and cycle arguments

For minimum spanning trees, the lightest edge across a cut is safe. This is also
greedy reasoning. Kruskal and Prim from Module 13 are greedy algorithms:

- Kruskal selects the cheapest edge that does not create a cycle.
- Prim selects the cheapest edge leaving the current tree.

## 9. Counterexamples are complete disproofs

One input on which a strategy is worse than a feasible alternative is enough to
disprove the claim that the strategy is always optimal.

A good counterexample is:

- small,
- completely verifiable by hand,
- tailored to the weakness of the selection rule,
- documented with both the greedy and better result.

## 10. Constructing a counterexample systematically

Ask which future opportunity the local choice could block.

1. Force greedy to choose an attractive candidate.
2. Place two or more jointly better candidates behind it.
3. Ensure that the first choice excludes that combination.
4. Compare the objective value, not intuition.

This pattern occurs in interval, coin, 0/1-knapsack, and scheduling problems.

---

# Part III — Interval scheduling

## 11. Problem

Given half-open intervals `[start, end)`, find a maximum-size subset without
overlap. An interval ending at 5 is compatible with one beginning at 5.

```python
from dataclasses import dataclass
from itertools import combinations, count
from collections import Counter
import heapq

@dataclass(frozen=True)
class Interval:
    name: str
    start: int
    end: int

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError("An interval requires end > start")

    @property
    def duration(self):
        return self.end - self.start

intervals = (
    Interval("A", 1, 4),
    Interval("B", 3, 5),
    Interval("C", 0, 6),
    Interval("D", 5, 7),
    Interval("E", 3, 9),
    Interval("F", 5, 9),
    Interval("G", 6, 10),
    Interval("H", 8, 11),
    Interval("I", 8, 12),
    Interval("J", 2, 14),
    Interval("K", 12, 16),
)
```

## 12. The correct greedy rule

> Always choose the compatible interval with the earliest finish time.

Why not earliest start? A long interval that starts very early may block the
entire timeline. The earliest finish leaves as much room as possible for the
remaining intervals.

```python
def interval_scheduling(candidates):
    ordered = sorted(candidates, key=lambda interval: (interval.end, interval.start))
    selected = []
    current_end = float("-inf")
    trace = []

    for interval in ordered:
        if interval.start >= current_end:
            selected.append(interval)
            current_end = interval.end
            trace.append(("select", interval.name, current_end))
        else:
            trace.append(("skip", interval.name, current_end))
    return tuple(selected), tuple(trace)

schedule, schedule_trace = interval_scheduling(intervals)
assert tuple(interval.name for interval in schedule) == ("A", "D", "H", "K")
```

## 13. Simulation

Sorted by finish time:

```text
A(1,4), B(3,5), C(0,6), D(5,7), E(3,9), F(5,9),
G(6,10), H(8,11), I(8,12), J(2,14), K(12,16)
```

| Candidate | Compatible? | Decision | Current end |
|---|---|---|---:|
| A [1,4) | yes | select | 4 |
| B [3,5) | no | skip | 4 |
| C [0,6) | no | skip | 4 |
| D [5,7) | yes | select | 7 |
| E [3,9) | no | skip | 7 |
| F [5,9) | no | skip | 7 |
| G [6,10) | no | skip | 7 |
| H [8,11) | yes | select | 11 |
| I [8,12) | no | skip | 11 |
| J [2,14) | no | skip | 11 |
| K [12,16) | yes | select | 16 |

The result contains four intervals.

## 14. A brute-force reference for small inputs

An exact reference implementation helps test greedy on many small cases. It
does not prove the strategy in general, but it can discover counterexamples.

```python
def intervals_are_compatible(candidate_schedule):
    ordered = sorted(candidate_schedule, key=lambda interval: interval.start)
    return all(
        first.end <= second.start
        for first, second in zip(ordered, ordered[1:])
    )

def optimal_interval_schedule_bruteforce(candidates):
    best = ()
    for size in range(len(candidates) + 1):
        for subset in combinations(candidates, size):
            if intervals_are_compatible(subset) and len(subset) > len(best):
                best = subset
    return best

optimal_schedule = optimal_interval_schedule_bruteforce(intervals)
assert len(schedule) == len(optimal_schedule) == 4
assert intervals_are_compatible(schedule)
```

## 15. Exchange argument for earliest finish

Let `G` be the interval with the earliest finish. An optimal solution starts
with an interval `O`.

- `end(G) <= end(O)` by definition.
- Replace `O` with `G`.
- Every later interval that could start after `O` can also start after `G`,
  because `G` ends no later.
- The number of selected intervals remains unchanged.

Thus, an optimal solution beginning with `G` exists. The intervals starting
after `G` form another instance of the same problem. Induction completes the
proof.

## 16. Incorrect rule: earliest start

```python
earliest_start_counterexample = (
    Interval("long", 0, 10),
    Interval("short-1", 1, 2),
    Interval("short-2", 2, 3),
    Interval("short-3", 3, 4),
)

def schedule_by_key(candidates, key):
    selected = []
    current_end = float("-inf")
    for interval in sorted(candidates, key=key):
        if interval.start >= current_end:
            selected.append(interval)
            current_end = interval.end
    return tuple(selected)

wrong_start = schedule_by_key(
    earliest_start_counterexample,
    key=lambda interval: interval.start,
)
right_start, _ = interval_scheduling(earliest_start_counterexample)
assert len(wrong_start) == 1
assert len(right_start) == 3
```

## 17. Incorrect rule: shortest duration

```python
shortest_duration_counterexample = (
    Interval("left", 0, 3),
    Interval("tempting", 2, 4),
    Interval("right", 3, 6),
)
wrong_duration = schedule_by_key(
    shortest_duration_counterexample,
    key=lambda interval: interval.duration,
)
right_duration, _ = interval_scheduling(shortest_duration_counterexample)
assert tuple(interval.name for interval in wrong_duration) == ("tempting",)
assert tuple(interval.name for interval in right_duration) == ("left", "right")
```

## 18. Weighted intervals

If intervals have different values and the goal is to maximize **total value**,
earliest finish is no longer optimal in general. One valuable interval can be
better than several cheap ones. Weighted interval scheduling is a classic
dynamic-programming problem.

## 19. Complexity

- Sorting: `O(n log n)`
- Selection scan: `O(n)`
- Auxiliary space: `O(n)` for the sorted copy and result

If the input is already sorted by finish time, `O(n)` is sufficient.

---

# Part IV — Coin change

## 20. The natural greedy rule

Repeatedly choose the largest coin that does not exceed the remaining amount.

```python
def greedy_change(coins, amount):
    if amount < 0:
        raise ValueError("Amount must be nonnegative")
    if any(coin <= 0 for coin in coins):
        raise ValueError("Coins must be positive")

    remaining = amount
    chosen = []
    for coin in sorted(set(coins), reverse=True):
        count_for_coin, remaining = divmod(remaining, coin)
        chosen.extend([coin] * count_for_coin)
    return tuple(chosen), remaining

euro_coins = (1, 2, 5, 10, 20, 50)
euro_change, euro_rest = greedy_change(euro_coins, 87)
assert euro_change == (50, 20, 10, 5, 2)
assert euro_rest == 0
```

For common euro denominations, this rule returns a minimum number of coins.
That success depends on the structure of the coin system, not on the greedy
scheme alone.

## 21. Counterexample `{1, 3, 4}` for amount 6

Greedy chooses:

```text
4 + 1 + 1 = 6    three coins
```

The optimum is:

```text
3 + 3 = 6        two coins
```

```python
def optimal_change_dp(coins, amount):
    if amount < 0:
        raise ValueError("Amount must be nonnegative")
    usable = tuple(sorted(set(coins), reverse=True))
    best = [None] * (amount + 1)
    best[0] = ()

    for subtotal in range(amount + 1):
        if best[subtotal] is None:
            continue
        for coin in usable:
            target = subtotal + coin
            if target <= amount:
                candidate = best[subtotal] + (coin,)
                if best[target] is None or len(candidate) < len(best[target]):
                    best[target] = candidate
    return best[amount]

bad_greedy, bad_rest = greedy_change((1, 3, 4), 6)
bad_optimal = optimal_change_dp((1, 3, 4), 6)
assert bad_rest == 0
assert bad_greedy == (4, 1, 1)
assert bad_optimal == (3, 3)
assert len(bad_greedy) > len(bad_optimal)
```

This counterexample disproves the statement that the largest fitting coin is
always safe.

## 22. Canonical coin systems

A coin system is canonical if greedy returns a minimum number of coins for every
representable amount. Testing many amounts builds confidence, but without an
additional bound it is not a general proof.

```python
def first_change_counterexample(coins, maximum):
    for amount in range(maximum + 1):
        greedy, rest = greedy_change(coins, amount)
        optimal = optimal_change_dp(coins, amount)
        if rest == 0 and optimal is not None and len(greedy) != len(optimal):
            return amount, greedy, optimal
    return None

assert first_change_counterexample(euro_coins, 200) is None
assert first_change_counterexample((1, 3, 4), 20) == (
    6,
    (4, 1, 1),
    (3, 3),
)
```

## 23. Why greedy fails here

Coin 4 looks locally best, but it leaves amount 2, which needs two unit coins.
Greedy does not account for the future remainder. Dynamic programming compares
the best solutions for all smaller amounts.

## 24. Unrepresentable amounts

Without a unit coin, a remainder may remain. A robust API must not report that
case as a complete solution.

```python
incomplete_change, incomplete_rest = greedy_change((4, 6), 7)
assert incomplete_change == (6,)
assert incomplete_rest == 1
assert optimal_change_dp((4, 6), 7) is None
```

---

# Part V — Fractional knapsack

## 25. Problem

Each item has weight `w` and value `v`; the knapsack has capacity `C`. Items may
be split. The goal is to maximize total value.

The key measure is value density:

```text
ratio = value / weight
```

## 26. Greedy by value density

```python
@dataclass(frozen=True)
class Item:
    name: str
    weight: float
    value: float

    def __post_init__(self):
        if self.weight <= 0 or self.value < 0:
            raise ValueError("Weight must be positive and value nonnegative")

    @property
    def density(self):
        return self.value / self.weight

def fractional_knapsack(items, capacity):
    if capacity < 0:
        raise ValueError("Capacity must be nonnegative")
    remaining = capacity
    total_value = 0.0
    selection = []

    for item in sorted(items, key=lambda candidate: candidate.density, reverse=True):
        if remaining == 0:
            break
        fraction = min(1.0, remaining / item.weight)
        selection.append((item, fraction))
        total_value += fraction * item.value
        remaining -= fraction * item.weight
    return total_value, tuple(selection), remaining

knapsack_items = (
    Item("A", 10, 60),
    Item("B", 20, 100),
    Item("C", 30, 120),
)
fractional_value, fractional_selection, unused_capacity = fractional_knapsack(
    knapsack_items, 50
)
assert fractional_value == 240
assert tuple((item.name, fraction) for item, fraction in fractional_selection) == (
    ("A", 1.0),
    ("B", 1.0),
    ("C", 2 / 3),
)
assert unused_capacity == 0
```

## 27. Simulation

| Item | Weight | Value | Density | Fraction | Value contribution |
|---|---:|---:|---:|---:|---:|
| A | 10 | 60 | 6 | 100% | 60 |
| B | 20 | 100 | 5 | 100% | 100 |
| C | 30 | 120 | 4 | 66.7% | 80 |
| **Total** | **50** | | | | **240** |

## 28. Exchange argument

Suppose a solution uses weight from a lower-density item while a higher-density
item is not yet fully used. Exchange a small amount of weight `δ`:

- remove value `δ * lower_density`,
- add value `δ * higher_density`,
- keep total weight unchanged,
- never reduce value, and increase it when the density is strictly higher.

Every optimal solution can therefore be transformed into density order.
Divisibility is the essential property.

## 29. Why 0/1 knapsack is different

Without divisibility, the locally best ratio can block a globally better
combination. In this example, the fractional optimum is 240, but the best
integral combination is B+C with value 220.

```python
def zero_one_knapsack_bruteforce(items, capacity):
    best_value = 0
    best_subset = ()
    for size in range(len(items) + 1):
        for subset in combinations(items, size):
            weight = sum(item.weight for item in subset)
            value = sum(item.value for item in subset)
            if weight <= capacity and value > best_value:
                best_value = value
                best_subset = subset
    return best_value, best_subset

integer_value, integer_selection = zero_one_knapsack_bruteforce(
    knapsack_items, 50
)
assert integer_value == 220
assert {item.name for item in integer_selection} == {"B", "C"}
assert integer_value < fractional_value
```

In general, 0/1 knapsack requires dynamic programming, branch-and-bound, or an
approximation; pure ratio-greedy does not guarantee an optimum.

## 30. Complexity

- Density sorting: `O(n log n)`
- Filling: `O(n)`
- Total: `O(n log n)`

---

# Part VI — Huffman coding

## 31. Objective

Huffman assigns short bit codes to frequent symbols and longer codes to rare
symbols. The codes are prefix-free: no valid code is the prefix of another.
Therefore, a bit stream can be decoded unambiguously from left to right.

The objective is to minimize weighted code length:

```text
sum over symbols: frequency(symbol) * code_length(symbol)
```

## 32. The greedy choice

> Repeatedly combine the two subtrees with the smallest frequencies.

The new parent receives the sum of both frequencies and returns to the min-heap.
The final remaining node is the root.

## 33. Data structure and tree construction

```python
@dataclass(frozen=True)
class HuffmanNode:
    frequency: int
    symbol: str | None = None
    left: "HuffmanNode | None" = None
    right: "HuffmanNode | None" = None

    @property
    def is_leaf(self):
        return self.symbol is not None

def build_huffman_tree(frequencies):
    if not frequencies:
        raise ValueError("At least one symbol is required")
    if any(frequency <= 0 for frequency in frequencies.values()):
        raise ValueError("Frequencies must be positive")

    order = count()
    heap = [
        (frequency, next(order), HuffmanNode(frequency, symbol))
        for symbol, frequency in sorted(frequencies.items())
    ]
    heapq.heapify(heap)
    trace = []

    while len(heap) > 1:
        first_frequency, _, first = heapq.heappop(heap)
        second_frequency, _, second = heapq.heappop(heap)
        parent_frequency = first_frequency + second_frequency
        parent = HuffmanNode(parent_frequency, None, first, second)
        heapq.heappush(heap, (parent_frequency, next(order), parent))
        trace.append((first_frequency, second_frequency, parent_frequency))
    return heap[0][2], tuple(trace)

classic_frequencies = {
    "A": 5,
    "B": 9,
    "C": 12,
    "D": 13,
    "E": 16,
    "F": 45,
}
huffman_root, huffman_trace = build_huffman_tree(classic_frequencies)
assert huffman_root.frequency == 100
assert huffman_trace == (
    (5, 9, 14),
    (12, 13, 25),
    (14, 16, 30),
    (25, 30, 55),
    (45, 55, 100),
)
```

## 34. Reading codes from the tree

A left edge represents `0`; a right edge represents `1`. With only one symbol,
use `0` so repeated occurrences still produce an explicit bit stream.

```python
def huffman_codes(root):
    codes = {}

    def visit(node, prefix):
        if node.is_leaf:
            codes[node.symbol] = prefix or "0"
            return
        if node.left is None or node.right is None:
            raise ValueError("Invalid Huffman tree")
        visit(node.left, prefix + "0")
        visit(node.right, prefix + "1")

    visit(root, "")
    return codes

classic_codes = huffman_codes(huffman_root)
assert classic_codes == {
    "F": "0",
    "C": "100",
    "D": "101",
    "A": "1100",
    "B": "1101",
    "E": "111",
}
```

Different tie-breakers may produce different but equally optimal codes when
frequencies are equal.

## 35. Checking prefix freedom

```python
def is_prefix_free(codes):
    values = tuple(codes.values())
    return all(
        first == second or not second.startswith(first)
        for first in values
        for second in values
    )

assert is_prefix_free(classic_codes)
```

## 36. Encoding and decoding

```python
def huffman_encode(text, codes):
    try:
        return "".join(codes[symbol] for symbol in text)
    except KeyError as error:
        raise ValueError(f"Unknown symbol: {error.args[0]!r}") from error

def huffman_decode(bits, root):
    if root.is_leaf:
        if any(bit != "0" for bit in bits):
            raise ValueError("Invalid bit stream")
        return root.symbol * len(bits)

    decoded = []
    node = root
    for bit in bits:
        if bit not in "01":
            raise ValueError("Bit stream may contain only 0 and 1")
        node = node.left if bit == "0" else node.right
        if node is None:
            raise ValueError("Bit stream leaves the Huffman tree")
        if node.is_leaf:
            decoded.append(node.symbol)
            node = root
    if node is not root:
        raise ValueError("Bit stream ends inside a code")
    return "".join(decoded)

sample_text = "FACE"
encoded_sample = huffman_encode(sample_text, classic_codes)
assert huffman_decode(encoded_sample, huffman_root) == sample_text
```

## 37. Compression effect

A fixed code for six symbols needs at least three bits per symbol. One hundred
symbol occurrences would therefore need 300 bits. Huffman needs:

```python
huffman_bits = sum(
    frequency * len(classic_codes[symbol])
    for symbol, frequency in classic_frequencies.items()
)
fixed_width_bits = sum(classic_frequencies.values()) * 3
assert huffman_bits == 224
assert fixed_width_bits == 300
assert huffman_bits / fixed_width_bits < 0.75
```

This calculation ignores storage for the tree or codebook. For very short text,
the header can consume the entire gain.

## 38. Why are the two smallest frequencies siblings?

In an optimal prefix tree, the two rarest symbols can be placed as siblings at
maximum depth. If more frequent symbols occupy those positions, exchanging them
with rarer ones cannot increase weighted length.

Combining the two siblings into a pseudo-symbol with their total frequency
leaves a smaller Huffman problem. This establishes the greedy-choice property
and optimal substructure.

## 39. Complexity

For `k` distinct symbols and text length `n`:

- Frequency analysis: `O(n)`
- `k-1` heap merges: `O(k log k)`
- Code generation: `O(k)` plus total code lengths
- Text encoding: `O(n)` dictionary lookups
- Space: `O(k)` for tree and codebook, plus output bits

---

# Part VII — Meeting rooms as another greedy pattern

## 40. Minimum number of rooms required simultaneously

Sort meetings by start time. A min-heap stores the end times of occupied rooms.
If the earliest room is free in time, reuse it; otherwise, open a new room.

```python
def minimum_meeting_rooms(meetings):
    active_end_times = []
    maximum_rooms = 0
    trace = []

    for meeting in sorted(meetings, key=lambda interval: interval.start):
        while active_end_times and active_end_times[0] <= meeting.start:
            freed_at = heapq.heappop(active_end_times)
            trace.append(("release", freed_at, meeting.start))
        heapq.heappush(active_end_times, meeting.end)
        maximum_rooms = max(maximum_rooms, len(active_end_times))
        trace.append(("allocate", meeting.name, tuple(sorted(active_end_times))))
    return maximum_rooms, tuple(trace)

meeting_sample = (
    Interval("M1", 0, 30),
    Interval("M2", 5, 10),
    Interval("M3", 15, 20),
    Interval("M4", 20, 25),
)
room_count, room_trace = minimum_meeting_rooms(meeting_sample)
assert room_count == 2
```

The maximum heap size equals the maximum overlap, which is a lower bound that
the algorithm reaches exactly. Project 03 turns this count into concrete room
assignments.

## 41. Selection versus resource demand

Do not confuse these objectives:

- Interval scheduling selects the maximum number of meetings for **one** room.
- Meeting rooms schedules **all** meetings with the minimum number of rooms.

The interval data looks similar, but the objective and greedy rule differ.

---

# Part VIII — Greedy versus dynamic programming

## 42. Similarities

Both often exploit optimal substructure and build solutions step by step. The
key difference is how many alternatives remain available.

## 43. Differences

| Aspect | Greedy | Dynamic programming |
|---|---|---|
| decision | locally best and final | compare multiple alternatives |
| revise earlier choice | no | indirectly through state comparison |
| typical runtime | often `O(n log n)` | often multidimensional or pseudopolynomial |
| memory | usually small | state table or memoization |
| proof obligation | local choice must be safe | recurrence must cover every case |
| example | fractional knapsack | 0/1 knapsack |

## 44. Related problems, different guarantees

| Problem | Greedy? | Reason or alternative |
|---|---|---|
| unweighted interval scheduling | yes | earliest finish plus exchange |
| weighted interval scheduling | generally no | DP over predecessor intervals |
| fractional knapsack | yes | density plus small exchanges |
| 0/1 knapsack | generally no | DP or branch-and-bound |
| coin change with arbitrary coins | generally no | DP over amounts |
| Huffman coding | yes | merge the two smallest frequencies |
| shortest path with nonnegative edges | yes | Dijkstra |
| shortest path with negative edges | not Dijkstra | Bellman–Ford |

## 45. Warning sign: a choice strongly changes the remainder

If a local choice produces a complicated remaining state and no exchangeability
is visible, dynamic programming or search is more promising. The remaining
amount matters in coin change; remaining capacity and combinations of whole
items matter in 0/1 knapsack.

---

# Part IX — Recognizing greedy suitability

## 46. Positive signals

- A clear local ranking exists.
- The local choice can replace the first choice of every optimal solution.
- Only the remaining set matters, not the detailed history.
- Feasible sets have a strong exchange structure.
- A cut, prefix, or dominance argument seems natural.
- The structure resembles intervals, MSTs, Huffman, or nonnegative shortest paths.

## 47. Negative signals

- One choice creates many qualitatively different remaining states.
- Parts are indivisible and combinations matter.
- Local measures ignore a critical future remainder.
- Small input changes break the strategy.
- Several plausible rules disagree, but no exchange argument works.
- A small brute-force program quickly finds counterexamples.

## 48. Evaluation process for a new strategy

```text
1. Specify the objective and constraints.
2. State the local rule precisely.
3. Test small instances by hand.
4. Construct or search for counterexamples.
5. Justify the greedy-choice property.
6. Identify optimal substructure.
7. Formulate a proof: exchange, stays ahead, cut, or contradiction.
8. Only then build the optimized implementation.
```

## 49. Brute force as a strategy tester

For small inputs, compare a proposed greedy rule against an exact reference.
This does not replace proof, but it is an excellent counterexample generator.

```python
def verify_interval_greedy_on_instance(candidates):
    greedy, _ = interval_scheduling(candidates)
    optimal = optimal_interval_schedule_bruteforce(candidates)
    return len(greedy) == len(optimal)

assert verify_interval_greedy_on_instance(intervals)
assert verify_interval_greedy_on_instance(earliest_start_counterexample)
assert verify_interval_greedy_on_instance(shortest_duration_counterexample)
```

---

# Part X — Common pitfalls

## 50. “Greedy is fast, so I will use greedy”

Runtime does not guarantee correctness. Analyze the structure first.

## 51. Describing the strategy imprecisely

“Take the best element” is not a rule. Best by which key? How are ties resolved?
When is a candidate feasible?

## 52. Treating a few successful examples as proof

Tests can reveal errors, but cannot establish universal correctness alone.

## 53. Confusing objectives

Maximum count, maximum value, minimum rooms, and minimum total duration are
different problems, even when they use identical input data.

## 54. Ignoring ties

Tie-breaking affects reproducibility. A correct algorithm should remain optimal
for every allowed tie-breaker or specify the rule explicitly.

## 55. Mutating the input

`list.sort()` changes its input. Educational and library code should often use
`sorted()` to create a copy.

## 56. Confusing fractional and 0/1 variants

Divisibility changes the mathematical structure completely.

## 57. Forgetting Huffman codebook costs

Compression measurements must include the header, tree, or codebook, especially
for short texts.

## 58. Invalid edge cases

- an empty candidate collection,
- intervals with `end <= start`,
- negative capacity,
- nonpositive coins or weights,
- an unrepresentable amount,
- text containing only one symbol,
- an unknown symbol during encoding,
- an incomplete bit code during decoding.

---

# Part XI — Correctness and testing

## 59. Interval-scheduling invariants

- Selected intervals are pairwise compatible.
- Their finish times increase.
- After each choice, all earlier-finishing candidates have been considered.
- The result is maximal under the strategy; the exchange argument additionally
  proves optimality.

```python
assert intervals_are_compatible(schedule)
assert [interval.end for interval in schedule] == sorted(
    interval.end for interval in schedule
)
assert len(schedule_trace) == len(intervals)
```

## 60. Fractional-knapsack invariants

- Every fraction lies in `[0,1]`.
- Total weight does not exceed capacity.
- Positive fractions occur in nonincreasing density order.
- At most one item is selected partially.

```python
fractions = [fraction for _, fraction in fractional_selection]
densities = [item.density for item, _ in fractional_selection]
used_weight = sum(
    item.weight * fraction for item, fraction in fractional_selection
)
assert all(0 <= fraction <= 1 for fraction in fractions)
assert densities == sorted(densities, reverse=True)
assert sum(0 < fraction < 1 for fraction in fractions) <= 1
assert used_weight <= 50
```

## 61. Huffman invariants

- Root frequency equals the sum of all frequencies.
- Every internal node has two children.
- The codebook contains exactly all symbols.
- Codes are prefix-free.
- `decode(encode(text)) == text`.
- Weighted bit length agrees with the codebook.

```python
assert huffman_root.frequency == sum(classic_frequencies.values())
assert set(classic_codes) == set(classic_frequencies)
assert is_prefix_free(classic_codes)
assert huffman_decode(
    huffman_encode("ABCDEF", classic_codes), huffman_root
) == "ABCDEF"
```

## 62. Property tests

Strong automated checks include:

- random small interval sets against brute force,
- greedy coin change against DP while recording counterexamples,
- fractional knapsack against linear optimization on small cases,
- Huffman round trips for seeded texts,
- prefix freedom for every generated codebook,
- meeting-room count against maximum overlap from a sweep line.

---

# Part XII — Data-science transfer

## 63. Greedy methods in data work

- allocate a budget step by step to actions with highest marginal benefit,
- select nonoverlapping training or maintenance windows,
- schedule jobs by deadline or priority,
- connect clusters or graph components through cheap edges,
- reduce storage with prefix codes,
- select features approximately under cost constraints,
- assign data-pipeline resources with heaps.

## 64. Be careful with heuristics

In data science, a local strategy is often called “greedy feature selection” or
“greedy search.” It may be practically useful without a global optimality
guarantee. Distinguish clearly between:

- **Algorithm with proof:** guaranteed optimal under stated assumptions.
- **Heuristic:** plausible and often fast, but without a universal guarantee.

A heuristic is not inherently bad; its claim simply needs to be honest.

## 65. Measurable quality

When optimality cannot be proved or exact optimization is too expensive:

- solve small instances exactly and measure the quality gap,
- compare several selection rules,
- investigate approximation ratios,
- test sensitivity to tie-breaking and random seeds,
- report runtime and solution quality separately.

---

# Part XIII — Review questions

## 66. Questions

1. What makes a decision greedy?
2. Why is optimal substructure alone insufficient?
3. What does an exchange argument establish?
4. Which rule solves unweighted interval scheduling optimally?
5. Why is earliest start incorrect?
6. What does coin system `{1,3,4}` disprove for amount 6?
7. Why does value density work for fractional knapsack?
8. Why does the same rule fail for 0/1 knapsack?
9. Which two elements does Huffman combine at every step?
10. Why is a Huffman code prefix-free?
11. What role does the min-heap play?
12. How does meeting-room counting differ from interval scheduling?
13. Which warning signs suggest dynamic programming?
14. Can testing replace a greedy proof?
15. When can a greedy heuristic still be useful?

## 67. Short answers

1. It makes a locally optimal feasible choice and never reverses it.
2. It does not show that one particular local choice is safe.
3. An optimal solution can adopt the greedy choice without losing quality.
4. Sort by earliest finish and select compatible intervals.
5. A long interval starting early may block many short intervals.
6. The largest fitting coin does not always minimize coin count.
7. Small weight amounts can be exchanged for material of higher density.
8. Whole items prevent arbitrarily small exchanges.
9. The two subtrees with the smallest frequencies.
10. Only leaves contain symbols, and no leaf is above another leaf.
11. It repeatedly returns the two smallest frequencies in `O(log k)`.
12. One selects as many meetings as possible for one room; the other schedules
    every meeting with as few rooms as possible.
13. Complex remaining state, indivisible combinations, and no exchange argument.
14. No; tests can find counterexamples and increase confidence.
15. When exact optimization is too costly and quality is measured empirically.

---

# Part XIV — Compact overview

## 68. Cheat sheet

| Problem | Greedy key | Structure | Guarantee |
|---|---|---|---|
| interval scheduling | earliest finish | exchange | optimal |
| arbitrary coin change | largest coin | system-dependent | not general |
| fractional knapsack | highest value density | divisibility plus exchange | optimal |
| 0/1 knapsack | highest value density | indivisible | not general |
| Huffman | two smallest frequencies | prefix tree plus induction | optimal |
| meeting rooms | earliest end time in heap | maximum overlap | optimal |
| Kruskal | lightest safe edge | cut plus Union-Find | optimal |
| Dijkstra | smallest unsettled distance | nonnegative edges | optimal |

## 69. One sentence per method

```text
Interval scheduling: Finish as early as possible and preserve future space.
Coin change: The largest coin is safe only in suitable coin systems.
Fractional knapsack: Buy value per unit of weight in descending order.
Huffman: Give rare symbols the greatest depth.
Meeting rooms: Always reuse the room that becomes free first.
```

## 70. Project preview

- **01-basic:** Implement interval scheduling, fractional knapsack, and coin
  change; demonstrate greedy failure explicitly against an exact reference.
- **02-medium:** Build a complete Huffman compressor for text files and measure
  compression rates including metadata.
- **03-final:** Plan a seeded calendar, determine the minimum room count, create
  concrete assignments, and compare them with a naive allocation.

The central greedy skill is not sorting. It is the ability to justify a local
rule through structure or reject it honestly with a small counterexample.

---

# Deutsche Fassung

# Modul 14: Greedy-Algorithmen

Ein Greedy-Algorithmus trifft in jedem Schritt die lokal attraktivste zulässige
Entscheidung und revidiert sie später nicht. Das klingt einfach – und genau
deshalb ist Greedy gefährlich: Eine plausible lokale Regel ist noch kein Beweis
für globale Optimalität.

Dieses Modul behandelt Greedy daher als Doppelaufgabe:

1. eine lokale Auswahlregel konstruieren,
2. beweisen oder widerlegen, dass sie immer optimal ist.

## Lernziele

Nach diesem Modul kannst du:

- Kandidaten, Zulässigkeit, Auswahlregel und Ziel einer Greedy-Lösung trennen,
- Greedy-Choice-Property und optimale Teilstruktur erklären,
- Gegenbeispiele systematisch konstruieren,
- Interval Scheduling nach frühestem Ende implementieren und simulieren,
- die Korrektheit über ein Exchange-Argument plausibilisieren,
- greedy Münzwechsel ausführen und seine Grenzen demonstrieren,
- den fraktionalen Rucksack nach Wertdichte optimal füllen,
- einen Huffman-Baum mit einem Min-Heap aufbauen,
- Texte mit präfixfreien Huffman-Codes en- und dekodieren,
- Greedy von Dynamic Programming abgrenzen,
- bei einem neuen Problem begründen, ob eine Greedy-Strategie tragfähig ist.

---

# Teil I — Das Greedy-Denkmodell

## 1. Vier Bestandteile

Fast jedes Greedy-Problem lässt sich mit vier Fragen beschreiben:

1. **Kandidaten:** Welche Elemente könnten gewählt werden?
2. **Zulässigkeit:** Welche Wahl verletzt keine Nebenbedingung?
3. **Lokale Priorität:** Welcher zulässige Kandidat sieht momentan am besten aus?
4. **Ziel:** Welche globale Größe soll optimiert werden?

Beispiel Interval Scheduling:

| Bestandteil | Bedeutung |
|---|---|
| Kandidaten | Zeitintervalle |
| zulässig | überschneidet sich nicht mit bereits gewählten Intervallen |
| lokale Priorität | kleinstes Enddatum |
| globales Ziel | maximale Anzahl kompatibler Intervalle |

Die lokale Priorität muss zum globalen Ziel passen. „Kürzeste Dauer“ klingt beim
Scheduling vernünftig, ist aber nicht immer korrekt.

## 2. Das typische Schema

```text
sort or prioritize the candidates
solution = empty

for candidate in greedy order:
    if candidate is feasible:
        add candidate permanently

return solution
```

Greedy verwendet häufig:

- Sortierung nach einem Schlüssel,
- einen Min- oder Max-Heap,
- eine Sweep-Line,
- Union-Find für schnelle Zulässigkeitsprüfungen,
- eine Menge oder einen Zähler für bereits verbrauchte Ressourcen.

## 3. Unwiderruflichkeit

Backtracking probiert eine Wahl und nimmt sie bei Bedarf zurück. Dynamic
Programming vergleicht mehrere Teilentscheidungen. Greedy sagt:

> Diese Wahl ist sicher. Ich muss nie wieder über sie nachdenken.

Genau diese Behauptung braucht eine Begründung.

## 4. Zwei strukturelle Eigenschaften

### Greedy-Choice-Property

Es existiert mindestens eine optimale Gesamtlösung, die mit der Greedy-Wahl
beginnt. Die lokale Wahl verbaut also nicht jede optimale Lösung.

### Optimale Teilstruktur

Nach der ersten sicheren Wahl bleibt ein kleineres Problem, dessen optimale
Lösung Teil einer optimalen Gesamtlösung ist.

Optimale Teilstruktur allein genügt nicht: Viele Dynamic-Programming-Probleme
besitzen sie, aber keine sichere Greedy-Wahl.

## 5. Laufzeit versus Korrektheit

Die Analyse besteht aus zwei unabhängigen Fragen:

- **Ist die Lösung optimal?** Beweis oder Gegenbeispiel.
- **Wie teuer ist der Algorithmus?** Meist Sortierung `O(n log n)` plus linearer
  Durchlauf.

Ein `O(n log n)`-Algorithmus kann sehr schnell eine falsche Lösung berechnen.

---

# Teil II — Wie beweist man Greedy?

## 6. Exchange-Argument

Das häufigste Muster:

1. Betrachte eine beliebige optimale Lösung `OPT`.
2. Falls `OPT` bereits mit der Greedy-Wahl beginnt, sind wir fertig.
3. Sonst tausche die erste Wahl von `OPT` gegen die Greedy-Wahl aus.
4. Zeige, dass die neue Lösung weiterhin zulässig und nicht schlechter ist.
5. Wiederhole das Argument auf dem Restproblem.

Der Austausch baut eine optimale Lösung schrittweise in eine Greedy-Lösung um,
ohne ihren Wert zu verschlechtern.

## 7. „Stays ahead“

Man zeigt nach jedem Schritt, dass Greedy mindestens so weit ist wie jede
Alternative. Beispiel: Nach `k` ausgewählten Intervallen endet der Greedy-Plan
nicht später als jeder andere kompatible Plan mit `k` Intervallen.

## 8. Cut- und Cycle-Argumente

Bei minimalen Spannbäumen ist die leichteste Kante über einen Schnitt sicher.
Das ist ebenfalls Greedy-Begründung. Kruskal und Prim aus Modul 13 sind bereits
Greedy-Algorithmen:

- Kruskal nimmt die billigste Kante, die keinen Zyklus erzeugt.
- Prim nimmt die billigste Kante aus dem aktuellen Baum heraus.

## 9. Gegenbeispiele sind vollwertige Beweise

Um „diese Strategie ist immer optimal“ zu widerlegen, genügt **eine** Eingabe,
auf der sie schlechter als eine zulässige Alternative ist.

Ein gutes Gegenbeispiel ist:

- klein,
- vollständig nachrechenbar,
- genau auf die Schwäche der Auswahlregel zugeschnitten,
- mit Greedy-Ergebnis und besserem Ergebnis dokumentiert.

## 10. Systematisch ein Gegenbeispiel bauen

Frage: Welche zukünftige Chance könnte die lokale Wahl blockieren?

1. Erzwinge, dass Greedy einen attraktiven Kandidaten wählt.
2. Lege dahinter zwei oder mehr gemeinsam bessere Kandidaten.
3. Sorge dafür, dass die erste Wahl diese Kombination ausschließt.
4. Vergleiche Zielfunktion, nicht Bauchgefühl.

Dieses Muster erscheint bei Intervallen, Münzen, 0/1-Rucksack und vielen
Scheduling-Problemen.

---

# Teil III — Interval Scheduling

## 11. Problem

Gegeben sind halb offene Intervalle `[start, end)`. Gesucht ist eine maximale
Teilmenge ohne Überlappung. Endet ein Intervall um 5 und beginnt ein anderes um
5, sind sie kompatibel.

```python
from dataclasses import dataclass
from itertools import combinations, count
from collections import Counter
import heapq

@dataclass(frozen=True)
class Interval:
    name: str
    start: int
    end: int

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError("An interval requires end > start")

    @property
    def duration(self):
        return self.end - self.start

intervals = (
    Interval("A", 1, 4),
    Interval("B", 3, 5),
    Interval("C", 0, 6),
    Interval("D", 5, 7),
    Interval("E", 3, 9),
    Interval("F", 5, 9),
    Interval("G", 6, 10),
    Interval("H", 8, 11),
    Interval("I", 8, 12),
    Interval("J", 2, 14),
    Interval("K", 12, 16),
)
```

## 12. Die richtige Greedy-Regel

> Wähle stets das kompatible Intervall mit dem frühesten Ende.

Warum nicht der früheste Start? Ein sehr früh beginnendes, langes Intervall kann
den ganzen Zeitbereich blockieren. Das früheste Ende lässt maximal viel Raum für
den Rest.

```python
def interval_scheduling(candidates):
    ordered = sorted(candidates, key=lambda interval: (interval.end, interval.start))
    selected = []
    current_end = float("-inf")
    trace = []

    for interval in ordered:
        if interval.start >= current_end:
            selected.append(interval)
            current_end = interval.end
            trace.append(("select", interval.name, current_end))
        else:
            trace.append(("skip", interval.name, current_end))
    return tuple(selected), tuple(trace)

schedule, schedule_trace = interval_scheduling(intervals)
assert tuple(interval.name for interval in schedule) == ("A", "D", "H", "K")
```

## 13. Simulation

Nach Endzeit sortiert:

```text
A(1,4), B(3,5), C(0,6), D(5,7), E(3,9), F(5,9),
G(6,10), H(8,11), I(8,12), J(2,14), K(12,16)
```

| Kandidat | kompatibel? | Entscheidung | aktuelles Ende |
|---|---|---|---:|
| A [1,4) | ja | nehmen | 4 |
| B [3,5) | nein | überspringen | 4 |
| C [0,6) | nein | überspringen | 4 |
| D [5,7) | ja | nehmen | 7 |
| E [3,9) | nein | überspringen | 7 |
| F [5,9) | nein | überspringen | 7 |
| G [6,10) | nein | überspringen | 7 |
| H [8,11) | ja | nehmen | 11 |
| I [8,12) | nein | überspringen | 11 |
| J [2,14) | nein | überspringen | 11 |
| K [12,16) | ja | nehmen | 16 |

Ergebnis: vier Intervalle.

## 14. Brute-Force-Referenz für kleine Eingaben

Eine Referenzimplementierung ist nützlich, um Greedy auf vielen kleinen Fällen
zu testen. Sie beweist die Strategie nicht allgemein, findet aber Gegenbeispiele.

```python
def intervals_are_compatible(candidate_schedule):
    ordered = sorted(candidate_schedule, key=lambda interval: interval.start)
    return all(
        first.end <= second.start
        for first, second in zip(ordered, ordered[1:])
    )

def optimal_interval_schedule_bruteforce(candidates):
    best = ()
    for size in range(len(candidates) + 1):
        for subset in combinations(candidates, size):
            if intervals_are_compatible(subset) and len(subset) > len(best):
                best = subset
    return best

optimal_schedule = optimal_interval_schedule_bruteforce(intervals)
assert len(schedule) == len(optimal_schedule) == 4
assert intervals_are_compatible(schedule)
```

## 15. Exchange-Argument für frühestes Ende

Sei `G` das Intervall mit dem frühesten Ende. Eine optimale Lösung beginnt mit
einem Intervall `O`.

- `end(G) <= end(O)` per Definition.
- Ersetze `O` durch `G`.
- Jedes spätere Intervall, das nach `O` starten konnte, kann auch nach dem nicht
  später endenden `G` starten.
- Die Anzahl gewählter Intervalle bleibt gleich.

Damit existiert eine optimale Lösung, die mit `G` beginnt. Nach `G` bleibt wieder
dasselbe Problem auf den danach startenden Intervallen. Induktion beendet den
Beweis.

## 16. Falsche Regel: frühester Start

```python
earliest_start_counterexample = (
    Interval("long", 0, 10),
    Interval("short-1", 1, 2),
    Interval("short-2", 2, 3),
    Interval("short-3", 3, 4),
)

def schedule_by_key(candidates, key):
    selected = []
    current_end = float("-inf")
    for interval in sorted(candidates, key=key):
        if interval.start >= current_end:
            selected.append(interval)
            current_end = interval.end
    return tuple(selected)

wrong_start = schedule_by_key(
    earliest_start_counterexample,
    key=lambda interval: interval.start,
)
right_start, _ = interval_scheduling(earliest_start_counterexample)
assert len(wrong_start) == 1
assert len(right_start) == 3
```

## 17. Falsche Regel: kürzeste Dauer

```python
shortest_duration_counterexample = (
    Interval("left", 0, 3),
    Interval("tempting", 2, 4),
    Interval("right", 3, 6),
)
wrong_duration = schedule_by_key(
    shortest_duration_counterexample,
    key=lambda interval: interval.duration,
)
right_duration, _ = interval_scheduling(shortest_duration_counterexample)
assert tuple(interval.name for interval in wrong_duration) == ("tempting",)
assert tuple(interval.name for interval in right_duration) == ("left", "right")
```

## 18. Gewichtete Intervalle

Wenn Intervalle unterschiedliche Werte besitzen und der **Gesamtwert** maximiert
werden soll, ist frühestes Ende nicht mehr allgemein optimal. Ein einzelnes
wertvolles Intervall kann besser sein als mehrere billige. Weighted Interval
Scheduling ist ein klassisches Dynamic-Programming-Problem.

## 19. Komplexität

- Sortierung: `O(n log n)`
- Auswahlscan: `O(n)`
- Zusatzspeicher: `O(n)` für sortierte Kopie und Ergebnis

Ist die Eingabe bereits nach Endzeit sortiert, genügt `O(n)`.

---

# Teil IV — Münzwechsel

## 20. Die naheliegende Greedy-Regel

Für einen Betrag wähle wiederholt die größte Münze, die noch hineinpasst.

```python
def greedy_change(coins, amount):
    if amount < 0:
        raise ValueError("Amount must be nonnegative")
    if any(coin <= 0 for coin in coins):
        raise ValueError("Coins must be positive")

    remaining = amount
    chosen = []
    for coin in sorted(set(coins), reverse=True):
        count_for_coin, remaining = divmod(remaining, coin)
        chosen.extend([coin] * count_for_coin)
    return tuple(chosen), remaining

euro_coins = (1, 2, 5, 10, 20, 50)
euro_change, euro_rest = greedy_change(euro_coins, 87)
assert euro_change == (50, 20, 10, 5, 2)
assert euro_rest == 0
```

Für gängige Euro-Nennwerte liefert diese Regel minimale Münzanzahlen. Das liegt
an der Struktur des Münzsystems, nicht am Greedy-Schema allein.

## 21. Gegenbeispiel `{1, 3, 4}` für Betrag 6

Greedy nimmt:

```text
4 + 1 + 1 = 6    three coins
```

Optimal ist:

```text
3 + 3 = 6        two coins
```

```python
def optimal_change_dp(coins, amount):
    if amount < 0:
        raise ValueError("Amount must be nonnegative")
    usable = tuple(sorted(set(coins), reverse=True))
    best = [None] * (amount + 1)
    best[0] = ()

    for subtotal in range(amount + 1):
        if best[subtotal] is None:
            continue
        for coin in usable:
            target = subtotal + coin
            if target <= amount:
                candidate = best[subtotal] + (coin,)
                if best[target] is None or len(candidate) < len(best[target]):
                    best[target] = candidate
    return best[amount]

bad_greedy, bad_rest = greedy_change((1, 3, 4), 6)
bad_optimal = optimal_change_dp((1, 3, 4), 6)
assert bad_rest == 0
assert bad_greedy == (4, 1, 1)
assert bad_optimal == (3, 3)
assert len(bad_greedy) > len(bad_optimal)
```

Das Gegenbeispiel widerlegt die Aussage „größte passende Münze ist immer sicher“.

## 22. Kanonische Münzsysteme

Ein Münzsystem heißt kanonisch, wenn Greedy für jeden darstellbaren Betrag eine
minimale Münzanzahl liefert. Das Testen vieler Beträge schafft Vertrauen, ist
aber ohne zusätzliche Schranke kein allgemeiner Beweis.

```python
def first_change_counterexample(coins, maximum):
    for amount in range(maximum + 1):
        greedy, rest = greedy_change(coins, amount)
        optimal = optimal_change_dp(coins, amount)
        if rest == 0 and optimal is not None and len(greedy) != len(optimal):
            return amount, greedy, optimal
    return None

assert first_change_counterexample(euro_coins, 200) is None
assert first_change_counterexample((1, 3, 4), 20) == (
    6,
    (4, 1, 1),
    (3, 3),
)
```

## 23. Woran scheitert Greedy?

Die Münze 4 sieht lokal am besten aus, hinterlässt aber Rest 2, der zwei
1er-Münzen benötigt. Greedy berücksichtigt den zukünftigen Restzustand nicht.
Dynamic Programming vergleicht dagegen die besten Lösungen aller kleineren
Beträge.

## 24. Wenn ein Betrag nicht darstellbar ist

Ohne Münze 1 kann ein Rest bleiben. Eine robuste API darf diesen Fall nicht als
vollständige Lösung ausgeben.

```python
incomplete_change, incomplete_rest = greedy_change((4, 6), 7)
assert incomplete_change == (6,)
assert incomplete_rest == 1
assert optimal_change_dp((4, 6), 7) is None
```

---

# Teil V — Fraktionaler Rucksack

## 25. Problem

Jedes Objekt besitzt Gewicht `w` und Wert `v`. Der Rucksack hat Kapazität `C`.
Objekte dürfen geteilt werden. Gesucht ist maximaler Gesamtwert.

Die entscheidende Kennzahl ist die Wertdichte:

```text
ratio = value / weight
```

## 26. Greedy nach Wertdichte

```python
@dataclass(frozen=True)
class Item:
    name: str
    weight: float
    value: float

    def __post_init__(self):
        if self.weight <= 0 or self.value < 0:
            raise ValueError("Weight must be positive and value nonnegative")

    @property
    def density(self):
        return self.value / self.weight

def fractional_knapsack(items, capacity):
    if capacity < 0:
        raise ValueError("Capacity must be nonnegative")
    remaining = capacity
    total_value = 0.0
    selection = []

    for item in sorted(items, key=lambda candidate: candidate.density, reverse=True):
        if remaining == 0:
            break
        fraction = min(1.0, remaining / item.weight)
        selection.append((item, fraction))
        total_value += fraction * item.value
        remaining -= fraction * item.weight
    return total_value, tuple(selection), remaining

knapsack_items = (
    Item("A", 10, 60),
    Item("B", 20, 100),
    Item("C", 30, 120),
)
fractional_value, fractional_selection, unused_capacity = fractional_knapsack(
    knapsack_items, 50
)
assert fractional_value == 240
assert tuple((item.name, fraction) for item, fraction in fractional_selection) == (
    ("A", 1.0),
    ("B", 1.0),
    ("C", 2 / 3),
)
assert unused_capacity == 0
```

## 27. Simulation

| Objekt | Gewicht | Wert | Dichte | Anteil | Wertbeitrag |
|---|---:|---:|---:|---:|---:|
| A | 10 | 60 | 6 | 100 % | 60 |
| B | 20 | 100 | 5 | 100 % | 100 |
| C | 30 | 120 | 4 | 66,7 % | 80 |
| **Summe** | **50** | | | | **240** |

## 28. Exchange-Argument

Angenommen, eine Lösung verwendet Gewicht von einem Objekt mit niedrigerer
Dichte, während ein Objekt mit höherer Dichte noch nicht vollständig verwendet
wird. Tausche eine kleine Gewichtsmenge `δ` aus:

- entfernt wird Wert `δ * niedrige_Dichte`,
- hinzu kommt Wert `δ * hohe_Dichte`,
- Gesamtgewicht bleibt gleich,
- Wert sinkt nicht und steigt bei strikt höherer Dichte.

Jede optimale Lösung kann so in Dichte-Reihenfolge umgebaut werden. Teilbarkeit
ist der Schlüssel.

## 29. Warum 0/1-Rucksack anders ist

Ohne Teilbarkeit kann das lokal beste Verhältnis eine global bessere Kombination
blockieren. Im Beispiel ist fraktional 240 möglich, aber ganzzahlig ist die beste
Kombination B+C mit Wert 220.

```python
def zero_one_knapsack_bruteforce(items, capacity):
    best_value = 0
    best_subset = ()
    for size in range(len(items) + 1):
        for subset in combinations(items, size):
            weight = sum(item.weight for item in subset)
            value = sum(item.value for item in subset)
            if weight <= capacity and value > best_value:
                best_value = value
                best_subset = subset
    return best_value, best_subset

integer_value, integer_selection = zero_one_knapsack_bruteforce(
    knapsack_items, 50
)
assert integer_value == 220
assert {item.name for item in integer_selection} == {"B", "C"}
assert integer_value < fractional_value
```

0/1-Rucksack benötigt im Allgemeinen Dynamic Programming, Branch-and-Bound oder
Approximationen; reines Verhältnis-Greedy garantiert kein Optimum.

## 30. Komplexität

- Dichtesortierung: `O(n log n)`
- Füllung: `O(n)`
- insgesamt `O(n log n)`

---

# Teil VI — Huffman-Codierung

## 31. Ziel

Huffman weist häufigen Symbolen kurze und seltenen Symbolen längere Bitcodes zu.
Die Codes sind präfixfrei: Kein gültiger Code ist Präfix eines anderen. Dadurch
kann ein Bitstrom eindeutig von links nach rechts dekodiert werden.

Das Ziel ist, die gewichtete Codelänge zu minimieren:

```text
sum over symbols: frequency(symbol) * code_length(symbol)
```

## 32. Die Greedy-Wahl

> Verbinde wiederholt die zwei Teilbäume mit den kleinsten Frequenzen.

Der neue Elternknoten erhält die Summe beider Frequenzen und kommt zurück in den
Min-Heap. Der letzte verbleibende Knoten ist die Baumwurzel.

## 33. Datenstruktur und Baumbau

```python
@dataclass(frozen=True)
class HuffmanNode:
    frequency: int
    symbol: str | None = None
    left: "HuffmanNode | None" = None
    right: "HuffmanNode | None" = None

    @property
    def is_leaf(self):
        return self.symbol is not None

def build_huffman_tree(frequencies):
    if not frequencies:
        raise ValueError("At least one symbol is required")
    if any(frequency <= 0 for frequency in frequencies.values()):
        raise ValueError("Frequencies must be positive")

    order = count()
    heap = [
        (frequency, next(order), HuffmanNode(frequency, symbol))
        for symbol, frequency in sorted(frequencies.items())
    ]
    heapq.heapify(heap)
    trace = []

    while len(heap) > 1:
        first_frequency, _, first = heapq.heappop(heap)
        second_frequency, _, second = heapq.heappop(heap)
        parent_frequency = first_frequency + second_frequency
        parent = HuffmanNode(parent_frequency, None, first, second)
        heapq.heappush(heap, (parent_frequency, next(order), parent))
        trace.append((first_frequency, second_frequency, parent_frequency))
    return heap[0][2], tuple(trace)

classic_frequencies = {
    "A": 5,
    "B": 9,
    "C": 12,
    "D": 13,
    "E": 16,
    "F": 45,
}
huffman_root, huffman_trace = build_huffman_tree(classic_frequencies)
assert huffman_root.frequency == 100
assert huffman_trace == (
    (5, 9, 14),
    (12, 13, 25),
    (14, 16, 30),
    (25, 30, 55),
    (45, 55, 100),
)
```

## 34. Codes aus dem Baum lesen

Linke Kante bedeutet `0`, rechte Kante `1`. Bei nur einem Symbol verwenden wir
`0`, damit auch Wiederholungen dieses Symbols einen expliziten Bitstrom besitzen.

```python
def huffman_codes(root):
    codes = {}

    def visit(node, prefix):
        if node.is_leaf:
            codes[node.symbol] = prefix or "0"
            return
        if node.left is None or node.right is None:
            raise ValueError("Invalid Huffman tree")
        visit(node.left, prefix + "0")
        visit(node.right, prefix + "1")

    visit(root, "")
    return codes

classic_codes = huffman_codes(huffman_root)
assert classic_codes == {
    "F": "0",
    "C": "100",
    "D": "101",
    "A": "1100",
    "B": "1101",
    "E": "111",
}
```

Bei gleichen Frequenzen können andere Tie-Breaker andere, aber gleich optimale
Codes erzeugen.

## 35. Präfixfreiheit prüfen

```python
def is_prefix_free(codes):
    values = tuple(codes.values())
    return all(
        first == second or not second.startswith(first)
        for first in values
        for second in values
    )

assert is_prefix_free(classic_codes)
```

## 36. En- und Dekodierung

```python
def huffman_encode(text, codes):
    try:
        return "".join(codes[symbol] for symbol in text)
    except KeyError as error:
        raise ValueError(f"Unknown symbol: {error.args[0]!r}") from error

def huffman_decode(bits, root):
    if root.is_leaf:
        if any(bit != "0" for bit in bits):
            raise ValueError("Invalid bit stream")
        return root.symbol * len(bits)

    decoded = []
    node = root
    for bit in bits:
        if bit not in "01":
            raise ValueError("Bit stream may contain only 0 and 1")
        node = node.left if bit == "0" else node.right
        if node is None:
            raise ValueError("Bit stream leaves the Huffman tree")
        if node.is_leaf:
            decoded.append(node.symbol)
            node = root
    if node is not root:
        raise ValueError("Bit stream ends inside a code")
    return "".join(decoded)

sample_text = "FACE"
encoded_sample = huffman_encode(sample_text, classic_codes)
assert huffman_decode(encoded_sample, huffman_root) == sample_text
```

## 37. Kompressionswirkung

Für sechs Symbole benötigt ein fixer Code mindestens drei Bits pro Symbol. Bei
100 Symbolvorkommen wären das 300 Bits. Huffman benötigt:

```python
huffman_bits = sum(
    frequency * len(classic_codes[symbol])
    for symbol, frequency in classic_frequencies.items()
)
fixed_width_bits = sum(classic_frequencies.values()) * 3
assert huffman_bits == 224
assert fixed_width_bits == 300
assert huffman_bits / fixed_width_bits < 0.75
```

Diese Rechnung ignoriert den Speicher für Baum oder Codebuch. Bei sehr kurzen
Texten kann der Header den Gewinn aufzehren.

## 38. Warum sind die zwei kleinsten Frequenzen Geschwister?

In einem optimalen Präfixbaum können die zwei seltensten Symbole auf maximaler
Tiefe als Geschwister angeordnet werden. Falls dort häufigere Symbole liegen,
kann man sie mit selteneren tauschen, ohne die gewichtete Länge zu erhöhen.

Fasst man diese zwei Geschwister zu einem Pseudosymbol mit Summenfrequenz
zusammen, bleibt ein kleineres Huffman-Problem. Das liefert Greedy-Choice-Property
und optimale Teilstruktur.

## 39. Komplexität

Bei `k` verschiedenen Symbolen und Textlänge `n`:

- Frequenzanalyse: `O(n)`
- `k-1` Heap-Merges: `O(k log k)`
- Codes erzeugen: `O(k)` plus gesamte Codelängen
- Text kodieren: `O(n)` Dictionary-Zugriffe
- Speicher: `O(k)` für Baum und Codebuch, zusätzlich Ausgabebits

---

# Teil VII — Meeting-Räume als weiteres Greedy-Muster

## 40. Minimale Anzahl gleichzeitig benötigter Räume

Sortiere Meetings nach Startzeit. Ein Min-Heap speichert Endzeiten belegter
Räume. Ist der früheste Raum rechtzeitig frei, wird er wiederverwendet;
andernfalls wird ein neuer geöffnet.

```python
def minimum_meeting_rooms(meetings):
    active_end_times = []
    maximum_rooms = 0
    trace = []

    for meeting in sorted(meetings, key=lambda interval: interval.start):
        while active_end_times and active_end_times[0] <= meeting.start:
            freed_at = heapq.heappop(active_end_times)
            trace.append(("release", freed_at, meeting.start))
        heapq.heappush(active_end_times, meeting.end)
        maximum_rooms = max(maximum_rooms, len(active_end_times))
        trace.append(("allocate", meeting.name, tuple(sorted(active_end_times))))
    return maximum_rooms, tuple(trace)

meeting_sample = (
    Interval("M1", 0, 30),
    Interval("M2", 5, 10),
    Interval("M3", 15, 20),
    Interval("M4", 20, 25),
)
room_count, room_trace = minimum_meeting_rooms(meeting_sample)
assert room_count == 2
```

Die maximale Heap-Größe entspricht der maximalen Überlappung und damit einer
unteren Schranke, die der Algorithmus exakt erreicht. Modulprojekt 03 baut daraus
konkrete Belegungspläne.

## 41. Auswahl versus Ressourcenbedarf

Nicht verwechseln:

- Interval Scheduling: maximale Zahl von Meetings in **einem** Raum auswählen.
- Meeting Rooms: **alle** Meetings durchführen und minimale Raumzahl bestimmen.

Ähnliche Intervalldaten, andere Zielfunktion, andere Greedy-Regel.

---

# Teil VIII — Greedy versus Dynamic Programming

## 42. Gemeinsamkeiten

Beide nutzen häufig optimale Teilstruktur. Beide können Zustände schrittweise
aufbauen. Der entscheidende Unterschied ist, wie viele Alternativen erhalten
bleiben.

## 43. Unterschiede

| Aspekt | Greedy | Dynamic Programming |
|---|---|---|
| Entscheidung | lokal beste, endgültig | mehrere Alternativen vergleichen |
| frühere Wahl ändern | nein | indirekt durch Zustandsvergleich |
| typische Zeit | oft `O(n log n)` | häufig mehrdimensional/pseudopolynomiell |
| Speicher | meist klein | Zustandstabelle oder Memoization |
| Beweislast | lokale Wahl muss sicher sein | Rekurrenz muss alle Fälle abdecken |
| Beispiel | fraktionaler Rucksack | 0/1-Rucksack |

## 44. Dasselbe Thema, andere Garantie

| Problem | Greedy möglich? | Begründung / Alternative |
|---|---|---|
| ungewichtetes Interval Scheduling | ja | frühestes Ende + Exchange |
| gewichtetes Interval Scheduling | allgemein nein | DP über Vorgängerintervalle |
| fraktionaler Rucksack | ja | Dichte + Austausch kleiner Mengen |
| 0/1-Rucksack | allgemein nein | DP / Branch-and-Bound |
| Münzwechsel, beliebige Münzen | allgemein nein | DP über Beträge |
| Huffman | ja | zwei kleinste Frequenzen zusammenfassen |
| kürzester Weg, nichtnegative Kanten | ja | Dijkstra |
| kürzester Weg, negative Kanten | Dijkstra nein | Bellman-Ford |

## 45. Warnsignal: Wahl beeinflusst schwer den Rest

Wenn eine lokale Wahl einen komplizierten Restzustand erzeugt und keine
Austauschbarkeit erkennbar ist, spricht das für DP oder Suche. Beim Münzwechsel
ist der verbleibende Betrag entscheidend; beim 0/1-Rucksack die verbleibende
Kapazität und Kombination ganzer Objekte.

---

# Teil IX — Greedy-Tauglichkeit erkennen

## 46. Positive Signale

- Es gibt eine klare lokale Rangfolge.
- Eine lokale Wahl lässt sich gegen die erste Wahl jeder optimalen Lösung
  austauschen.
- Nur die verbleibende Menge, nicht die detaillierte Historie, ist relevant.
- Die zulässigen Mengen besitzen starke Austauschbarkeit.
- Ein Schnitt-, Präfix- oder Dominanzargument liegt nahe.
- Bekannte Struktur: Intervalle, MST, Huffman, positive kürzeste Wege.

## 47. Negative Signale

- Eine Wahl erzeugt viele qualitativ verschiedene Restzustände.
- Teile sind unteilbar und Kombinationen zählen.
- Lokale Kennzahlen ignorieren einen kritischen zukünftigen Rest.
- Kleine Änderungen der Eingabe zerstören die Strategie.
- Mehrere plausible Regeln liefern unterschiedliche Ergebnisse, aber kein
  Exchange-Argument greift.
- Ein kleines Brute-Force-Programm findet schnell Gegenbeispiele.

## 48. Prüfprozess für eine neue Strategie

```text
1. Specify the objective and constraints.
2. State the local rule precisely.
3. Test small instances by hand.
4. Construct or search for counterexamples.
5. Justify the greedy-choice property.
6. Identify optimal substructure.
7. Formulate a proof: exchange, stays ahead, cut, or contradiction.
8. Only then build the optimized implementation.
```

## 49. Brute Force als Strategietester

Für kleine Eingaben kann man eine vermutete Greedy-Regel gegen eine exakte
Referenz prüfen. Das ersetzt keinen Beweis, ist aber ein hervorragender
Gegenbeispielgenerator.

```python
def verify_interval_greedy_on_instance(candidates):
    greedy, _ = interval_scheduling(candidates)
    optimal = optimal_interval_schedule_bruteforce(candidates)
    return len(greedy) == len(optimal)

assert verify_interval_greedy_on_instance(intervals)
assert verify_interval_greedy_on_instance(earliest_start_counterexample)
assert verify_interval_greedy_on_instance(shortest_duration_counterexample)
```

---

# Teil X — Typische Fallstricke

## 50. „Greedy ist schnell, also nehme ich Greedy“

Laufzeit ist keine Korrektheitsgarantie. Erst Struktur, dann Algorithmus.

## 51. Strategie unpräzise beschreiben

„Nimm das beste Element“ ist keine Regel. Beste nach welchem Schlüssel? Wie
werden Gleichstände behandelt? Wann ist ein Kandidat zulässig?

## 52. Ein paar erfolgreiche Beispiele als Beweis ansehen

Tests können Fehler zeigen, aber universelle Korrektheit nicht allein beweisen.

## 53. Zielfunktion verwechseln

Maximale Anzahl, maximaler Wert, minimale Räume und minimale Gesamtdauer sind
verschiedene Probleme – selbst bei identischen Eingabedaten.

## 54. Gleichstände ignorieren

Tie-Breaking kann reproduzierbare Ausgaben beeinflussen. Ein korrekter
Algorithmus sollte bei jedem zulässigen Tie-Breaker optimal bleiben oder die
Regel spezifizieren.

## 55. Eingabe mutieren

`list.sort()` verändert die Eingabe. Lern- und Bibliothekscode sollte häufig mit
`sorted()` eine Kopie erzeugen.

## 56. Fraktional und 0/1 verwechseln

Teilbarkeit ändert die mathematische Struktur vollständig.

## 57. Huffman-Codebuchkosten vergessen

Kompressionsrate muss Header, Baum oder Codebuch einbeziehen, besonders bei
kurzen Texten.

## 58. Ungültige Randfälle

- leere Kandidatenmenge,
- Intervalle mit `end <= start`,
- negative Kapazität,
- nichtpositive Münzen oder Gewichte,
- nicht darstellbarer Betrag,
- Text mit nur einem Symbol,
- unbekanntes Symbol beim Kodieren,
- unvollständiger Bitcode beim Dekodieren.

---

# Teil XI — Korrektheits- und Testdenken

## 59. Invarianten für Interval Scheduling

- Gewählte Intervalle sind paarweise kompatibel.
- Endzeiten der gewählten Intervalle steigen.
- Nach jeder Wahl wurden alle früher endenden Kandidaten berücksichtigt.
- Die Lösung ist maximal im Sinne der Strategie; Optimalität folgt zusätzlich
  aus dem Exchange-Argument.

```python
assert intervals_are_compatible(schedule)
assert [interval.end for interval in schedule] == sorted(
    interval.end for interval in schedule
)
assert len(schedule_trace) == len(intervals)
```

## 60. Invarianten für den fraktionalen Rucksack

- jeder Anteil liegt in `[0,1]`,
- Gesamtgewicht überschreitet die Kapazität nicht,
- positive Anteile erscheinen in nicht steigender Dichte,
- höchstens ein Objekt ist teilweise gewählt.

```python
fractions = [fraction for _, fraction in fractional_selection]
densities = [item.density for item, _ in fractional_selection]
used_weight = sum(
    item.weight * fraction for item, fraction in fractional_selection
)
assert all(0 <= fraction <= 1 for fraction in fractions)
assert densities == sorted(densities, reverse=True)
assert sum(0 < fraction < 1 for fraction in fractions) <= 1
assert used_weight <= 50
```

## 61. Invarianten für Huffman

- Wurzelfrequenz ist Summe aller Häufigkeiten.
- Jeder innere Knoten besitzt zwei Kinder.
- Codebuch enthält genau alle Symbole.
- Codes sind präfixfrei.
- `decode(encode(text)) == text`.
- gewichtete Bitlänge stimmt mit dem Codebuch überein.

```python
assert huffman_root.frequency == sum(classic_frequencies.values())
assert set(classic_codes) == set(classic_frequencies)
assert is_prefix_free(classic_codes)
assert huffman_decode(
    huffman_encode("ABCDEF", classic_codes), huffman_root
) == "ABCDEF"
```

## 62. Property-Tests

Starke automatische Prüfungen:

- zufällige kleine Intervallmengen gegen Brute Force,
- Greedy-Münzwechsel gegen DP und Gegenbeispiel protokollieren,
- fraktionalen Rucksack gegen lineare Optimierung auf kleinen Fällen,
- Huffman-Roundtrips für Seed-Texte,
- Präfixfreiheit jedes erzeugten Codebuchs,
- Meeting-Raum-Zahl gegen maximale Überlappung einer Sweep-Line.

---

# Teil XII — Data-Science-Transfer

## 63. Greedy im Datenalltag

- Budget schrittweise auf Maßnahmen mit höchstem Grenznutzen verteilen,
- nicht überlappende Trainings- oder Wartungsfenster wählen,
- Jobs nach Deadline oder Priorität planen,
- Cluster oder Graphkomponenten über billige Kanten verbinden,
- Speicher durch Präfixcodes reduzieren,
- Features unter Kostenbeschränkungen approximativ auswählen,
- Datenpipeline-Ressourcen mit Heaps zuordnen.

## 64. Vorsicht bei Heuristiken

In Data Science nennt man eine lokale Strategie oft „Greedy Feature Selection“
oder „Greedy Search“. Das kann praktisch nützlich sein, ohne globale
Optimalitätsgarantie. Dann sauber unterscheiden:

- **Algorithmus mit Beweis:** garantiert optimal unter Voraussetzungen.
- **Heuristik:** plausible, oft schnelle Lösung ohne universelle Garantie.

Eine Heuristik ist nicht schlecht – nur ihre Aussage muss ehrlich sein.

## 65. Messbare Qualität

Wenn Optimalität nicht beweisbar oder exakte Lösung zu teuer ist:

- kleine Fälle exakt lösen und Qualitätslücke messen,
- mehrere Auswahlregeln vergleichen,
- Approximation Ratio untersuchen,
- Sensitivität gegenüber Tie-Breaking und Seeds prüfen,
- Laufzeit und Lösungsqualität getrennt berichten.

---

# Teil XIII — Kontrollfragen

## 66. Fragen

1. Was macht eine Entscheidung „greedy“?
2. Warum genügt optimale Teilstruktur allein nicht?
3. Was zeigt ein Exchange-Argument?
4. Welche Regel löst ungewichtetes Interval Scheduling optimal?
5. Warum ist frühester Start nicht korrekt?
6. Was widerlegt das Münzsystem `{1,3,4}` bei Betrag 6?
7. Warum funktioniert Wertdichte beim fraktionalen Rucksack?
8. Warum scheitert dieselbe Regel beim 0/1-Rucksack?
9. Welche zwei Elemente verbindet Huffman jeweils?
10. Warum ist Huffmans Code präfixfrei?
11. Welche Rolle spielt der Min-Heap?
12. Wie unterscheidet sich Meeting-Room-Zählung von Interval Scheduling?
13. Welche Warnsignale sprechen für Dynamic Programming?
14. Kann Testen einen Greedy-Beweis ersetzen?
15. Wann darf eine Greedy-Heuristik trotzdem sinnvoll sein?

## 67. Kurzantworten

1. Sie wählt lokal optimal, bleibt zulässig und wird nicht zurückgenommen.
2. Sie sagt nicht, dass eine einzelne lokale Wahl sicher ist.
3. Eine optimale Lösung lässt sich ohne Qualitätsverlust an die Greedy-Wahl
   anpassen.
4. Nach frühestem Ende sortieren und kompatible Intervalle nehmen.
5. Ein früh startendes langes Intervall kann viele kurze blockieren.
6. Größte passende Münze minimiert bei beliebigen Systemen nicht immer die Anzahl.
7. Kleine Gewichtsmengen lassen sich gegen höhere Dichte tauschen.
8. Ganze Objekte verhindern diesen beliebig kleinen Austausch.
9. Die zwei Teilbäume mit kleinsten Frequenzen.
10. Nur Blätter tragen Symbole; kein Blatt liegt über einem anderen Blatt.
11. Er liefert wiederholt die zwei kleinsten Frequenzen in `O(log k)`.
12. Das eine wählt möglichst viele für einen Raum, das andere plant alle mit
    minimal vielen Räumen.
13. Komplexer Restzustand, unteilbare Kombinationen, fehlendes Exchange-Argument.
14. Nein; es kann Gegenbeispiele finden und Vertrauen erhöhen.
15. Wenn exakte Optimierung zu teuer ist und Qualität empirisch bewertet wird.

---

# Teil XIV — Kompakte Gesamtübersicht

## 68. Spickzettel

| Problem | Greedy-Schlüssel | Struktur | Garantie |
|---|---|---|---|
| Interval Scheduling | frühestes Ende | Exchange | optimal |
| beliebiger Münzwechsel | größte Münze | systemabhängig | nicht allgemein |
| fraktionaler Rucksack | höchste Wertdichte | Teilbarkeit + Exchange | optimal |
| 0/1-Rucksack | höchste Wertdichte | unteilbar | nicht allgemein |
| Huffman | zwei kleinste Frequenzen | Präfixbaum + Induktion | optimal |
| Meeting Rooms | früheste Endzeit im Heap | maximale Überlappung | optimal |
| Kruskal | leichteste sichere Kante | Cut + Union-Find | optimal |
| Dijkstra | kleinste offene Distanz | nichtnegative Kanten | optimal |

## 69. Ein Satz pro Verfahren

```text
Interval scheduling: Finish as early as possible and preserve future space.
Coin change: The largest coin is safe only in suitable coin systems.
Fractional knapsack: Buy value per unit of weight in descending order.
Huffman: Give rare symbols the greatest depth.
Meeting rooms: Always reuse the room that becomes free first.
```

## 70. Ausblick auf die Projekte

- **01-basic:** Interval Scheduling, fraktionaler Rucksack und Münzwechsel
  implementieren; Greedy-Scheitern explizit gegen eine exakte Referenz zeigen.
- **02-medium:** vollständigen Huffman-Kompressor für Textdateien bauen und
  Kompressionsraten einschließlich Metadaten messen.
- **03-final:** Seed-Kalender planen, minimale Raumzahl bestimmen, konkrete
  Belegungspläne erzeugen und gegen eine naive Zuordnung vergleichen.

Die zentrale Greedy-Kompetenz ist nicht das Sortieren. Sie ist die Fähigkeit,
eine lokale Regel entweder durch Struktur zu rechtfertigen oder durch ein kleines
Gegenbeispiel ehrlich zu verwerfen.
