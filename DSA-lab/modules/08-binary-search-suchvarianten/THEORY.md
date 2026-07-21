# Module 08 — Binary Search & Search Variants

Binary Search is small enough to fit into a few lines of code, and at the same time rich
enough to train central algorithmic skills: define a search space precisely, get an
invariant, master edge cases, and translate an optimization problem into a monotonic
yes-no question. Precisely the short implementation is deceptive. A wrongly selected
interval or an inconsistent limit creates errors that remain invisible in ordinary
input.

This script builds the theme in three levels:

1. **Intuition:** Why halving works and when pre-sorting is worthwhile.
2. **Simulation:** How intervals change in the exact search, in limits, in rotated
   arrays and in the search on the answer.
3. **Formalization:** Which invariants the variants carry, why they terminate correctly
   and what complexity actually arises.

In the end, a single code template should not be memorized. It is crucial to be able to
choose a consistent search interval, a monotonic decision function and a suitable return
meaning for a new problem.

---

# Level I — Intuition

## 1. The phone book and the halving idea

A printed telephone book is sorted by name. If you're looking for "Nguyen", you don't
start on page one and read every name. You open up about the middle:

- If there is "cramer", the left half can be excluded.
- If there is "Schulz", the right half can be excluded.
- The same decision is then repeated on the remaining half.

The sorting transforms a comparison into information about an entire area. In an
unsorted list, `middle_value < target` doesn't say anything about which side the target
is on. Binary Search is therefore not a faster replacement for linear search on any
data. The orderly structure is its decisive precondition.

After each unsuccessful comparison, at most half of the previous search space remains.
1,024 candidates shall not exceed:

```text
1024 → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
```

It's only 10 halvings. Generally, there are only `n / 2^k` candidates left after `k`
steps. As soon as this expression is at most one, `k ≈ log₂(n)` applies.

## 2. What are they looking for?

"Binary Search" can refer to three different intentions:

1. **Exact search:** Does a value occur? If so, which index?
2. **Border search:** Where does an area of equal or appropriate values begin or end?
3. **Answer search:** What is the smallest or largest value of a monotonic condition?

The third form is the most important generalization. The search space doesn't have to be
an array. It may consist of capacity, time budgets, thresholds or other ordered
candidates. Binary Search then searches for the transition between "not possible" and
"possible".

## 3. When is Sorting plus Searching worth it?

A linear search costs `O(n)` per request. Unique sorting costs `O(n log n)`, then every
binary search costs `O(log n)`. For `q` searches, the following results are rough:

```text
linear:             O(q · n)
sort + search: O(n log n + q log n)
```

With a single request, the linear search usually wins because the sorting costs are not
amortized. For many requests on a largely static data stock, the sorted representation
can be worthwhile. If only exact membership counts, a hash map or a set is often even
more appropriate. The sorting also offers area queries, predecessors, successors and
ordered output.

A small experiment shows the difference between the comparative figures. It does not
measure the entire real runtime, but isolates the algorithmic idea.

```python
def linear_search_with_count(values, target):
    """Return the target index and number of equality comparisons."""
    for index, value in enumerate(values):
        if value == target:
            return index, index + 1
    return -1, len(values)


def binary_search_with_count(values, target):
    """Return the target index and number of midpoint comparisons."""
    left, right = 0, len(values) - 1
    comparisons = 0
    while left <= right:
        middle = left + (right - left) // 2
        comparisons += 1
        if values[middle] == target:
            return middle, comparisons
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    return -1, comparisons


sample = list(range(1_000_000))
assert linear_search_with_count(sample, 999_999) == (999_999, 1_000_000)
index, comparisons = binary_search_with_count(sample, 999_999)
assert index == 999_999
assert comparisons <= 20
print(f"Linear: 1,000,000 comparisons; binary: {comparisons}")
```

---

# Level II — Simulation

## 4. Exact search step by step

Search for `23` in this sorted array:

```text
Index: 0   1   2   3   4   5   6   7   8
Value:   3   7  11  15  18  23  29  31  42
```

We use a **closed interval** `[left, right]`. Both borders belong to the search area.

| Step | `left` | `right` | `middle` | Value | Decision |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 8 | 4 | 18 | `18 < 23`, i.e. `left = 5` |
| 2 | 5 | 8 | 6 | 29 | `29 > 23`, i.e. `right = 5` |
| 3 | 5 | 5 | 5 | 23 | found |

Before each comparison, the search space contains all the indices that may still be
`23`. After the comparison in step 1, the indices `0` to `4` can be excluded including.
`middle` must not remain in the next interval: its value has already been checked.
Therefore, the new limit is `middle + 1` and not `middle`.

```python
def binary_search(values, target):
    """Return an index of target in sorted values, or -1 if absent."""
    left, right = 0, len(values) - 1
    while left <= right:
        middle = left + (right - left) // 2
        if values[middle] == target:
            return middle
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    return -1


values = [3, 7, 11, 15, 18, 23, 29, 31, 42]
assert binary_search(values, 23) == 5
assert binary_search(values, 3) == 0
assert binary_search(values, 42) == 8
assert binary_search(values, 16) == -1
assert binary_search([], 10) == -1
```

### 4.1 Unsuccessful Search

Search for now `16` in the same array:

| Step | Interval | Center/value | new interval |
|---:|---|---|---|
| 1 | `[0, 8]` | `4 / 18` | `[0, 3]` |
| 2 | `[0, 3]` | `1 / 7` | `[2, 3]` |
| 3 | `[2, 3]` | `2 / 11` | `[3, 3]` |
| 4 | `[3, 3]` | `3 / 15` | `[4, 3]` |

The interval `[4, 3]` is empty because `left > right`. This is exactly why the equal
sign in `while left <= right` belongs to the closed interval. A one-element interval
like `[3, 3]` is still a valid search space and must be checked.

## 5. Make the track visible

A trace function helps to check the interval semantics instead of just the result.

```python
def binary_search_trace(values, target):
    """Return visited closed intervals and midpoint decisions."""
    left, right = 0, len(values) - 1
    trace = []
    while left <= right:
        middle = left + (right - left) // 2
        relation = "equal"
        if values[middle] < target:
            relation = "too_small"
        elif values[middle] > target:
            relation = "too_large"
        trace.append((left, right, middle, values[middle], relation))
        if relation == "equal":
            break
        if relation == "too_small":
            left = middle + 1
        else:
            right = middle - 1
    return trace


trace = binary_search_trace([3, 7, 11, 15, 18, 23, 29, 31, 42], 23)
assert trace == [
    (0, 8, 4, 18, "too_small"),
    (5, 8, 6, 29, "too_large"),
    (5, 5, 5, 23, "equal"),
]
```

## 6. Not just any hit, but the first position

For duplicates, the exact search may provide any appropriate index. For area requests,
however, the **first position with `value >= target`** is often searched. This position
is called Lower Bound or Insert Position.

For `[2, 4, 4, 4, 7, 9]` and target `4` the half-open interval `[left, right)` is used:

| Step | Interval | `middle` | Value | Decision |
|---:|---|---:|---:|---|
| 1 | `[0, 6)` | 3 | 4 | Candidate; search right including center: `[0, 3)` |
| 2 | `[0, 3)` | 1 | 4 | Candidate; `[0, 1)` |
| 3 | `[0, 1)` | 0 | 2 | too small; `[1, 1)` |

At the end is `left == right == 1`. Unlike the exact search, equality does not end. The
found position could still have an equivalent predecessor.

```python
def lower_bound(values, target):
    """Return the first index whose value is greater than or equal to target."""
    left, right = 0, len(values)
    while left < right:
        middle = left + (right - left) // 2
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle
    return left


values = [2, 4, 4, 4, 7, 9]
assert lower_bound(values, 4) == 1
assert lower_bound(values, 5) == 4
assert lower_bound(values, 1) == 0
assert lower_bound(values, 10) == len(values)
assert lower_bound([], 4) == 0
```

The return is always a valid insertion position from `0` to `len(values)`. She's not
automatically a hit. For an exact search via the Lower Bound, a separate test follows.

```python
def find_via_lower_bound(values, target):
    """Return the first target index, or -1 if target is absent."""
    index = lower_bound(values, target)
    if index < len(values) and values[index] == target:
        return index
    return -1


assert find_via_lower_bound([2, 4, 4, 4, 7, 9], 4) == 1
assert find_via_lower_bound([2, 4, 4, 4, 7, 9], 5) == -1
```

## 7. Upper Bound and last occurrence

The Upper Bound is the first position with `value > target`. The only algorithmic
difference to the Lower Bound lies in equality treatment.

```python
def upper_bound(values, target):
    """Return the first index whose value is strictly greater than target."""
    left, right = 0, len(values)
    while left < right:
        middle = left + (right - left) // 2
        if values[middle] <= target:
            left = middle + 1
        else:
            right = middle
    return left


def equal_range(values, target):
    """Return the half-open range containing all occurrences of target."""
    return lower_bound(values, target), upper_bound(values, target)


def last_occurrence(values, target):
    """Return the last target index, or -1 if target is absent."""
    index = upper_bound(values, target) - 1
    if index >= 0 and values[index] == target:
        return index
    return -1


values = [2, 4, 4, 4, 7, 9]
assert upper_bound(values, 4) == 4
assert equal_range(values, 4) == (1, 4)
assert last_occurrence(values, 4) == 3
assert last_occurrence(values, 5) == -1
assert equal_range(values, 5) == (4, 4)
```

The number of occurrences is obtained without linear passage of the block as
`upper_bound - lower_bound`.

## 8. Pythons `bisect` module

The standard library provides the two border searches:

- `bisect_left(values, target)` is equal to `lower_bound`.
- `bisect_right(values, target)` is equal to `upper_bound`.
- Add `insort_left` and `insort_right` at the respective limit.

Searching for the insertion position costs `O(log n)`. Inserting into a Python list
still costs `O(n)` because the following elements have to be moved.

```python
from bisect import bisect_left, bisect_right, insort_left


values = [2, 4, 4, 4, 7, 9]
assert bisect_left(values, 4) == lower_bound(values, 4) == 1
assert bisect_right(values, 4) == upper_bound(values, 4) == 4

insort_left(values, 5)
assert values == [2, 4, 4, 4, 5, 7, 9]
```

Since Python 3.10, the search functions accept a `key` argument. Important is the
asymmetrical meaning: `key` is applied to list elements, not to the search value passed
separately. The search value must already be in the key room.

```python
from bisect import bisect_left


experiments = [
    {"learning_rate": 0.001, "run": "a"},
    {"learning_rate": 0.01, "run": "b"},
    {"learning_rate": 0.1, "run": "c"},
]
position = bisect_left(experiments, 0.02, key=lambda row: row["learning_rate"])
assert position == 2
```

## 9. Binary Search on the Answer

Suppose packages must be sent in their given order within `D` days. Search for not an
element of an array, but the smallest daily capacity. For each candidate capacity, the
question can be answered:

> Is this capacity sufficient to send all packets in a maximum of `D` days?

The answers are monotonic:

```text
Capacity: 3 4 5 6 7 8 9 10 11...
Doable:    N N N N J J J   J   J...
```

Search for the first `J`. The lower limit is the heaviest single package; the upper
limit is the sum of all weights, i.e. a capacity for a single day.

For weights `[3, 2, 2, 4, 1, 4]` and three days:

| Candidate | days needed | Is it possible? | Next area |
|---:|---:|---|---|
| 10 | 2 | Yes | Smaller half |
| 7 | 3 | Yes | Smaller half |
| 5 | 4 | yes | larger half |
| 6 | 3 | Yes | Smaller half |

The smallest feasible capacity is `6`.

```python
def required_days(weights, capacity):
    """Return days needed when items must retain their original order."""
    days = 1
    current_load = 0
    for weight in weights:
        if weight > capacity:
            return float("inf")
        if current_load + weight > capacity:
            days += 1
            current_load = 0
        current_load += weight
    return days


def minimum_shipping_capacity(weights, day_limit):
    """Return the smallest capacity that ships all weights within day_limit."""
    if not weights or day_limit < 1:
        raise ValueError("weights must be non-empty and day_limit positive")
    left, right = max(weights), sum(weights)
    while left < right:
        middle = left + (right - left) // 2
        if required_days(weights, middle) <= day_limit:
            right = middle
        else:
            left = middle + 1
    return left


weights = [3, 2, 2, 4, 1, 4]
assert [required_days(weights, capacity) for capacity in (5, 6, 7, 10)] == [
    4,
    3,
    3,
    2,
]
assert minimum_shipping_capacity(weights, 3) == 6
assert minimum_shipping_capacity(weights, 1) == sum(weights)
```

### 9.1 A reusable "first true" template

Many integer responses can be reduced to the same border search. The precondition is: In
the inclusive area `[low, high]` there is at least one true candidate, and after the
first true candidate all others also become true.

```python
def first_true(low, high, predicate):
    """Return the first integer in [low, high] satisfying a monotone predicate."""
    if low > high or not predicate(high):
        raise ValueError("the search range must contain a feasible candidate")
    while low < high:
        middle = low + (high - low) // 2
        if predicate(middle):
            high = middle
        else:
            low = middle + 1
    return low


assert first_true(0, 100, lambda value: value * value >= 30) == 6
assert first_true(7, 7, lambda value: value >= 7) == 7
```

The integer square root is another example. Search for the largest value with `root² <=
number`. This can be formulated either with a mirrored "last true" template or over the
first too large value.

```python
def integer_square_root(number):
    """Return floor(sqrt(number)) without calling math.sqrt."""
    if number < 0:
        raise ValueError("number must not be negative")
    left, right = 0, number + 1
    while left + 1 < right:
        middle = left + (right - left) // 2
        if middle * middle <= number:
            left = middle
        else:
            right = middle
    return left


assert [integer_square_root(value) for value in range(10)] == [
    0,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    3,
]
assert integer_square_root(10**12) == 10**6
```

### 9.2 Real Search Spaces

On floating point numbers cannot be searched meaningfully up to `left == right`.
Instead, the loop ends after a fixed number of iterations or as soon as the interval is
less than a desired tolerance. A fixed number of iterations is often easier to analyze:
each round halves the error range.

```python
def approximate_square_root(number, iterations=80):
    """Approximate a non-negative square root by bisection."""
    if number < 0:
        raise ValueError("number must not be negative")
    left, right = 0.0, max(1.0, float(number))
    for _ in range(iterations):
        middle = (left + right) / 2.0
        if middle * middle < number:
            left = middle
        else:
            right = middle
    return (left + right) / 2.0


root = approximate_square_root(2)
assert abs(root * root - 2) < 1e-12
assert approximate_square_root(0) < 1e-12
```

## 10. Search in the rotated sorted array

An ascending array may have been rotated in one place:

```text
original: [1, 3, 5, 7, 9, 11, 13]
Rotated:      [9, 11, 13, 1, 3, 5, 7]
```

The entire array is no longer sorted, but for different values at least one half is
sorted in each round. If the target is within the value range of this half, it will
continue to be searched for; otherwise, in the other half.

Wanted `3`:

| Interval | Center/value | sorted half | Decision |
|---|---|---|---|
| `[0, 6]` | `3 / 1` | right `[1, 3, 5, 7]` | `3` is on the right |
| `[4, 6]` | `5 / 5` | left `[3, 5]` | `3` is left |
| `[4, 4]` | `4 / 3` | — | found |

```python
def search_rotated(values, target):
    """Search a rotated sorted sequence containing distinct values."""
    left, right = 0, len(values) - 1
    while left <= right:
        middle = left + (right - left) // 2
        if values[middle] == target:
            return middle

        if values[left] <= values[middle]:
            if values[left] <= target < values[middle]:
                right = middle - 1
            else:
                left = middle + 1
        else:
            if values[middle] < target <= values[right]:
                left = middle + 1
            else:
                right = middle - 1
    return -1


rotated = [9, 11, 13, 1, 3, 5, 7]
assert search_rotated(rotated, 3) == 4
assert search_rotated(rotated, 11) == 1
assert search_rotated(rotated, 8) == -1
assert search_rotated([], 8) == -1
```

For many duplicates, the statement "at least one half is clearly sorted" can be
unusable. If `values[left]`, `values[middle]` and `values[right]` are the same, it is
not visible on which side the rotation is located. The boundaries can then be carefully
shifted by one. As a result, the worst case worsens to `O(n)`.

```python
def search_rotated_with_duplicates(values, target):
    """Return whether target occurs in a rotated sorted sequence with duplicates."""
    left, right = 0, len(values) - 1
    while left <= right:
        middle = left + (right - left) // 2
        if values[middle] == target:
            return True
        if values[left] == values[middle] == values[right]:
            left += 1
            right -= 1
        elif values[left] <= values[middle]:
            if values[left] <= target < values[middle]:
                right = middle - 1
            else:
                left = middle + 1
        else:
            if values[middle] < target <= values[right]:
                left = middle + 1
            else:
                right = middle - 1
    return False


assert search_rotated_with_duplicates([2, 2, 2, 3, 4, 2], 3)
assert not search_rotated_with_duplicates([2, 2, 2, 3, 4, 2], 5)
```

## 11. Exponential search at unknown range

Binary Search usually needs a right border. With a sorted current, an API with index-
based access or a very large list, the interesting position may be close to the
beginning and the meaningful range is unknown. Exponential Search first finds an
interval:

```text
Check index: 1, 2, 4, 8, 16, 32,...
```

As soon as the value at the limit is at least equal to the target, a normal binary
search will follow in the last skipped area. If the target is at position `p`, both the
border search and the subsequent search costs `O(log p)`.

```python
def binary_search_between(values, target, left, right):
    """Search target in the inclusive index range [left, right]."""
    while left <= right:
        middle = left + (right - left) // 2
        if values[middle] == target:
            return middle
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    return -1


def exponential_search(values, target):
    """Search sorted values by first discovering an exponential range."""
    if not values:
        return -1
    if values[0] == target:
        return 0
    bound = 1
    while bound < len(values) and values[bound] < target:
        bound *= 2
    return binary_search_between(
        values,
        target,
        bound // 2,
        min(bound, len(values) - 1),
    )


values = list(range(0, 10_000, 3))
assert exponential_search(values, 0) == 0
assert exponential_search(values, 27) == 9
assert exponential_search(values, 9_999) == len(values) - 1
assert exponential_search(values, 10_000) == -1
```

## 12. Outlook: hyperparameters and thresholds

Binary Search is only correct if the checked condition is monotonic along the search
space. This is not automatically the case with hyperparameters. The validation Accuracy
of a model can only rise with growing regularization and fall later. On such an unknown,
intoxicated curve, Binary Search is not a general optimizer.

On the other hand, the idea is suitable for monotonic secondary conditions. In the case
of a classification threshold:

- A higher positive threshold predicts less positive examples.
- The Recall curve is therefore not increasing; if defined cleanly, it falls monotonicly
  or remains the same.
- Precision can often rise, but must not be strictly monotonic on finite data with
  bindings.

Another example is resource-restricted tuning: "Is a model with at most this memory
budget executable?" can be monotonic, even if the model quality to be maximized is not.
The principle behind the search for answers therefore remains useful, but monotony must
be professionally justified and tested on the basis of edge cases. Project `03-final`
applies this caution to a simulated score record.

---

# Level III — Formalisation

## 13. Search intervals are contracts

Most off-by-one errors are not caused by the center formula, but by the mixing of two
interval contracts.

### 13.1 Closed interval `[left, right]`

- Initialization: `left = 0`, `right = n - 1`
- Not empty: `left <= right`
- Size: `right - left + 1`
- Discard center: `left = middle + 1` or `right = middle - 1`
- Typical return in case of failure: `-1`

This shape fits well with the exact search, because every possible index is contained in
the interval.

### 13.2 Half-open interval `[left, right)`

- Initialization: `left = 0`, `right = n`
- Not empty: `left < right`
- Size: `right - left`
- Keep left half including center: `right = middle`
- Discard center: `left = middle + 1`
- Typical return: Insert position `left` in the area `0..n`

This shape fits particularly well with Lower Bound and Upper Bound. The empty interval
is clearly represented by `left == right`.

The following edge case test makes the differences visible:

```python
edge_cases = [
    ([], 5),
    ([5], 5),
    ([5], 4),
    ([5], 6),
    ([1, 5], 1),
    ([1, 5], 5),
    ([1, 5], 3),
]

for edge_values, edge_target in edge_cases:
    exact = binary_search(edge_values, edge_target)
    insertion = lower_bound(edge_values, edge_target)
    assert exact == -1 or edge_values[exact] == edge_target
    assert all(value < edge_target for value in edge_values[:insertion])
    assert all(value >= edge_target for value in edge_values[insertion:])
```

## 14. The invariant of the exact search

For the closed exact search a suitable invariant is:

> If `target` occurs in the array and has not yet been found, at least
> an occurrence in the current interval `[left, right]`.

### Initialization

Before the first round is the interval `[0, n - 1]`. It contains all valid indices. The
statement therefore applies.

### Maintenance

Be `middle` the checked index.

- If `values[middle] < target`, neither `middle` nor smaller indices can contain the
  target due to sorting. With `left = middle + 1` all possible hits are preserved.
- If `values[middle] > target`, `middle` and all larger indices can be excluded. The
  invariant is retained with `right = middle - 1`.
- In equality, a correct match is found.

### Termination

Each unsuccessful round removes `middle` and at least one more half from the search
area. The non-negative interval size `right - left + 1` is strictly reduced. The loop is
scheduled. If `left > right` then the interval is empty. After the invariant, the target
cannot appear in the array.

## 15. The Invariant of the Lower Bound

For `[left, right)` you can understand the border search over already classified areas:

- All indices `< left` contain values `< target`.
- All indices `>= right` contain values `>= target`.
- Only `[left, right)` is not yet classified.

If `values[middle] < target`, the center belongs safely to the left area, i.e. `left =
middle + 1`. Otherwise, the center belongs to the right area; it could be the first
index you are looking for and will be kept with `right = middle`. With `left == right`
nothing is unclassified. Both statements come together at the same limit, so `left` is
the first position with a value of at least `target`.

This invariant explains why the border search does not return early on equality.

## 16. Centre and progress

The mathematically obvious formula is `(left + right) // 2`. Python integers do not run
over, but in languages with fixed integer width, the sum can overflow. This form avoids
the problem:

```text
middle = left + (right - left) // 2
```

Even more important is guaranteed progress. If the interval contains two elements and
rounds down `middle`, an update must not restore the same state. At a closed interval,
for example, `left = middle` easily leads to an endless loop. Consistent templates use
`middle + 1`, `middle - 1` or change the other side to `middle` with half-open borders.

## 17. Comparing Complexities

| Procedure | Prerequisite | Time | Add-on memory | Special feature |
|---|---|---:|---:|---|
| Linear search | No order | `O(n)` | `O(1)` | a request often sufficient |
| Binary Search | Sorted Random Access | `O(log n)` | `O(1)` iterative | Exact hit |
| Lower/Upper Bound | Sorted Random Access | `O(log n)` | `O(1)` | Boundaries and areas |
| `bisect` Search | sorted python sequence | `O(log n)` | `O(1)` | Standard Library |
| `insort` in list | Sorted Python List | `O(n)` | `O(1)` additional | Search fast, move linear |
| Exponential Search | sorted, range unclear | `O(log p)` | `O(1)` | `p` is target position |
| Rotated search, unique | Rotated Sorted | `O(log n)` | `O(1)` | one half is ordered |
| Rotated search, duplicates | Rotated Sorted | Worst Case `O(n)` | `O(1)` | Order can be ambiguous |
| Searching for answers | Monotone predicate | `O(log R · C)` | Problem-dependent | `R` Search room, `C` Test routine |

Binary Search requires efficient index-based access. On a linked list, reaching the
center will cost `O(n)`, thus losing the advantage. For data on slow external media,
B-trees or block-oriented indices can be more appropriate because they minimize the
number of expensive accesses.

## 18. Binary Search on the answer formal

Be `P(x)` a predicate on an ordered integer range. For the search for the smallest true
value we need:

```text
x < y and P(x) = true  ⇒  P(y) = true
```

So the truth values have the form

```text
wrong, wrong,..., wrong, true, true,..., true
```

The invariant of the `first_true` template is:

> The interval `[low, high]` contains the first true candidate, and `high` is
> A true candidate.

If `P(middle)` is true, `middle` may already be the limit; therefore it is retained with
`high = middle`. If it is wrong, the first true candidate and all other options are
strictly right; therefore follow `low = middle + 1`.

The total cost is not just `O(log R)`. If a feasibility check requires `C` time, `O(C
log R)` is obtained. In the shipping example, the test runs over all `n` packages, so
the solution costs `O(n log(sum(weights)))`.

## 19. Classic faults and antidote

### Error 1: Unsorted input

A correct looking algorithm provides incorrect results on unsorted data. The order is a
precondition, not an optimization.

### Error 2: Mixed interval models

`right = len(values)` belongs to the semi-open model. `while left <= right` belongs to
the closed model. The combination can access outside the list.

### Error 3: No guarantee of progress

An update to `left = middle` can get stuck at the center rounded down. After each round,
the draw has to shrink strictly.

### Error 4: Interpret insert position as hit

`lower_bound` also provides a valid position for a missing target. An exact hit needs
`index < n and values[index] == target` in addition.

### Error 5: Only suspect monotony

When searching for answers, it must follow from the problem logic that after a
transition all further answers remain the same. Some empirical samples are not proof of
correctness.

### Error 6: Wrong upper limit

A `first_true` search needs a guaranteed right candidate. If the range is unknown, it
can be found similar to exponential search by doubling.

## 20. Test strategy

A robust implementation not only checks typical hits. Tests shall include at least the
following classes:

- empty sequence and sequence with an element,
- hits on the first and last index,
- Target smaller than all and larger than all elements,
- missing objective between two elements,
- straight and odd lengths,
- many duplicates and an array of duplicates only,
- Lower/Upper Bound on both outer edges,
- minimal answer search space with only one candidate,
- transition directly at the lower or upper response limit,
- Rotated arrays without rotation and with rotation at the edge.

Property tests can check the variants against established references. For example, for a
sorted random list:

```python
import random
from bisect import bisect_left, bisect_right


rng = random.Random(80801)  # Fixed seed makes failures reproducible.
for _ in range(200):
    random_values = sorted(rng.randrange(-20, 21) for _ in range(rng.randrange(50)))
    random_target = rng.randrange(-25, 26)
    assert lower_bound(random_values, random_target) == bisect_left(
        random_values, random_target
    )
    assert upper_bound(random_values, random_target) == bisect_right(
        random_values, random_target
    )
    found = binary_search(random_values, random_target)
    assert (found != -1) == (random_target in random_values)
```

## 21. Decision Guide

Before implementing these questions:

1. **Is the search space arranged or can a monotonic condition be created?** If no,
   Binary Search is not applicable.
2. **Am I looking for any hit or limit?** Any hit allowed early return; no limit.
3. **What is the meaning of my interval?** Leave `[left, right]` or `[left, right)` as a
   comment or in the docstring.
4. **What is guaranteed after the loop?** `-1`, a insertion position, the first true
   candidate or the last false candidate are different contracts.
5. **Why does the search space shrink in each round?** Check especially intervals with
   one and two elements.
6. **How expensive is the predicate?** When searching for answers, it is often executed
   logarithmically.
7. **Is it worth preparing?** Considered sorting costs, number of requests, updates and
   alternative data structures.

## 22. Executive summary

Binary Search uses order to rule out about half of the search space per comparison. The
technique becomes reliable when the interval is understood as a contract and the
invariant as a justification:

- Exact search works comfortably on `[left, right]` and ends at `left > right`.
- Lower and Upper Bound of course work on `[left, right)` and deliver insertion
  positions or range boundaries.
- Pythons `bisect` provides these border searches, but does not make list insertions
  sublinear.
- Binary Search on the answer seeks the transition of a monotonic predicate; search
  space and feasibility testing must be technically justified.
- Rotated arrays get enough local order for logarithmic search for unique values;
  duplicates can linearize the worst case.
- Exponential Search detects an unknown range by doubling.
- Sorting plus repeated search is especially worthwhile for many requests on relatively
  static data.

Anyone who can say in any variant what is true before, during and after the loop over
the search space can not only implement Binary Search by heart, but also transfer it to
new problems.

---

# Deutsche Fassung

# Modul 08 — Binary Search & Suchvarianten

Binary Search ist klein genug, um auf wenige Zeilen Code zu passen, und zugleich
reich genug, um zentrale algorithmische Fähigkeiten zu trainieren: einen
Suchraum präzise definieren, eine Invariante erhalten, Randfälle beherrschen und
ein Optimierungsproblem in eine monotone Ja-Nein-Frage übersetzen. Gerade die
kurze Implementierung ist trügerisch. Ein falsch gewähltes Intervall oder eine
inkonsistente Grenze erzeugt Fehler, die bei gewöhnlichen Eingaben unsichtbar
bleiben.

Dieses Skript baut das Thema in drei Ebenen auf:

1. **Intuition:** Warum Halbierung funktioniert und wann sich Vorsortieren lohnt.
2. **Simulation:** Wie sich Intervalle bei der exakten Suche, bei Grenzen, in
   rotierten Arrays und bei der Suche auf einer Antwort verändern.
3. **Formalisierung:** Welche Invarianten die Varianten tragen, warum sie korrekt
   terminieren und welche Komplexität tatsächlich entsteht.

Am Ende soll nicht eine einzelne Codevorlage auswendig gelernt sein. Entscheidend
ist, für ein neues Problem selbst ein konsistentes Suchintervall, eine monotone
Entscheidungsfunktion und eine passende Rückgabebedeutung wählen zu können.

---

# Ebene I — Intuition

## 1. Das Telefonbuch und die Halbierungsidee

Ein gedrucktes Telefonbuch ist nach Namen sortiert. Wer „Nguyen“ sucht, beginnt
nicht auf Seite eins und liest jeden Namen. Man schlägt ungefähr die Mitte auf:

- Steht dort „Kramer“, kann die linke Hälfte ausgeschlossen werden.
- Steht dort „Schulz“, kann die rechte Hälfte ausgeschlossen werden.
- Danach wird dieselbe Entscheidung auf der verbleibenden Hälfte wiederholt.

Die Sortierung verwandelt einen Vergleich in Information über einen ganzen
Bereich. Bei einer unsortierten Liste sagt `middle_value < target` nichts darüber
aus, auf welcher Seite das Ziel liegt. Binary Search ist deshalb kein schnellerer
Ersatz für lineare Suche auf beliebigen Daten. Die geordnete Struktur ist seine
entscheidende Vorbedingung.

Nach jedem erfolglosen Vergleich bleibt höchstens die Hälfte des vorherigen
Suchraums übrig. Aus 1.024 Kandidaten werden höchstens

```text
1024 → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
```

Das sind nur zehn Halbierungen. Allgemein sind nach `k` Schritten höchstens
`n / 2^k` Kandidaten übrig. Sobald dieser Ausdruck höchstens eins ist, gilt
`k ≈ log₂(n)`.

## 2. Was wird eigentlich gesucht?

„Binary Search“ kann drei verschiedene Absichten bezeichnen:

1. **Exakte Suche:** Kommt ein Wert vor? Falls ja, an welchem Index?
2. **Grenzsuche:** Wo beginnt oder endet ein Bereich gleicher beziehungsweise
   geeigneter Werte?
3. **Antwortsuche:** Welcher kleinste oder größte Wert erfüllt eine monotone
   Bedingung?

Die dritte Form ist die wichtigste Verallgemeinerung. Der Suchraum muss kein
Array sein. Er kann aus Kapazitäten, Zeitbudgets, Schwellenwerten oder anderen
geordneten Kandidaten bestehen. Binary Search sucht dann den Übergang zwischen
„nicht möglich“ und „möglich“.

## 3. Wann lohnt sich Sortieren plus Suchen?

Eine lineare Suche kostet pro Anfrage `O(n)`. Einmaliges Sortieren kostet
`O(n log n)`, anschließend kostet jede Binary Search `O(log n)`. Für `q`
Suchanfragen ergeben sich grob:

```text
linear:             O(q · n)
sort + search: O(n log n + q log n)
```

Bei einer einzigen Anfrage gewinnt meist die lineare Suche, weil die
Sortierkosten nicht amortisiert werden. Bei vielen Anfragen auf einem weitgehend
statischen Datenbestand kann sich die sortierte Repräsentation lohnen. Wenn nur
exakte Mitgliedschaft zählt, ist eine Hash Map beziehungsweise ein Set oft noch
passender. Die Sortierung bietet zusätzlich Bereichsanfragen, Vorgänger,
Nachfolger und geordnete Ausgabe.

Ein kleines Experiment zeigt den Unterschied der Vergleichszahlen. Es misst
nicht die gesamte reale Laufzeit, isoliert aber die algorithmische Idee.

```python
def linear_search_with_count(values, target):
    """Return the target index and number of equality comparisons."""
    for index, value in enumerate(values):
        if value == target:
            return index, index + 1
    return -1, len(values)


def binary_search_with_count(values, target):
    """Return the target index and number of midpoint comparisons."""
    left, right = 0, len(values) - 1
    comparisons = 0
    while left <= right:
        middle = left + (right - left) // 2
        comparisons += 1
        if values[middle] == target:
            return middle, comparisons
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    return -1, comparisons


sample = list(range(1_000_000))
assert linear_search_with_count(sample, 999_999) == (999_999, 1_000_000)
index, comparisons = binary_search_with_count(sample, 999_999)
assert index == 999_999
assert comparisons <= 20
print(f"Linear: 1,000,000 comparisons; binary: {comparisons}")
```

---

# Ebene II — Simulation

## 4. Exakte Suche Schritt für Schritt

Gesucht ist `23` in diesem sortierten Array:

```text
Index: 0   1   2   3   4   5   6   7   8
Value:   3   7  11  15  18  23  29  31  42
```

Wir verwenden ein **geschlossenes Intervall** `[left, right]`. Beide Grenzen
gehören zum Suchraum.

| Schritt | `left` | `right` | `middle` | Wert | Entscheidung |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 8 | 4 | 18 | `18 < 23`, also `left = 5` |
| 2 | 5 | 8 | 6 | 29 | `29 > 23`, also `right = 5` |
| 3 | 5 | 5 | 5 | 23 | gefunden |

Der Suchraum enthält vor jedem Vergleich alle Indizes, an denen `23` noch liegen
kann. Nach dem Vergleich in Schritt 1 können die Indizes `0` bis `4`
einschließlich ausgeschlossen werden. `middle` darf nicht im nächsten Intervall
bleiben: Sein Wert wurde bereits geprüft. Deshalb lautet die neue Grenze
`middle + 1` und nicht `middle`.

```python
def binary_search(values, target):
    """Return an index of target in sorted values, or -1 if absent."""
    left, right = 0, len(values) - 1
    while left <= right:
        middle = left + (right - left) // 2
        if values[middle] == target:
            return middle
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    return -1


values = [3, 7, 11, 15, 18, 23, 29, 31, 42]
assert binary_search(values, 23) == 5
assert binary_search(values, 3) == 0
assert binary_search(values, 42) == 8
assert binary_search(values, 16) == -1
assert binary_search([], 10) == -1
```

### 4.1 Erfolglose Suche

Gesucht ist nun `16` im selben Array:

| Schritt | Intervall | Mitte/Wert | neues Intervall |
|---:|---|---|---|
| 1 | `[0, 8]` | `4 / 18` | `[0, 3]` |
| 2 | `[0, 3]` | `1 / 7` | `[2, 3]` |
| 3 | `[2, 3]` | `2 / 11` | `[3, 3]` |
| 4 | `[3, 3]` | `3 / 15` | `[4, 3]` |

Das Intervall `[4, 3]` ist leer, weil `left > right`. Genau deshalb gehört beim
geschlossenen Intervall das Gleichheitszeichen in `while left <= right`. Ein
Ein-Element-Intervall wie `[3, 3]` ist noch ein gültiger Suchraum und muss geprüft
werden.

## 5. Die Spur sichtbar machen

Eine Trace-Funktion hilft, die Intervallsemantik statt nur das Ergebnis zu
prüfen.

```python
def binary_search_trace(values, target):
    """Return visited closed intervals and midpoint decisions."""
    left, right = 0, len(values) - 1
    trace = []
    while left <= right:
        middle = left + (right - left) // 2
        relation = "equal"
        if values[middle] < target:
            relation = "too_small"
        elif values[middle] > target:
            relation = "too_large"
        trace.append((left, right, middle, values[middle], relation))
        if relation == "equal":
            break
        if relation == "too_small":
            left = middle + 1
        else:
            right = middle - 1
    return trace


trace = binary_search_trace([3, 7, 11, 15, 18, 23, 29, 31, 42], 23)
assert trace == [
    (0, 8, 4, 18, "too_small"),
    (5, 8, 6, 29, "too_large"),
    (5, 5, 5, 23, "equal"),
]
```

## 6. Nicht irgendein Treffer, sondern die erste Position

Bei Duplikaten darf die exakte Suche jeden passenden Index liefern. Für
Bereichsanfragen wird dagegen oft die **erste Position mit `value >= target`**
gesucht. Diese Position heißt Lower Bound oder Insert-Position.

Für `[2, 4, 4, 4, 7, 9]` und Ziel `4` wird das halb offene Intervall
`[left, right)` verwendet:

| Schritt | Intervall | `middle` | Wert | Entscheidung |
|---:|---|---:|---:|---|
| 1 | `[0, 6)` | 3 | 4 | Kandidat; rechts einschließlich Mitte weitersuchen: `[0, 3)` |
| 2 | `[0, 3)` | 1 | 4 | Kandidat; `[0, 1)` |
| 3 | `[0, 1)` | 0 | 2 | zu klein; `[1, 1)` |

Am Ende ist `left == right == 1`. Anders als bei der exakten Suche wird bei
Gleichheit nicht beendet. Die gefundene Position könnte noch einen gleichwertigen
Vorgänger besitzen.

```python
def lower_bound(values, target):
    """Return the first index whose value is greater than or equal to target."""
    left, right = 0, len(values)
    while left < right:
        middle = left + (right - left) // 2
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle
    return left


values = [2, 4, 4, 4, 7, 9]
assert lower_bound(values, 4) == 1
assert lower_bound(values, 5) == 4
assert lower_bound(values, 1) == 0
assert lower_bound(values, 10) == len(values)
assert lower_bound([], 4) == 0
```

Die Rückgabe ist immer eine gültige Einfügeposition von `0` bis `len(values)`.
Sie ist nicht automatisch ein Treffer. Für eine exakte Suche über den Lower
Bound folgt deshalb eine separate Prüfung.

```python
def find_via_lower_bound(values, target):
    """Return the first target index, or -1 if target is absent."""
    index = lower_bound(values, target)
    if index < len(values) and values[index] == target:
        return index
    return -1


assert find_via_lower_bound([2, 4, 4, 4, 7, 9], 4) == 1
assert find_via_lower_bound([2, 4, 4, 4, 7, 9], 5) == -1
```

## 7. Upper Bound und letztes Vorkommen

Der Upper Bound ist die erste Position mit `value > target`. Der einzige
algorithmische Unterschied zum Lower Bound liegt in der Gleichheitsbehandlung.

```python
def upper_bound(values, target):
    """Return the first index whose value is strictly greater than target."""
    left, right = 0, len(values)
    while left < right:
        middle = left + (right - left) // 2
        if values[middle] <= target:
            left = middle + 1
        else:
            right = middle
    return left


def equal_range(values, target):
    """Return the half-open range containing all occurrences of target."""
    return lower_bound(values, target), upper_bound(values, target)


def last_occurrence(values, target):
    """Return the last target index, or -1 if target is absent."""
    index = upper_bound(values, target) - 1
    if index >= 0 and values[index] == target:
        return index
    return -1


values = [2, 4, 4, 4, 7, 9]
assert upper_bound(values, 4) == 4
assert equal_range(values, 4) == (1, 4)
assert last_occurrence(values, 4) == 3
assert last_occurrence(values, 5) == -1
assert equal_range(values, 5) == (4, 4)
```

Die Anzahl der Vorkommen ergibt sich ohne lineares Durchlaufen des Blocks als
`upper_bound - lower_bound`.

## 8. Pythons `bisect`-Modul

Die Standardbibliothek stellt die beiden Grenzsuchen bereit:

- `bisect_left(values, target)` entspricht `lower_bound`.
- `bisect_right(values, target)` entspricht `upper_bound`.
- `insort_left` und `insort_right` fügen an der jeweiligen Grenze ein.

Die Suche nach der Einfügeposition kostet `O(log n)`. Das Einfügen in eine
Python-Liste kostet trotzdem `O(n)`, weil die nachfolgenden Elemente verschoben
werden müssen.

```python
from bisect import bisect_left, bisect_right, insort_left


values = [2, 4, 4, 4, 7, 9]
assert bisect_left(values, 4) == lower_bound(values, 4) == 1
assert bisect_right(values, 4) == upper_bound(values, 4) == 4

insort_left(values, 5)
assert values == [2, 4, 4, 4, 5, 7, 9]
```

Seit Python 3.10 akzeptieren die Suchfunktionen ein `key`-Argument. Wichtig ist
die asymmetrische Bedeutung: `key` wird auf Listenelemente angewendet, nicht auf
den separat übergebenen Suchwert. Der Suchwert muss bereits im Schlüsselraum
liegen.

```python
from bisect import bisect_left


experiments = [
    {"learning_rate": 0.001, "run": "a"},
    {"learning_rate": 0.01, "run": "b"},
    {"learning_rate": 0.1, "run": "c"},
]
position = bisect_left(experiments, 0.02, key=lambda row: row["learning_rate"])
assert position == 2
```

## 9. Binary Search auf der Antwort

Angenommen, Pakete müssen in ihrer gegebenen Reihenfolge innerhalb von `D` Tagen
verschickt werden. Gesucht ist nicht ein Element eines Arrays, sondern die
kleinste tägliche Kapazität. Für jede Kandidatenkapazität lässt sich die Frage
beantworten:

> Reicht diese Kapazität aus, um alle Pakete in höchstens `D` Tagen zu senden?

Die Antworten sind monoton:

```text
Capacity: 3 4 5 6 7 8 9 10 11...
Doable:    N N N N J J J   J   J...
```

Gesucht ist das erste `J`. Die untere Grenze ist das schwerste einzelne Paket;
die obere Grenze ist die Summe aller Gewichte, also eine Kapazität für einen
einzigen Tag.

Für Gewichte `[3, 2, 2, 4, 1, 4]` und drei Tage:

| Kandidat | benötigte Tage | machbar? | nächster Bereich |
|---:|---:|---|---|
| 10 | 2 | ja | kleinere Hälfte |
| 7 | 3 | ja | kleinere Hälfte |
| 5 | 4 | nein | größere Hälfte |
| 6 | 3 | ja | kleinere Hälfte |

Die kleinste machbare Kapazität ist `6`.

```python
def required_days(weights, capacity):
    """Return days needed when items must retain their original order."""
    days = 1
    current_load = 0
    for weight in weights:
        if weight > capacity:
            return float("inf")
        if current_load + weight > capacity:
            days += 1
            current_load = 0
        current_load += weight
    return days


def minimum_shipping_capacity(weights, day_limit):
    """Return the smallest capacity that ships all weights within day_limit."""
    if not weights or day_limit < 1:
        raise ValueError("weights must be non-empty and day_limit positive")
    left, right = max(weights), sum(weights)
    while left < right:
        middle = left + (right - left) // 2
        if required_days(weights, middle) <= day_limit:
            right = middle
        else:
            left = middle + 1
    return left


weights = [3, 2, 2, 4, 1, 4]
assert [required_days(weights, capacity) for capacity in (5, 6, 7, 10)] == [
    4,
    3,
    3,
    2,
]
assert minimum_shipping_capacity(weights, 3) == 6
assert minimum_shipping_capacity(weights, 1) == sum(weights)
```

### 9.1 Eine wiederverwendbare „first true“-Schablone

Viele ganzzahlige Antwortsuchen lassen sich auf dieselbe Grenzsuche reduzieren.
Die Vorbedingung lautet: Im inklusiven Bereich `[low, high]` existiert mindestens
ein wahrer Kandidat, und nach dem ersten wahren Kandidaten werden alle weiteren
ebenfalls wahr.

```python
def first_true(low, high, predicate):
    """Return the first integer in [low, high] satisfying a monotone predicate."""
    if low > high or not predicate(high):
        raise ValueError("the search range must contain a feasible candidate")
    while low < high:
        middle = low + (high - low) // 2
        if predicate(middle):
            high = middle
        else:
            low = middle + 1
    return low


assert first_true(0, 100, lambda value: value * value >= 30) == 6
assert first_true(7, 7, lambda value: value >= 7) == 7
```

Die ganzzahlige Quadratwurzel ist ein weiteres Beispiel. Gesucht ist hier der
größte Wert mit `root² <= number`. Das kann entweder mit einer gespiegelten
„last true“-Vorlage oder über den ersten zu großen Wert formuliert werden.

```python
def integer_square_root(number):
    """Return floor(sqrt(number)) without calling math.sqrt."""
    if number < 0:
        raise ValueError("number must not be negative")
    left, right = 0, number + 1
    while left + 1 < right:
        middle = left + (right - left) // 2
        if middle * middle <= number:
            left = middle
        else:
            right = middle
    return left


assert [integer_square_root(value) for value in range(10)] == [
    0,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    3,
]
assert integer_square_root(10**12) == 10**6
```

### 9.2 Reelle Suchräume

Auf Gleitkommazahlen kann nicht sinnvoll bis `left == right` gesucht werden.
Stattdessen endet die Schleife nach einer festen Iterationszahl oder sobald das
Intervall kleiner als eine gewünschte Toleranz ist. Eine feste Iterationszahl
ist oft leichter zu analysieren: Jede Runde halbiert den Fehlerbereich.

```python
def approximate_square_root(number, iterations=80):
    """Approximate a non-negative square root by bisection."""
    if number < 0:
        raise ValueError("number must not be negative")
    left, right = 0.0, max(1.0, float(number))
    for _ in range(iterations):
        middle = (left + right) / 2.0
        if middle * middle < number:
            left = middle
        else:
            right = middle
    return (left + right) / 2.0


root = approximate_square_root(2)
assert abs(root * root - 2) < 1e-12
assert approximate_square_root(0) < 1e-12
```

## 10. Suche im rotierten sortierten Array

Ein aufsteigend sortiertes Array kann an einer Stelle rotiert worden sein:

```text
original: [1, 3, 5, 7, 9, 11, 13]
Rotated:      [9, 11, 13, 1, 3, 5, 7]
```

Das gesamte Array ist nicht mehr sortiert, aber bei verschiedenen Werten ist in
jeder Runde mindestens eine Hälfte sortiert. Liegt das Ziel im Wertebereich
dieser Hälfte, wird dort weitergesucht; andernfalls in der anderen Hälfte.

Gesucht ist `3`:

| Intervall | Mitte/Wert | sortierte Hälfte | Entscheidung |
|---|---|---|---|
| `[0, 6]` | `3 / 1` | rechts `[1, 3, 5, 7]` | `3` liegt rechts |
| `[4, 6]` | `5 / 5` | links `[3, 5]` | `3` liegt links |
| `[4, 4]` | `4 / 3` | — | gefunden |

```python
def search_rotated(values, target):
    """Search a rotated sorted sequence containing distinct values."""
    left, right = 0, len(values) - 1
    while left <= right:
        middle = left + (right - left) // 2
        if values[middle] == target:
            return middle

        if values[left] <= values[middle]:
            if values[left] <= target < values[middle]:
                right = middle - 1
            else:
                left = middle + 1
        else:
            if values[middle] < target <= values[right]:
                left = middle + 1
            else:
                right = middle - 1
    return -1


rotated = [9, 11, 13, 1, 3, 5, 7]
assert search_rotated(rotated, 3) == 4
assert search_rotated(rotated, 11) == 1
assert search_rotated(rotated, 8) == -1
assert search_rotated([], 8) == -1
```

Bei vielen Duplikaten kann die Aussage „mindestens eine Hälfte ist eindeutig
sortiert“ unbrauchbar werden. Sind `values[left]`, `values[middle]` und
`values[right]` gleich, ist nicht erkennbar, auf welcher Seite die Rotation
liegt. Die Grenzen können dann vorsichtig um eins verschoben werden. Dadurch
verschlechtert sich der Worst Case auf `O(n)`.

```python
def search_rotated_with_duplicates(values, target):
    """Return whether target occurs in a rotated sorted sequence with duplicates."""
    left, right = 0, len(values) - 1
    while left <= right:
        middle = left + (right - left) // 2
        if values[middle] == target:
            return True
        if values[left] == values[middle] == values[right]:
            left += 1
            right -= 1
        elif values[left] <= values[middle]:
            if values[left] <= target < values[middle]:
                right = middle - 1
            else:
                left = middle + 1
        else:
            if values[middle] < target <= values[right]:
                left = middle + 1
            else:
                right = middle - 1
    return False


assert search_rotated_with_duplicates([2, 2, 2, 3, 4, 2], 3)
assert not search_rotated_with_duplicates([2, 2, 2, 3, 4, 2], 5)
```

## 11. Exponential Search bei unbekannter Reichweite

Binary Search benötigt normalerweise eine rechte Grenze. Bei einem sortierten
Strom, einer API mit indexbasiertem Zugriff oder einer sehr großen Liste ist die
interessante Position möglicherweise nahe am Anfang und die sinnvolle Reichweite
unbekannt. Exponential Search findet zuerst ein Intervall:

```text
Check index: 1, 2, 4, 8, 16, 32,...
```

Sobald der Wert an der Grenze mindestens dem Ziel entspricht, folgt eine normale
Binary Search im zuletzt übersprungenen Bereich. Liegt das Ziel an Position `p`,
kostet sowohl die Grenzsuche als auch die anschließende Suche `O(log p)`.

```python
def binary_search_between(values, target, left, right):
    """Search target in the inclusive index range [left, right]."""
    while left <= right:
        middle = left + (right - left) // 2
        if values[middle] == target:
            return middle
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    return -1


def exponential_search(values, target):
    """Search sorted values by first discovering an exponential range."""
    if not values:
        return -1
    if values[0] == target:
        return 0
    bound = 1
    while bound < len(values) and values[bound] < target:
        bound *= 2
    return binary_search_between(
        values,
        target,
        bound // 2,
        min(bound, len(values) - 1),
    )


values = list(range(0, 10_000, 3))
assert exponential_search(values, 0) == 0
assert exponential_search(values, 27) == 9
assert exponential_search(values, 9_999) == len(values) - 1
assert exponential_search(values, 10_000) == -1
```

## 12. Ausblick: Hyperparameter und Schwellenwerte

Binary Search ist nur dann korrekt, wenn die geprüfte Bedingung entlang des
Suchraums monoton ist. Das ist bei Hyperparametern nicht automatisch der Fall.
Die Validation Accuracy eines Modells kann mit wachsender Regularisierung erst
steigen und später fallen. Auf solch einer unbekannten, verrauschten Kurve ist
Binary Search kein allgemeiner Optimierer.

Geeignet ist die Idee dagegen für monotone Nebenbedingungen. Bei einem
Klassifikationsschwellenwert gilt typischerweise:

- Ein höherer positiver Schwellenwert sagt weniger Beispiele positiv voraus.
- Die Recall-Kurve ist deshalb nicht steigend; bei sauberer Definition fällt sie
  monoton oder bleibt gleich.
- Die Precision kann häufig steigen, muss auf endlichen Daten mit Bindungen aber
  nicht streng monoton sein.

Ein weiteres Beispiel ist ressourcenbeschränktes Tuning: „Ist ein Modell mit
höchstens diesem Speicherbudget ausführbar?“ kann monoton sein, auch wenn die zu
maximierende Modellqualität es nicht ist. Das Prinzip hinter der Antwortsuche
bleibt also nützlich, aber die Monotonie muss fachlich begründet und anhand von
Randfällen geprüft werden. Projekt `03-final` wendet diese Vorsicht auf einen
simulierten Score-Datensatz an.

---

# Ebene III — Formalisierung

## 13. Suchintervalle sind Verträge

Die meisten Off-by-One-Fehler entstehen nicht durch die Mittelpunktformel,
sondern durch das Vermischen zweier Intervallverträge.

### 13.1 Geschlossenes Intervall `[left, right]`

- Initialisierung: `left = 0`, `right = n - 1`
- Nicht leer: `left <= right`
- Größe: `right - left + 1`
- Mitte verwerfen: `left = middle + 1` oder `right = middle - 1`
- Typische Rückgabe bei Misserfolg: `-1`

Diese Form passt gut zur exakten Suche, weil jeder noch mögliche Index im
Intervall enthalten ist.

### 13.2 Halb offenes Intervall `[left, right)`

- Initialisierung: `left = 0`, `right = n`
- Nicht leer: `left < right`
- Größe: `right - left`
- Linke Hälfte einschließlich Mitte behalten: `right = middle`
- Mitte verwerfen: `left = middle + 1`
- Typische Rückgabe: Einfügeposition `left` im Bereich `0..n`

Diese Form passt besonders gut zu Lower Bound und Upper Bound. Das leere
Intervall wird eindeutig durch `left == right` dargestellt.

Folgender Randfalltest macht die Unterschiede sichtbar:

```python
edge_cases = [
    ([], 5),
    ([5], 5),
    ([5], 4),
    ([5], 6),
    ([1, 5], 1),
    ([1, 5], 5),
    ([1, 5], 3),
]

for edge_values, edge_target in edge_cases:
    exact = binary_search(edge_values, edge_target)
    insertion = lower_bound(edge_values, edge_target)
    assert exact == -1 or edge_values[exact] == edge_target
    assert all(value < edge_target for value in edge_values[:insertion])
    assert all(value >= edge_target for value in edge_values[insertion:])
```

## 14. Die Invariante der exakten Suche

Für die geschlossene exakte Suche lautet eine geeignete Invariante:

> Falls `target` im Array vorkommt und noch nicht gefunden wurde, liegt mindestens
> ein Vorkommen im aktuellen Intervall `[left, right]`.

### Initialisierung

Vor der ersten Runde ist das Intervall `[0, n - 1]`. Es enthält alle gültigen
Indizes. Die Aussage gilt daher.

### Erhaltung

Sei `middle` der geprüfte Index.

- Wenn `values[middle] < target`, können aufgrund der Sortierung weder `middle`
  noch kleinere Indizes das Ziel enthalten. Mit `left = middle + 1` bleiben alle
  möglichen Treffer erhalten.
- Wenn `values[middle] > target`, können `middle` und alle größeren Indizes
  ausgeschlossen werden. Mit `right = middle - 1` bleibt die Invariante erhalten.
- Bei Gleichheit ist ein korrekter Treffer gefunden.

### Terminierung

Jede erfolglose Runde entfernt `middle` und mindestens eine weitere Hälfte aus
dem Suchraum. Die nichtnegative Intervallgröße `right - left + 1` wird strikt
kleiner. Die Schleife terminiert. Ist danach `left > right`, ist das Intervall
leer. Nach der Invariante kann das Ziel dann nicht im Array vorkommen.

## 15. Die Invariante des Lower Bound

Für `[left, right)` lässt sich die Grenzsuche über bereits klassifizierte Bereiche
verstehen:

- Alle Indizes `< left` enthalten Werte `< target`.
- Alle Indizes `>= right` enthalten Werte `>= target`.
- Nur `[left, right)` ist noch nicht klassifiziert.

Ist `values[middle] < target`, gehört die Mitte sicher zum linken Bereich, also
`left = middle + 1`. Andernfalls gehört die Mitte zum rechten Bereich; sie könnte
der gesuchte erste Index sein und wird mit `right = middle` behalten. Bei
`left == right` ist nichts unklassifiziert. Beide Aussagen treffen an derselben
Grenze zusammen, also ist `left` die erste Position mit Wert mindestens `target`.

Diese Invariante erklärt, warum bei der Grenzsuche nicht früh bei Gleichheit
zurückgegeben wird.

## 16. Mittelpunkt und Fortschritt

Die mathematisch naheliegende Formel lautet `(left + right) // 2`. Python-Integer
laufen nicht über, aber in Sprachen mit fester Integer-Breite kann die Summe
überlaufen. Diese Form vermeidet das Problem:

```text
middle = left + (right - left) // 2
```

Noch wichtiger ist garantierter Fortschritt. Wenn das Intervall zwei Elemente
enthält und `middle` nach unten rundet, darf ein Update nicht denselben Zustand
wiederherstellen. Bei einem geschlossenen Intervall führt beispielsweise
`left = middle` leicht zu einer Endlosschleife. Konsistente Schablonen verwenden
`middle + 1`, `middle - 1` oder ändern bei halb offenen Grenzen die andere Seite
auf `middle`.

## 17. Komplexitäten im Vergleich

| Verfahren | Voraussetzung | Zeit | Zusatzspeicher | Besonderheit |
|---|---|---:|---:|---|
| Lineare Suche | keine Ordnung | `O(n)` | `O(1)` | eine Anfrage oft ausreichend |
| Binary Search | sortierter Random Access | `O(log n)` | `O(1)` iterativ | exakter Treffer |
| Lower/Upper Bound | sortierter Random Access | `O(log n)` | `O(1)` | Grenzen und Bereiche |
| `bisect`-Suche | sortierte Python-Sequenz | `O(log n)` | `O(1)` | Standardbibliothek |
| `insort` in Liste | sortierte Python-Liste | `O(n)` | `O(1)` zusätzlich | Suche schnell, Verschieben linear |
| Exponential Search | sortiert, Reichweite unklar | `O(log p)` | `O(1)` | `p` ist Zielposition |
| Rotierte Suche, eindeutig | rotiert sortiert | `O(log n)` | `O(1)` | eine Hälfte ist geordnet |
| Rotierte Suche, Duplikate | rotiert sortiert | Worst Case `O(n)` | `O(1)` | Ordnung kann mehrdeutig sein |
| Antwortsuche | monotones Prädikat | `O(log R · C)` | problemabhängig | `R` Suchraum, `C` Prüfroutine |

Binary Search setzt effizienten indexbasierten Zugriff voraus. Auf einer Linked
List kostet das Erreichen der Mitte `O(n)`, wodurch der Vorteil verloren geht.
Bei Daten auf langsamen externen Medien können B-Bäume oder blockorientierte
Indizes geeigneter sein, weil sie die Zahl teurer Zugriffe minimieren.

## 18. Binary Search auf der Antwort formal

Sei `P(x)` ein Prädikat auf einem geordneten ganzzahligen Bereich. Für die Suche
nach dem kleinsten wahren Wert benötigen wir:

```text
x < y and P(x) = true  ⇒  P(y) = true
```

Die Wahrheitswerte haben also die Form

```text
wrong, wrong,..., wrong, true, true,..., true
```

Die Invariante der `first_true`-Schablone lautet:

> Das Intervall `[low, high]` enthält den ersten wahren Kandidaten, und `high` ist
> ein wahrer Kandidat.

Ist `P(middle)` wahr, kann `middle` bereits die Grenze sein; deshalb bleibt es mit
`high = middle` erhalten. Ist es falsch, liegen der erste wahre Kandidat und alle
weiteren Möglichkeiten strikt rechts; deshalb folgt `low = middle + 1`.

Die Gesamtkosten sind nicht bloß `O(log R)`. Wenn eine Machbarkeitsprüfung
`C` Zeit benötigt, ergibt sich `O(C log R)`. Im Versandbeispiel läuft die Prüfung
über alle `n` Pakete, also kostet die Lösung `O(n log(sum(weights)))`.

## 19. Klassische Fehler und Gegenmittel

### Fehler 1: Unsortierte Eingabe

Ein korrekt aussehender Algorithmus liefert auf unsortierten Daten falsche
Ergebnisse. Die Ordnung ist eine Vorbedingung, keine Optimierung.

### Fehler 2: Vermischte Intervallmodelle

`right = len(values)` gehört zum halb offenen Modell. `while left <= right`
gehört zum geschlossenen Modell. Die Kombination kann außerhalb der Liste
zugreifen.

### Fehler 3: Keine Fortschrittsgarantie

Ein Update auf `left = middle` kann bei nach unten gerundeter Mitte stecken
bleiben. Nach jeder Runde muss der unentschiedene Bereich strikt schrumpfen.

### Fehler 4: Insert-Position als Treffer interpretieren

`lower_bound` liefert auch für ein fehlendes Ziel eine gültige Position. Ein
exakter Treffer braucht zusätzlich `index < n and values[index] == target`.

### Fehler 5: Monotonie nur vermuten

Bei Antwortsuche muss aus der Problemlogik folgen, dass nach einem Übergang alle
weiteren Antworten gleich bleiben. Einige empirische Stichproben sind kein
Korrektheitsbeweis.

### Fehler 6: Falsche obere Grenze

Eine `first_true`-Suche braucht einen garantiert machbaren rechten Kandidaten.
Ist die Reichweite unbekannt, kann sie ähnlich wie bei Exponential Search durch
Verdopplung gefunden werden.

## 20. Teststrategie

Eine belastbare Implementierung prüft nicht nur typische Treffer. Mindestens
folgende Klassen gehören in Tests:

- leere Sequenz und Sequenz mit einem Element,
- Treffer am ersten und letzten Index,
- Ziel kleiner als alle und größer als alle Elemente,
- fehlendes Ziel zwischen zwei Elementen,
- gerade und ungerade Längen,
- viele Duplikate und ein Array nur aus Duplikaten,
- Lower/Upper Bound an beiden Außenrändern,
- minimaler Antwortsuchraum mit nur einem Kandidaten,
- Übergang direkt an der unteren oder oberen Antwortgrenze,
- rotierte Arrays ohne Rotation und mit Rotation am Rand.

Property-Tests können die Varianten gegen etablierte Referenzen prüfen. Für eine
sortierte Zufallsliste gilt beispielsweise:

```python
import random
from bisect import bisect_left, bisect_right


rng = random.Random(80801)  # Fixed seed makes failures reproducible.
for _ in range(200):
    random_values = sorted(rng.randrange(-20, 21) for _ in range(rng.randrange(50)))
    random_target = rng.randrange(-25, 26)
    assert lower_bound(random_values, random_target) == bisect_left(
        random_values, random_target
    )
    assert upper_bound(random_values, random_target) == bisect_right(
        random_values, random_target
    )
    found = binary_search(random_values, random_target)
    assert (found != -1) == (random_target in random_values)
```

## 21. Entscheidungsleitfaden

Stelle vor der Implementierung diese Fragen:

1. **Ist der Suchraum geordnet oder lässt sich eine monotone Bedingung bilden?**
   Falls nein, ist Binary Search nicht anwendbar.
2. **Suche ich einen beliebigen Treffer oder eine Grenze?** Ein beliebiger
   Treffer erlaubt frühe Rückgabe; eine Grenze nicht.
3. **Welche Bedeutung hat mein Intervall?** Schreibe `[left, right]` oder
   `[left, right)` als Kommentar beziehungsweise in den Docstring.
4. **Was ist nach der Schleife garantiert?** `-1`, eine Einfügeposition, der
   erste wahre Kandidat oder der letzte falsche Kandidat sind unterschiedliche
   Verträge.
5. **Warum schrumpft der Suchraum in jeder Runde?** Prüfe besonders Intervalle
   mit einem und zwei Elementen.
6. **Wie teuer ist das Prädikat?** Bei Antwortsuche wird es logarithmisch oft
   ausgeführt.
7. **Lohnt sich die Vorbereitung?** Berücksichtige Sortierkosten, Zahl der
   Anfragen, Aktualisierungen und alternative Datenstrukturen.

## 22. Zusammenfassung

Binary Search nutzt Ordnung, um pro Vergleich ungefähr die Hälfte des Suchraums
auszuschließen. Die Technik wird zuverlässig, wenn das Intervall als Vertrag und
die Invariante als Begründung verstanden werden:

- Exakte Suche arbeitet bequem auf `[left, right]` und endet bei `left > right`.
- Lower und Upper Bound arbeiten natürlich auf `[left, right)` und liefern
  Einfügepositionen beziehungsweise Bereichsgrenzen.
- Pythons `bisect` stellt diese Grenzsuchen bereit, macht Listeneinfügungen aber
  nicht sublinear.
- Binary Search auf der Antwort sucht den Übergang eines monotonen Prädikats;
  Suchraum und Machbarkeitsprüfung müssen fachlich begründet sein.
- Rotierte Arrays erhalten bei eindeutigen Werten genug lokale Ordnung für
  logarithmische Suche; Duplikate können den Worst Case linearisieren.
- Exponential Search entdeckt eine unbekannte Reichweite durch Verdopplung.
- Sortieren plus wiederholtes Suchen lohnt sich vor allem bei vielen Anfragen auf
  relativ statischen Daten.

Wer bei jeder Variante sagen kann, was vor, während und nach der Schleife über
den Suchraum wahr ist, kann Binary Search nicht nur auswendig implementieren,
sondern auf neue Probleme übertragen.
