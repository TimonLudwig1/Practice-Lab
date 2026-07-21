# Module 07: Sorting algorithms

Sorting is the textbook problem par excellence. Not because modern programs should
implement each sorting algorithm themselves, but because very different ideas can be
compared with the same clear task: local exchanges, repeated selection, incremental
construction, divide and conquer, partitioning and using additional information about
the value range.

This script develops sorting methods in three levels:

1. **Intuition:** Why an order simplifies follow-up problems and which properties count
   alongside runtime.
2. **Simulation:** Bubble, Selection, Insertion, Merge and Quicksort are passed step by
   step on small arrays. Counting, Bucket and Radix Sort show when the comparison lower
   bound may be bypassed.
3. **Formalization:** Invariants, correctness arguments, stability, space requirement,
   worst cases, the barrier Ω(n log n) and Pythons Timsort.

After this module, you should implement at least five procedures, justify their
properties and be able to make a suitable choice for almost sorted data, many duplicates
or a limited value range.

---

## Level 1: Intuition

### 1. Why Sorting Prepares So Many Problems

An unsorted sequence contains the same information as its sorted version, but its
structure is more difficult to use. After sorting, many tasks become easier:

- Binary search reduces a search from O(n) to O(log n).
- Duplicates stand next to each other and can be detected in a single pass.
- Minimum, maximum and quantile are in predictable positions.
- Two sorted data streams can be combined linearly.
- Area queries and ranking sequences become directly accessible.

Sorting is therefore often not a final task, but a one-time preprocessing that
accelerates many subsequent operations.

### 2. What does "sorted" mean?

For an ascending sequence `a`:

```text
a[0] ≤ a[1] <=... ≤ a[n - 1]
```

A correct sorting algorithm must meet two conditions:

1. **Order:** The output is sorted according to the desired relation.
2. **Permutation:** The output contains exactly the same elements with the same
   frequencies as the input.

Just checking the order is not enough. A faulty algorithm could simply turn `[3, 1, 2]`
into `[1, 2]`; the result would be sorted, but an element would be lost.

```python
from collections import Counter


def is_sorted(values: list[int]) -> bool:
    """Return whether values are in nondecreasing order."""
    return all(left <= right for left, right in zip(values, values[1:]))


def is_sorted_permutation(original: list[int], result: list[int]) -> bool:
    """Check both required properties of a sorting result."""
    return is_sorted(result) and Counter(original) == Counter(result)


assert is_sorted([1, 1, 3, 8])
assert not is_sorted([1, 4, 2])
assert not is_sorted_permutation([3, 1, 2], [1, 2])
assert is_sorted_permutation([3, 1, 2, 1], [1, 1, 2, 3])
```

### 3. More than Big-O: the property profile

Two methods with the same asymptotic runtime can differ in practice. At least these
questions are important for a well-founded choice:

- **Worst/Average/Best Case:** Which type of input triggers what costs?
- **Stability:** Does the relative order of equivalent elements remain?
- **In-Place:** Does the procedure require only constant or logarithmic additional
  memory instead of a second array of size n?
- **Adaptivity:** Is an existing order being exploited?
- **Comparatively based:** Does the algorithm only learn by comparison or can it
  interpret values as indices/digits?
- **Data access:** Does he work cache-friendly on connected memory?

There is therefore no "best sorting algorithm" without context.

### 4. The paradigms at a glance

| Idea | Representatives | Key question |
|---|---|---|
| Local neighbouring repair | Bubble Sort | Are adjacent elements wrong around? |
| Repeated selection | Selection Sort | Which smallest element is missing in the prefix? |
| Expand Ordinary Prefix | Insertion Sort | Where does the next element belong? |
| Share and merge orderly | Merge Sort | How do I combine two sorted halves? |
| To partition a pivot | Quicksort | What is left or right? |
| Use heap property | Heap Sort | Which extreme can I repeatedly draw from? |
| Count/break value range | Counting/Bucket/Radix | Which additional structure of the keys can I use? |

---

## Level 2: Simulation

### 5. Bubble Sort: big values rise to the right

Bubble Sort compares neighbors and swaps them if they are in wrong order. After a
complete run, the largest unsorted element is at its final position.

Input: `[5, 2, 4, 1]`

**Pass 1:**

| Comparison | Action | Condition |
|---|---|---|
| 5 and 2 | Exchange | `[2, 5, 4, 1]` |
| 5 and 4 | Exchange | `[2, 4, 5, 1]` |
| 5 and 1 | Exchange | `[2, 4, 1, 5]` |

The five is ready. **Pass 2:**

| Comparison | Action | Condition |
|---|---|---|
| 2 and 4 | none | `[2, 4, 1, 5]` |
| 4 and 1 | Exchange | `[2, 1, 4, 5]` |

The four is ready. **Pass 3:** 2 and 1 are exchanged; result `[1, 2, 4, 5]`.

```python
def bubble_sort(values: list[int]) -> list[int]:
    """Return a sorted copy using stable adjacent swaps."""
    result = values.copy()
    for end in range(len(result) - 1, 0, -1):
        swapped = False
        for index in range(end):
            if result[index] > result[index + 1]:
                result[index], result[index + 1] = (
                    result[index + 1],
                    result[index],
                )
                swapped = True
        if not swapped:
            break
    return result


bubble_input = [5, 2, 4, 1]
assert bubble_sort(bubble_input) == [1, 2, 4, 5]
assert bubble_input == [5, 2, 4, 1]
```

The `swapped` marker makes Bubble Sort adaptive: With already sorted input, it ends
after a linear pass. Without this rule, the best case would be quadratic.

**Schleifeninvariant:** Before each new pass, the suffix is sorted to the right of `end`
and contains the largest elements in their final positions.

### 6. Selection Sort: select the minimum

Selection Sort divides the array into a finished prefix and an unsorted rest. It
searches for the minimum in the rest and exchanges it to the next vacant position.

Input: `[5, 2, 4, 1]`

| Round | Unsorted area | Minimum | Exchange | Condition |
|---:|---|---:|---|---|
| 0 | `[5, 2, 4, 1]` | 1 to Index 3 | Index 0 ↔ 3 | `[1, 2, 4, 5]` |
| 1 | `[2, 4, 5]` | 2 to Index 1 | with yourself | `[1, 2, 4, 5]` |
| 2 | `[4, 5]` | 4 to Index 2 | with yourself | `[1, 2, 4, 5]` |

```python
def selection_sort(values: list[int]) -> list[int]:
    """Return a sorted copy by repeatedly selecting the minimum."""
    result = values.copy()
    for start in range(len(result) - 1):
        minimum_index = start
        for index in range(start + 1, len(result)):
            if result[index] < result[minimum_index]:
                minimum_index = index
        result[start], result[minimum_index] = (
            result[minimum_index],
            result[start],
        )
    return result


assert selection_sort([5, 2, 4, 1]) == [1, 2, 4, 5]
```

Selection Sort performs approximately `n(n-1)/2` comparisons regardless of input. Its
advantage is the small number of exchanges: at most n-1. If writing operations are
particularly expensive, this property can be relevant.

The direct exchange over long distances makes the standard variant unstable. One
equivalent element can be moved over another.

**Schleifeninvariant:** Before round `start` contains the prefix `result[:start]` the
`start` smallest elements in correct order.

### 7. Insertion Sort: place the next element

Insertion Sort is similar to sorting playing cards on hand. The left prefix is already
sorted. The next element is saved; larger prefix elements move to the right until the
insertion position is free.

Input: `[5, 2, 4, 1]`

| Key | sorted prefix before | Shifts | State after |
|---:|---|---|---|
| 2 | `[5]` | 5 to the right | `[2, 5, 4, 1]` |
| 4 | `[2, 5]` | 5 to the right | `[2, 4, 5, 1]` |
| 1 | `[2, 4, 5]` | 5, 4, 2 turn right | `[1, 2, 4, 5]` |

```python
def insertion_sort(values: list[int]) -> list[int]:
    """Return a sorted copy by extending a sorted prefix."""
    result = values.copy()
    for index in range(1, len(result)):
        key = result[index]
        position = index - 1
        while position >= 0 and result[position] > key:
            result[position + 1] = result[position]
            position -= 1
        result[position + 1] = key
    return result


assert insertion_sort([5, 2, 4, 1]) == [1, 2, 4, 5]
assert insertion_sort([1, 2, 3, 4]) == [1, 2, 3, 4]
```

Important is the strict comparison `>` instead of `>=`. Equivalent elements are not
pushed past each other; thus insertion Sort is stable.

For almost sorted data there are only a few **Inversions**. An inversion is an index
pair `(i, j)` with `i < j`, but `a[i] > a[j]`. Any shift of insertion Sort eliminates an
inversion. The runtime can therefore be expressed as O(n + I), where I is the number of
inversions. This explains its good practical performance on small or almost sorted runs.

**Schleifeninvariant:** Before iteration with `index` `result[:index]` is sorted and
contains exactly the original elements of this prefix.

### 8. Measuring the elementary methods

All three methods are in the worst case O(n2), but react differently to input forms. We
count comparisons instead of just time, so that the result does not depend on the
computer.

```python
def insertion_comparisons(values: list[int]) -> int:
    """Count key comparisons performed by insertion sort."""
    result = values.copy()
    comparisons = 0
    for index in range(1, len(result)):
        key = result[index]
        position = index - 1
        while position >= 0:
            comparisons += 1
            if result[position] <= key:
                break
            result[position + 1] = result[position]
            position -= 1
        result[position + 1] = key
    return comparisons


comparison_size = 100
sorted_comparisons = insertion_comparisons(list(range(comparison_size)))
reverse_comparisons = insertion_comparisons(list(range(comparison_size, 0, -1)))
print(
    f"Insertion comparisons: sorted={sorted_comparisons}, "
    f"reverse={reverse_comparisons}"
)
assert sorted_comparisons == comparison_size - 1
assert reverse_comparisons == comparison_size * (comparison_size - 1) // 2
```

### 9. Merge Sort: split and merge in order

Merge Sort follows Divide and Conquer:

1. Divide the sequence into two halves.
2. Sort both halves recursively.
3. Merge the sorted halves in linear time.

For `[5, 2, 4, 1]` the recursion tree is created:

```text
                [5, 2, 4, 1]
               /            \
          [5, 2]            [4, 1]
          /    \            /    \
        [5]    [2]        [4]    [1]
          \    /            \    /
          [2, 5]            [1, 4]
               \            /
                [1, 2, 4, 5]
```

The decisive step is merge. For `[2, 5]` and `[1, 4]`:

| left front | front right | Subsequent output |
|---:|---:|---|
| 2 | 1 | `[1]` |
| 2 | 4 | `[1, 2]` |
| 5 | 4 | `[1, 2, 4]` |
| 5 | empty | `[1, 2, 4, 5]` |

```python
def merge(left: list[int], right: list[int]) -> list[int]:
    """Stably merge two sorted lists."""
    result: list[int] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            result.append(left[left_index])
            left_index += 1
        else:
            result.append(right[right_index])
            right_index += 1
    result.extend(left[left_index:])
    result.extend(right[right_index:])
    return result


def merge_sort(values: list[int]) -> list[int]:
    """Return a stable merge-sorted copy."""
    if len(values) <= 1:
        return values.copy()
    middle = len(values) // 2
    left = merge_sort(values[:middle])
    right = merge_sort(values[middle:])
    return merge(left, right)


assert merge([2, 5], [1, 4]) == [1, 2, 4, 5]
assert merge_sort([5, 2, 4, 1]) == [1, 2, 4, 5]
```

At each recursive level, n elements are mined in total. There are about `log₂ n` levels.
Therefore, Merge Sort costs O(n log n) in Best, Average and Worst Case. The array
variant shown requires O(n) additional memory.

The comparison `<=` prefers the element from the left half in the case of equality.
Since left elements used to be in the input, the relative order of equivalent elements
remains: Merge Sort is stable.

### 10. Quicksort: to partition a pivot

Quicksort selects a pivot and partitions the sequence so that smaller or same elements
are left and larger right. After that both areas are sorted recursively. Unlike Merge
Sort, it does the crucial work of sharing, not merging.

We simulate the Lomuto partition to `[4, 2, 7, 3, 1, 6]` with pivot 6. `boundary` marks
the end of the range with values `<= pivot`.

| Read | Comparison with 6 | Action | Condition | Boundary |
|---:|---|---|---|---:|
| 4 | Small | Exchange with index 0 | `[4, 2, 7, 3, 1, 6]` | 1 |
| 2 | Small | Exchange with index 1 | `[4, 2, 7, 3, 1, 6]` | 2 |
| 7 | Larger | none | `[4, 2, 7, 3, 1, 6]` | 2 |
| 3 | Small | Exchange with index 2 | `[4, 2, 3, 7, 1, 6]` | 3 |
| 1 | Small | Exchange with index 3 | `[4, 2, 3, 1, 7, 6]` | 4 |
| Pivot | – | Exchange with boundary | `[4, 2, 3, 1, 6, 7]` | Pivot index 4 |

The 6 is now final. To the left are only values `<= 6`, to the right are values `> 6`.
The left side is not yet fully sorted.

```python
def lomuto_partition(values: list[int], low: int, high: int) -> int:
    """Partition values[low:high+1] and return the final pivot index."""
    pivot = values[high]
    boundary = low
    for index in range(low, high):
        if values[index] <= pivot:
            values[boundary], values[index] = values[index], values[boundary]
            boundary += 1
    values[boundary], values[high] = values[high], values[boundary]
    return boundary


def _quick_sort_range(values: list[int], low: int, high: int) -> None:
    if low >= high:
        return
    pivot_index = lomuto_partition(values, low, high)
    _quick_sort_range(values, low, pivot_index - 1)
    _quick_sort_range(values, pivot_index + 1, high)


def quick_sort(values: list[int]) -> list[int]:
    """Return an in-place-partitioned quick-sorted copy."""
    result = values.copy()
    _quick_sort_range(result, 0, len(result) - 1)
    return result


partition_example = [4, 2, 7, 3, 1, 6]
partition_index = lomuto_partition(partition_example, 0, len(partition_example) - 1)
assert partition_example == [4, 2, 3, 1, 6, 7]
assert partition_index == 4
assert quick_sort([5, 2, 4, 1]) == [1, 2, 4, 5]
```

**Partition invariant:** Before each iteration, the range to the left of `boundary`
contains only values `<= pivot`; between `boundary` and the current index are already
verified values `> pivot`.

With approximately halving pivots, the recursion tree has O(log n) levels, on each of
them a total of O(n) elements are partitioned: Average Case O(n log n).

### 11. Pivot choice and Quicksorts Worst Case

If the last element is always used as a pivot, an already sorted sequence is fatal:

```text
n Elements -> Partitions n-1 and 0
n-1        -> Partitions n-2 and 0
...
```

The work is `(n-1) + (n-2) + ... + 1 = O(n²)`, and the recursion depth becomes O(n). In
Python, this can even reach the recursion limit.

Typical countermeasures:

- random pivot,
- Median-of-three from first, middle and last value,
- 3-way partitioning in `< pivot`, `== pivot`, `> pivot` with many duplicates,
- Recursion first on the smaller partition and iterative treatment of the larger to
  limit the stack depth,
- Switch to a worst-case-safe procedure for deep recursion (introsort idea).

The usual array implementation partitions in-place and requires stackframes in the
Average Case O(log n). However, it is usually not stable because far-off elements are
exchanged.

### 12. Heap Sort: the principle

A **Max Heap** holds the largest element at the root. Heap Sort:

1. builds a Max Heap in O(n),
2. replaces the root with the last unsorted element,
3. reduces the heap area and restores the heap property in O(log n),
4. repeats this n-1 times.

This guarantees O(n log n) time, O(1) explicit additional memory with array-based in-
place implementation and no stability. The details of heap setup, tilt-down and priority
queues follow in module 11. Here is important: Heap Sort combines a guaranteed
Quicksort-like space requirement with a worse cache locality and larger constants than
many practical alternatives.

### 13. Counting Sort: Count values instead of compare

The limit Ω(n log n) applies to algorithms that use only comparisons between elements.
If Keys are integers in a small known area, they can be used directly as count indices.

For `[3, 1, 2, 1, 0]`:

```text
Value:        0  1  2  3
Frequency: 1 2 1 1
Issue:    [0, 1, 1, 2, 3]
```

```python
def counting_sort(values: list[int]) -> list[int]:
    """Sort integers, including negatives, by counting their range."""
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    counts = [0] * (maximum - minimum + 1)
    for value in values:
        counts[value - minimum] += 1
    result: list[int] = []
    for offset, count in enumerate(counts):
        result.extend([offset + minimum] * count)
    return result


assert counting_sort([3, 1, 2, 1, 0]) == [0, 1, 1, 2, 3]
assert counting_sort([-2, 3, -2, 0]) == [-2, -2, 0, 3]
```

For n values and range k, counting Sort O(n + k) costs time and O(k) additional memory.
With n = 100 and keys between 0 and 1012, it would be absurd to create an array with a
trillion buckets. The procedure is only useful if k is limited to n.

The compact implementation above sorted naked integers. To maintain a stable order of
associated records, the classic stable variant uses cumulative frequencies and places
elements in a separate output array.

### 14. Bucket Sort: Distribution in intervals

Bucket Sort distributes values in orderly intervals, sort each bucket and chain it. For
evenly distributed floating point numbers in `[0, 1)`, n Buckets can remain very small
in the expected case.

```python
def bucket_sort_unit_interval(values: list[float]) -> list[float]:
    """Sort floats from the half-open interval [0, 1)."""
    if any(value < 0 or value >= 1 for value in values):
        raise ValueError("all values must be in [0, 1)")
    if not values:
        return []
    buckets: list[list[float]] = [[] for _ in values]
    for value in values:
        buckets[int(value * len(buckets))].append(value)
    result: list[float] = []
    for bucket in buckets:
        # Small buckets make insertion sort an appropriate local strategy.
        local = bucket.copy()
        for index in range(1, len(local)):
            key = local[index]
            position = index - 1
            while position >= 0 and local[position] > key:
                local[position + 1] = local[position]
                position -= 1
            local[position + 1] = key
        result.extend(local)
    return result


bucket_values = [0.78, 0.12, 0.44, 0.41, 0.05]
assert bucket_sort_unit_interval(bucket_values) == sorted(bucket_values)
```

With appropriate, approximately uniform distribution, the expected runtime O(n) is. If
all values are concentrated in a bucket, the worst case depends on the local procedure
and can be placed with Insertion Sort O(n2). Bucket Sort therefore needs a technically
plausible distribution acceptance.

### 15. Radix Sort: point for point

Radix Sort sorts composite keys in several stable runs. The LSD variant starts at the
lowest value. After each pass, the values are correctly arranged with respect to the
digits processed so far.

For `[170, 45, 75, 90, 802, 24, 2, 66]`:

```text
One:   [170, 90, 802, 2, 24, 45, 75, 66]
Tens: [802, 2, 24, 45, 66, 170, 75, 90]
Hundreds:[2, 24, 45, 66, 75, 90, 170, 802]
```

The stability of each digit pass is indispensable: the order of previously processed
digits must not be destroyed with the same current digits.

```python
def radix_sort_nonnegative(values: list[int]) -> list[int]:
    """Sort nonnegative integers with stable base-10 LSD passes."""
    if any(value < 0 for value in values):
        raise ValueError("radix sort example accepts only nonnegative integers")
    result = values.copy()
    maximum = max(result, default=0)
    place = 1
    while maximum // place > 0:
        buckets: list[list[int]] = [[] for _ in range(10)]
        for value in result:
            digit = (value // place) % 10
            buckets[digit].append(value)
        result = [value for bucket in buckets for value in bucket]
        place *= 10
    return result


radix_values = [170, 45, 75, 90, 802, 24, 2, 66]
assert radix_sort_nonnegative(radix_values) == sorted(radix_values)
```

For d digits and base b, Radix Sort O(d(n + b)) costs time. The information is only
better than O(n log n) if d and b are appropriately limited. Strings of fixed length or
integer with known bit width are typical application fields.

---

## Level 3: Formalisation

### 16. Correctness through invariants

A loop invariant describes a statement that:

1. before the first iteration,
2. is preserved by any iteration; and
3. together with the termination condition provides the desired condition.

| Procedure | central invariant |
|---|---|
| Bubble | The finished suffix contains the largest elements finally sorted. |
| Selection | The prefix contains the smallest elements finally sorted. |
| Insertion | The prefix is a sorted permutation of the input prefix. |
| Merge | The output always contains the smallest elements already viewed. |
| Quicksort partition | To the left of the limit are only values `<= pivot`. |

Invariants not only explain **that** an algorithm works. They help implement: Every
mutation must receive the statement.

### 17. Complexity table

| Procedure | Best | Average | Worst | Add-on memory | Stable | typical in-place |
|---|---:|---:|---:|---:|---|---|
| Bubble with demolition | O(n) | O(n2) | O(n2) | O(1) | Yes | Yes |
| Selection | O(n2) | O(n2) | O(n2) | O(1) | yes | Yes |
| Insertion | O(n) | O(n2) | O(n2) | O(1) | Yes | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) for arrays | Yes | yes |
| Quicksort | O(n log n) | O(n log n) | O(n2) | O(log n) average stack | yes | Yes |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | yes | Yes |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) to O(n+k) | possible | yes |
| Bucket Sort | O(n) expected | Depending on distribution | O(n2) | O(n+k) | Locally dependent | yes |
| Radix Sort | O(d(n+b)) | O(d(n+b)) | O(d(n+b)) | O(n+b) | yes with stable passports | yes |

Table values are not a substitute for assumptions. `k` is the width of the value range
for counting Sort, `d` the number number and `b` the base for the radix Sort.

### 18. Stability in concrete terms

If a database sort lines by department after they have already been sorted by name, the
name order should often be retained within the same departments. This is achieved with a
stable second sort.

```python
employees = [
    {"name": "Ada", "team": "B"},
    {"name": "Barbara", "team": "A"},
    {"name": "Edsger", "team": "B"},
    {"name": "Grace", "team": "A"},
]
by_name = sorted(employees, key=lambda row: row["name"])
by_team_then_name = sorted(by_name, key=lambda row: row["team"])
assert [row["name"] for row in by_team_then_name] == [
    "Barbara",
    "Grace",
    "Ada",
    "Edsger",
]
```

Stability only applies to elements with the same sorting key. She doesn't say anything
about whether different keys are correctly arranged.

An unstable procedure can be stabilized by decorating with the original index, but pays
for additional memory and an extended comparison key.

### 19. Specify in-place

"In-place" is not used identically everywhere. Usually it means O(1) extra memory
outside the input. Recursive algorithms need stackframes:

- Insertion and Selection Sort: O(1) Additional memory.
- Classic In-Place Quicksort: O(log n) Stack in Average Case, O(n) in Worst Case.
- the merge-sort array variant shown: O(n) temporary memory plus recursion.

A function can also create a copy, although its internal algorithm works in-place. Our
learning implementations often intentionally return a copy so that tests can check the
unchanged input. The property profile of the internal process must be considered
separately.

### 20. Why comparison sorting needs Ω(n log n)

A comparison-based algorithm receives only one binary information per yes/no comparison.
There are possible sequences for n different elements `n!`. The algorithm must be able
to distinguish everyone.

You can see all possible comparison processes as a binary decision tree:

- Every inner node is a comparison.
- Each edge is one of the two results.
- Each sheet represents a possible input order.

A binary tree of height h has at most `2^h` leaves. It is therefore necessary to apply:

```text
2^h=n!
h >= log2(n!)
```

And `log₂(n!)` grows in Θ (n log n). Intuitive already contains the product of the
largest n/2 factors at least `(n/2)^(n/2)`:

```text
N! >= (n/2)^(n/2)
log2(n!) >=(n/2) * log2(N/2) = Ω(n log n)
```

```python
from math import factorial, log2


for decision_size in (4, 8, 16):
    required_depth = log2(factorial(decision_size))
    reference_growth = decision_size * log2(decision_size)
    print(
        f"n={decision_size:2d}: log2(n!)={required_depth:7.2f}, "
        f"n*log2(n)={reference_growth:7.2f}"
    )
    assert required_depth <= reference_growth
```

The limit applies to the worst case of any general comparison sorting. Counting or Radix
Sort do not contradict it: they use additional information about keys, not just pair-by-
pair comparisons. Almost sorted entries also form a limited problem class on which
adaptive procedures can be linear.

### 21. Timsort: what Python actually uses

`list.sort()` and `sorted()` use **Timsort**, a stable hybrid method. It was designed
for real data that often contain ordered sub-sequences.

Conceptually:

1. Timsort detects ascending or descending **Runs**.
2. Descendant runs are turned over without injuring stability on the same keys.
3. Short runs are extended with an insertion-sort-like technique.
4. Runs are brought together in a stable manner according to size rules.
5. For long winning series of a merge page, "Galloping" can look block by block instead
   of always taking over only one element.

The result combines:

- O(n) Best case for highly pre-sorted data,
- O(n log n) Worst Case,
- Stability and
- O(n) temporary memory in the worst case.

Timsort is not a simple change "with small n insertion, otherwise merge". Its run
detection and merge invariants are the core of adaptability.

#### `sorted()` and `list.sort()`

- `sorted(iterable)` always creates a new list and accepts each iterable.
- `list.sort()` modifies a list in-place and intentionally returns `None`.
- Both accept `key=` and `reverse=`.
- The key function is calculated once per element and its result is used internally for
  sorting.
- Both are stable.

```python
words = ["pear", "fig", "banana", "kiwi", "plum"]
by_length = sorted(words, key=len)
assert by_length == ["fig", "pear", "kiwi", "plum", "banana"]
assert words == ["pear", "fig", "banana", "kiwi", "plum"]

mutable_words = words.copy()
sort_return = mutable_words.sort(key=len, reverse=True)
assert sort_return is None
assert mutable_words == ["banana", "pear", "kiwi", "plum", "fig"]
```

If the length is equal, the original order is retained. For several criteria, a Tuple
key is often clearer:

```python
records = [
    {"priority": 2, "timestamp": 10},
    {"priority": 1, "timestamp": 30},
    {"priority": 1, "timestamp": 20},
]
ordered_records = sorted(
    records,
    key=lambda row: (row["priority"], row["timestamp"]),
)
assert [(row["priority"], row["timestamp"]) for row in ordered_records] == [
    (1, 20),
    (1, 30),
    (2, 10),
]
```

### 22. Scenarios to decide well-founded

#### Small or almost sorted arrays

Insertion Sort is simple, cache-friendly and adaptive. Even large sorting methods often
use it for small areas.

#### Guaranteed worst case and stability

Merge Sort provides O(n log n) and stability, but requires O(n) additional memory for
arrays. Merge can be particularly natural for linked lists.

#### Average fast in-place sorting

Well implemented Quicksort has small constants and good cache locality. Pivot strategy
and protection against degenerated recursion are crucial.

#### Many duplicates

3-way-quicksort can efficiently isolate equality areas. Counting Sort is attractive if
the duplicates come from a small integer area at the same time.

#### Limited Integrity

Counting Sort costs O(n+k). It is excellent when k is small enough, but unusable with
huge, thinly occupied range of values.

#### Fixed number/byte width

Radix Sort can sort integers or fixed strings in linearly many stable passes.
Representation and base determine memory and constants.

#### Unknown real python data

`sorted()` or `.sort()` are usually the right choice. Timsort is stable, robust and uses
existing runs. Self-implementations serve the understanding or special domain
conditions, not the replacement of the standard library.

### 23. Typical Errors

#### Overlapping Quicksort Areas

After partitioning at index p, the recursive areas are `[low, p-1]` and `[p+1, high]`.
If p is included again, the recursion can stagnate.

#### Merge loses leftovers

Once one half is empty, all remaining elements of the other half must be attached.

#### Destroy stability by wrong comparison

In the merge, the left half must be chosen first. In the case of insertion only strictly
larger elements are moved.

#### Test random entries only

Important test forms are: empty, one element, already sorted, vice versa, identical
values, many duplicates, negative values and sizes just around division boundaries.

#### Measure runtimes without result check

A false algorithm can be particularly fast. Each benchmark must first check against a
trusted reference such as `sorted()`.

#### Use Counting Sort at large value range

Not n alone, but `maximum - minimum + 1` determines the memory.

### 24. Small empirical sample of growth

The following measurement is not a precise benchmark suite, but makes the distance
between O(n2) and O(n log n) visible. A fixed seed ensures comparable input.

```python
import random
from time import perf_counter


def timed(function, values: list[int]) -> float:
    """Validate one sorting run and return elapsed seconds."""
    expected = sorted(values)
    start = perf_counter()
    result = function(values)
    elapsed = perf_counter() - start
    assert result == expected
    return elapsed


growth_rng = random.Random(707)
for growth_size in (100, 200, 400):
    growth_values = [growth_rng.randrange(10_000) for _ in range(growth_size)]
    insertion_time = timed(insertion_sort, growth_values)
    merge_time = timed(merge_sort, growth_values)
    quick_time = timed(quick_sort, growth_values)
    print(
        f"n={growth_size:4d}: insertion={insertion_time:.6f}s, "
        f"merge={merge_time:.6f}s, quick={quick_time:.6f}s"
    )
```

For reliable statements it takes warm-ups, repetitions, several input types and
sufficiently large measuring times. This is exactly what is being implemented in the
Sort Benchmark project.

### 25. Self-control

You've understood the core ideas if you can explain these questions:

1. Which two properties must each sorting output fulfil?
2. Why is insertion Sort on almost sorted data often fast?
3. Which invariant makes Selection Sort correct?
4. Why does Merge Sort stay in Worst Case O(n log n)?
5. How does a bad pivot choice O(n2) create at Quicksort?
6. What does stability mean on two data sets with the same key?
7. Why does Ω(n log n) not apply to Counting Sort?
8. When is the space requirement of Counting Sort unacceptable?
9. Why does LSD Radix Sort require stable number passes?
10. What real data structure does Timsort use?
11. Why is Heap Sort treated here only as a principle and only in module 11 in detail?
12. What procedure would you choose for 10 million status codes in the range 100 to 599,
    and what is the reason for the choice?

---

## Executive summary

Sorting algorithms solve the same task with fundamentally different paradigms. Bubble,
Selection and Insertion Sort make local invariants visible, but are quadratic in the
worst case. Merge Sort guarantees O(n log n) by balancing and linear merge. Quicksort
usually partitions very quickly in-place, but retains O(n2) as a worst case without
protection. Heap Sort guarantees O(n log n) with constant additional memory and is
constructively deepened in module 11.

The limit Ω(n log n) intuitively follows that comparison n! to distinguish possible
input sequences. Counting, Bucket and Radix Sort are allowed to bypass them because they
know more about keys and distribution than a general comparison algorithm. These
additional assumptions also determine their limits.

Stability, in-place property, adaptability, worst case and value range are not side
details, but part of the algorithm choice. With Timsort, Python uses a stable, adaptive
hybrid method that detects existing runs. In production code `sorted()` is almost always
the starting point; the self-implementations provide the model with which you can assess
its strengths and alternatives professionally.

---

# Deutsche Fassung

# Modul 07: Sortieralgorithmen

Sortieren ist das Lehrbuchproblem schlechthin. Nicht weil moderne Programme
jeden Sortieralgorithmus selbst implementieren sollten, sondern weil sich an
derselben klaren Aufgabe sehr unterschiedliche Ideen vergleichen lassen:
lokale Vertauschungen, wiederholte Auswahl, inkrementeller Aufbau, Divide and
Conquer, Partitionierung und das Ausnutzen zusätzlicher Informationen über den
Wertebereich.

Dieses Skript entwickelt Sortierverfahren in drei Ebenen:

1. **Intuition:** Warum eine Ordnung Folgeprobleme vereinfacht und welche
   Eigenschaften neben der Laufzeit zählen.
2. **Simulation:** Bubble, Selection, Insertion, Merge und Quicksort werden auf
   kleinen Arrays Schritt für Schritt durchlaufen. Counting, Bucket und Radix
   Sort zeigen, wann die Vergleichsschranke umgangen werden darf.
3. **Formalisierung:** Invarianten, Korrektheitsargumente, Stabilität,
   Speicherbedarf, Worst Cases, die Schranke Ω(n log n) und Pythons Timsort.

Nach dem Modul sollst du mindestens fünf Verfahren implementieren, ihre
Eigenschaften begründen und für fast sortierte Daten, viele Duplikate oder einen
begrenzten Wertebereich eine passende Wahl treffen können.

---

## Ebene 1: Intuition

### 1. Warum Sortieren so viele Probleme vorbereitet

Eine unsortierte Folge enthält dieselben Informationen wie ihre sortierte
Version, aber ihre Struktur ist schwerer nutzbar. Nach dem Sortieren werden viele
Aufgaben einfacher:

- Binäre Suche reduziert eine Suche von O(n) auf O(log n).
- Duplikate stehen nebeneinander und lassen sich in einem Durchlauf erkennen.
- Minimum, Maximum und Quantile liegen an berechenbaren Positionen.
- Zwei sortierte Datenströme können linear zusammengeführt werden.
- Bereichsanfragen und Rangfolgen werden direkt zugänglich.

Sortieren ist daher oft keine Endaufgabe, sondern eine einmalige Vorverarbeitung,
die viele spätere Operationen beschleunigt.

### 2. Was bedeutet „sortiert“?

Für eine aufsteigende Folge `a` gilt:

```text
a[0] ≤ a[1] <=... ≤ a[n - 1]
```

Ein korrekter Sortieralgorithmus muss zwei Bedingungen erfüllen:

1. **Ordnung:** Die Ausgabe ist gemäß der gewünschten Relation sortiert.
2. **Permutation:** Die Ausgabe enthält exakt dieselben Elemente mit denselben
   Häufigkeiten wie die Eingabe.

Nur die Ordnung zu prüfen reicht nicht. Ein fehlerhafter Algorithmus könnte aus
`[3, 1, 2]` einfach `[1, 2]` machen; die Folge wäre sortiert, aber ein Element
wäre verloren.

```python
from collections import Counter


def is_sorted(values: list[int]) -> bool:
    """Return whether values are in nondecreasing order."""
    return all(left <= right for left, right in zip(values, values[1:]))


def is_sorted_permutation(original: list[int], result: list[int]) -> bool:
    """Check both required properties of a sorting result."""
    return is_sorted(result) and Counter(original) == Counter(result)


assert is_sorted([1, 1, 3, 8])
assert not is_sorted([1, 4, 2])
assert not is_sorted_permutation([3, 1, 2], [1, 2])
assert is_sorted_permutation([3, 1, 2, 1], [1, 1, 2, 3])
```

### 3. Mehr als Big-O: das Eigenschaftsprofil

Zwei Verfahren mit derselben asymptotischen Laufzeit können sich praktisch stark
unterscheiden. Für eine fundierte Wahl sind mindestens diese Fragen wichtig:

- **Worst/Average/Best Case:** Welche Eingabeform löst welche Kosten aus?
- **Stabilität:** Bleibt die relative Reihenfolge gleichwertiger Elemente
  erhalten?
- **In-Place:** Benötigt das Verfahren nur konstanten oder logarithmischen
  Zusatzspeicher statt eines zweiten Arrays der Größe n?
- **Adaptivität:** Wird eine bereits vorhandene Ordnung ausgenutzt?
- **Vergleichsbasiert:** Lernt der Algorithmus nur durch Vergleiche oder darf er
  Werte als Indizes/Ziffern interpretieren?
- **Datenzugriff:** Arbeitet er cache-freundlich auf zusammenhängendem Speicher?

Es gibt deshalb nicht „den besten Sortieralgorithmus“ ohne Kontext.

### 4. Die Paradigmen auf einen Blick

| Idee | Vertreter | Leitfrage |
|---|---|---|
| Lokale Nachbarreparatur | Bubble Sort | Sind benachbarte Elemente falsch herum? |
| Wiederholte Auswahl | Selection Sort | Welches kleinste Element fehlt im Präfix? |
| Geordnetes Präfix erweitern | Insertion Sort | Wo gehört das nächste Element hinein? |
| Teilen und geordnet verschmelzen | Merge Sort | Wie kombiniere ich zwei sortierte Hälften? |
| Um ein Pivot partitionieren | Quicksort | Was gehört links beziehungsweise rechts? |
| Heap-Eigenschaft nutzen | Heap Sort | Welches Extrem kann ich wiederholt entnehmen? |
| Wertebereich zählen/zerlegen | Counting/Bucket/Radix | Welche Zusatzstruktur der Keys darf ich nutzen? |

---

## Ebene 2: Simulation

### 5. Bubble Sort: große Werte steigen nach rechts

Bubble Sort vergleicht Nachbarn und vertauscht sie, wenn sie in falscher
Reihenfolge stehen. Nach einem vollständigen Durchlauf liegt das größte noch
unsortierte Element an seiner endgültigen Position.

Eingabe: `[5, 2, 4, 1]`

**Pass 1:**

| Vergleich | Aktion | Zustand |
|---|---|---|
| 5 und 2 | tauschen | `[2, 5, 4, 1]` |
| 5 und 4 | tauschen | `[2, 4, 5, 1]` |
| 5 und 1 | tauschen | `[2, 4, 1, 5]` |

Die 5 ist fertig. **Pass 2:**

| Vergleich | Aktion | Zustand |
|---|---|---|
| 2 und 4 | keine | `[2, 4, 1, 5]` |
| 4 und 1 | tauschen | `[2, 1, 4, 5]` |

Die 4 ist fertig. **Pass 3:** 2 und 1 werden getauscht; Ergebnis
`[1, 2, 4, 5]`.

```python
def bubble_sort(values: list[int]) -> list[int]:
    """Return a sorted copy using stable adjacent swaps."""
    result = values.copy()
    for end in range(len(result) - 1, 0, -1):
        swapped = False
        for index in range(end):
            if result[index] > result[index + 1]:
                result[index], result[index + 1] = (
                    result[index + 1],
                    result[index],
                )
                swapped = True
        if not swapped:
            break
    return result


bubble_input = [5, 2, 4, 1]
assert bubble_sort(bubble_input) == [1, 2, 4, 5]
assert bubble_input == [5, 2, 4, 1]
```

Der `swapped`-Marker macht Bubble Sort adaptiv: Bei bereits sortierter Eingabe
endet es nach einem linearen Pass. Ohne diese Abbruchregel wäre auch der Best
Case quadratisch.

**Schleifeninvariante:** Vor jedem neuen Pass ist das Suffix rechts von `end`
sortiert und enthält die größten Elemente an ihren endgültigen Positionen.

### 6. Selection Sort: das Minimum auswählen

Selection Sort teilt das Array in ein fertiges Präfix und einen unsortierten
Rest. Es sucht im Rest das Minimum und tauscht es an die nächste freie Stelle.

Eingabe: `[5, 2, 4, 1]`

| Runde | unsortierter Bereich | Minimum | Tausch | Zustand |
|---:|---|---:|---|---|
| 0 | `[5, 2, 4, 1]` | 1 an Index 3 | Index 0 ↔ 3 | `[1, 2, 4, 5]` |
| 1 | `[2, 4, 5]` | 2 an Index 1 | mit sich selbst | `[1, 2, 4, 5]` |
| 2 | `[4, 5]` | 4 an Index 2 | mit sich selbst | `[1, 2, 4, 5]` |

```python
def selection_sort(values: list[int]) -> list[int]:
    """Return a sorted copy by repeatedly selecting the minimum."""
    result = values.copy()
    for start in range(len(result) - 1):
        minimum_index = start
        for index in range(start + 1, len(result)):
            if result[index] < result[minimum_index]:
                minimum_index = index
        result[start], result[minimum_index] = (
            result[minimum_index],
            result[start],
        )
    return result


assert selection_sort([5, 2, 4, 1]) == [1, 2, 4, 5]
```

Selection Sort führt unabhängig von der Eingabe ungefähr `n(n-1)/2` Vergleiche
aus. Sein Vorteil ist die geringe Anzahl an Tauschen: höchstens n-1. Wenn
Schreiboperationen besonders teuer sind, kann diese Eigenschaft relevant sein.

Der direkte Tausch über große Distanzen macht die Standardvariante instabil.
Ein gleichwertiges Element kann über ein anderes hinwegbewegt werden.

**Schleifeninvariante:** Vor Runde `start` enthält das Präfix
`result[:start]` die `start` kleinsten Elemente in korrekter Reihenfolge.

### 7. Insertion Sort: das nächste Element einordnen

Insertion Sort ähnelt dem Sortieren von Spielkarten auf der Hand. Das linke
Präfix ist bereits sortiert. Das nächste Element wird gespeichert; größere
Präfixelemente rücken nach rechts, bis die Einfügeposition frei ist.

Eingabe: `[5, 2, 4, 1]`

| Key | sortiertes Präfix vorher | Verschiebungen | Zustand danach |
|---:|---|---|---|
| 2 | `[5]` | 5 nach rechts | `[2, 5, 4, 1]` |
| 4 | `[2, 5]` | 5 nach rechts | `[2, 4, 5, 1]` |
| 1 | `[2, 4, 5]` | 5, 4, 2 nach rechts | `[1, 2, 4, 5]` |

```python
def insertion_sort(values: list[int]) -> list[int]:
    """Return a sorted copy by extending a sorted prefix."""
    result = values.copy()
    for index in range(1, len(result)):
        key = result[index]
        position = index - 1
        while position >= 0 and result[position] > key:
            result[position + 1] = result[position]
            position -= 1
        result[position + 1] = key
    return result


assert insertion_sort([5, 2, 4, 1]) == [1, 2, 4, 5]
assert insertion_sort([1, 2, 3, 4]) == [1, 2, 3, 4]
```

Wichtig ist der strikte Vergleich `>` statt `>=`. Gleichwertige Elemente werden
nicht aneinander vorbeigeschoben; dadurch ist Insertion Sort stabil.

Bei fast sortierten Daten gibt es nur wenige **Inversionen**. Eine Inversion ist
ein Indexpaar `(i, j)` mit `i < j`, aber `a[i] > a[j]`. Jede Verschiebung von
Insertion Sort beseitigt eine Inversion. Die Laufzeit lässt sich daher als
O(n + I) ausdrücken, wobei I die Zahl der Inversionen ist. Das erklärt seine gute
Praxisleistung auf kleinen oder fast sortierten Runs.

**Schleifeninvariante:** Vor der Iteration mit `index` ist
`result[:index]` sortiert und enthält genau die ursprünglichen Elemente dieses
Präfixes.

### 8. Die elementaren Verfahren messen

Alle drei Verfahren sind im Worst Case O(n²), reagieren aber verschieden auf
Eingabeformen. Wir zählen Vergleiche statt nur Zeit, damit das Ergebnis nicht vom
Rechner abhängt.

```python
def insertion_comparisons(values: list[int]) -> int:
    """Count key comparisons performed by insertion sort."""
    result = values.copy()
    comparisons = 0
    for index in range(1, len(result)):
        key = result[index]
        position = index - 1
        while position >= 0:
            comparisons += 1
            if result[position] <= key:
                break
            result[position + 1] = result[position]
            position -= 1
        result[position + 1] = key
    return comparisons


comparison_size = 100
sorted_comparisons = insertion_comparisons(list(range(comparison_size)))
reverse_comparisons = insertion_comparisons(list(range(comparison_size, 0, -1)))
print(
    f"Insertion comparisons: sorted={sorted_comparisons}, "
    f"reverse={reverse_comparisons}"
)
assert sorted_comparisons == comparison_size - 1
assert reverse_comparisons == comparison_size * (comparison_size - 1) // 2
```

### 9. Merge Sort: teilen und geordnet verschmelzen

Merge Sort folgt Divide and Conquer:

1. Teile die Folge in zwei Hälften.
2. Sortiere beide Hälften rekursiv.
3. Führe die sortierten Hälften in linearer Zeit zusammen.

Für `[5, 2, 4, 1]` entsteht der Rekursionsbaum:

```text
                [5, 2, 4, 1]
               /            \
          [5, 2]            [4, 1]
          /    \            /    \
        [5]    [2]        [4]    [1]
          \    /            \    /
          [2, 5]            [1, 4]
               \            /
                [1, 2, 4, 5]
```

Der entscheidende Schritt ist Merge. Für `[2, 5]` und `[1, 4]`:

| links vorn | rechts vorn | Ausgabe danach |
|---:|---:|---|
| 2 | 1 | `[1]` |
| 2 | 4 | `[1, 2]` |
| 5 | 4 | `[1, 2, 4]` |
| 5 | leer | `[1, 2, 4, 5]` |

```python
def merge(left: list[int], right: list[int]) -> list[int]:
    """Stably merge two sorted lists."""
    result: list[int] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            result.append(left[left_index])
            left_index += 1
        else:
            result.append(right[right_index])
            right_index += 1
    result.extend(left[left_index:])
    result.extend(right[right_index:])
    return result


def merge_sort(values: list[int]) -> list[int]:
    """Return a stable merge-sorted copy."""
    if len(values) <= 1:
        return values.copy()
    middle = len(values) // 2
    left = merge_sort(values[:middle])
    right = merge_sort(values[middle:])
    return merge(left, right)


assert merge([2, 5], [1, 4]) == [1, 2, 4, 5]
assert merge_sort([5, 2, 4, 1]) == [1, 2, 4, 5]
```

Auf jeder Rekursionsebene werden insgesamt n Elemente gemergt. Es gibt ungefähr
`log₂ n` Ebenen. Daher kostet Merge Sort in Best, Average und Worst Case
O(n log n). Die gezeigte Arrayvariante benötigt O(n) Zusatzspeicher.

Der Vergleich `<=` bevorzugt bei Gleichheit das Element aus der linken Hälfte.
Da linke Elemente in der Eingabe früher standen, bleibt die relative Reihenfolge
gleichwertiger Elemente erhalten: Merge Sort ist stabil.

### 10. Quicksort: um ein Pivot partitionieren

Quicksort wählt ein Pivot und partitioniert die Folge so, dass kleinere oder
gleiche Elemente links und größere rechts stehen. Danach werden beide Bereiche
rekursiv sortiert. Anders als Merge Sort erledigt es die entscheidende Arbeit
beim Teilen, nicht beim Zusammenführen.

Wir simulieren die Lomuto-Partition auf
`[4, 2, 7, 3, 1, 6]` mit Pivot 6. `boundary` markiert das Ende des Bereichs mit
Werten `<= pivot`.

| gelesen | Vergleich mit 6 | Aktion | Zustand | boundary |
|---:|---|---|---|---:|
| 4 | kleiner | mit Index 0 tauschen | `[4, 2, 7, 3, 1, 6]` | 1 |
| 2 | kleiner | mit Index 1 tauschen | `[4, 2, 7, 3, 1, 6]` | 2 |
| 7 | größer | keine | `[4, 2, 7, 3, 1, 6]` | 2 |
| 3 | kleiner | mit Index 2 tauschen | `[4, 2, 3, 7, 1, 6]` | 3 |
| 1 | kleiner | mit Index 3 tauschen | `[4, 2, 3, 1, 7, 6]` | 4 |
| Pivot | – | mit boundary tauschen | `[4, 2, 3, 1, 6, 7]` | Pivotindex 4 |

Die 6 steht nun endgültig. Links liegen nur Werte `<= 6`, rechts nur Werte `> 6`.
Die linke Seite ist noch nicht vollständig sortiert.

```python
def lomuto_partition(values: list[int], low: int, high: int) -> int:
    """Partition values[low:high+1] and return the final pivot index."""
    pivot = values[high]
    boundary = low
    for index in range(low, high):
        if values[index] <= pivot:
            values[boundary], values[index] = values[index], values[boundary]
            boundary += 1
    values[boundary], values[high] = values[high], values[boundary]
    return boundary


def _quick_sort_range(values: list[int], low: int, high: int) -> None:
    if low >= high:
        return
    pivot_index = lomuto_partition(values, low, high)
    _quick_sort_range(values, low, pivot_index - 1)
    _quick_sort_range(values, pivot_index + 1, high)


def quick_sort(values: list[int]) -> list[int]:
    """Return an in-place-partitioned quick-sorted copy."""
    result = values.copy()
    _quick_sort_range(result, 0, len(result) - 1)
    return result


partition_example = [4, 2, 7, 3, 1, 6]
partition_index = lomuto_partition(partition_example, 0, len(partition_example) - 1)
assert partition_example == [4, 2, 3, 1, 6, 7]
assert partition_index == 4
assert quick_sort([5, 2, 4, 1]) == [1, 2, 4, 5]
```

**Partition-Invariante:** Vor jeder Iteration enthält der Bereich links von
`boundary` nur Werte `<= pivot`; zwischen `boundary` und dem aktuellen Index
liegen bereits geprüfte Werte `> pivot`.

Bei ungefähr halbierenden Pivots hat der Rekursionsbaum O(log n) Ebenen, auf
jeder werden insgesamt O(n) Elemente partitioniert: Average Case O(n log n).

### 11. Pivot-Wahl und Quicksorts Worst Case

Wird immer das letzte Element als Pivot verwendet, ist eine bereits sortierte
Folge fatal:

```text
n Elements -> Partitions n-1 and 0
n-1        -> Partitions n-2 and 0
...
```

Die Arbeit ist `(n-1) + (n-2) + ... + 1 = O(n²)`, und die Rekursionstiefe wird
O(n). In Python kann das sogar das Rekursionslimit erreichen.

Typische Gegenmaßnahmen:

- zufälliges Pivot,
- Median-of-three aus erstem, mittlerem und letztem Wert,
- 3-Wege-Partitionierung in `< pivot`, `== pivot`, `> pivot` bei vielen
  Duplikaten,
- Rekursion zuerst auf der kleineren Partition und iterative Behandlung der
  größeren, um die Stacktiefe zu begrenzen,
- Wechsel zu einem Worst-Case-sicheren Verfahren bei zu tiefer Rekursion
  (Introsort-Idee).

Die übliche Arrayimplementierung partitioniert in-place und benötigt im Average
Case O(log n) Stackframes. Sie ist aber normalerweise nicht stabil, weil weit
voneinander entfernte Elemente getauscht werden.

### 12. Heap Sort: das Prinzip

Ein **Max Heap** hält das größte Element an der Wurzel. Heap Sort:

1. baut in O(n) einen Max Heap,
2. tauscht die Wurzel mit dem letzten unsortierten Element,
3. verkleinert den Heapbereich und stellt die Heap-Eigenschaft in O(log n)
   wieder her,
4. wiederholt dies n-1 Mal.

Damit entstehen garantiert O(n log n) Zeit, O(1) expliziter Zusatzspeicher bei
arraybasierter In-Place-Implementierung und keine Stabilität. Die Details von
Heapaufbau, Sift-down und Priority Queues folgen in Modul 11. Hier ist wichtig:
Heap Sort verbindet einen garantierten Quicksort-ähnlichen Speicherbedarf mit
einer schlechteren Cache-Lokalität und größeren Konstanten als viele praktische
Alternativen.

### 13. Counting Sort: Werte zählen statt vergleichen

Die Schranke Ω(n log n) gilt für Algorithmen, die nur Vergleiche zwischen
Elementen verwenden. Sind Keys ganze Zahlen in einem kleinen bekannten Bereich,
können sie direkt als Zählindizes dienen.

Für `[3, 1, 2, 1, 0]`:

```text
Value:        0  1  2  3
Frequency: 1 2 1 1
Issue:    [0, 1, 1, 2, 3]
```

```python
def counting_sort(values: list[int]) -> list[int]:
    """Sort integers, including negatives, by counting their range."""
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    counts = [0] * (maximum - minimum + 1)
    for value in values:
        counts[value - minimum] += 1
    result: list[int] = []
    for offset, count in enumerate(counts):
        result.extend([offset + minimum] * count)
    return result


assert counting_sort([3, 1, 2, 1, 0]) == [0, 1, 1, 2, 3]
assert counting_sort([-2, 3, -2, 0]) == [-2, -2, 0, 3]
```

Für n Werte und Wertebereichsbreite k kostet Counting Sort O(n + k) Zeit und
O(k) Zusatzspeicher. Bei n = 100 und Keys zwischen 0 und 10¹² wäre es absurd,
ein Array mit einer Billion Buckets anzulegen. Das Verfahren ist nur sinnvoll,
wenn k im Verhältnis zu n begrenzt ist.

Die kompakte Implementierung oben sortiert nackte Integer. Soll eine stabile
Reihenfolge zugehöriger Datensätze erhalten bleiben, verwendet die klassische
stabile Variante kumulierte Häufigkeiten und platziert Elemente in ein separates
Ausgabearray.

### 14. Bucket Sort: Verteilung in Intervalle

Bucket Sort verteilt Werte in geordnete Intervalle, sortiert jeden Bucket und
verkettet sie. Für gleichmäßig verteilte Fließkommazahlen in `[0, 1)` können n
Buckets im erwarteten Fall sehr klein bleiben.

```python
def bucket_sort_unit_interval(values: list[float]) -> list[float]:
    """Sort floats from the half-open interval [0, 1)."""
    if any(value < 0 or value >= 1 for value in values):
        raise ValueError("all values must be in [0, 1)")
    if not values:
        return []
    buckets: list[list[float]] = [[] for _ in values]
    for value in values:
        buckets[int(value * len(buckets))].append(value)
    result: list[float] = []
    for bucket in buckets:
        # Small buckets make insertion sort an appropriate local strategy.
        local = bucket.copy()
        for index in range(1, len(local)):
            key = local[index]
            position = index - 1
            while position >= 0 and local[position] > key:
                local[position + 1] = local[position]
                position -= 1
            local[position + 1] = key
        result.extend(local)
    return result


bucket_values = [0.78, 0.12, 0.44, 0.41, 0.05]
assert bucket_sort_unit_interval(bucket_values) == sorted(bucket_values)
```

Bei geeigneter, ungefähr gleichmäßiger Verteilung ist die erwartete Laufzeit
O(n). Konzentrieren sich alle Werte in einem Bucket, hängt der Worst Case vom
lokalen Verfahren ab und kann mit Insertion Sort O(n²) werden. Bucket Sort
benötigt daher eine fachlich plausible Verteilungsannahme.

### 15. Radix Sort: Ziffer für Ziffer

Radix Sort sortiert zusammengesetzte Keys in mehreren stabilen Durchläufen. Die
LSD-Variante beginnt bei der niedrigstwertigen Ziffer. Nach jedem Pass sind die
Werte bezüglich der bisher verarbeiteten Ziffern korrekt geordnet.

Für `[170, 45, 75, 90, 802, 24, 2, 66]`:

```text
One:   [170, 90, 802, 2, 24, 45, 75, 66]
Tens: [802, 2, 24, 45, 66, 170, 75, 90]
Hundreds:[2, 24, 45, 66, 75, 90, 170, 802]
```

Die Stabilität jedes Ziffernpasses ist unverzichtbar: Die Ordnung früher
verarbeiteter Ziffern darf bei gleichen aktuellen Ziffern nicht zerstört werden.

```python
def radix_sort_nonnegative(values: list[int]) -> list[int]:
    """Sort nonnegative integers with stable base-10 LSD passes."""
    if any(value < 0 for value in values):
        raise ValueError("radix sort example accepts only nonnegative integers")
    result = values.copy()
    maximum = max(result, default=0)
    place = 1
    while maximum // place > 0:
        buckets: list[list[int]] = [[] for _ in range(10)]
        for value in result:
            digit = (value // place) % 10
            buckets[digit].append(value)
        result = [value for bucket in buckets for value in bucket]
        place *= 10
    return result


radix_values = [170, 45, 75, 90, 802, 24, 2, 66]
assert radix_sort_nonnegative(radix_values) == sorted(radix_values)
```

Bei d Ziffern und Basis b kostet Radix Sort O(d(n + b)) Zeit. Die Angabe ist nur
dann besser als O(n log n), wenn d und b passend begrenzt sind. Strings fester
Länge oder Integer mit bekannter Bitbreite sind typische Anwendungsfelder.

---

## Ebene 3: Formalisierung

### 16. Korrektheit durch Invarianten

Eine Schleifeninvariante beschreibt eine Aussage, die:

1. vor der ersten Iteration gilt,
2. durch jede Iteration erhalten bleibt und
3. zusammen mit der Abbruchbedingung die gewünschte Nachbedingung liefert.

| Verfahren | zentrale Invariante |
|---|---|
| Bubble | Das fertige Suffix enthält die größten Elemente endgültig sortiert. |
| Selection | Das Präfix enthält die kleinsten Elemente endgültig sortiert. |
| Insertion | Das Präfix ist eine sortierte Permutation des Eingabepräfixes. |
| Merge | Die Ausgabe enthält stets die kleinsten bereits betrachteten Elemente. |
| Quicksort-Partition | Links der Grenze liegen nur Werte `<= pivot`. |

Invarianten erklären nicht nur, **dass** ein Algorithmus funktioniert. Sie helfen
beim Implementieren: Jede Mutation muss die Aussage erhalten.

### 17. Komplexitätstabelle

| Verfahren | Best | Average | Worst | Zusatzspeicher | stabil | typisch in-place |
|---|---:|---:|---:|---:|---|---|
| Bubble mit Abbruch | O(n) | O(n²) | O(n²) | O(1) | ja | ja |
| Selection | O(n²) | O(n²) | O(n²) | O(1) | nein | ja |
| Insertion | O(n) | O(n²) | O(n²) | O(1) | ja | ja |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) bei Arrays | ja | nein |
| Quicksort | O(n log n) | O(n log n) | O(n²) | O(log n) average Stack | nein | ja |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | nein | ja |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) bis O(n+k) | möglich | nein |
| Bucket Sort | O(n) erwartet | abhängig von Verteilung | O(n²) | O(n+k) | abhängig lokal | nein |
| Radix Sort | O(d(n+b)) | O(d(n+b)) | O(d(n+b)) | O(n+b) | ja bei stabilen Pässen | nein |

Tabellenwerte sind kein Ersatz für Annahmen. `k` ist beim Counting Sort die
Breite des Wertebereichs, `d` die Ziffernzahl und `b` die Basis beim Radix Sort.

### 18. Stabilität konkret

Sortiert eine Datenbank Zeilen nach Abteilung, nachdem sie bereits nach Name
sortiert wurden, soll innerhalb gleicher Abteilungen oft die Namensreihenfolge
erhalten bleiben. Das gelingt mit einem stabilen zweiten Sort.

```python
employees = [
    {"name": "Ada", "team": "B"},
    {"name": "Barbara", "team": "A"},
    {"name": "Edsger", "team": "B"},
    {"name": "Grace", "team": "A"},
]
by_name = sorted(employees, key=lambda row: row["name"])
by_team_then_name = sorted(by_name, key=lambda row: row["team"])
assert [row["name"] for row in by_team_then_name] == [
    "Barbara",
    "Grace",
    "Ada",
    "Edsger",
]
```

Stabilität betrifft nur Elemente mit gleichem Sortierschlüssel. Sie sagt nichts
darüber aus, ob verschiedene Keys korrekt geordnet sind.

Ein instabiles Verfahren kann durch Dekorieren mit dem ursprünglichen Index
stabilisiert werden, bezahlt dafür aber zusätzlichen Speicher und einen
erweiterten Vergleichsschlüssel.

### 19. In-Place präzisieren

„In-place“ wird nicht überall identisch verwendet. Meist bedeutet es O(1)
zusätzlichen Speicher außerhalb der Eingabe. Rekursive Algorithmen benötigen
aber Stackframes:

- Insertion und Selection Sort: O(1) Zusatzspeicher.
- klassischer In-Place-Quicksort: O(log n) Stack im Average Case, O(n) im Worst
  Case.
- die gezeigte Merge-Sort-Arrayvariante: O(n) temporärer Speicher plus Rekursion.

Eine Funktion kann außerdem eine Kopie anlegen, obwohl ihr innerer Algorithmus
in-place arbeitet. Unsere Lernimplementierungen geben häufig absichtlich eine
Kopie zurück, damit Tests die unveränderte Eingabe prüfen können. Das
Eigenschaftsprofil des inneren Verfahrens muss davon getrennt betrachtet werden.

### 20. Warum Vergleichssortierung Ω(n log n) braucht

Ein vergleichsbasierter Algorithmus erfährt pro Ja/Nein-Vergleich höchstens eine
binäre Information. Für n verschiedene Elemente existieren `n!` mögliche
Reihenfolgen. Der Algorithmus muss jede unterscheiden können.

Man kann alle möglichen Vergleichsverläufe als binären Entscheidungsbaum sehen:

- Jeder innere Knoten ist ein Vergleich.
- Jede Kante ist eines der zwei Ergebnisse.
- Jedes Blatt steht für eine mögliche Eingabereihenfolge.

Ein binärer Baum der Höhe h besitzt höchstens `2^h` Blätter. Daher muss gelten:

```text
2^h=n!
h >= log2(n!)
```

Und `log₂(n!)` wächst in Θ(n log n). Intuitiv enthält schon das Produkt der
größten n/2 Faktoren mindestens `(n/2)^(n/2)`:

```text
N! >= (n/2)^(n/2)
log2(n!) >=(n/2) * log2(N/2) = Ω(n log n)
```

```python
from math import factorial, log2


for decision_size in (4, 8, 16):
    required_depth = log2(factorial(decision_size))
    reference_growth = decision_size * log2(decision_size)
    print(
        f"n={decision_size:2d}: log2(n!)={required_depth:7.2f}, "
        f"n*log2(n)={reference_growth:7.2f}"
    )
    assert required_depth <= reference_growth
```

Die Schranke gilt für den Worst Case jeder allgemeinen Vergleichssortierung.
Counting oder Radix Sort widersprechen ihr nicht: Sie verwenden zusätzliche
Information über Keys, nicht nur paarweise Vergleiche. Auch fast sortierte
Eingaben bilden eine eingeschränkte Problemklasse, auf der adaptive Verfahren
linear sein können.

### 21. Timsort: was Python tatsächlich nutzt

`list.sort()` und `sorted()` verwenden **Timsort**, ein stabiles hybrides
Verfahren. Es wurde für reale Daten entworfen, die häufig bereits geordnete
Teilfolgen enthalten.

Konzeptionell:

1. Timsort erkennt auf- oder absteigende **Runs**.
2. Absteigende Runs werden umgedreht, ohne Stabilität bei gleichen Keys zu
   verletzen.
3. Kurze Runs werden mit einer Insertion-Sort-ähnlichen Technik erweitert.
4. Runs werden nach Größenregeln stabil zusammengeführt.
5. Bei langen Gewinnserien einer Merge-Seite kann „Galloping“ blockweise suchen
   statt immer nur ein Element zu übernehmen.

Das Ergebnis kombiniert:

- O(n) Best Case auf stark vorsortierten Daten,
- O(n log n) Worst Case,
- Stabilität und
- O(n) temporären Speicher im Worst Case.

Timsort ist kein simpler Wechsel „bei kleinem n Insertion, sonst Merge“. Seine
Run-Erkennung und Merge-Invarianten sind der Kern der Adaptivität.

#### `sorted()` und `list.sort()`

- `sorted(iterable)` erzeugt immer eine neue Liste und akzeptiert jedes Iterable.
- `list.sort()` verändert eine Liste in-place und liefert absichtlich `None`.
- Beide akzeptieren `key=` und `reverse=`.
- Die Key-Funktion wird pro Element einmal berechnet und ihr Ergebnis intern für
  die Sortierung genutzt.
- Beide sind stabil.

```python
words = ["pear", "fig", "banana", "kiwi", "plum"]
by_length = sorted(words, key=len)
assert by_length == ["fig", "pear", "kiwi", "plum", "banana"]
assert words == ["pear", "fig", "banana", "kiwi", "plum"]

mutable_words = words.copy()
sort_return = mutable_words.sort(key=len, reverse=True)
assert sort_return is None
assert mutable_words == ["banana", "pear", "kiwi", "plum", "fig"]
```

Bei Gleichstand der Länge bleibt die ursprüngliche Reihenfolge erhalten. Für
mehrere Kriterien ist oft ein Tuple-Key klarer:

```python
records = [
    {"priority": 2, "timestamp": 10},
    {"priority": 1, "timestamp": 30},
    {"priority": 1, "timestamp": 20},
]
ordered_records = sorted(
    records,
    key=lambda row: (row["priority"], row["timestamp"]),
)
assert [(row["priority"], row["timestamp"]) for row in ordered_records] == [
    (1, 20),
    (1, 30),
    (2, 10),
]
```

### 22. Szenarien begründet entscheiden

#### Kleine oder fast sortierte Arrays

Insertion Sort ist einfach, cache-freundlich und adaptiv. Selbst große
Sortierverfahren verwenden es häufig für kleine Teilbereiche.

#### Garantierter Worst Case und Stabilität

Merge Sort bietet O(n log n) und Stabilität, benötigt bei Arrays aber O(n)
Zusatzspeicher. Für Linked Lists kann Merge besonders natürlich sein.

#### Durchschnittlich schnelle In-Place-Sortierung

Gut implementierter Quicksort hat kleine Konstanten und gute Cache-Lokalität.
Pivotstrategie und Schutz vor degenerierter Rekursion sind entscheidend.

#### Viele Duplikate

3-Wege-Quicksort kann Gleichheitsbereiche effizient isolieren. Counting Sort ist
attraktiv, wenn die Duplikate zugleich aus einem kleinen Integerbereich stammen.

#### Begrenzter Integerbereich

Counting Sort kostet O(n+k). Es ist hervorragend, wenn k klein genug ist, aber
unbrauchbar bei riesigem dünn besetztem Wertebereich.

#### Feste Ziffern-/Bytebreite

Radix Sort kann Integer oder feste Strings in linear vielen stabilen Pässen
sortieren. Repräsentation und Basis bestimmen Speicher und Konstanten.

#### Unbekannte reale Python-Daten

`sorted()` beziehungsweise `.sort()` sind normalerweise die richtige Wahl.
Timsort ist stabil, robust und nutzt vorhandene Runs. Eigenimplementierungen
dienen dem Verständnis oder speziellen Domänenbedingungen, nicht dem Ersatz der
Standardbibliothek.

### 23. Typische Fehler

#### Quicksort-Bereiche überlappen

Nach Partitionierung am Index p lauten die rekursiven Bereiche `[low, p-1]` und
`[p+1, high]`. Wird p erneut eingeschlossen, kann die Rekursion stagnieren.

#### Merge verliert Reste

Sobald eine Hälfte leer ist, müssen alle verbleibenden Elemente der anderen
Hälfte angehängt werden.

#### Stabilität durch falschen Vergleich zerstören

Beim Merge muss bei Gleichheit die linke Hälfte zuerst gewählt werden. Bei
Insertion werden nur strikt größere Elemente verschoben.

#### Nur zufällige Eingaben testen

Wichtige Testformen sind: leer, ein Element, bereits sortiert, umgekehrt,
identische Werte, viele Duplikate, negative Werte und Größen knapp um
Teilungsgrenzen.

#### Laufzeiten ohne Ergebnisprüfung messen

Ein falscher Algorithmus kann besonders schnell sein. Jeder Benchmark muss
zuerst gegen eine vertrauenswürdige Referenz wie `sorted()` prüfen.

#### Counting Sort bei großem Wertebereich einsetzen

Nicht n allein, sondern `maximum - minimum + 1` bestimmt den Speicher.

### 24. Kleine empirische Wachstumsprobe

Die folgende Messung ist keine präzise Benchmark-Suite, macht aber den Abstand
zwischen O(n²) und O(n log n) sichtbar. Ein fixer Seed sorgt für vergleichbare
Eingaben.

```python
import random
from time import perf_counter


def timed(function, values: list[int]) -> float:
    """Validate one sorting run and return elapsed seconds."""
    expected = sorted(values)
    start = perf_counter()
    result = function(values)
    elapsed = perf_counter() - start
    assert result == expected
    return elapsed


growth_rng = random.Random(707)
for growth_size in (100, 200, 400):
    growth_values = [growth_rng.randrange(10_000) for _ in range(growth_size)]
    insertion_time = timed(insertion_sort, growth_values)
    merge_time = timed(merge_sort, growth_values)
    quick_time = timed(quick_sort, growth_values)
    print(
        f"n={growth_size:4d}: insertion={insertion_time:.6f}s, "
        f"merge={merge_time:.6f}s, quick={quick_time:.6f}s"
    )
```

Für belastbare Aussagen braucht es Warm-ups, Wiederholungen, mehrere
Eingabetypen und ausreichend große Messzeiten. Genau das wird im
Sortier-Benchmark-Projekt umgesetzt.

### 25. Selbstkontrolle

Du hast die Kernideen verstanden, wenn du diese Fragen begründen kannst:

1. Welche zwei Eigenschaften muss jede Sortierausgabe erfüllen?
2. Warum ist Insertion Sort auf fast sortierten Daten oft schnell?
3. Welche Invariante macht Selection Sort korrekt?
4. Warum bleibt Merge Sort auch im Worst Case O(n log n)?
5. Wie erzeugt eine schlechte Pivot-Wahl O(n²) bei Quicksort?
6. Was bedeutet Stabilität an zwei Datensätzen mit gleichem Key?
7. Warum gilt Ω(n log n) nicht für Counting Sort?
8. Wann ist der Speicherbedarf von Counting Sort unvertretbar?
9. Warum benötigt LSD Radix Sort stabile Ziffernpässe?
10. Welche reale Datenstruktur nutzt Timsort aus?
11. Weshalb ist Heap Sort hier nur als Prinzip und erst in Modul 11 im Detail
    behandelt?
12. Welches Verfahren würdest du für 10 Millionen Statuscodes im Bereich 100
    bis 599 wählen, und welche Annahme begründet die Wahl?

---

## Zusammenfassung

Sortieralgorithmen lösen dieselbe Aufgabe mit grundverschiedenen Paradigmen.
Bubble, Selection und Insertion Sort machen lokale Invarianten sichtbar, sind im
Worst Case aber quadratisch. Merge Sort garantiert O(n log n) durch balanciertes
Teilen und lineares Mergen. Quicksort partitioniert meist sehr schnell in-place,
behält ohne Schutz jedoch O(n²) als Worst Case. Heap Sort garantiert O(n log n)
bei konstantem Zusatzspeicher und wird in Modul 11 konstruktiv vertieft.

Die Schranke Ω(n log n) folgt intuitiv daraus, dass Vergleiche n! mögliche
Eingabereihenfolgen unterscheiden müssen. Counting, Bucket und Radix Sort dürfen
sie umgehen, weil sie mehr über Keys und Verteilung wissen als ein allgemeiner
Vergleichsalgorithmus. Diese Zusatzannahmen bestimmen zugleich ihre Grenzen.

Stabilität, In-Place-Eigenschaft, Adaptivität, Worst Case und Wertebereich sind
keine Nebendetails, sondern Teil der Algorithmuswahl. Python nutzt mit Timsort
ein stabiles, adaptives Hybridverfahren, das vorhandene Runs erkennt. In
Produktionscode ist `sorted()` fast immer der Ausgangspunkt; die
Eigenimplementierungen liefern das Modell, mit dem du seine Stärken und die
Alternativen fachlich beurteilen kannst.
