# Module 02 — Arrays & Strings

Arrays are among the most basic data structures. They combine a simple idea – elements
are in a fixed order – with a decisive property: an index can be used to access a
position directly. Python lists appear flexible and convenient, but are internally based
on exactly this array principle.

Strings are also indexable sequences. In contrast to lists, however, they are immutable.
This one design decision changes which operations are favorable and which silently
create many copies.

This script follows three levels:

1. **Intuition:** Why contiguous storage allows direct access and why growth or
   immutability require special strategies.
2. **Simulation:** How shifts, resizes, prefix sums, in-place patterns and matrix
   traversals proceed step by step.
3. **Formalization:** Which invariants and complexities are behind the most important
   operations and array patterns.

In the end, you should be able to justify the cost of any usual list operation, use
prefix sums safely and explain why NumPy processes numerical arrays much more
efficiently than general Python lists.

---

## 1. Intuition: Numbered drawers

### 1.1 Direct access through fixed positions

Imagine a cabinet with drawers of the same size, right next to each other. Each drawer
has a number. To open drawer 700, you don't have to search the first 699. You calculate
their position from starting point, number and drawer width:

\[
\text{Address}(i)=\text{Start address}+i\cdot\text{Element width}.
\]

This is the core idea of an array. Because each field has the same width and the fields
are coherent, index access costs constant time regardless of the array length.

A linked list works differently: your elements can be distributed and point to the next
one. Access to position 700 requires the following of up to 700 references. Arrays
exchange flexible local changes for quick position access.

### 1.2 Why inserting in the middle is expensive

If the drawers are completely occupied and a new drawer should be inserted at position
2, all later drawers must move one position to the right.

From

\[
[A,B,C,D,E]
\]

will be gradually:

| Step | Condition |
|---:|---|
| Start | [A, B, C, D, E, _] |
| Move E | [A, B, C, D, _, E] |
| Move D | [A, B, C, _, D, E] |
| Move C | [A, B, _, C, D, E] |
| Use X | [A, B, X, C, D, E] |

The closer the insertion position is at the beginning, the more elements are moved.
Direct index access remains cheap; structural changes within the array are linear.

### 1.3 The capacity problem

A static array has a fixed number of fields. If it is full, it cannot be guaranteed to
reserve additional storage directly behind it. A dynamic array solves the problem by
three steps:

1. reserve a larger contiguous buffer;
2. Copy existing elements,
3. release the old buffer.

If the buffer would only grow by exactly one field for each append, the copy costs would
arise for \(n\) apps

\[
1+2+3+\dots+(n-1)=\Theta(n^2).
\]

Instead, dynamic arrays reserve more space than currently required. This over-allocation
usually makes later apps constant and resizes rare.

### 1.4 Strings: A labeled sign instead of a board

A list is like a writing board: a single field can be changed. A string resembles a
printed shield. If a sign is to be different, a new sign will be made.

~~~python
values = ["D", "S", "A"]
values[0] = "X"

text = "DSA"
# text[0] = "X"  # TypeError: strings are immutable
changed = "X" + text[1:]
~~~

This immutability has advantages: Strings are safe to use as a dictionary key, can be
shared and do not change surprisingly. But it also means that apparent changes create
new objects.

---

## 2. Simulation: How Python Lists Grow

### 2.1 Static and dynamic array

A static array does not separate **Length** and **Capacity**: Both are fixed. In a
dynamic array, the terms mean:

- **Length:** Number of actually saved elements.
- **Capacity:** Number of elements for which the current buffer provides space.

The invariant always applies

\[
0\le \text{Length}\le \text{Capacity}.
\]

Python provides a dynamic array with list. A list does not internally store all any
Python objects directly in a row, but references to these objects. The references have a
uniform width; the actual objects can be located at other storage points.

Conceptually:

~~~text
list buffer: [ref] [ref" [ref") [free] [free]
                 \     |     /
objects:          42 "DSA" Customer(...)

length = 3
capacity = 5
~~~

Therefore, the same list may contain values of different types. However, this creates
additional indirection and property costs for numerical calculations.

### 2.2 Resize with doubling by hand

For the simulation we use a simplified growth rule: With full buffer the capacity is
doubled. Real Python versions use other, smaller growth factors; the principle remains
the same.

| append | Length before | Capacity before | Action | Copies | Capacity afterwards |
|---:|---:|---:|---|---:|---:|
| A | 0 | 1 | Write directly | 0 | 1 |
| B | 1 | 1 | Resize | 1 | 2 |
| C | 2 | 2 | Resize | 2 | 4 |
| D | 3 | 4 | Write directly | 0 | 4 |
| E | 4 | 4 | Resize | 4 | 8 |
| F | 5 | 8 | Write directly | 0 | 8 |

A single resize is linear, because all existing references are copied. Over a long
append sequence, however, the copying quantities form the geometric sum

\[
1+2+4+\dots < 2n.
\]

Together with \(n\) write operations, the overall work for \(n\) apps remains linear. An
app therefore costs **amortized \(\Theta(1)\)**.

### 2.3 Watch over-allocation in CPython

Python does not publish internal capacity as a normal list property. But with
sys.getsizeof you can see when the memory block gets bigger:

~~~python
import sys


def observe_list_growth(limit=80):
    """Print points at which a list's allocated size changes."""
    values = []
    previous_bytes = sys.getsizeof(values)
    print(f"length={len(values):2d}, bytes={previous_bytes}")

    for value in range(limit):
        values.append(value)
        current_bytes = sys.getsizeof(values)
        if current_bytes != previous_bytes:
            print(f"length={len(values):2d}, bytes={current_bytes}")
            previous_bytes = current_bytes


observe_list_growth()
~~~

The output byte values depend on Python version and platform. The pattern is relevant:
The size does not change after every append, but rather abruptly. Between two jumps the
list uses already reserved free places.

sys.getsizeof also measures only the list object with reference buffer, not recursively
the memory of all referenced objects.

### 2.4 Insert step by step

~~~python
values = [10, 20, 30, 40]
values.insert(1, 15)
~~~

For insertion position 1, the references are moved from right to left:

| Operation | Buffer |
|---|---|
| Start | [10, 20, 30, 40, _] |
| 40 by index 4 | [10, 20, 30, _, 40] |
| 30 by index 3 | [10, 20, _, 30, 40] |
| 20 by index 2 | [10, _, 20, 30, 40] |
| 15 by index 1 | [10, 15, 20, 30, 40] |

For length \(n\) and position \(i\) approximately \(n-i\) references are moved. This
follows \(O(n-i)\), in the worst case \(O(n)\).

### 2.5 Delete step by step

The removal from the center creates the reverse shift:

~~~python
values = [10, 15, 20, 30, 40]
del values[1]
~~~

| Operation | logical content |
|---|---|
| Start | [10, 15, 20, 30, 40] |
| 20 by index 1 | [10, 20, 20, 30, 40] |
| 30 by index 2 | [10, 20, 30, 30, 40] |
| 40 by index 3 | [10, 20, 30, 40, 40] |
| Reduce length | [10, 20, 30, 40] |

pop() at the end does not require such shift and is amortized constant. pop(0) shifts
almost the entire list and is linear.

### 2.6 compare runtimes empirically

The following experiment compares append at the end with insert at the beginning. The
list is re-created for each measurement so that both functions get the same initial
state:

~~~python
from statistics import median
from time import perf_counter


def measure_operation(operation, size, repeats=9):
    """Return median runtime for one operation on a fresh list."""
    durations = []
    for _ in range(repeats):
        values = list(range(size))
        start = perf_counter()
        operation(values)
        durations.append(perf_counter() - start)
    return median(durations)


sizes = [1_000, 2_000, 4_000, 8_000, 16_000]
for size in sizes:
    append_time = measure_operation(lambda values: values.append(-1), size)
    front_time = measure_operation(lambda values: values.insert(0, -1), size)
    print(size, append_time, front_time)
~~~

A single append measurement point can randomly contain a resize. A batch benchmark is
often more meaningful about many operations. insert(0, x) shows the linear displacement
work already per operation.

---

## 3. The cost model of the Python list

### 3.1 Overview

For a list of length \(n\) in CPython typically apply:

| Operation | typical time | Reasons |
|---|---:|---|
| values[i] | \(\Theta(1)\) | The address of the reference will be calculated |
| values[i] = x | \(\Theta(1)\) | replace a reference |
| len(values) | \(\Theta(1)\) | Save length |
| append(x) | amortised \(\Theta(1)\) | mostly free space, rare resize |
| pop() | amortised \(\Theta(1)\) | last element, no shift |
| insert(i, x) | \(O(n-i)\) | Moving Suffix |
| pop(i), del values[i] | \(O(n-i)\) | Moving Suffix |
| x in values | \(O(n)\) | Sequential comparison |
| values.index(x) | \(O(n)\) | sequential search |
| values.count(x) | \(\Theta(n)\) | All elements will be tested |
| values[a:b] | \(\Theta(k)\) | \(k\) Copying references |
| values.copy() | \(\Theta(n)\) | flat copy of all references |
| values.extend(other) | \(\Theta(m)\) amortised | \(m\) Attach references |
| values + other | \(\Theta(n+m)\) | Create and copy new list |
| values.reverse() | \(\Theta(n)\) | Swap elements in-place |
| reversed(values) | \(\Theta(1)\) Setup | Iterator, no list copy |
| sort() | \(O(n\log n)\) Worst Case | Timsort, in-place |
| Sorted(values) | \(O(n\log n)\) Worst Case | new sorted list |

\(k\) is the Slice length and \(m\) the length of the second container.

### 3.2 Index access is not search

~~~python
indexed_values = list(range(1_000))
value = indexed_values[700]
~~~

is a direct access and constant.

~~~python
target = 700
position = indexed_values.index(target)
~~~

must compare values until target is found. Best case is \(\Theta(1)\), worst case
\(\Theta(n)\).

The index is a known position. Search first determines which position belongs to a
value.

### 3.3 Negative indices

Python translates a negative index conceptually by

\[
i_{\text{Normalized}}=n+i.
\]

values[-1] becomes values[n-1]. The calculation remains constant; a negative index does
not trigger a reverse flow.

### 3.4 Slicing copied

~~~python
prefix = values[:100]
suffix = values[100:]
copy = values[:]
~~~

Each expression creates a new list. A slice in a loop can therefore cause an unexpected
sum of copy costs:

~~~python
def consume_by_slicing(values):
    """Repeatedly copy shrinking suffixes."""
    remaining = values
    total = 0
    while remaining:
        total += remaining[0]
        remaining = remaining[1:]
    return total
~~~

The copied lengths are \(n-1,n-2,\dots,1\). The time is \(\Theta(n^2)\), although only
one visible while loop exists.

### 3.5 Shallow Copy and shared objects

A list copy copies references, not recursively the referenced objects:

~~~python
original = [[1, 2], [3, 4]]
copied = original.copy()
copied[0].append(99)

assert original == [[1, 2, 99], [3, 4]]
assert copied[0] is original[0]
~~~

The outer lists are different, the inner lists are shared. For completely independent
nested structures, a targeted deep copy is required depending on the data type.

---

## 4. Multidimensional arrays and memory layout

### 4.1 A matrix as a list of lines

Python does not have a built-in special matrix type. A common representation is a list
of lines:

~~~python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

assert matrix[1][2] == 6
~~~

The first index selects a row list, the second an element in it. The lines are
independent python lists and do not even have to have the same length. For matrix
algorithms, rectangular form should therefore be explicitly checked.

~~~python
def validate_rectangular(matrix):
    """Raise ValueError if rows have different lengths."""
    if not matrix:
        return
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
~~~

### 4.2 The Alias Trap in Initialization

~~~python
wrong = [[0] * 3] * 3
wrong[0][0] = 7
~~~

The result is:

~~~text
[[7, 0, 0],
 [7, 0, 0],
 [7, 0, 0]]
~~~

The multiplication has referenced the same internal list three times. Right:

~~~python
correct = [[0 for _ in range(3)] for _ in range(3)]
correct[0][0] = 7

assert correct == [
    [7, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]
~~~

Each run of the outer comprehension creates a new row list.

### 4.3 Row-Major traversal

For a real contiguous two-dimensional array in Row Major layout, the values of a row are
directly in succession:

~~~text
Matrix:            linear memory:
[a b c]            [a b c d e f]
[d e f]
~~~

For \(r\) rows and \(c\) columns, element \((i,j)\) has the linear index

\[
i\cdot c+j.
\]

A line-by-line traversal follows this layout:

~~~python
def row_major_sum(matrix):
    """Sum a rectangular matrix row by row."""
    validate_rectangular(matrix)
    total = 0
    for row in matrix:
        for value in row:
            total += value
    return total
~~~

For coherent numerical arrays, this order usually uses the CPU cache better than column-
wise jumps. With Python lists of lists, storage is less compact, but the line run
remains the natural pattern.

### 4.4 Why NumPy Is Different

A python list of numerical values stores references to individual Python objects. Each
object bears type and administrative information.

On the other hand, a NumPy array with dtype int64 stores similar 64-bit values compactly
in a typed data buffer:

~~~text
Python list: [ref] -> PyInt object
             [ref] -> PyInt object
             [ref] -> PyInt object

NumPy int64: [8 bytes][8 bytes ][8 byte]
~~~

This presentation brings three key advantages:

1. **Less memory overhead:** Values lie directly and equally wide in the buffer.
2. **Better Cache Locality:** Neighboring values are loaded together into fast cache
   lines.
3. **Vectorization:** A loop runs in compiled code over many values instead of
   performing Python bytecode and dynamic type checks for each element.

### 4.5 Python loop against NumPy vectorization

~~~python
import numpy as np


def python_square(values):
    """Square Python numbers in an explicit loop."""
    return [value * value for value in values]


def numpy_square(values):
    """Square a typed NumPy array in compiled code."""
    return values * values


python_values = list(range(1_000))
numpy_values = np.arange(1_000, dtype=np.int64)

assert python_square(python_values) == numpy_square(numpy_values).tolist()
~~~

Both variants perform conceptually \(n\) multiplications and are \(\Theta(n)\). NumPy
does not win through a better Big-O class, but through compact data, low interpreter
costs, optimized machine instructions and SIMD if necessary.

Vectorization is not free:

- Generating or converting an array takes time and memory.
- Interim expressions can create temporary arrays.
- For very small data, Python setup or NumPy call costs can dominate.
- An unsuitable dtype can overflow or consume unnecessary memory.

The correct statement is therefore not "NumPy is always faster", but: For sufficiently
large, homogeneous numerical data, NumPy can perform the same linear work with
significantly smaller constant costs.

### 4.6 Views and Contigue Memory

Many NumPy slices are views on the same data buffer:

~~~python
array = np.array([10, 20, 30, 40])
view = array[1:3]
view[0] = 99

assert array.tolist() == [10, 99, 30, 40]
~~~

This is different from the list slicing that creates a new list. Whether a NumPy array
is C-contiguous depends on form and strides. Transposed or sliced views may have
disconnected access patterns, although they do not copy data.

---

## 5. Strings as immutable sequences

### 5.1 Indexing, Slicing and Replace

Strings support many sequence operations:

~~~python
text = "algorithm"

assert text[0] == "a"
assert text[-1] == "m"
assert text[1:4] == "lgo"
assert len(text) == 9
~~~

Index access and len are constant. A slice of length \(k\) creates a new string and
costs \(\Theta(k)\). A change is made from unchanged parts to a new string:

~~~python
def replace_character(text, index, replacement):
    """Return text with one position replaced."""
    if len(replacement) != 1:
        raise ValueError("replacement must be one character")
    if not 0 <= index < len(text):
        raise IndexError("index out of range")
    return text[:index] + replacement + text[index + 1 :]
~~~

The function copies in total proportional to the text length and requires \(\Theta(n)\)
time and \(\Theta(n)\) space for the result.

### 5.2 Repeated Concatenation

~~~python
def concatenate_in_loop(parts):
    """Build text through repeated concatenation."""
    result = ""
    for part in parts:
        result += part
    return result
~~~

Strings are immutable. Conceptually, each concatenation can copy the previous content
and the new part into a new buffer. With \(n\) equal length parts, potentially quadratic
total copies are created.

CPython optimizes certain local += patterns if the old string is not referenced
elsewhere. This implementation optimization is not a good basis for general or portable
algorithm design.

### 5.3 Collect and run once join

~~~python
def join_parts(parts):
    """Join all text parts in one planned allocation."""
    return "".join(parts)
~~~

join knows all parts, can determine the total length, reserve a matching result buffer
and copy each character essentially once. If \(L\) is the sum of all sublengths, runtime
is \(\Theta(L)\).

The separator is in front of join:

~~~python
words = ["data", "structures", "algorithms"]
sentence = " ".join(words)

assert sentence == "data structures algorithms"
~~~

### 5.4 The right input size for text

For string algorithms, the number of words is often not enough as a size. Two lists can
contain ten words each, but have very different overall character lengths. A precise
analysis uses, for example:

- \(p\): number of parts,
- \(L\): Sum of all characters in these parts.

join works in \(\Theta(L)\), not just \(\Theta(p)\).

### 5.5 Unicode and practical limits

Python strings consist conceptually of Unicode code points. A visible graph can consist
of several code points, such as a letter and a combining accent. len therefore does not
reliably measure the number of visually perceived characters.

For algorithmic tasks with ASCII small letters, an array of 26 counters can suffice. For
general Unicode, a hash map of the actual characters is usually more appropriate.

---

## 6. Basic pattern I: In-place operations

### 6.1 What "in-place" means

An in-place algorithm changes the existing container and uses only constant or at least
no proportional auxiliary storage.

~~~python
def reverse_in_place(values):
    """Reverse a mutable sequence using two indices."""
    left = 0
    right = len(values) - 1

    while left < right:
        values[left], values[right] = values[right], values[left]
        left += 1
        right -= 1
~~~

Trace for [A, B, C, D, E]:

| Step | left | right | Condition |
|---:|---:|---:|---|
| Start | 0 | 4 | [A, B, C, D, E] |
| Exchange | 1 | 3 | [E, B, C, D, A] |
| Exchange | 2 | 2 | [E, D, C, B, A] |
| End | — | — | [E, D, C, B, A] |

The loop invariant is:

> Before each run all positions are outside the interval
> [left, right] already at their final reversed position.

Two positions are final per run. Therefore, the loop ends after at most \(\lfloor
n/2\rfloor\) swapping. Time: \(\Theta(n)\), Auxiliary Space: \(\Theta(1)\).

### 6.2 Rotation by three reversals

A right-hand rotation around \(k\) positions should

\[
[A,B,C,D,E,F,G]
\]

for \(k=3\) in

\[
[E,F,G,A,B,C,D]
\]

It's all right.

Without additional array works:

1. reverse the entire list,
2. reverse the first \(k\) elements,
3. reverse the rest.

~~~python
def reverse_range(values, left, right):
    """Reverse an inclusive range in place."""
    while left < right:
        values[left], values[right] = values[right], values[left]
        left += 1
        right -= 1


def rotate_right(values, steps):
    """Rotate values to the right in O(n) time and O(1) extra space."""
    if not values:
        return

    steps %= len(values)
    reverse_range(values, 0, len(values) - 1)
    reverse_range(values, 0, steps - 1)
    reverse_range(values, steps, len(values) - 1)
~~~

Simulation:

| Phase | Condition |
|---|---|
| Start | [A, B, C, D, E, F, G] |
| To reverse everything | [G, F, E, D, C, B, A] |
| Prefix of length 3 | [E, F, G, D, C, B, A] |
| Turn the rest around | [E, F, G, A, B, C, D] |

Each element is moved constantly often. The total runtime remains \(\Theta(n)\).

### 6.3 Trade-off: mutation against memory

In-place saves an additional array, but changes the input. This is only correct if
callers expect this mutation. A clear API names mutation in the function name or
documentation and often returns None, such as list.sort.

---

## 7. Basic pattern II: prefix sums

### 7.1 Motivation: many area totals

Daily sales are given:

\[
[4,2,7,1,3,6].
\]

A single sum of index 1 to 4 can be calculated directly:

\[
2+7+1+3=13.
\]

In many queries, however, the naive method repeats the same additions over and over
again. prefix sums move this work into a one-time preparation.

### 7.2 Structure with Sentinel

We define prefix so that prefix[i] contains the sum of the first \(i\) elements.
prefix[0] is 0:

| read value | Prefix array |
|---:|---|
| Start | [0] |
| 4 | [0, 4] |
| 2 | [0, 4, 6] |
| 7 | [0, 4, 6, 13] |
| 1 | [0, 4, 6, 13, 14] |
| 3 | [0, 4, 6, 13, 14, 17] |
| 6 | [0, 4, 6, 13, 14, 17, 23] |

~~~python
def build_prefix_sums(values):
    """Return prefix sums with a leading zero sentinel."""
    prefix = [0]
    for value in values:
        prefix.append(prefix[-1] + value)
    return prefix
~~~

### 7.3 Area query by subtraction

For the included interval [left, right]:

\[
\text{Total}(left,right)
=prefix[right+1]-prefix[left].
\]

prefix[right+1] contains everything up to and including right. prefix[left] contains
exactly the portion before left and is deducted.

~~~python
def range_sum(prefix, left, right):
    """Return the inclusive range sum from a prefix array."""
    value_count = len(prefix) - 1
    if not 0 <= left <= right < value_count:
        raise IndexError("invalid inclusive range")
    return prefix[right + 1] - prefix[left]


sales = [4, 2, 7, 1, 3, 6]
sales_prefix = build_prefix_sums(sales)

assert range_sum(sales_prefix, 1, 4) == 13
assert range_sum(sales_prefix, 0, 5) == 23
~~~

### 7.4 Correctness invariant

After processing the first \(i\) values:

\[
prefix[i]=\sum_{j=0}^{i-1} values[j].
\]

The start value prefix[0]=0 meets the statement for zero processed values. When
attaching prefix[-1]+value, exactly the next value is added. Thus, the invariant
inductive applies to all prefix positions.

### 7.5 Complexity and Break-even

- Structure: \(\Theta(n)\) Time and \(\Theta(n)\) Additional memory.
- A query: \(\Theta(1)\).
- \(q\) Total queries: \(\Theta(n+q)\).

Naive range sums can cost up to \(\Theta(n)\) per query, total \(\Theta(qn)\). prefix
sums are particularly worthwhile for many queries on unchanged data.

If a value changes, all subsequent prefix values are affected. For frequent updates and
queries, other structures such as Fenwick Trees or Segment Trees are more suitable;
these follow in later DSA contexts.

---

## 8. Basic pattern III: matrix traversal

### 8.1 Full line run

~~~python
def matrix_positions(matrix):
    """Return all positions and values in row-major order."""
    validate_rectangular(matrix)
    positions = []
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            positions.append((row_index, column_index, value))
    return positions
~~~

For \(r\) rows and \(c\) columns \(rc\) elements are visited. The runtime is
\(\Theta(rc)\). Since the function returns all positions, the output \(\Theta(rc)\) also
requires memory.

### 8.2 Four neighbors of a grid field

Many matrix problems treat a grid as implicit graphs. Instead of saving a neighboring
list for each field, valid neighbors are calculated from four direction vectors:

~~~python
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def orthogonal_neighbors(matrix, row, column):
    """Return valid up, down, left, and right neighbors."""
    validate_rectangular(matrix)
    if not matrix or not matrix[0]:
        return []

    row_count = len(matrix)
    column_count = len(matrix[0])
    if not 0 <= row < row_count or not 0 <= column < column_count:
        raise IndexError("matrix position out of range")

    neighbors = []
    for row_delta, column_delta in DIRECTIONS:
        next_row = row + row_delta
        next_column = column + column_delta
        if 0 <= next_row < row_count and 0 <= next_column < column_count:
            neighbors.append((next_row, next_column))
    return neighbors
~~~

Each position has at most four orthogonal neighbors. The calculation for a field is
therefore \(\Theta(1)\).

### 8.3 explicitly treat boundary conditions

Matrix algorithms often fail not at the basic idea, but at:

- empty matrix,
- empty lines,
- non-rectangular lines;
- a single row or column;
- false comparisons at the last valid position.

The dimensions should be determined once and all indices should be checked against
\(0\le row<r\) or \(0\le column<c\).

---

## 9. Classic array and string patterning

### 9.1 Deduplication of a sorted array

Duplicates stand side by side in a sorted array. A read index runs through all values, a
write index marks the next position for a new unique value.

~~~python
def deduplicate_sorted(values):
    """Deduplicate a sorted list in place and return its logical length."""
    if not values:
        return 0

    write = 1
    for read in range(1, len(values)):
        if values[read] != values[write - 1]:
            values[write] = values[read]
            write += 1
    return write
~~~

Trace for [1, 1, 2, 2, 4]:

| read | read value | write before | Action | valid prefix |
|---:|---:|---:|---|---|
| 1 | 1 | 1 | Duplicate | [1] |
| 2 | 2 | 1 | Write to Index 1 | [1, 2] |
| 3 | 2 | 2 | Duplicate | [1, 2] |
| 4 | 2 | 2 | Duplicate | [1, 2] |
| 5 | 4 | 2 | Write to Index 2 | [1, 2, 4] |

The invariant:

> Before each read step, values[:write] contains exactly the unique values from
> the prefix already read, in sorted order.

After the run, only values[:write] are logically valid. The physical list remains the
same as long as the rest is not deleted. Time: \(\Theta(n)\), Auxiliary Space:
\(\Theta(1)\).

### 9.2 Merge Two Sorted Arrays

~~~python
def merge_sorted(left_values, right_values):
    """Merge two sorted sequences into a new sorted list."""
    left_index = 0
    right_index = 0
    merged = []

    while (
        left_index < len(left_values)
        and right_index < len(right_values)
    ):
        if left_values[left_index] <= right_values[right_index]:
            merged.append(left_values[left_index])
            left_index += 1
        else:
            merged.append(right_values[right_index])
            right_index += 1

    merged.extend(left_values[left_index:])
    merged.extend(right_values[right_index:])
    return merged
~~~

Trace for left=[2, 5, 8] and right=[1], 5, 9]:

| Comparison | Selection | Result |
|---|---:|---|
| 2 against 1 | 1 right | [1] |
| 2 against 5 | 2 left | [1, 2] |
| 5 against 5 | 5 left | [1, 2, 5] |
| 8 against 5 | 5 right | [1, 2, 5, 5] |
| 8 against 9 | 8 left | [1, 2, 5, 5, 8] |
| Rest | 9 right | [1, 2, 5, 5, 8, 9] |

The invariant:

> merged is sorted and contains exactly the already consumed elements
> both inputs.

Each of the \(n+m\) values is appended exactly once. Time and result place are
\(\Theta(n+m)\). The <= ensures that the left element is taken over first with the same
keys; thus a merge can be stable.

### 9.3 Anagram check by frequency

Two texts are anagrams if each character occurs equally often. A hash map is suitable
for general characters:

~~~python
def are_anagrams(left_text, right_text):
    """Return whether two strings contain the same character counts."""
    if len(left_text) != len(right_text):
        return False

    counts = {}
    for character in left_text:
        counts[character] = counts.get(character, 0) + 1

    for character in right_text:
        remaining = counts.get(character, 0) - 1
        if remaining < 0:
            return False
        counts[character] = remaining

    return True
~~~

Under usual hashing assumptions, the function costs \(\Theta(n)\) time. The additional
memory is \(O(k)\), where \(k\) indicates the number of different characters.

Sorting both strings would also be correct with \(\Theta(n\log n)\) time, but
asymptotically slower. For a fixed alphabet, for example exactly 26 ASCII small letters,
the hash map can be replaced by an array of fixed sizes.

### 9.4 Rotation: Modulo and empty input

In the case of rotations, \(k\) must be normalized to the list length:

\[
k_{\text{effective}}=k\bmod n.
\]

A rotation by \(n\), \(2n\) or \(3n\) positions does not change anything. However,
Modulo must not be executed with an empty list because the division is undefined by
zero. Therefore, rotate_right first checks whether values are empty.

---

## 10. Formalisation

### 10.1 Array abstraction

A one-dimensional array \(A\) of length \(n\) is a sequence

\[
A[0],A[1],\dots,A[n-1]
\]

with direct assignment of each valid index to a storage position. For a static array of
equally wide elements:

\[
\operatorname{address}(A[i])
=\operatorname{base}(A)+i\cdot w,
\]

where \(w\) is the element width.

The index invariant is:

\[
0\le i<n.
\]

Access outside this area is invalid. Python checks the limit and throws IndexError;
lower languages can show undefined behavior without testing.

### 10.2 Dynamic array

A dynamic array complements:

- length \(n\),
- capacity \(C\),
- a buffer of \(C\) positions,

with

\[
0\le n\le C.
\]

append writes to \(n<C\) directly to position \(n\). With \(n=C\) a larger buffer is
created. If the capacity grows geometrically by a factor \(\alpha>1\), the resize copies
add up via \(n\) appends to \(O(n)\). Thus append is amortized \(O(1)\).

### 10.3 In-place invariants

In-place procedures overwrite input positions. Their correctness requires a statement
about which area is already final and which is still unprocessed.

For deduplication:

\[
values[0:write]
\]

contains exactly its unique values after processing values[0:read]. Since read only runs
to the right and never writes larger than read, the function does not overwrite any
unread value.

### 10.4 Prefix Sum Definition

For \(A\) of length \(n\) we define \(P\) of length\(n+1\):

\[
P[0]=0
\]

and

\[
P[i+1]=P[i]+A[i].
\]

Then an inclusive interval \(0\le l\le r<n\) applies:

\[
\sum_{i=l}^{r}A[i]=P[r+1]-P[l].
\]

The additional zero avoids a special case for \(l=0\).

### 10.5 Matrix index

For a rectangular \(r\times c\) matrix in the Row Major layout, position \((i,j)\) on
the linear index

\[
i\cdot c+j
\]

is meant. Valid

\[
0\le i<r,\qquad 0\le j<c.
\]

A complete run is \(\Theta(rc)\) because each matrix element must be considered at least
once.

---

## 11. Common Errors

### 11.1 Keep each list operation for O(1)

Index access is constant, search and shift are not. values.insert(0, x), pop(0),
remove(x) and x in values can be linear.

### 11.2 use list.insert(0, x) as queue

Repeated removal or insertion at the top of the list leads to quadratic shifts. For a
queue, collections.deque with operations at both ends is more suitable; the data
structure follows in module 05.

### 11.3 Keep slices for free views

Python list licenses copy references. NumPy slices are often views. The same syntax does
not mean the same memory behavior here.

### 11.4 Structural changes during iteration

~~~python
def remove_negatives_wrong(values):
    """Demonstrate skipped elements during mutation."""
    for value in values:
        if value < 0:
            values.remove(value)
~~~

After a remove, the next element moves to the left, while the iterator continues. This
allows values to be skipped. Safe alternatives are:

- Create new list by Comprehension,
- run backwards over indices,
- a write index for in-place filtering.

~~~python
def remove_negatives_in_place(values):
    """Filter negative numbers with a write index."""
    write = 0
    for value in values:
        if value >= 0:
            values[write] = value
            write += 1
    del values[write:]
~~~

### 11.5 Create nested lists with multiplication

[[0] * columns] * rows divides the same line. A Comprehension creates independent lines.

### 11.6 Confound in-place and copy

- values.sort() changes values and returns None.
- sorted(values) creates a new list.
- values.reverse() changes values.
- reversed(values) creates an iterator.

APIs should use these differences consciously.

### 11.7 Mix prefix sum limits

Some implementations use included, other half-open intervals. This script uses for
queries [left, right] included and therefore prefix[right+1]-prefix[left]. A clear
convention prevents off-by-one errors.

### 11.8 Build strings with += uncritical in large loops

Even if CPython optimizes local cases, join is the explicit linear pattern for many
known parts. For streaming or very large outputs io.StringIO or direct writing into a
stream are also suitable.

### 11.9 Confound NumPy Speedup with Better Big-O

A vectorized and a python-based element-wise operation can be both \(\Theta(n)\). NumPy
reduces the constant cost. For real improvements in complexity, the number of asymptotic
work steps must change.

---

## 12. Systematic approach to array tasks

### Step 1: Name properties of the input

Is it sorted? Can it be changed? Are values homogeneous? Are there many queries or
updates? Is the matrix guaranteed to be rectangular?

### Step 2: Clear desired issue and space budget

Does a new array have to be created or does a logical length suffice? Is \(O(n)\)
auxiliary storage allowed? Does the original order have to be preserved?

### Step 3: Mark Expensive Operations

Watch Slicing, Front Insert, Linear Membership Tests, Repeated Concatenation, and Hidden
Sorting.

### Step 4: Recognize matching pattern

- Many unchangeable area sums: prefix sums.
- Sorted input and pairwise comparison: Two Pointers or Merge.
- Mutation without additional array: read/write indices or swap.
- Rectangular data: nested traversal with clear boundaries.
- Character frequencies: Hash Map or fixed counter array.

### Step 5: Formulate invariant

Describe which prefix, interval or range of results is already correct. A good invariant
often leads directly to implementation.

### Step 6: Test edge cases

Empty input, one element, all values equal, no duplicates, rotation multiples of \(n\),
single matrix line and Unicode text.

### Step 7: Analyze time and memory separately

Considered result size, slices, copies, views and temporary arrays. Name their sizes
separately for multiple entries.

---

## 13. Self-control

### Task 1

Determine time and auxiliary space:

~~~python
def every_second(values):
    return values[::2]
~~~

### Task 2

Why is the following matrix initialization flawed?

~~~python
matrix = [[None] * 4] * 5
~~~

### Task 3

A list has \(n\) values and \(q\) ranges are queried. Compare naive calculation and
prefix sums.

### Task 4

What is the complexity of the function in \(n=\text{len(left)}\) and
\(m=\text{len(right)}\)?

~~~python
def concatenate_lists(left, right):
    result = left.copy()
    result.extend(right)
    return result
~~~

### Task 5

Explain why a NumPy vectorization can hit a Python loop strongly, although both require
\(\Theta(n)\) time.

### Task 6

Which invariant makes deduplicate_sorted correct and why is \(write\le read\) valid?

### Solutions

1. The Slice copies about \(n/2\) references. Time and result place are \(\Theta(n)\).
2. All five external positions refer to the same internal list. A mutation of a line
   appears in all lines. Use a Comprehension that recreates each line.
3. Naiv costs a query up to \(\Theta(n)\), i.e. total \(\Theta(qn)\). prefix sums
   require \(\Theta(n)\) preparation, \(\Theta(n)\) memory and then \(\Theta(1)\) per
   query, total \(\Theta(n+q)\).
4. copy costs \(\Theta(n)\), extend \(\Theta(m)\). Total time and result place:
   \(\Theta(n+m)\).
5. NumPy stores typed values compactly, uses Cache Locality and executes the loop in
   optimized compiled code, if necessary with SIMD. The asymptotic number of processed
   values remains linear, the constant costs decrease.
6. values[:write] contains exactly the unique values of the prefix already read. write
   grows at most once per read step and does not start right from read. Therefore, no
   unread value is overwritten.

---

## 14. Executive summary

Arrays allow direct index access because positions are computed from a coherent, evenly
built memory layout. Python lists extend this principle to dynamic arrays from object
references. Over-allocation makes append amortized constant, while inserts and deletes
within the list cause linear shifts.

The most important findings are:

- Index access, assignment and len are \(\Theta(1)\).
- append and pop at the end are amortized \(\Theta(1)\).
- Search, front operations and changes in the middle are in the worst case
  \(\Theta(n)\).
- List slicing and copies require time proportional to the copied length.
- Strings are immutable; many parts are connected linearly to join planbar.
- In-place patterns use indices and clear invariants to save additional memory.
- prefix sums swap \(\Theta(n)\) preparation and memory for constant area queries.
- A complete \(r\times c\) matrix traversal requires \(\Theta(rc)\).
- Rotation, deduplication and merge are based on controlled index ranges instead of
  unnecessary slices or shifts.
- NumPy usually achieves the same Big-O class with homogeneous numerical data as Python
  loops, but with compact memory, better cache usage and vectorized execution.

If you can justify the cost of standard operations from memory movements, derive prefix
sums correctly and explain the practical NumPy advantage without false Big-O statements,
the qualification goals of this section of theory are reached.

---

# Deutsche Fassung

# Modul 02 — Arrays & Strings

Arrays gehören zu den grundlegendsten Datenstrukturen. Sie verbinden eine
einfache Idee – Elemente liegen in einer festen Reihenfolge – mit einer
entscheidenden Eigenschaft: Über einen Index lässt sich direkt auf eine Position
zugreifen. Python-Listen wirken flexibel und bequem, beruhen intern aber auf
genau diesem Array-Prinzip.

Strings sind ebenfalls indexierbare Sequenzen. Im Gegensatz zu Listen sind sie
jedoch immutable. Diese eine Designentscheidung verändert, welche Operationen
günstig sind und welche unbemerkt viele Kopien erzeugen.

Dieses Skript folgt drei Ebenen:

1. **Intuition:** Warum zusammenhängende Speicherung direkten Zugriff ermöglicht
   und weshalb Wachstum beziehungsweise Immutability besondere Strategien
   benötigen.
2. **Simulation:** Wie Verschiebungen, Resizes, Prefix Sums, In-Place-Muster und
   Matrix-Traversierungen Schritt für Schritt ablaufen.
3. **Formalisierung:** Welche Invarianten und Komplexitäten hinter den
   wichtigsten Operationen und Array-Patterns stehen.

Am Ende sollst du die Kosten jeder üblichen Listenoperation begründen, Prefix
Sums sicher einsetzen und erklären können, warum NumPy numerische Arrays
wesentlich effizienter verarbeitet als allgemeine Python-Listen.

---

## 1. Intuition: Nummerierte Schubladen

### 1.1 Direkter Zugriff durch feste Positionen

Stell dir einen Schrank mit gleich großen, direkt nebeneinanderliegenden
Schubladen vor. Jede Schublade besitzt eine Nummer. Um Schublade 700 zu öffnen,
musst du nicht die ersten 699 durchsuchen. Du berechnest ihre Position aus
Startpunkt, Nummer und Schubladenbreite:

\[
\text{Adresse}(i)=\text{Startadresse}+i\cdot\text{Elementbreite}.
\]

Das ist die Kernidee eines Arrays. Weil jedes Feld dieselbe Breite besitzt und
die Felder zusammenhängend liegen, kostet der Indexzugriff unabhängig von der
Array-Länge konstante Zeit.

Eine Linked List funktioniert anders: Ihre Elemente können verteilt liegen und
verweisen jeweils auf das nächste. Der Zugriff auf Position 700 erfordert dort
das Folgen von bis zu 700 Verweisen. Arrays tauschen flexible lokale Änderungen
gegen schnellen Positionszugriff.

### 1.2 Warum Einfügen in der Mitte teuer ist

Sind die Schubladen lückenlos belegt und soll an Position 2 eine neue Schublade
eingeschoben werden, müssen alle späteren Schubladen um eine Position nach
rechts rücken.

Aus

\[
[A,B,C,D,E]
\]

wird schrittweise:

| Schritt | Zustand |
|---:|---|
| Start | [A, B, C, D, E, _] |
| E verschieben | [A, B, C, D, _, E] |
| D verschieben | [A, B, C, _, D, E] |
| C verschieben | [A, B, _, C, D, E] |
| X einsetzen | [A, B, X, C, D, E] |

Je näher die Einfügeposition am Anfang liegt, desto mehr Elemente werden
verschoben. Der direkte Indexzugriff bleibt billig; strukturelle Änderungen
innerhalb des Arrays sind dagegen linear.

### 1.3 Das Kapazitätsproblem

Ein statisches Array besitzt eine feste Anzahl von Feldern. Ist es voll, kann
nicht garantiert direkt dahinter weiterer Speicher reserviert werden. Ein
dynamisches Array löst das Problem durch drei Schritte:

1. einen größeren zusammenhängenden Puffer reservieren,
2. vorhandene Elemente kopieren,
3. den alten Puffer freigeben.

Würde der Puffer bei jedem append nur um genau ein Feld wachsen, entstünden für
\(n\) appends die Kopierkosten

\[
1+2+3+\dots+(n-1)=\Theta(n^2).
\]

Stattdessen reservieren dynamische Arrays mehr Platz als aktuell benötigt.
Dieses Over-Allocation macht spätere appends meistens konstant und Resizes
selten.

### 1.4 Strings: Ein beschriftetes Schild statt einer Tafel

Eine Liste ist wie eine beschreibbare Tafel: Ein einzelnes Feld kann verändert
werden. Ein String gleicht einem gedruckten Schild. Soll ein Zeichen anders
lauten, wird ein neues Schild hergestellt.

~~~python
values = ["D", "S", "A"]
values[0] = "X"

text = "DSA"
# text[0] = "X"  # TypeError: strings are immutable
changed = "X" + text[1:]
~~~

Diese Immutability hat Vorteile: Strings sind sicher als Dictionary-Schlüssel
nutzbar, können geteilt werden und ändern sich nicht überraschend. Sie bedeutet
aber auch, dass scheinbare Änderungen neue Objekte erzeugen.

---

## 2. Simulation: Wie Python-Listen wachsen

### 2.1 Statisches und dynamisches Array

Ein statisches Array trennt **Länge** und **Kapazität** nicht: Beide sind fest.
Bei einem dynamischen Array bedeuten die Begriffe:

- **Länge:** Anzahl tatsächlich gespeicherter Elemente.
- **Kapazität:** Anzahl der Elemente, für die der aktuelle Puffer Platz bietet.

Es gilt stets die Invariante

\[
0\le \text{Länge}\le \text{Kapazität}.
\]

Python stellt mit list ein dynamisches Array bereit. Eine Liste speichert intern
nicht alle beliebigen Python-Objekte direkt hintereinander, sondern
Referenzen auf diese Objekte. Die Referenzen besitzen eine einheitliche Breite;
die eigentlichen Objekte können an anderen Speicherstellen liegen.

Konzeptionell:

~~~text
list buffer: [ref] [ref" [ref") [free] [free]
                 \     |     /
objects:          42 "DSA" Customer(...)

length = 3
capacity = 5
~~~

Darum kann dieselbe Liste Werte verschiedener Typen enthalten. Für numerische
Berechnungen entstehen dadurch allerdings zusätzliche Indirektion und
Objektkosten.

### 2.2 Resize mit Verdopplung von Hand

Für die Simulation verwenden wir eine vereinfachte Wachstumsregel: Bei vollem
Puffer wird die Kapazität verdoppelt. Reale Python-Versionen verwenden andere,
kleinere Wachstumsfaktoren; das Prinzip bleibt dasselbe.

| append | Länge vorher | Kapazität vorher | Aktion | Kopien | Kapazität nachher |
|---:|---:|---:|---|---:|---:|
| A | 0 | 1 | direkt schreiben | 0 | 1 |
| B | 1 | 1 | Resize | 1 | 2 |
| C | 2 | 2 | Resize | 2 | 4 |
| D | 3 | 4 | direkt schreiben | 0 | 4 |
| E | 4 | 4 | Resize | 4 | 8 |
| F | 5 | 8 | direkt schreiben | 0 | 8 |

Ein einzelnes Resize ist linear, denn alle vorhandenen Referenzen werden
kopiert. Über eine lange append-Folge bilden die Kopiermengen aber die
geometrische Summe

\[
1+2+4+\dots < 2n.
\]

Zusammen mit \(n\) Schreiboperationen bleibt die Gesamtarbeit für \(n\) appends
linear. Ein append kostet daher **amortisiert \(\Theta(1)\)**.

### 2.3 Over-Allocation in CPython beobachten

Python veröffentlicht die interne Kapazität nicht als normale Listen-Eigenschaft.
Mit sys.getsizeof lässt sich aber beobachten, wann der Speicherblock größer
wird:

~~~python
import sys


def observe_list_growth(limit=80):
    """Print points at which a list's allocated size changes."""
    values = []
    previous_bytes = sys.getsizeof(values)
    print(f"length={len(values):2d}, bytes={previous_bytes}")

    for value in range(limit):
        values.append(value)
        current_bytes = sys.getsizeof(values)
        if current_bytes != previous_bytes:
            print(f"length={len(values):2d}, bytes={current_bytes}")
            previous_bytes = current_bytes


observe_list_growth()
~~~

Die ausgegebenen Bytewerte hängen von Python-Version und Plattform ab. Relevant
ist das Muster: Die Größe ändert sich nicht nach jedem append, sondern
sprunghaft. Zwischen zwei Sprüngen nutzt die Liste bereits reservierte freie
Plätze.

sys.getsizeof misst außerdem nur das Listenobjekt samt Referenzpuffer, nicht
rekursiv den Speicher aller referenzierten Objekte.

### 2.4 Insert Schritt für Schritt

~~~python
values = [10, 20, 30, 40]
values.insert(1, 15)
~~~

Für die Einfügeposition 1 werden die Referenzen von rechts nach links
verschoben:

| Operation | Puffer |
|---|---|
| Start | [10, 20, 30, 40, _] |
| 40 nach Index 4 | [10, 20, 30, _, 40] |
| 30 nach Index 3 | [10, 20, _, 30, 40] |
| 20 nach Index 2 | [10, _, 20, 30, 40] |
| 15 nach Index 1 | [10, 15, 20, 30, 40] |

Bei Länge \(n\) und Position \(i\) werden ungefähr \(n-i\) Referenzen
verschoben. Daraus folgt \(O(n-i)\), im Worst Case \(O(n)\).

### 2.5 Delete Schritt für Schritt

Das Entfernen aus der Mitte erzeugt die umgekehrte Verschiebung:

~~~python
values = [10, 15, 20, 30, 40]
del values[1]
~~~

| Operation | logischer Inhalt |
|---|---|
| Start | [10, 15, 20, 30, 40] |
| 20 nach Index 1 | [10, 20, 20, 30, 40] |
| 30 nach Index 2 | [10, 20, 30, 30, 40] |
| 40 nach Index 3 | [10, 20, 30, 40, 40] |
| Länge reduzieren | [10, 20, 30, 40] |

pop() am Ende benötigt keine solche Verschiebung und ist amortisiert konstant.
pop(0) verschiebt dagegen fast die gesamte Liste und ist linear.

### 2.6 Laufzeiten empirisch vergleichen

Das folgende Experiment vergleicht append am Ende mit insert am Anfang. Die
Liste wird für jede Messung neu erzeugt, damit beide Funktionen denselben
Ausgangszustand erhalten:

~~~python
from statistics import median
from time import perf_counter


def measure_operation(operation, size, repeats=9):
    """Return median runtime for one operation on a fresh list."""
    durations = []
    for _ in range(repeats):
        values = list(range(size))
        start = perf_counter()
        operation(values)
        durations.append(perf_counter() - start)
    return median(durations)


sizes = [1_000, 2_000, 4_000, 8_000, 16_000]
for size in sizes:
    append_time = measure_operation(lambda values: values.append(-1), size)
    front_time = measure_operation(lambda values: values.insert(0, -1), size)
    print(size, append_time, front_time)
~~~

Ein einzelner append-Messpunkt kann zufällig ein Resize enthalten. Über viele
Operationen ist deshalb ein Batch-Benchmark oft aussagekräftiger. insert(0, x)
zeigt dagegen bereits pro Operation die lineare Verschiebungsarbeit.

---

## 3. Das Kostenmodell der Python-Liste

### 3.1 Übersicht

Für eine Liste der Länge \(n\) gelten in CPython typischerweise:

| Operation | typische Zeit | Begründung |
|---|---:|---|
| values[i] | \(\Theta(1)\) | Adresse der Referenz wird berechnet |
| values[i] = x | \(\Theta(1)\) | eine Referenz wird ersetzt |
| len(values) | \(\Theta(1)\) | Länge wird gespeichert |
| append(x) | amortisiert \(\Theta(1)\) | meist freier Platz, selten Resize |
| pop() | amortisiert \(\Theta(1)\) | letztes Element, keine Verschiebung |
| insert(i, x) | \(O(n-i)\) | Suffix wird verschoben |
| pop(i), del values[i] | \(O(n-i)\) | Suffix wird verschoben |
| x in values | \(O(n)\) | sequenzieller Vergleich |
| values.index(x) | \(O(n)\) | sequenzielle Suche |
| values.count(x) | \(\Theta(n)\) | alle Elemente werden geprüft |
| values[a:b] | \(\Theta(k)\) | \(k\) Referenzen werden kopiert |
| values.copy() | \(\Theta(n)\) | flache Kopie aller Referenzen |
| values.extend(other) | \(\Theta(m)\) amortisiert | \(m\) Referenzen anhängen |
| values + other | \(\Theta(n+m)\) | neue Liste erzeugen und kopieren |
| values.reverse() | \(\Theta(n)\) | Elemente in-place vertauschen |
| reversed(values) | \(\Theta(1)\) Setup | Iterator, keine Listenkopie |
| sort() | \(O(n\log n)\) Worst Case | Timsort, in-place |
| sorted(values) | \(O(n\log n)\) Worst Case | neue sortierte Liste |

\(k\) ist die Slice-Länge und \(m\) die Länge des zweiten Containers.

### 3.2 Indexzugriff ist nicht Suche

~~~python
indexed_values = list(range(1_000))
value = indexed_values[700]
~~~

ist ein direkter Zugriff und konstant.

~~~python
target = 700
position = indexed_values.index(target)
~~~

muss dagegen Werte vergleichen, bis target gefunden wird. Best Case ist
\(\Theta(1)\), Worst Case \(\Theta(n)\).

Der Index ist eine bekannte Position. Suche ermittelt erst, welche Position zu
einem Wert gehört.

### 3.3 Negative Indizes

Python übersetzt einen negativen Index konzeptionell durch

\[
i_{\text{normalisiert}}=n+i.
\]

values[-1] wird damit zu values[n-1]. Die Berechnung bleibt konstant; ein
negativer Index löst keinen Rückwärtsdurchlauf aus.

### 3.4 Slicing kopiert

~~~python
prefix = values[:100]
suffix = values[100:]
copy = values[:]
~~~

Jeder Ausdruck erzeugt eine neue Liste. Ein Slice in einer Schleife kann deshalb
eine unerwartete Summe von Kopierkosten verursachen:

~~~python
def consume_by_slicing(values):
    """Repeatedly copy shrinking suffixes."""
    remaining = values
    total = 0
    while remaining:
        total += remaining[0]
        remaining = remaining[1:]
    return total
~~~

Die kopierten Längen sind \(n-1,n-2,\dots,1\). Die Zeit ist
\(\Theta(n^2)\), obwohl nur eine sichtbare while-Schleife existiert.

### 3.5 Shallow Copy und geteilte Objekte

Eine Listenkopie kopiert Referenzen, nicht rekursiv die referenzierten Objekte:

~~~python
original = [[1, 2], [3, 4]]
copied = original.copy()
copied[0].append(99)

assert original == [[1, 2, 99], [3, 4]]
assert copied[0] is original[0]
~~~

Die äußeren Listen sind verschieden, die inneren Listen werden geteilt. Für
vollständig unabhängige verschachtelte Strukturen ist je nach Datentyp eine
gezielte tiefe Kopie erforderlich.

---

## 4. Mehrdimensionale Arrays und Speicherlayout

### 4.1 Eine Matrix als Liste von Zeilen

Python besitzt keinen eingebauten speziellen Matrix-Typ. Eine verbreitete
Darstellung ist eine Liste von Zeilen:

~~~python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

assert matrix[1][2] == 6
~~~

Der erste Index wählt eine Zeilenliste, der zweite ein Element darin. Die Zeilen
sind eigenständige Python-Listen und müssen nicht einmal dieselbe Länge haben.
Für Matrixalgorithmen sollte rechteckige Form deshalb explizit geprüft werden.

~~~python
def validate_rectangular(matrix):
    """Raise ValueError if rows have different lengths."""
    if not matrix:
        return
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
~~~

### 4.2 Die Alias-Falle bei der Initialisierung

~~~python
wrong = [[0] * 3] * 3
wrong[0][0] = 7
~~~

Das Ergebnis ist:

~~~text
[[7, 0, 0],
 [7, 0, 0],
 [7, 0, 0]]
~~~

Die Multiplikation hat dieselbe innere Liste dreimal referenziert. Richtig ist:

~~~python
correct = [[0 for _ in range(3)] for _ in range(3)]
correct[0][0] = 7

assert correct == [
    [7, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]
~~~

Jeder Durchlauf der äußeren Comprehension erzeugt eine neue Zeilenliste.

### 4.3 Row-Major Traversierung

Bei einem echten zusammenhängenden zweidimensionalen Array in Row-Major-Layout
liegen die Werte einer Zeile direkt hintereinander:

~~~text
Matrix:            linear memory:
[a b c]            [a b c d e f]
[d e f]
~~~

Für \(r\) Zeilen und \(c\) Spalten besitzt Element \((i,j)\) den linearen Index

\[
i\cdot c+j.
\]

Eine zeilenweise Traversierung folgt diesem Layout:

~~~python
def row_major_sum(matrix):
    """Sum a rectangular matrix row by row."""
    validate_rectangular(matrix)
    total = 0
    for row in matrix:
        for value in row:
            total += value
    return total
~~~

Bei zusammenhängenden numerischen Arrays nutzt diese Reihenfolge den CPU-Cache
meist besser als spaltenweise Sprünge. Bei Python-Listen von Listen ist die
Speicherung weniger kompakt, doch der Zeilendurchlauf bleibt das natürliche
Muster.

### 4.4 Warum NumPy anders ist

Eine Python-Liste numerischer Werte speichert Referenzen auf einzelne
Python-Objekte. Jedes Objekt trägt Typ- und Verwaltungsinformationen.

Ein NumPy-Array mit dtype int64 speichert dagegen gleichartige 64-Bit-Werte
kompakt in einem typisierten Datenpuffer:

~~~text
Python list: [ref] -> PyInt object
             [ref] -> PyInt object
             [ref] -> PyInt object

NumPy int64: [8 bytes][8 bytes ][8 byte]
~~~

Diese Darstellung bringt drei zentrale Vorteile:

1. **Weniger Speicher-Overhead:** Werte liegen direkt und gleich breit im
   Puffer.
2. **Bessere Cache Locality:** Benachbarte Werte werden gemeinsam in schnelle
   Cache-Zeilen geladen.
3. **Vektorisierung:** Eine Schleife läuft in kompiliertem Code über viele Werte,
   statt für jedes Element Python-Bytecode und dynamische Typprüfungen
   auszuführen.

### 4.5 Python-Schleife gegen NumPy-Vektorisierung

~~~python
import numpy as np


def python_square(values):
    """Square Python numbers in an explicit loop."""
    return [value * value for value in values]


def numpy_square(values):
    """Square a typed NumPy array in compiled code."""
    return values * values


python_values = list(range(1_000))
numpy_values = np.arange(1_000, dtype=np.int64)

assert python_square(python_values) == numpy_square(numpy_values).tolist()
~~~

Beide Varianten führen konzeptionell \(n\) Multiplikationen aus und sind
\(\Theta(n)\). NumPy gewinnt nicht durch eine bessere Big-O-Klasse, sondern durch
kompakte Daten, geringe Interpreterkosten, optimierte Maschineninstruktionen und
gegebenenfalls SIMD.

Vektorisierung ist nicht kostenlos:

- Das Erzeugen oder Konvertieren eines Arrays kostet Zeit und Speicher.
- Zwischenausdrücke können temporäre Arrays anlegen.
- Für sehr kleine Daten können Python-Setup oder NumPy-Aufrufkosten dominieren.
- Ein ungeeigneter dtype kann überlaufen oder unnötig Speicher verbrauchen.

Die richtige Aussage lautet deshalb nicht „NumPy ist immer schneller“, sondern:
Für ausreichend große, homogene numerische Daten kann NumPy dieselbe lineare
Arbeit mit wesentlich kleineren konstanten Kosten ausführen.

### 4.6 Views und Contiguous Memory

Viele NumPy-Slices sind Views auf denselben Datenpuffer:

~~~python
array = np.array([10, 20, 30, 40])
view = array[1:3]
view[0] = 99

assert array.tolist() == [10, 99, 30, 40]
~~~

Das unterscheidet sich vom List Slicing, das eine neue Liste erzeugt. Ob ein
NumPy-Array C-contiguous ist, hängt von Form und Strides ab. Transponierte oder
geslicte Views können disconnectede Zugriffsmuster besitzen, obwohl sie
keine Daten kopieren.

---

## 5. Strings als immutable Sequenzen

### 5.1 Indexieren, Slicen und Ersetzen

Strings unterstützen viele Sequenzoperationen:

~~~python
text = "algorithm"

assert text[0] == "a"
assert text[-1] == "m"
assert text[1:4] == "lgo"
assert len(text) == 9
~~~

Indexzugriff und len sind konstant. Ein Slice der Länge \(k\) erzeugt einen
neuen String und kostet \(\Theta(k)\). Eine Änderung wird aus unveränderten
Teilen zu einem neuen String zusammengesetzt:

~~~python
def replace_character(text, index, replacement):
    """Return text with one position replaced."""
    if len(replacement) != 1:
        raise ValueError("replacement must be one character")
    if not 0 <= index < len(text):
        raise IndexError("index out of range")
    return text[:index] + replacement + text[index + 1 :]
~~~

Die Funktion kopiert insgesamt proportional zur Textlänge und benötigt
\(\Theta(n)\) Zeit sowie \(\Theta(n)\) Platz für das Ergebnis.

### 5.2 Wiederholte Konkatenation

~~~python
def concatenate_in_loop(parts):
    """Build text through repeated concatenation."""
    result = ""
    for part in parts:
        result += part
    return result
~~~

Strings sind immutable. Konzeptionell kann jede Konkatenation den bisherigen
Inhalt und das neue Teil in einen neuen Puffer kopieren. Bei \(n\) gleich langen
Teilen entstehen potenziell quadratische Gesamtkopien.

CPython optimiert bestimmte lokale +=-Muster, wenn der alte String nicht
anderweitig referenziert wird. Diese Implementierungsoptimierung ist keine gute
Grundlage für allgemeinen oder portablen Algorithmusentwurf.

### 5.3 Sammeln und einmal join ausführen

~~~python
def join_parts(parts):
    """Join all text parts in one planned allocation."""
    return "".join(parts)
~~~

join kennt alle Teile, kann die Gesamtlänge bestimmen, einen passenden
Ergebnispuffer reservieren und jedes Zeichen im Wesentlichen einmal kopieren.
Ist \(L\) die Summe aller Teillängen, beträgt die Laufzeit \(\Theta(L)\).

Der Separator steht vor join:

~~~python
words = ["data", "structures", "algorithms"]
sentence = " ".join(words)

assert sentence == "data structures algorithms"
~~~

### 5.4 Die richtige Eingabegröße für Text

Bei Stringalgorithmen reicht die Anzahl der Wörter oft nicht als Größe. Zwei
Listen können je zehn Wörter enthalten, aber sehr unterschiedliche
Gesamtzeichenlängen besitzen. Eine präzise Analyse verwendet beispielsweise:

- \(p\): Anzahl der Teile,
- \(L\): Summe aller Zeichen in diesen Teilen.

join arbeitet in \(\Theta(L)\), nicht bloß \(\Theta(p)\).

### 5.5 Unicode und praktische Grenzen

Python-Strings bestehen konzeptionell aus Unicode-Codepoints. Ein sichtbares
Graphem kann aus mehreren Codepoints bestehen, beispielsweise einem Buchstaben
und einem kombinierenden Akzent. len misst daher nicht zuverlässig die Anzahl
visuell wahrgenommener Zeichen.

Für algorithmische Aufgaben mit ASCII-Kleinbuchstaben kann ein Array aus 26
Zählern genügen. Für allgemeines Unicode ist eine Hash Map der tatsächlich
auftretenden Zeichen meist angemessener.

---

## 6. Grundmuster I: In-Place-Operationen

### 6.1 Was „in-place“ bedeutet

Ein In-Place-Algorithmus verändert den vorhandenen Container und verwendet nur
konstanten oder zumindest keinen zur Eingabe proportionalen Hilfsspeicher.

~~~python
def reverse_in_place(values):
    """Reverse a mutable sequence using two indices."""
    left = 0
    right = len(values) - 1

    while left < right:
        values[left], values[right] = values[right], values[left]
        left += 1
        right -= 1
~~~

Trace für [A, B, C, D, E]:

| Schritt | left | right | Zustand |
|---:|---:|---:|---|
| Start | 0 | 4 | [A, B, C, D, E] |
| Tausch | 1 | 3 | [E, B, C, D, A] |
| Tausch | 2 | 2 | [E, D, C, B, A] |
| Ende | — | — | [E, D, C, B, A] |

Die Schleifeninvariante lautet:

> Vor jedem Durchlauf stehen alle Positionen außerhalb des Intervalls
> [left, right] bereits an ihrer endgültigen umgekehrten Position.

Pro Durchlauf werden zwei Positionen endgültig. Daher endet die Schleife nach
höchstens \(\lfloor n/2\rfloor\) Tauschen. Zeit: \(\Theta(n)\), Auxiliary Space:
\(\Theta(1)\).

### 6.2 Rotation durch drei Umkehrungen

Eine Rechtsrotation um \(k\) Positionen soll

\[
[A,B,C,D,E,F,G]
\]

für \(k=3\) in

\[
[E,F,G,A,B,C,D]
\]

verwandeln.

Ohne Zusatzarray funktioniert:

1. gesamte Liste umkehren,
2. die ersten \(k\) Elemente umkehren,
3. den Rest umkehren.

~~~python
def reverse_range(values, left, right):
    """Reverse an inclusive range in place."""
    while left < right:
        values[left], values[right] = values[right], values[left]
        left += 1
        right -= 1


def rotate_right(values, steps):
    """Rotate values to the right in O(n) time and O(1) extra space."""
    if not values:
        return

    steps %= len(values)
    reverse_range(values, 0, len(values) - 1)
    reverse_range(values, 0, steps - 1)
    reverse_range(values, steps, len(values) - 1)
~~~

Simulation:

| Phase | Zustand |
|---|---|
| Start | [A, B, C, D, E, F, G] |
| alles umkehren | [G, F, E, D, C, B, A] |
| Prefix der Länge 3 | [E, F, G, D, C, B, A] |
| Rest umkehren | [E, F, G, A, B, C, D] |

Jedes Element wird konstant oft bewegt. Die Gesamtlaufzeit bleibt
\(\Theta(n)\).

### 6.3 Trade-off: Mutation gegen Speicher

In-place spart ein zusätzliches Array, verändert aber die Eingabe. Das ist nur
dann korrekt, wenn Aufrufer diese Mutation erwarten. Eine klare API benennt
Mutation im Funktionsnamen oder in der Dokumentation und gibt häufig None
zurück, wie list.sort.

---

## 7. Grundmuster II: Prefix Sums

### 7.1 Motivation: viele Bereichssummen

Gegeben seien tägliche Verkäufe:

\[
[4,2,7,1,3,6].
\]

Eine einzelne Summe von Index 1 bis 4 lässt sich direkt berechnen:

\[
2+7+1+3=13.
\]

Bei sehr vielen Abfragen wiederholt die naive Methode jedoch immer wieder
dieselben Additionen. Prefix Sums verschieben diese Arbeit in eine einmalige
Vorbereitung.

### 7.2 Aufbau mit Sentinel

Wir definieren prefix so, dass prefix[i] die Summe der ersten \(i\) Elemente
enthält. prefix[0] ist 0:

| gelesener Wert | Prefix-Array |
|---:|---|
| Start | [0] |
| 4 | [0, 4] |
| 2 | [0, 4, 6] |
| 7 | [0, 4, 6, 13] |
| 1 | [0, 4, 6, 13, 14] |
| 3 | [0, 4, 6, 13, 14, 17] |
| 6 | [0, 4, 6, 13, 14, 17, 23] |

~~~python
def build_prefix_sums(values):
    """Return prefix sums with a leading zero sentinel."""
    prefix = [0]
    for value in values:
        prefix.append(prefix[-1] + value)
    return prefix
~~~

### 7.3 Bereichsabfrage durch Subtraktion

Für das inklusive Intervall [left, right] gilt:

\[
\text{sum}(left,right)
=prefix[right+1]-prefix[left].
\]

prefix[right+1] enthält alles bis einschließlich right. prefix[left] enthält
genau den Anteil vor left und wird abgezogen.

~~~python
def range_sum(prefix, left, right):
    """Return the inclusive range sum from a prefix array."""
    value_count = len(prefix) - 1
    if not 0 <= left <= right < value_count:
        raise IndexError("invalid inclusive range")
    return prefix[right + 1] - prefix[left]


sales = [4, 2, 7, 1, 3, 6]
sales_prefix = build_prefix_sums(sales)

assert range_sum(sales_prefix, 1, 4) == 13
assert range_sum(sales_prefix, 0, 5) == 23
~~~

### 7.4 Korrektheitsinvariante

Nach Verarbeitung der ersten \(i\) Werte gilt:

\[
prefix[i]=\sum_{j=0}^{i-1} values[j].
\]

Der Startwert prefix[0]=0 erfüllt die Aussage für null verarbeitete Werte. Beim
Anhängen von prefix[-1]+value wird genau der nächste Wert ergänzt. Damit gilt die
Invariante induktiv für alle Prefix-Positionen.

### 7.5 Komplexität und Break-even

- Aufbau: \(\Theta(n)\) Zeit und \(\Theta(n)\) Zusatzspeicher.
- Eine Abfrage: \(\Theta(1)\).
- \(q\) Abfragen insgesamt: \(\Theta(n+q)\).

Naive Bereichssummen können pro Abfrage bis zu \(\Theta(n)\) kosten, insgesamt
\(\Theta(qn)\). Prefix Sums lohnen sich besonders bei vielen Abfragen auf
unveränderten Daten.

Ändert sich ein Wert, sind alle späteren Prefix-Werte betroffen. Für häufige
Updates und Abfragen sind andere Strukturen wie Fenwick Trees oder Segment Trees
geeigneter; diese folgen in späteren DSA-Kontexten.

---

## 8. Grundmuster III: Matrix-Traversierung

### 8.1 Vollständiger Zeilendurchlauf

~~~python
def matrix_positions(matrix):
    """Return all positions and values in row-major order."""
    validate_rectangular(matrix)
    positions = []
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            positions.append((row_index, column_index, value))
    return positions
~~~

Für \(r\) Zeilen und \(c\) Spalten werden \(rc\) Elemente besucht. Die
Laufzeit ist \(\Theta(rc)\). Da die Funktion alle Positionen zurückgibt, benötigt
auch die Ausgabe \(\Theta(rc)\) Speicher.

### 8.2 Vier Nachbarn eines Grid-Feldes

Viele Matrixprobleme behandeln ein Grid als impliziten Graphen. Statt für jedes
Feld eine Nachbarliste zu speichern, werden gültige Nachbarn aus vier
Richtungsvektoren berechnet:

~~~python
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def orthogonal_neighbors(matrix, row, column):
    """Return valid up, down, left, and right neighbors."""
    validate_rectangular(matrix)
    if not matrix or not matrix[0]:
        return []

    row_count = len(matrix)
    column_count = len(matrix[0])
    if not 0 <= row < row_count or not 0 <= column < column_count:
        raise IndexError("matrix position out of range")

    neighbors = []
    for row_delta, column_delta in DIRECTIONS:
        next_row = row + row_delta
        next_column = column + column_delta
        if 0 <= next_row < row_count and 0 <= next_column < column_count:
            neighbors.append((next_row, next_column))
    return neighbors
~~~

Jede Position besitzt höchstens vier orthogonale Nachbarn. Die Berechnung für
ein Feld ist daher \(\Theta(1)\).

### 8.3 Randbedingungen explizit behandeln

Matrixalgorithmen scheitern häufig nicht an der Grundidee, sondern an:

- leerer Matrix,
- leeren Zeilen,
- nicht rechteckigen Zeilen,
- einer einzelnen Zeile oder Spalte,
- falschen Vergleichen an der letzten gültigen Position.

Die Dimensionen sollten einmal bestimmt und alle Indizes gegen
\(0\le row<r\) beziehungsweise \(0\le column<c\) geprüft werden.

---

## 9. Klassische Array- und String-Patterns

### 9.1 Deduplizierung eines sortierten Arrays

In einem sortierten Array stehen Duplikate nebeneinander. Ein Read-Index
durchläuft alle Werte, ein Write-Index markiert die nächste Position für einen
neuen eindeutigen Wert.

~~~python
def deduplicate_sorted(values):
    """Deduplicate a sorted list in place and return its logical length."""
    if not values:
        return 0

    write = 1
    for read in range(1, len(values)):
        if values[read] != values[write - 1]:
            values[write] = values[read]
            write += 1
    return write
~~~

Trace für [1, 1, 2, 2, 2, 4]:

| read | gelesener Wert | write vorher | Aktion | gültiger Prefix |
|---:|---:|---:|---|---|
| 1 | 1 | 1 | Duplikat | [1] |
| 2 | 2 | 1 | an Index 1 schreiben | [1, 2] |
| 3 | 2 | 2 | Duplikat | [1, 2] |
| 4 | 2 | 2 | Duplikat | [1, 2] |
| 5 | 4 | 2 | an Index 2 schreiben | [1, 2, 4] |

Die Invariante:

> Vor jedem read-Schritt enthält values[:write] genau die eindeutigen Werte aus
> dem bereits gelesenen Prefix, in sortierter Reihenfolge.

Nach dem Lauf ist nur values[:write] logisch gültig. Die physische Liste bleibt
gleich lang, solange der Rest nicht gelöscht wird. Zeit: \(\Theta(n)\),
Auxiliary Space: \(\Theta(1)\).

### 9.2 Zwei sortierte Arrays zusammenführen

~~~python
def merge_sorted(left_values, right_values):
    """Merge two sorted sequences into a new sorted list."""
    left_index = 0
    right_index = 0
    merged = []

    while (
        left_index < len(left_values)
        and right_index < len(right_values)
    ):
        if left_values[left_index] <= right_values[right_index]:
            merged.append(left_values[left_index])
            left_index += 1
        else:
            merged.append(right_values[right_index])
            right_index += 1

    merged.extend(left_values[left_index:])
    merged.extend(right_values[right_index:])
    return merged
~~~

Trace für left=[2, 5, 8] und right=[1, 5, 9]:

| Vergleich | Auswahl | Ergebnis |
|---|---:|---|
| 2 gegen 1 | 1 rechts | [1] |
| 2 gegen 5 | 2 links | [1, 2] |
| 5 gegen 5 | 5 links | [1, 2, 5] |
| 8 gegen 5 | 5 rechts | [1, 2, 5, 5] |
| 8 gegen 9 | 8 links | [1, 2, 5, 5, 8] |
| Rest | 9 rechts | [1, 2, 5, 5, 8, 9] |

Die Invariante:

> merged ist sortiert und enthält genau die bereits konsumierten Elemente
> beider Eingaben.

Jeder der \(n+m\) Werte wird genau einmal angehängt. Zeit und Ergebnisplatz sind
\(\Theta(n+m)\). Das <= sorgt dafür, dass bei gleichen Schlüsseln das linke
Element zuerst übernommen wird; dadurch kann ein Merge stabil sein.

### 9.3 Anagramm-Check durch Häufigkeiten

Zwei Texte sind Anagramme, wenn jedes Zeichen gleich oft vorkommt. Für allgemeine
Zeichen eignet sich eine Hash Map:

~~~python
def are_anagrams(left_text, right_text):
    """Return whether two strings contain the same character counts."""
    if len(left_text) != len(right_text):
        return False

    counts = {}
    for character in left_text:
        counts[character] = counts.get(character, 0) + 1

    for character in right_text:
        remaining = counts.get(character, 0) - 1
        if remaining < 0:
            return False
        counts[character] = remaining

    return True
~~~

Unter üblichen Hashing-Annahmen kostet die Funktion \(\Theta(n)\) Zeit. Der
Zusatzspeicher ist \(O(k)\), wobei \(k\) die Anzahl unterschiedlicher Zeichen
bezeichnet.

Sortieren beider Strings wäre mit \(\Theta(n\log n)\) Zeit ebenfalls korrekt,
aber asymptotisch langsamer. Für ein festes Alphabet, beispielsweise genau 26
ASCII-Kleinbuchstaben, kann die Hash Map durch ein Array fester Größe ersetzt
werden.

### 9.4 Rotation: Modulo und leere Eingaben

Bei Rotationen muss \(k\) auf die Listenlänge normalisiert werden:

\[
k_{\text{effektiv}}=k\bmod n.
\]

Eine Rotation um \(n\), \(2n\) oder \(3n\) Positionen verändert nichts.
Modulo darf bei einer leeren Liste jedoch nicht ausgeführt werden, weil die
Division durch null undefiniert ist. Deshalb prüft rotate_right zuerst, ob
values leer ist.

---

## 10. Formalisierung

### 10.1 Array-Abstraktion

Ein eindimensionales Array \(A\) der Länge \(n\) ist eine Folge

\[
A[0],A[1],\dots,A[n-1]
\]

mit direkter Zuordnung jedes gültigen Index zu einer Speicherposition. Für ein
statisches Array gleich breiter Elemente gilt:

\[
\operatorname{address}(A[i])
=\operatorname{base}(A)+i\cdot w,
\]

wobei \(w\) die Elementbreite ist.

Die Indexinvariante lautet:

\[
0\le i<n.
\]

Ein Zugriff außerhalb dieses Bereichs ist ungültig. Python prüft die Grenze und
wirft IndexError; niedrigere Sprachen können ohne Prüfung undefiniertes Verhalten
zeigen.

### 10.2 Dynamisches Array

Ein dynamisches Array ergänzt:

- length \(n\),
- capacity \(C\),
- einen Puffer mit \(C\) Positionen,

mit

\[
0\le n\le C.
\]

append schreibt bei \(n<C\) direkt an Position \(n\). Bei \(n=C\) wird ein
größerer Puffer erzeugt. Wächst die Kapazität geometrisch um einen Faktor
\(\alpha>1\), summieren sich die Resize-Kopien über \(n\) appends zu \(O(n)\).
Damit ist append amortisiert \(O(1)\).

### 10.3 In-Place-Invarianten

In-Place-Verfahren überschreiben Eingabepositionen. Ihre Korrektheit benötigt
eine Aussage darüber, welcher Bereich bereits endgültig und welcher noch
unverarbeitet ist.

Für die Deduplizierung:

\[
values[0:write]
\]

enthält nach Verarbeitung von values[0:read] genau dessen eindeutige Werte. Da
read nur nach rechts läuft und write nie größer als read wird, überschreibt die
Funktion keinen noch ungelesenen Wert.

### 10.4 Prefix-Sum-Definition

Für \(A\) der Länge \(n\) definieren wir \(P\) der Länge \(n+1\):

\[
P[0]=0
\]

und

\[
P[i+1]=P[i]+A[i].
\]

Dann gilt für ein inklusives Intervall \(0\le l\le r<n\):

\[
\sum_{i=l}^{r}A[i]=P[r+1]-P[l].
\]

Die zusätzliche Null vermeidet einen Sonderfall für \(l=0\).

### 10.5 Matrix-Index

Für eine rechteckige \(r\times c\)-Matrix im Row-Major-Layout wird Position
\((i,j)\) auf den linearen Index

\[
i\cdot c+j
\]

abgebildet. Gültig sind

\[
0\le i<r,\qquad 0\le j<c.
\]

Ein vollständiger Durchlauf ist \(\Theta(rc)\), weil jedes Matrixelement
mindestens einmal betrachtet werden muss.

---

## 11. Häufige Fehler

### 11.1 Jede Listenoperation für O(1) halten

Der Indexzugriff ist konstant, Suche und Verschiebung sind es nicht.
values.insert(0, x), pop(0), remove(x) und x in values können linear sein.

### 11.2 list.insert(0, x) als Queue verwenden

Wiederholtes Entfernen oder Einfügen am Listenanfang führt zu quadratischen
Verschiebungen. Für eine Queue ist collections.deque mit Operationen an beiden
Enden geeigneter; die Datenstruktur folgt in Modul 05.

### 11.3 Slices für kostenlose Views halten

Python-Listenslices kopieren Referenzen. NumPy-Slices sind häufig Views. Gleiche
Syntax bedeutet hier nicht dasselbe Speicherverhalten.

### 11.4 Während der Iteration strukturell verändern

~~~python
def remove_negatives_wrong(values):
    """Demonstrate skipped elements during mutation."""
    for value in values:
        if value < 0:
            values.remove(value)
~~~

Nach einem remove rückt das nächste Element nach links, während der Iterator
weitergeht. Dadurch können Werte übersprungen werden. Sichere Alternativen sind:

- neue Liste per Comprehension erzeugen,
- rückwärts über Indizes laufen,
- ein Write-Index für In-Place-Filterung.

~~~python
def remove_negatives_in_place(values):
    """Filter negative numbers with a write index."""
    write = 0
    for value in values:
        if value >= 0:
            values[write] = value
            write += 1
    del values[write:]
~~~

### 11.5 Verschachtelte Listen mit Multiplikation erzeugen

[[0] * columns] * rows teilt dieselbe Zeile. Eine Comprehension erzeugt
unabhängige Zeilen.

### 11.6 In-place und Kopie verwechseln

- values.sort() verändert values und gibt None zurück.
- sorted(values) erzeugt eine neue Liste.
- values.reverse() verändert values.
- reversed(values) erzeugt einen Iterator.

APIs sollten diese Unterschiede bewusst nutzen.

### 11.7 Prefix-Sum-Grenzen vermischen

Einige Implementierungen verwenden inklusive, andere halb offene Intervalle.
Dieses Skript verwendet für Abfragen [left, right] inklusive und deshalb
prefix[right+1]-prefix[left]. Eine klare Konvention verhindert Off-by-One-Fehler.

### 11.8 Strings mit += unkritisch in großen Schleifen aufbauen

Auch wenn CPython lokale Fälle optimiert, ist join das explizite lineare Muster
für viele bekannte Teile. Bei Streaming oder sehr großen Ausgaben eignen sich
zusätzlich io.StringIO oder direktes Schreiben in einen Stream.

### 11.9 NumPy-Speedup mit besserer Big-O verwechseln

Eine vektorisierte und eine Python-basierte elementweise Operation können beide
\(\Theta(n)\) sein. NumPy reduziert die konstanten Kosten. Für echte
Komplexitätsverbesserungen muss sich die Anzahl asymptotischer Arbeitsschritte
ändern.

---

## 12. Systematisches Vorgehen bei Array-Aufgaben

### Schritt 1: Eigenschaften der Eingabe benennen

Ist sie sortiert? Darf sie verändert werden? Sind Werte homogen? Gibt es viele
Abfragen oder Updates? Ist die Matrix garantiert rechteckig?

### Schritt 2: Gewünschte Ausgabe und Platzbudget klären

Muss ein neues Array entstehen oder genügt eine logische Länge? Ist
\(O(n)\)-Hilfsspeicher erlaubt? Muss die ursprüngliche Reihenfolge erhalten
bleiben?

### Schritt 3: Teure Operationen markieren

Achte auf Slicing, Front-Insert, lineare Membership-Tests, wiederholte
Konkatenation und versteckte Sortierung.

### Schritt 4: Passendes Muster erkennen

- Viele unveränderliche Bereichssummen: Prefix Sums.
- Sortierte Eingabe und paarweiser Vergleich: Two Pointers oder Merge.
- Mutation ohne Zusatzarray: Read-/Write-Indizes oder Swap.
- Rechteckige Daten: geschachtelte Traversierung mit klaren Grenzen.
- Zeichenhäufigkeiten: Hash Map oder festes Zählerarray.

### Schritt 5: Invariante formulieren

Beschreibe, welcher Prefix, welches Intervall oder welcher Ergebnisbereich
bereits korrekt ist. Eine gute Invariante führt oft direkt zur Implementierung.

### Schritt 6: Kantenfälle testen

Leere Eingabe, ein Element, alle Werte gleich, keine Duplikate, Rotation um
Vielfache von \(n\), einzelne Matrixzeile und Unicode-Text.

### Schritt 7: Zeit und Speicher getrennt analysieren

Berücksichtige Ergebnisgröße, Slices, Kopien, Views und temporäre Arrays. Nenne
bei mehreren Eingaben ihre Größen getrennt.

---

## 13. Selbstkontrolle

### Aufgabe 1

Bestimme Zeit und Auxiliary Space:

~~~python
def every_second(values):
    return values[::2]
~~~

### Aufgabe 2

Warum ist die folgende Matrixinitialisierung fehlerhaft?

~~~python
matrix = [[None] * 4] * 5
~~~

### Aufgabe 3

Eine Liste besitzt \(n\) Werte und es werden \(q\) Bereichssummen abgefragt.
Vergleiche naive Berechnung und Prefix Sums.

### Aufgabe 4

Welche Komplexität besitzt die Funktion in \(n=\text{len(left)}\) und
\(m=\text{len(right)}\)?

~~~python
def concatenate_lists(left, right):
    result = left.copy()
    result.extend(right)
    return result
~~~

### Aufgabe 5

Erkläre, warum eine NumPy-Vektorisierung eine Python-Schleife stark schlagen
kann, obwohl beide \(\Theta(n)\) Zeit benötigen.

### Aufgabe 6

Welche Invariante macht deduplicate_sorted korrekt und warum gilt
\(write\le read\)?

### Lösungen

1. Das Slice kopiert ungefähr \(n/2\) Referenzen. Zeit und Ergebnisplatz sind
   \(\Theta(n)\).
2. Alle fünf äußeren Positionen referenzieren dieselbe innere Liste. Eine
   Mutation einer Zeile erscheint in allen Zeilen. Verwende eine
   Comprehension, die jede Zeile neu erzeugt.
3. Naiv kostet eine Abfrage bis zu \(\Theta(n)\), also insgesamt
   \(\Theta(qn)\). Prefix Sums benötigen \(\Theta(n)\) Vorbereitung,
   \(\Theta(n)\) Speicher und danach \(\Theta(1)\) pro Abfrage, insgesamt
   \(\Theta(n+q)\).
4. copy kostet \(\Theta(n)\), extend \(\Theta(m)\). Insgesamt Zeit und
   Ergebnisplatz: \(\Theta(n+m)\).
5. NumPy speichert typisierte Werte kompakt, nutzt Cache Locality und führt die
   Schleife in optimiertem kompiliertem Code, gegebenenfalls mit SIMD, aus. Die
   asymptotische Zahl verarbeiteter Werte bleibt linear, die konstanten Kosten
   sinken.
6. values[:write] enthält genau die eindeutigen Werte des bereits gelesenen
   Prefix. write wächst höchstens einmal pro read-Schritt und startet nicht
   rechts von read. Daher wird kein ungelesener Wert überschrieben.

---

## 14. Zusammenfassung

Arrays ermöglichen direkten Indexzugriff, weil Positionen rechnerisch aus einem
zusammenhängenden, gleichmäßig aufgebauten Speicherlayout bestimmt werden.
Python-Listen erweitern dieses Prinzip zu dynamischen Arrays aus
Objektreferenzen. Over-Allocation macht append amortisiert konstant, während
Insert und Delete innerhalb der Liste lineare Verschiebungen verursachen.

Die wichtigsten Erkenntnisse:

- Indexzugriff, Zuweisung und len sind \(\Theta(1)\).
- append und pop am Ende sind amortisiert \(\Theta(1)\).
- Suche, Frontoperationen und Änderungen in der Mitte sind im Worst Case
  \(\Theta(n)\).
- List Slicing und Kopien benötigen Zeit proportional zur kopierten Länge.
- Strings sind immutable; viele Teile werden mit join planbar linear verbunden.
- In-Place-Muster verwenden Indizes und klare Invarianten, um Zusatzspeicher zu
  sparen.
- Prefix Sums tauschen \(\Theta(n)\) Vorbereitung und Speicher gegen konstante
  Bereichsabfragen.
- Eine vollständige \(r\times c\)-Matrix-Traversierung benötigt
  \(\Theta(rc)\).
- Rotation, Deduplizierung und Merge beruhen auf kontrollierten Indexbereichen
  statt unnötigen Slices oder Verschiebungen.
- NumPy erreicht bei homogenen numerischen Daten meist dieselbe Big-O-Klasse wie
  Python-Schleifen, aber mit kompaktem Speicher, besserer Cache-Nutzung und
  vektorisierter Ausführung.

Wenn du die Kosten der Standardoperationen aus Speicherbewegungen begründen,
Prefix Sums korrekt herleiten und den praktischen NumPy-Vorteil ohne falsche
Big-O-Aussagen erklären kannst, sind die Qualifikationsziele dieses
Theorieabschnitts erreicht.
