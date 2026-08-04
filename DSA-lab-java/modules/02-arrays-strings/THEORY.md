# Module 02 — Arrays and Strings

Arrays combine a fixed logical order with direct indexed access. Strings add
text semantics and immutability. This module explains memory layout, dynamic
growth, matrices, Unicode, and the core patterns that make array and string
problems efficient in Java.

By the end, you should be able to distinguish access from search, derive shift
and resize costs, reason about row-major traversal, implement in-place methods
with invariants, use prefix sums, and select appropriate Java types.

---

## 1. Intuition: numbered drawers

### 1.1 Direct access through fixed positions

Imagine equally sized drawers numbered from 0. If the first drawer starts at
address \(b\) and each entry occupies \(s\) bytes, index \(i\) starts at

\[
b+i\cdot s.
\]

This constant-size calculation explains \(O(1)\) array access. It does not mean
that finding an unknown value is constant: unsorted search may inspect every
position and costs \(O(n)\).

### 1.2 Why middle insertion is expensive

Inserting X at index 2 into `[A, B, C, D, E, _]` requires shifting E, D, and C
one position right. Copying must proceed right to left so unread values are not
overwritten. Inserting near the front costs \(\Theta(n)\); inserting at the end
is \(\Theta(1)\) when capacity is available.

### 1.3 The capacity problem

A fixed array cannot grow. A dynamic array keeps both a logical `size` and a
physical `capacity`. When full, it allocates a larger array and copies active
references. One append can therefore be linear, while a geometric growth policy
makes a long sequence amortized constant per append.

### 1.4 Strings are immutable signs

An array is like a board whose cells can be rewritten. A `String` is more like
a finished sign: an apparent modification creates a different string.

~~~java
String text = "data";
char first = text.charAt(0);
String changed = "D" + text.substring(1);
~~~

Use `StringBuilder` when many incremental modifications are required.

---

## 2. Simulation: how Java dynamic arrays grow

### 2.1 Static and dynamic arrays

~~~java
Integer[] fixed = new Integer[4];
fixed[0] = 10;

java.util.List<Integer> dynamic = new java.util.ArrayList<>();
dynamic.add(10);
~~~

The array has fixed length. `ArrayList` presents a growable interface backed by
an internal array. Its exact growth factor is an implementation detail; the API
does not promise doubling.

### 2.2 Simplified doubling by hand

With initial capacity 1, the capacity sequence is 1, 2, 4, 8, 16. Resizes copy
1, 2, 4, 8, ... old elements. Before \(n\) appends, total copies satisfy

\[
1+2+4+\dots<2n.
\]

Adding the \(n\) new writes gives \(O(n)\) work for the complete sequence and
amortized \(O(1)\) per append.

### 2.3 Capacity is not part of the ArrayList contract

Java exposes `ensureCapacity` and `trimToSize`, but not the current backing-array
length. Avoid depending on reflective access to implementation internals. The
custom dynamic-array project records its own capacity so the model is observable.

### 2.4 Insert step by step

~~~java
static void insert(int[] buffer, int size, int index, int value) {
    for (int position = size; position > index; position--) {
        buffer[position] = buffer[position - 1];
    }
    buffer[index] = value;
}
~~~

The suffix `[index, size)` shifts right. Cost is proportional to
`size - index`, plus a possible resize.

### 2.5 Delete step by step

~~~java
static int delete(int[] buffer, int size, int index) {
    int removed = buffer[index];
    for (int position = index; position < size - 1; position++) {
        buffer[position] = buffer[position + 1];
    }
    buffer[size - 1] = 0;
    return removed;
}
~~~

For reference arrays, clear the final active slot with `null` so it does not
keep an object reachable unnecessarily.

### 2.6 Measure shifts carefully

To compare front and end operations, prepare inputs outside the measured region,
warm up the JVM, repeat samples, consume results, and use several sizes. JMH is
the preferred tool for production-quality Java microbenchmarks. A classroom
timer is evidence about a range, not proof of complexity.

---

## 3. Cost model of Java arrays and ArrayList

### 3.1 Overview

| Operation | Array | ArrayList |
|---|---:|---:|
| read/write known index | \(O(1)\) | \(O(1)\) |
| search unsorted value | \(O(n)\) | \(O(n)\) |
| append with free capacity | not applicable | \(O(1)\) |
| append with growth | not applicable | \(O(n)\) |
| append over a sequence | not applicable | amortized \(O(1)\) |
| insert/delete near front | manual \(O(n)\) shift | \(O(n)\) |
| copy a range | \(O(k)\) | \(O(k)\) |

### 3.2 Index access is not search

~~~java
int value = values[500];                     // known position: O(1)
boolean found = java.util.Arrays.stream(values).anyMatch(x -> x == target); // O(n)
~~~

### 3.3 Java does not support negative indices

Python-style `values[-1]` is invalid in Java. The final array index is
`values.length - 1`; the final list index is `values.size() - 1`. An educational
wrapper may normalize negative indices explicitly, but that is not standard Java.

### 3.4 Range copies allocate

~~~java
int[] copy = java.util.Arrays.copyOfRange(values, start, end);
java.util.List<Integer> listCopy = new java.util.ArrayList<>(values.subList(start, end));
~~~

Both copy \(k=end-start\) values or references and need \(O(k)\) time and space.
`subList` alone is a backed view rather than an independent copy; structural
changes and lifetime coupling therefore require care.

### 3.5 Shallow copies share referenced objects

Copying `Widget[]` copies references, not the widgets. Replacing an array slot is
local to that array, but mutating a shared widget is visible through every copy.
Deep copying requires a domain-specific definition of how each object is cloned.

---

## 4. Multidimensional arrays and memory layout

### 4.1 A matrix as an array of rows

~~~java
int[][] matrix = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};
int centre = matrix[1][1];
~~~

Java's `int[][]` is an array of references to independent `int[]` rows. Rows may
have different lengths, so rectangular algorithms should validate their shape.

### 4.2 Aliasing rows

~~~java
int[] shared = new int[3];
int[][] wrong = {shared, shared, shared};
wrong[0][0] = 7; // every row observes the same array

int[][] correct = new int[3][3];
~~~

The first structure contains three references to one row. The second allocates
three distinct row arrays.

### 4.3 Row-major traversal

Although Java rows are separate arrays, visiting all values within one row before
moving to the next usually has better locality than repeatedly jumping between rows.

~~~java
long total = 0;
for (int[] row : matrix) {
    for (int value : row) {
        total += value;
    }
}
~~~

### 4.4 Primitive arrays versus boxed collections

An `int[]` stores primitive values compactly. An `ArrayList<Integer>` stores
references to boxed `Integer` objects. Primitive arrays usually require less
memory, avoid boxing, and offer better locality for homogeneous numerical work.

~~~text
int[]:                [4 bytes][4 bytes][4 bytes]
ArrayList<Integer>:   [reference] -> Integer object
~~~

This often changes constant factors without changing Big O. Specialized vector
libraries can further improve throughput, but asymptotic analysis still counts
the same number of elements.

### 4.5 Loops and streams

~~~java
static int[] squareLoop(int[] values) {
    int[] result = new int[values.length];
    for (int index = 0; index < values.length; index++) {
        result[index] = values[index] * values[index];
    }
    return result;
}

static int[] squareStream(int[] values) {
    return java.util.Arrays.stream(values).map(value -> value * value).toArray();
}
~~~

Both are \(\Theta(n)\). Streams express a pipeline but do not guarantee a
speedup. Measure realistic workloads and do not confuse syntax with complexity.

### 4.6 Copies, views, and contiguous storage

Arrays are contiguous primitive or reference buffers. `ArrayList` stores its
references in a contiguous internal array, while linked structures do not.
`subList` is a view; `Arrays.copyOfRange` is a copy. Always establish ownership,
mutation, and lifetime semantics before treating a range operation as cheap.

---

## 5. Strings as immutable sequences

### 5.1 Indexing, substring, and replacement

~~~java
String text = "algorithm";
char firstUnit = text.charAt(0);
String middle = text.substring(2, 6);
String replaced = text.replace('o', 'O');
~~~

Modern JDKs create independent substring content rather than keeping the entire
original backing storage alive. Creating or transforming a substring is not a
free view and generally costs proportional to produced text.

### 5.2 Repeated concatenation

~~~java
String result = "";
for (String word : words) {
    result = result + word;
}
~~~

Each iteration can copy the existing prefix. With steadily growing output, total
character copying can become quadratic.

### 5.3 Use StringBuilder

~~~java
StringBuilder builder = new StringBuilder();
for (String word : words) {
    builder.append(word);
}
String result = builder.toString();
~~~

For total output length \(L\), building is amortized \(O(L)\), excluding any
work needed to compute the parts.

### 5.4 Choose the right input size

For text algorithms, the number of strings may be less useful than the number
of UTF-16 code units, Unicode code points, bytes in a chosen encoding, or user-
perceived grapheme clusters. State which measure your complexity uses.

### 5.5 Unicode in Java

`char` is one UTF-16 code unit, not necessarily one Unicode code point. A
supplementary character occupies a surrogate pair.

~~~java
String text = "A😀B";
int codeUnits = text.length();
int codePoints = text.codePointCount(0, text.length());
int[] points = text.codePoints().toArray();
~~~

Even code points do not exactly match visible grapheme clusters. Locale-aware
case conversion and normalization can also change text length.

---

## 6. Pattern I: in-place operations

### 6.1 Meaning of in-place

An in-place algorithm mutates its input using \(O(1)\) auxiliary space beyond
fixed local variables. Its output may still occupy the original \(O(n)\) input
storage. Mutation must be part of the method contract.

~~~java
static void reverse(int[] values) {
    int left = 0;
    int right = values.length - 1;
    while (left < right) {
        int temporary = values[left];
        values[left++] = values[right];
        values[right--] = temporary;
    }
}
~~~

### 6.2 Rotation by three reversals

To rotate right by \(k\): normalize \(k\) with `Math.floorMod`, reverse the
whole array, reverse the first \(k\) values, then reverse the remainder.

~~~java
static void rotateRight(int[] values, int steps) {
    if (values.length < 2) return;
    int k = Math.floorMod(steps, values.length);
    reverseRange(values, 0, values.length - 1);
    reverseRange(values, 0, k - 1);
    reverseRange(values, k, values.length - 1);
}
~~~

Time is \(O(n)\), auxiliary space \(O(1)\).

### 6.3 Mutation versus memory

In-place code avoids allocation but changes caller-owned data, can make APIs
harder to reason about, and may prevent safe sharing. Returning a copy costs
\(O(n)\) space but provides simpler ownership. Choose deliberately.

---

## 7. Pattern II: prefix sums

### 7.1 Motivation

Repeatedly summing ranges directly costs \(O(qn)\) in the worst case for \(q\)
queries. One prefix index moves shared work into preprocessing.

### 7.2 Sentinel construction

~~~java
static long[] prefixSums(int[] values) {
    long[] prefix = new long[values.length + 1];
    for (int index = 0; index < values.length; index++) {
        prefix[index + 1] = prefix[index] + values[index];
    }
    return prefix;
}
~~~

The leading zero makes empty prefixes and half-open ranges uniform.

### 7.3 Range query

~~~java
static long rangeSum(long[] prefix, int start, int end) {
    return prefix[end] - prefix[start];
}
~~~

For values `[4, -1, 7, 3, 2]`, prefix is `[0, 4, 3, 10, 13, 15]`.
The range `[1, 4)` equals `13 - 4 = 9`.

### 7.4 Correctness invariant

For every valid \(i\),

\[
prefix[i]=\sum_{j=0}^{i-1}values[j].
\]

Subtracting two prefix values cancels everything before `start` and leaves
exactly `[start, end)`.

### 7.5 Complexity and break-even

Build time and memory are \(O(n)\); each query is \(O(1)\). Prefix sums are most
valuable when the underlying values remain unchanged across many queries. An
update invalidates later prefix entries; Fenwick trees or segment trees support
dynamic updates more efficiently.

---

## 8. Pattern III: matrix traversal

### 8.1 Full row traversal

~~~java
for (int row = 0; row < matrix.length; row++) {
    for (int column = 0; column < matrix[row].length; column++) {
        visit(matrix[row][column]);
    }
}
~~~

For \(r\) rows and \(c\) columns, a rectangular traversal is \(\Theta(rc)\).

### 8.2 Four neighbors

~~~java
int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
for (int[] direction : directions) {
    int nextRow = row + direction[0];
    int nextColumn = column + direction[1];
    if (0 <= nextRow && nextRow < rows &&
            0 <= nextColumn && nextColumn < columns) {
        visit(matrix[nextRow][nextColumn]);
    }
}
~~~

### 8.3 Treat boundaries explicitly

Empty matrices, empty rows, one row, one column, ragged rows, and corners should
be deliberate cases. Do not use exceptions as normal boundary control flow.

---

## 9. Classic array and string patterns

### 9.1 Deduplicate a sorted array

~~~java
static int removeDuplicates(int[] values) {
    if (values.length == 0) return 0;
    int write = 1;
    for (int read = 1; read < values.length; read++) {
        if (values[read] != values[write - 1]) {
            values[write++] = values[read];
        }
    }
    return write;
}
~~~

The prefix before `write` contains exactly the unique processed values. Time is
\(O(n)\), auxiliary space \(O(1)\).

### 9.2 Merge two sorted arrays

If the first array has a trailing buffer, merge from right to left. The largest
unwritten element belongs at the current final position, and writing there does
not destroy unread input.

~~~java
static void merge(int[] target, int valid, int[] other) {
    int left = valid - 1;
    int right = other.length - 1;
    int write = target.length - 1;
    while (right >= 0) {
        if (left >= 0 && target[left] > other[right]) {
            target[write--] = target[left--];
        } else {
            target[write--] = other[right--];
        }
    }
}
~~~

### 9.3 Anagrams by frequency

Count code points from one string and subtract those from the other. Under normal
hashing assumptions, time is expected \(O(n)\), with \(O(k)\) memory for \(k\)
distinct code points. Define whether case, whitespace, normalization, and locale matter.

### 9.4 Rotation edge cases

Normalize only after checking for an empty input, because modulo zero is invalid.
Use `Math.floorMod(steps, length)` so negative steps rotate in the intended direction.

---

## 10. Formalization

### 10.1 Array abstraction

An array of length \(n\) maps each integer index in `[0, n)` to exactly one
stored value. Direct access is constant because address computation uses a fixed
number of arithmetic operations.

### 10.2 Dynamic-array invariants

For logical size \(s\) and capacity \(c\): \(0\le s\le c\); active elements
occupy `[0, s)` in order; operations never expose `[s, c)`; resizing preserves
the active sequence.

### 10.3 In-place invariants

An invariant describes what portion is already final and what remains unread.
For backward merge, every position after `write` is final, and positions at or
before the input pointers remain available.

### 10.4 Prefix definition

With \(P[0]=0\) and \(P[i+1]=P[i]+A[i]\), every half-open sum is
\(P[end]-P[start]\).

### 10.5 Matrix index mapping

A truly contiguous row-major \(r\times c\) buffer maps `(row, column)` to
`row * c + column`. Java's nested arrays use row references, but flattening into
one primitive array can provide this representation explicitly.

---

## 11. Common errors

### 11.1 Assuming every list operation is O(1)

Known-index access is constant; search, front insertion, front deletion, copying,
and sorting are not.

### 11.2 Using ArrayList.remove(0) as a queue

Every removal shifts the remaining references. Repeating it can cost
\(\Theta(n^2)\). Use `ArrayDeque` for queue operations.

### 11.3 Treating every range as a free view

Know whether an API returns a backed view or a copy. Views can retain larger
structures and share mutations; copies cost time and memory but isolate ownership.

### 11.4 Structural modification during iteration

~~~java
for (Integer value : values) {
    if (value < 0) values.remove(value); // unsafe structural modification
}
~~~

Use `removeIf`, an explicit `Iterator.remove`, or a separate result collection.

### 11.5 Accidentally sharing matrix rows

Reusing one row reference creates aliases. Allocate each row separately unless
sharing is intentional.

### 11.6 Confusing in-place with copy

Method names and documentation should say whether caller-owned input changes.
Tests should verify both result and mutation behavior.

### 11.7 Mixing inclusive and half-open ranges

Choose one convention. Java APIs commonly use `[start, end)`, which allows length
`end - start` and empty ranges where `start == end`.

### 11.8 Building large strings with `+` in a loop

Use `StringBuilder` to avoid repeated prefix copies.

### 11.9 Confusing constant-factor speedups with better Big O

Primitive arrays, cache-friendly traversal, streams, vector libraries, and tuned
JDK methods can change runtime greatly while leaving the asymptotic class unchanged.

---

## 12. Systematic workflow for array tasks

1. Name input properties: length, sortedness, uniqueness, rectangular shape,
   mutability, numeric range, and Unicode interpretation.
2. Clarify output and space budget: mutate, return a copy, or return a view?
3. Mark expensive operations: shifts, copies, nested scans, boxing, sorting,
   concatenation, and library calls.
4. Recognize a pattern: two pointers, sliding window, prefix sum, backward merge,
   rolling aggregate, matrix boundaries, or frequency map.
5. State an invariant before writing the loop.
6. Test empty, singleton, all-equal, all-zero, negative, boundary, overflow,
   malformed, and large inputs as applicable.
7. Analyze time, auxiliary space, and output space separately.

---

## 13. Self-check

### Exercise 1

What are the time and auxiliary-space costs of copying an array range of length \(k\)?

### Exercise 2

Why can repeated `ArrayList.remove(0)` turn processing \(n\) items into quadratic time?

### Exercise 3

Build the prefix array for `[3, -2, 5, 4]` and answer `[1, 4)`.

### Exercise 4

Trace right rotation of `[1, 2, 3, 4, 5]` by 2 using three reversals.

### Exercise 5

Explain why iterating an `int[][]` row by row is usually more cache-friendly
than repeatedly jumping across rows.

### Exercise 6

Why can a primitive-array loop and a library-based implementation both be
\(\Theta(n)\) yet have substantially different durations?

### Solutions

1. \(\Theta(k)\) time and \(\Theta(k)\) new storage.
2. Each removal shifts the remaining suffix; costs sum to
   \((n-1)+(n-2)+\dots+1=\Theta(n^2)\).
3. Prefix is `[0, 3, 1, 6, 10]`; `[1, 4)` is `10 - 3 = 7`.
4. Reverse all: `[5,4,3,2,1]`; reverse first two: `[4,5,3,2,1]`; reverse
   the remainder: `[4,5,1,2,3]`.
5. Consecutive accesses remain within one compact row array, improving locality.
6. Big O describes growth. Boxing, allocation, dispatch, JIT optimization,
   memory layout, vectorization, and cache behavior change constant factors.

---

## 14. Summary

- Arrays provide constant known-index access through fixed positions.
- Dynamic arrays trade occasional linear resizes for amortized constant append.
- Middle insertions and deletions require linear shifts.
- Primitive arrays are compact; boxed collections add references and objects.
- Java strings are immutable UTF-16 sequences; code units are not code points.
- In-place algorithms require explicit mutation contracts and invariants.
- Prefix sums exchange \(O(n)\) preprocessing and space for \(O(1)\) range queries.
- Two pointers, rolling windows, backward merge, frequency maps, and boundary
  traversal solve many common tasks in linear time.
- Practical speedups and asymptotic improvements are different claims.

You have met the module goals when you can derive these costs from Java code,
state the invariants, implement the patterns, and test their edge cases.
