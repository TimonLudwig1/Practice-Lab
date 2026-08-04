# 01-basic — Build a Dynamic Array

## Goal

Implement and investigate the central mechanism behind Java's `ArrayList`: a
dynamic array backed by a fixed contiguous reference buffer. The custom class
supports indexed access, append with capacity doubling, insertion and deletion
with shifts, real buffer replacement, and an immutable resize log.

The element store is an `Object[]`, not an `ArrayList`. A small `ArrayList` is
used only for diagnostic growth events and never stores user elements.

## Structure and commands

~~~text
01-basic/
├── README.md
├── SOLUTION.md
├── pom.xml
├── src/main/java/lab/dsa/module02/basic/
│   ├── DynamicArray.java
│   └── GrowthExperiment.java
├── src/test/java/lab/dsa/module02/basic/DynamicArrayTest.java
└── results/
    ├── growth_log.csv
    └── capacity_and_costs.png
~~~

Install JDK 21 and Maven 3.9 or newer, then run:

~~~bash
mvn test
mvn exec:java
mvn exec:java -Dexec.args="--appends 256 --initial-capacity 2"
~~~

## Tasks

1. State the invariants relating size, capacity, active positions, and resize behavior.
2. Starting with capacity 1, trace appending A through I, including copies.
3. Simulate `insert(1, "X")` on `[A, B, C, D, _, _, _, _]` and explain
   why shifting must proceed from right to left.
4. Simulate deleting index 2 and record the returned value, shifts, new size,
   and cleared buffer position.
5. Analyze `resize`, `ensureCapacity`, `append`, `insert`, `delete`, and index
   normalization for time and auxiliary-space complexity.

The deterministic experiment assigns one cost unit to the new write and one to
each reference copied during resize. A normal append costs 1; growing from size
32 to 33 costs 33. It writes a CSV log and a PNG showing capacity steps, resize
spikes, and cumulative average cost.

## Complexity table

Fill this before reading `SOLUTION.md`:

| Operation | Best case | Worst case | Amortized | Auxiliary space |
|---|---:|---:|---:|---:|
| indexed access |  |  |  |  |
| append |  |  |  |  |
| insert at start |  |  |  |  |
| insert at end |  |  |  |  |
| delete at start |  |  |  |  |
| delete at end |  |  |  |  |
| resize |  |  |  |  |

## Questions

Explain where resizes occur, why their spacing follows powers of two, which
value the cumulative average approaches, why amortized is not average-case
analysis, why deletion does not automatically shrink capacity, and what would
break if insertion shifted left to right.

## Done when

- the four core invariants are stated in your own words,
- append, insert, delete, and resize are understood without using `ArrayList`
  as element storage,
- all tests pass and both artifacts are generated,
- resize points and copy counts for at least 16 appends can be predicted,
- the geometric-series proof for amortized \(\Theta(1)\) append is understood.
