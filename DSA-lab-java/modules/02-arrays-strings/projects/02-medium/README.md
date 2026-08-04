# 02-medium — Array and String Pattern Catalog

## Goal

Implement ten reusable Java patterns and explain their invariants, complexity,
and trade-offs: three-reversal rotation, backward in-place merge, prefix sums,
anagram counts, two-pointer deduplication, stable zero movement, product except
self, sliding-window substring search, spiral traversal, and run compression.

## Structure and commands

~~~text
02-medium/
├── README.md
├── SOLUTION.md
├── pom.xml
├── src/main/java/lab/dsa/module02/medium/
│   ├── PatternCatalog.java
│   └── CatalogDemo.java
└── src/test/java/lab/dsa/module02/medium/PatternCatalogTest.java
~~~

~~~bash
mvn test
mvn exec:java
~~~

For each method, state the input size, loop invariant, time, auxiliary space,
mutation behavior, and important preconditions. Trace the read/write indices for
the in-place algorithms. Pay particular attention to why merge writes from the
right, why prefix preprocessing pays off across many queries, why the sliding
window never moves backward, and how multi-digit run counts are written.

Read `SOLUTION.md` only after completing your own analysis.

## Done when

- all JUnit edge cases pass,
- every method has a justified time and space bound,
- you can trace mutation and index movement by hand,
- Unicode code points, empty inputs, duplicates, zeros, and rectangular-matrix
  validation are understood.
