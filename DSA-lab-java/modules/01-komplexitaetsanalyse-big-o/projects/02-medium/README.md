# 02-medium — Complexity Detective

## Goal

Analyze ten short Java methods. Some hide costs in array copying, membership
tests, string concatenation, sorting, or list operations. Others contain nested
loops without being quadratic.

Work as if performing a performance audit:

1. name the input size and relevant operations,
2. derive time and auxiliary-space complexity,
3. check the prediction with controlled measurements,
4. explain differences between theory and observations.

## Project structure

~~~text
02-medium/
├── README.md
├── SOLUTION.md
├── pom.xml
├── src/main/java/lab/dsa/module01/medium/ComplexityDetective.java
├── src/test/java/lab/dsa/module01/medium/ComplexityDetectiveTest.java
└── results/
    ├── measurements.csv
    └── normalized_growth.png
~~~

## Part 1: predictions

Read only `case01` through `case10` before running the benchmark.

| Case | Time class | Auxiliary space | Dominant operation | Assumptions |
|---|---|---|---|---|
| 01 |  |  |  |  |
| 02 |  |  |  |  |
| 03 |  |  |  |  |
| 04 |  |  |  |  |
| 05 |  |  |  |  |
| 06 |  |  |  |  |
| 07 |  |  |  |  |
| 08 |  |  |  |  |
| 09 |  |  |  |  |
| 10 |  |  |  |  |

Treat `size` as \(n\). Mark visible loops, determine the costs of operations
such as `Arrays.copyOfRange`, `contains`, `add(0, value)`, string concatenation,
and `Arrays.sort`, then combine the costs. Distinguish peak live memory from all
allocations made over the complete run.

## Part 2: run the benchmark

Install JDK 21 and Maven 3.9 or newer, then run:

~~~bash
mvn test
mvn exec:java
~~~

The default run measures all ten cases over seven doubling input sizes. It
calibrates fast cases, stores sample medians in `results/measurements.csv`, and
creates a normalized log-log plot. Every curve starts at 1, making relative
growth comparable despite different constants.

Useful variants:

~~~bash
mvn exec:java -Dexec.args="--sizes 32 64 128 256 --repeats 3 --min-sample-ms 2"
mvn exec:java -Dexec.args="--cases 05 06 07"
mvn exec:java -Dexec.args="--help"
~~~

## Part 3: detective report

After measuring, compare predicted classes with empirical slopes. Explain which
cases contain quadratic copying or shifting, why nested loops are not always
quadratic, how the collection type changes membership cost, why JVM and library
constants affect small inputs, and which case has quadratic peak auxiliary
space.

Read `SOLUTION.md` only after completing your report.

## Hints

- Sequential linear loops give \(O(n+n)=O(n)\).
- Geometrically shrinking work can sum to \(n+n/2+n/4+\dots=O(n)\).
- Hash-set membership is expected constant time under normal hashing assumptions.
- The irregular deterministic sort input avoids an already sorted best case.
- A benchmark supports a derivation; it does not replace it.

## Done when

- all ten time and space classes are justified before measuring,
- hidden array-copy, string-copy, front-insertion, membership, and sorting costs
  are understood,
- all JUnit tests pass and both output artifacts are generated,
- every measured curve is compared with its prediction,
- at least three deviations are explained using constants, JVM effects, or the
  limited measurement range.
