# 01-basic — Runtime Lab

## Goal

In this project, you investigate five functions whose work ranges from constant
to quadratic growth. First derive their complexity classes from the Java code,
then compare your predictions with reproducible runtime measurements.

The goal is not to infer Big O from a single duration. Instead, connect three
views:

1. the number of operations executed by the code,
2. the shape of the curve in the log-log plot,
3. the runtime change when the input size doubles.

## Why a Java application?

Benchmarks must be repeatable under the same conditions, save their results as
CSV, and create a plot without a manual notebook cell order. JUnit tests verify
the deterministic parts independently of noisy timing data.

## Project structure

~~~text
01-basic/
├── README.md
├── SOLUTION.md
├── pom.xml
├── src/main/java/lab/dsa/module01/basic/RuntimeLab.java
├── src/test/java/lab/dsa/module01/basic/RuntimeLabTest.java
└── results/              # generated at runtime and not versioned
    ├── measurements.csv
    └── runtime_growth.png
~~~

## Preparation

Install JDK 21 and Maven 3.9 or newer. From this project directory, run:

~~~bash
mvn test
~~~

## Tasks

### 1. Derive predictions from the code

Open `RuntimeLab.java` and read `curveA` through `curveE`. Do not run the
application yet. Record one prediction and code-based justification per curve.

| Curve | Predicted class | Justification from the code |
|---|---|---|
| A |  |  |
| B |  |  |
| C |  |  |
| D |  |  |
| E |  |  |

Choose from exactly these classes: \(O(1)\), \(O(\log n)\), \(O(n)\),
\(O(n\log n)\), and \(O(n^2)\).

For every function, identify which loop depends on \(n\), how its control
variable changes, and whether loops execute sequentially or are nested.

### 2. Run the lab

~~~bash
mvn exec:java
~~~

The application measures each function for six increasing input sizes,
calibrates repeated calls for fast functions, uses the median of several
samples, writes `results/measurements.csv`, creates
`results/runtime_growth.png` with logarithmic axes, and prints an empirical
log-log slope for every curve.

Custom sizes and a shorter trial run are available:

~~~bash
mvn exec:java -Dexec.args="--sizes 64 128 256 512 1024"
mvn exec:java -Dexec.args="--sizes 32 64 128 256 --repeats 3 --min-sample-ms 2"
~~~

Use `mvn exec:java -Dexec.args="--help"` to list every option.

### 3. Interpret the measurements

| Curve | Measured slope | Doubling behavior | Final class |
|---|---:|---|---|
| A |  |  |  |
| B |  |  |  |
| C |  |  |  |
| D |  |  |  |
| E |  |  |  |

Answer these questions in your own words:

1. Which curve stays nearly flat?
2. Which curve grows by roughly a factor of four when \(n\) doubles?
3. Why does \(O(\log n)\) have no fixed positive slope in a log-log plot?
4. Why may measured slopes differ from theoretical ideal values?
5. Which conclusion follows from code analysis, and which only from the experiment?

Read `SOLUTION.md` only after writing down your own mapping and reasoning.

## Measurement method

Very fast functions approach the resolution of a single timing measurement.
The lab therefore invokes them repeatedly within one sample until a minimum
duration is reached, then divides the elapsed duration by the call count. The
median of several samples is less sensitive to short disturbances than a single
measurement or the mean.

The default sizes are powers of two so doubling behavior is directly visible.
The tests intentionally avoid fixed runtime thresholds because results depend
on the computer and its current load. They verify exact abstract operation
counts, configuration validation, slope estimation, a small benchmark series,
and CSV export.

## Hints

- A nested loop is quadratic only if both execution counts grow proportionally to \(n\).
- Repeated halving suggests logarithmic growth.
- Linearithmic work combines a linear and a logarithmic factor.
- Constant call and measurement overhead often dominates small inputs.
- A plot supports a theoretical analysis but does not prove it.

## Done when

- you predicted and justified all five classes before measuring,
- all JUnit tests pass,
- the application generates the CSV and PNG artifacts,
- you classified all five curves using code and measurements,
- you can explain deviations between empirical slopes and theoretical classes,
- you explained why \(O(n\log n)\) eventually beats \(O(n^2)\).
