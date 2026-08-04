# 03-final — Time-Series Toolkit Without a Dataframe Library

## Goal

Build a complete sensor-analysis pipeline using Java arrays and lists. It
combines a rolling window for moving averages, prefix sums for repeated range
queries, and linear passes for z-score outlier detection.

The synthetic temperature series contains slow drift, a daily cycle, Gaussian
noise, and deliberately injected positive and negative anomalies. Seed
`20260716` makes equal configurations reproducible; injected positions are kept
as separate ground truth and are unknown to the detector.

## Structure and commands

~~~text
03-final/
├── README.md
├── SOLUTION.md
├── pom.xml
├── src/main/java/lab/dsa/module02/finalproject/
│   ├── GenerateSensorData.java
│   ├── SensorToolkit.java
│   └── SensorBenchmark.java
├── src/test/java/lab/dsa/module02/finalproject/SensorToolkitTest.java
├── data/sensor_readings.csv
└── results/
    ├── benchmark_results.csv
    └── sensor_and_runtime_comparison.png
~~~

~~~bash
mvn test
mvn exec:java
mvn exec:java -Dexec.args="--size 10000 --queries 2000 --repetitions 2"
~~~

## Tasks

1. Generate deterministic data with a local `Random`, then inject known spikes.
2. Implement a moving average in \(O(n)\) using one rolling sum.
3. Build an immutable prefix index in \(O(n)\), answer `[start, end)` in \(O(1)\),
   and process \(q\) queries in \(O(q)\).
4. Detect values whose absolute population z-score reaches an inclusive threshold.
5. Combine the results into immutable snapshots without consuming query input twice.
6. Validate the optimized methods against independent Java reference methods,
   then benchmark them separately and write CSV and PNG artifacts.

The Python edition compares against NumPy. Because NumPy has no direct Java
counterpart, this edition uses transparent Java reference implementations. It
does not promise a particular speedup; the comparison demonstrates correctness,
measurement boundaries, constants, and the effect of algorithm choice.

## Done when

- equal seeds generate identical series and retain anomaly ground truth,
- moving average is \(O(n)\), prefix queries are \(O(1)\), and z-score detection
  is \(O(n)\),
- empty, constant, invalid, and boundary cases are tested,
- optimized and reference results agree within documented floating-point tolerances,
- CSV files and the comparison plot are generated,
- benchmark limitations and the statistical limits of a global z-score are understood.
