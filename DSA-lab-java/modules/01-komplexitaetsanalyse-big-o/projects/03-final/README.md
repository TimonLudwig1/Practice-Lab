# 03-final — Performance Audit of a Data Pipeline

## Scenario

An e-commerce team aggregates event data into customer summaries. The existing
pipeline is correct but becomes slow for larger CSV files. Perform a complete
performance audit: locate bottlenecks, derive their time and space complexity,
design an efficient implementation, prove result equality, benchmark both over
growing datasets, and document the speedup.

The synthetic dataset is generated locally from a fixed seed; nothing is downloaded.

## Project structure

~~~text
03-final/
├── README.md
├── SOLUTION.md
├── pom.xml
├── src/main/java/lab/dsa/module01/finalproject/
│   ├── AuditPipeline.java
│   ├── GenerateData.java
│   └── RunAudit.java
├── src/test/java/lab/dsa/module01/finalproject/AuditPipelineTest.java
├── data/events.csv                  # generated, not versioned
└── results/                         # generated, not versioned
    ├── customer_summary.csv
    ├── performance_audit.csv
    ├── performance_audit.png
    └── AUDIT_REPORT.md
~~~

## Data model

Each event has `event_id`, `customer_id`, `category`, `amount_cents`,
`discount_cents`, `status`, and `event_timestamp`. Only `completed` events count.
For each customer, the pipeline returns completed-event count, net revenue,
integer average net amount, unique-category count, and latest completed timestamp.

Integer cents avoid rounding differences between implementations.

## Phase 1: analyze the baseline

Read only `inefficientPipeline` in `AuditPipeline.java`. Define \(n\) as all
events and \(u\) as customers with at least one completed event. Determine the
cost of list membership, the number of full scans, the worst-case runtime in
terms of \(n\) and \(u\), and the temporary collections.

| Pipeline | Expected time | Expected auxiliary space |
|---|---|---|
| Baseline |  |  |
| Refactoring target |  |  |

## Phase 2: design the refactoring

Design a single-pass aggregation. Decide how to access one customer accumulator,
update metrics incrementally, store unique categories, and retain deterministic
output order. Then compare your design with `optimizedPipeline`. Both methods
must return exactly equal immutable `CustomerSummary` records.

## Phase 3: test

Install JDK 21 and Maven 3.9 or newer, then run `mvn test`. Tests cover seeded
generation, CSV loading, equivalent results on constructed and generated data,
status filtering, benchmark validation, and report exports.

## Phase 4: benchmark before and after

~~~bash
mvn exec:java
mvn exec:java -Dexec.args="--rows 1000 --sizes 250 500 1000 --repeats 2"
mvn exec:java -Dexec.args="--reuse-data"
~~~

The default run generates 8,000 events with seed 20260716 and measures five
growing prefixes. Generation and CSV parsing stay outside the timed region.
Before every measurement, complete result equality is checked.

## Phase 5: interpret the report

Explain doubling behavior, whether speedup grows with \(n\), when the difference
becomes practically relevant, which change yields the asymptotic improvement,
which costs remain, why correctness outranks speed alone, and which limitations
the synthetic benchmark has.

Read `SOLUTION.md` only after your own analysis.

## Measurement rules

- Absolute durations apply only to the executing machine.
- Medians reduce short disturbances.
- Both variants receive the same immutable event records.
- Generation and CSV parsing are excluded from aggregation timing.
- Speedup is baseline time divided by optimized time.
- Faster code counts as success only when all business results remain equal.

## Done when

- the baseline is derived as \(O(nu)\), hence \(O(n^2)\) in the worst case,
- list membership and repeated-scan costs are explained,
- all tests pass and generation is reproducible,
- both pipelines match for every benchmark size,
- all four output artifacts are created,
- the optimized cost \(O(n+u\log u)\) is justified,
- results and audit limitations are documented in your own words.
