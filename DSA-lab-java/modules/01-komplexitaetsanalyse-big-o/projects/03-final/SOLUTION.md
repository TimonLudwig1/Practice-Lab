# Solution — Performance Audit

## Baseline analysis

Let \(n\) be all events and \(u\) the customers with completed events. The
baseline first copies its input in \(O(n)\). While building `customerIds`, every
completed event performs list membership costing up to \(O(u)\), for \(O(nu)\).
It then scans all \(n\) events again for each of \(u\) customers. Sorting output
costs \(O(u\log u)\). Total time is

\[
O(nu+u\log u),
\]

or \(O(n^2)\) when \(u\in O(n)\). Peak auxiliary space is \(O(n+u)\).

## Optimization strategy

The optimized pipeline maps each customer ID directly to one accumulator.
Every completed event increments its count, adds net revenue, adds a category to
a set, and updates the latest timestamp. Under ordinary hashing assumptions,
the pass takes expected \(O(n)\); sorting \(u\) keys gives total expected time
\(O(n+u\log u)\). Auxiliary space is \(O(u+c)\), where \(c\) is the total
number of category entries stored across customer sets.

## Correctness and measurement design

The audit compares complete immutable output lists before timing every size.
Tests add constructed edge cases and generated input. Generation and parsing are
outside the timed region, both implementations receive the same records, and
medians reduce transient noise. The speedup is

\[
\text{speedup}(n)=\frac{T_{\text{baseline}}(n)}{T_{\text{optimized}}(n)}.
\]

The speedup should generally grow because the baseline has the stronger growth
rate, although cache behavior, allocation, JIT compilation, garbage collection,
CPU frequency, and system load can perturb individual points.

## Limits

A production pipeline may be limited by networks, disks, databases,
serialization, or concurrency. Synthetic data cannot reproduce every real skew
or seasonal pattern. The transferable result is narrower: repeated full scans
and linear membership scale worse than single-pass hash aggregation. Absolute
seconds are machine-specific; output equality and growth behavior are not.
