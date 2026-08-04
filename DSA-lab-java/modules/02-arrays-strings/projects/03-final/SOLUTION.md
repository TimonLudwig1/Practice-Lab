# Solution and Interpretation

The generator computes baseline plus drift, periodic motion, and Gaussian noise,
then adds known offsets. A local seeded random generator isolates reproducibility.

The rolling-sum invariant says that before emitting each average, `rollingSum`
equals exactly the current window. After the first \(O(w)\) sum, each step
subtracts the departing value and adds the entering one. Total time is \(O(n)\),
with \(O(1)\) working space beyond output.

For prefix sums, `prefix[i]` equals the sum before index `i`. Therefore
`prefix[end] - prefix[start]` answers `[start, end)` in \(O(1)\). Building the
index costs \(O(n)\) time and space; \(q\) queries cost \(O(q)\).

Z-score detection uses three linear phases: mean, population variance, then
classification. A constant series has zero deviation and therefore returns no
outliers. The threshold comparison is inclusive. Apart from \(a\) returned
outliers, working space is \(O(1)\).

The benchmark validates shapes, values, and outlier indices before timing. Small
floating-point differences can arise because addition is not associative, so
numerical arrays use tight tolerances. Generation, validation, CSV writing, and
plotting stay outside measured calls.

| Operation | Build/time | Query | Working space without output |
|---|---:|---:|---:|
| sensor generation | \(O(n)\) | — | \(O(1)\) |
| moving average | \(O(n)\) | — | \(O(1)\) |
| prefix index | \(O(n)\) | \(O(1)\) | \(O(n)\) |
| batch of \(q\) ranges | — | \(O(q)\) | \(O(1)\) |
| z-score outliers | \(O(n)\) | — | \(O(1)\) |

A global z-score can be misleading with strong trends, changing variance,
seasonality, or many anomalies. Local windows, median absolute deviation, or
seasonal decomposition are possible extensions. Benchmark durations also depend
on JVM warm-up, CPU load, garbage collection, hardware, and input size.
