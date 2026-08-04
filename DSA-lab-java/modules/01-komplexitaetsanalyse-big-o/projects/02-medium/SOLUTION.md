# Solution — Complexity Detective

Use this file only after your own code analysis and measurement.

## Mapping

| Case | Time | Auxiliary space | Decisive observation |
|---|---|---|---|
| 01 | \(\Theta(1)\) | \(\Theta(1)\) | Array size and index are independent of \(n\). |
| 02 | \(\Theta(\log n)\) | \(\Theta(1)\) | `remaining` is halved each step. |
| 03 | \(\Theta(n)\) | \(\Theta(1)\) | Sequential passes add to \(n+n\). |
| 04 | \(\Theta(n)\) | \(\Theta(1)\) | Inner work sums to \(n+n/2+n/4+\dots<2n\). |
| 05 | expected \(\Theta(n)\) | \(\Theta(n)\) | Set construction is linear and membership is expected constant. |
| 06 | \(\Theta(n^2)\) | \(\Theta(n)\) | Each of \(n\) unsuccessful list searches scans \(n\) elements. |
| 07 | \(\Theta(n^2)\) | \(\Theta(n)\) peak | Suffix copies have lengths \(n-1,n-2,\dots,1\). |
| 08 | \(\Theta(n^2)\) | \(\Theta(n)\) | Each front insertion shifts the current list. |
| 09 | \(\Theta(n^2)\) | \(\Theta(n^2)\) | All old immutable string prefixes stay reachable. |
| 10 | typical \(\Theta(n\log n)\) | \(\Theta(n)\) | Sorting irregular generated values dominates. |

Case 04 is linear because its dependent inner lengths form a geometric series.
Cases 05 and 06 demonstrate that `contains` depends on the collection: expected
constant time for `HashSet`, linear time for `ArrayList`. Case 07 performs
quadratic total copying, although only two arrays are live at an assignment.
Case 08 shifts \(0+1+\dots+(n-1)\) references. Case 09 retains every prefix, so
both copied characters and live characters sum quadratically.

Empirical slopes need not be exactly 0, 1, or 2. Constant method and timer costs,
JIT compilation, garbage collection, cache effects, optimized library code, and
a limited size range affect results. An \(n\log n\) curve also has a local
log-log slope only slightly above 1. The correct conclusion is that measurements
are compatible with the derived classes; the classes follow from code analysis.
