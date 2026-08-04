# Solution and interpretation

Read this file only after analyzing all five functions and running the lab.

## Mapping

| Curve | Class | Reasoning |
|---|---|---|
| A | \(\Theta(1)\) | The loop always executes exactly 32 times, independently of \(n\). |
| B | \(\Theta(\log n)\) | Every outer step halves the problem size; the inner loop has the constant length 32. |
| C | \(\Theta(n)\) | A constant block of eight operations runs once per input unit. |
| D | \(\Theta(n\log n)\) | The problem size is repeatedly halved for each of the \(n\) elements. |
| E | \(\Theta(n^2)\) | The inner lengths sum to \(0+1+\dots+(n-1)=n(n-1)/2\). |

The functions return their abstract operation counts, so small inputs can verify
the derivation exactly. The benchmark still measures real JVM execution time.
Loop management, method calls, just-in-time compilation, garbage collection,
and other runtime effects influence the observations.

## Expected measurement shapes

Curve A should remain nearly flat. In a log-log plot, curve C approaches slope
1 and curve E approaches slope 2. Curve D lies between them and has a slope
slightly greater than the linear curve because of its logarithmic factor.

For curve B, \(T(n)=\log n\) gives \(\log T(n)=\log(\log n)\), not a straight
line with a fixed positive slope. It grows slowly and becomes relatively
flatter for larger \(n\). Over a limited range, constant runtime overhead can
hide even more of that logarithmic contribution.

## Doubling ratios

The theoretical approximate factors are A: 1, B: just above 1, C: 2, D:
slightly above 2, and E: 4. Measurements fluctuate, especially for fast curves.
Timer resolution, scheduling, CPU frequency, cache behavior, JIT warm-up, and
garbage collection all matter. Batching calls and taking a median reduce, but do
not eliminate, this noise.

## Why n log n eventually beats n²

Consider the ratio

\[
\frac{n^2}{n\log n}=\frac{n}{\log n}.
\]

The numerator grows much faster than the logarithm in the denominator, so the
ratio grows without bound. Constant factors only shift the input size at which
the advantage becomes visible.

## What the experiment can establish

Code analysis justifies the asymptotic classes. The measurements show that the
observed growth over the tested range is compatible with those classes. They do
not prove the classes: finitely many data points cannot rule out different
behavior outside the measured range.
