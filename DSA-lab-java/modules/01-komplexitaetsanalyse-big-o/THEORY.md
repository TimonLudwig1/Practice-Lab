# Module 01 — Complexity Analysis and Big O

Complexity analysis does not primarily ask how many milliseconds a program
takes. It asks how its resource requirements change as the input grows. That
perspective lets us compare algorithms across computers, programming languages,
and concrete inputs.

This chapter develops the topic at three levels:

1. **Intuition:** why growth matters more than one runtime measurement.
2. **Experiment:** how growth can be observed in code and measurement series.
3. **Formalization:** how Big O, Omega, Theta, time, and space complexity are defined.

By the end, you should be able to analyze unfamiliar Java code, interpret
measurements critically, and explain why an asymptotically better algorithm
eventually wins for sufficiently large inputs.

---

## 1. Intuition: growth matters more than a stopwatch

### 1.1 One problem, three solutions

Suppose an array contains measurements and we want to know whether any value
occurs more than once.

~~~java
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

static boolean containsDuplicatePairs(int[] values) {
    for (int left = 0; left < values.length; left++) {
        for (int right = left + 1; right < values.length; right++) {
            if (values[left] == values[right]) {
                return true;
            }
        }
    }
    return false;
}

static boolean containsDuplicateSorted(int[] values) {
    int[] ordered = Arrays.copyOf(values, values.length);
    Arrays.sort(ordered);
    for (int index = 1; index < ordered.length; index++) {
        if (ordered[index - 1] == ordered[index]) {
            return true;
        }
    }
    return false;
}

static boolean containsDuplicateSet(int[] values) {
    Set<Integer> seen = new HashSet<>();
    for (int value : values) {
        if (!seen.add(value)) {
            return true;
        }
    }
    return false;
}
~~~

All three methods return the same logical result, but their work grows differently:

- Pair comparison checks about \(n(n-1)/2\) pairs in the worst case. Doubling
  \(n\) almost quadruples the work.
- Sorting costs \(O(n\log n)\), followed by an \(O(n)\) scan.
- The set solution visits every element once. Under the usual hashing
  assumptions, its expected runtime is \(O(n)\).

For ten values, every version may appear instant. For ten million values, their
growth rates separate them dramatically. Complexity analysis deliberately
abstracts away from an isolated millisecond value.

### 1.2 Why a fast small measurement can mislead

A rough model for real runtime is

\[
T(n)=\text{number of elementary operations}\times\text{cost per operation}.
\]

The second factor depends on hardware, JVM version, just-in-time compilation,
cache behavior, the operating system, and implementation details. For small
inputs, optimized quadratic code can beat a linear implementation with a large
constant overhead. That does not change which one scales better eventually.

Complexity classes ignore constant factors and lower-order terms. For example,

\[
T(n)=3n^2+20n+400
\]

belongs to \(\Theta(n^2)\), because the quadratic term dominates as \(n\) grows.

### 1.3 The doubling experiment

A useful question is: what happens when the input size doubles?

| Growth | Work at \(n\) | Work at \(2n\) | Approximate factor |
|---|---:|---:|---:|
| constant | \(1\) | \(1\) | \(1\) |
| logarithmic | \(\log_2 n\) | \(\log_2 n+1\) | near \(1\) |
| linear | \(n\) | \(2n\) | \(2\) |
| linearithmic | \(n\log_2 n\) | \(2n(\log_2 n+1)\) | slightly above \(2\) |
| quadratic | \(n^2\) | \(4n^2\) | \(4\) |
| exponential | \(2^n\) | \(2^{2n}\) | \(2^n\) |

This table is not a proof, but it is a strong plausibility check for measurements.

---

## 2. Experiment: making growth visible

### 2.1 Count operations before measuring time

Runtime measurements are noisy. Counting abstract operations reveals the
structural growth without hardware effects.

~~~java
static long countPairChecks(int size) {
    long checks = 0;
    for (int left = 0; left < size; left++) {
        for (int right = left + 1; right < size; right++) {
            checks++;
        }
    }
    return checks;
}

for (int size : new int[] {10, 20, 40, 80}) {
    System.out.println(size + " " + countPairChecks(size));
}
~~~

| \(n\) | Pair checks \(n(n-1)/2\) | Factor from previous row |
|---:|---:|---:|
| 10 | 45 | — |
| 20 | 190 | 4.22 |
| 40 | 780 | 4.11 |
| 80 | 3,160 | 4.05 |

The factor approaches four because the linear part of
\((n^2-n)/2\) becomes relatively insignificant.

### 2.2 Binary search by hand

Logarithmic growth often occurs when each step reduces the remaining problem by
a constant factor. Binary search halves a sorted interval.

Searching for 31 in `[3, 7, 12, 18, 24, 31, 42, 56, 63]` gives:

| Step | Left | Right | Middle | Value | Decision |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 8 | 4 | 24 | continue right |
| 2 | 5 | 8 | 6 | 42 | continue left |
| 3 | 5 | 5 | 5 | 31 | found |

About one million sorted elements require only
\(\log_2(1{,}000{,}000)\approx20\) halvings.

~~~java
static int binarySearch(int[] values, int target) {
    int left = 0;
    int right = values.length - 1;

    while (left <= right) {
        int middle = left + (right - left) / 2;
        if (values[middle] == target) {
            return middle;
        }
        if (values[middle] < target) {
            left = middle + 1;
        } else {
            right = middle - 1;
        }
    }
    return -1;
}
~~~

### 2.3 Measuring Java runtime responsibly

`System.nanoTime()` is appropriate for measuring elapsed time. A useful
measurement series should also follow these rules:

- Create inputs outside the measured section.
- Warm up the code before recording values so the JVM can compile hot methods.
- Repeat measurements and use a robust statistic such as the median.
- Consume or verify results so the work cannot be optimized away.
- Cover several doublings of the input size.
- Batch very fast calls to exceed timer noise.

~~~java
import java.util.Arrays;
import java.util.function.Predicate;

static double benchmark(Predicate<int[]> function, int[] values, int repeats) {
    function.test(values);
    long[] durations = new long[repeats];
    for (int repeat = 0; repeat < repeats; repeat++) {
        long start = System.nanoTime();
        boolean result = function.test(values);
        durations[repeat] = System.nanoTime() - start;
        if (result) {
            throw new AssertionError("The generated input must be duplicate-free");
        }
    }
    Arrays.sort(durations);
    return durations[durations.length / 2] / 1_000_000_000.0;
}
~~~

The generated input should contain no duplicates so all variants perform their
worst-case work for this task. If the first two positions were equal, early
termination would make the scenarios incomparable.

For serious Java microbenchmarks, use JMH. The small laboratory benchmark is
intentionally transparent so its calibration and statistics can be studied.

### 2.4 Log-log plots

For \(T(n)=cn^k\), taking logarithms gives

\[
\log T(n)=\log c+k\log n.
\]

A polynomial therefore appears approximately as a straight line in a log-log
plot, with slope \(k\). Linear growth approaches slope 1 and quadratic growth
approaches slope 2. The function \(n\log n\) is not a pure power, so its curve
bends slightly between the two.

### 2.5 Doubling ratios

~~~java
static double[] doublingRatios(double[] durations) {
    double[] ratios = new double[Math.max(0, durations.length - 1)];
    for (int index = 1; index < durations.length; index++) {
        ratios[index - 1] = durations[index] / durations[index - 1];
    }
    return ratios;
}
~~~

Real measurements do not produce perfect factors. Small inputs are dominated by
constants and timer resolution; large inputs can expose caches, memory
allocation, garbage collection, CPU frequency changes, or system load.
Experiments provide evidence, not mathematical proof.

### 2.6 Keep theory and measurement separate

A careful report distinguishes:

1. **Theory:** pair comparison performs quadratic work in the worst case.
2. **Observation:** its measured doubling ratio approaches four over the tested range.
3. **Conclusion:** the observations are compatible with \(\Theta(n^2)\).

Finitely many measurements cannot prove asymptotic behavior outside their range.

---

## 3. Main growth classes

### 3.1 O(1): constant

~~~java
static Integer firstOrNull(int[] values) {
    return values.length == 0 ? null : values[0];
}
~~~

Array access is constant time. \(O(1)\) does not mean one operation or always
fast; it means the work does not grow with \(n\).

### 3.2 O(log n): logarithmic

Binary search is the standard example. After \(k\) halvings, roughly
\(n/2^k\) candidates remain. The search ends when

\[
\frac{n}{2^k}\le1,
\]

so \(k\ge\log_2 n\). Logarithm bases differ only by a constant factor and are
therefore omitted in asymptotic notation.

### 3.3 O(n): linear

~~~java
static int maximum(int[] values) {
    if (values.length == 0) {
        throw new IllegalArgumentException("Values must not be empty");
    }
    int largest = values[0];
    for (int index = 1; index < values.length; index++) {
        largest = Math.max(largest, values[index]);
    }
    return largest;
}
~~~

Every element must be considered in the worst case, so time is \(\Theta(n)\).
Only a fixed number of variables are used, so auxiliary space is \(\Theta(1)\).

### 3.4 O(n log n): linearithmic

Efficient comparison sorts such as merge sort typically have \(\log_2 n\)
levels and process all \(n\) elements on every level:

\[
n+n+\dots+n \text{ on } \log_2n \text{ levels}=n\log_2n.
\]

`Arrays.sort(int[])` uses a tuned dual-pivot quicksort and provides
\(O(n\log n)\) expected performance. The exact library algorithm and guarantee
depend on the overload and Java version, so library documentation belongs to a
precise analysis.

### 3.5 O(n²): quadratic

~~~java
record Pair(int left, int right) {}

static java.util.List<Pair> allOrderedPairs(int[] values) {
    java.util.List<Pair> pairs = new java.util.ArrayList<>();
    for (int left : values) {
        for (int right : values) {
            pairs.add(new Pair(left, right));
        }
    }
    return pairs;
}
~~~

The inner loop runs \(n\) times for each of \(n\) outer iterations. The output
also contains \(n^2\) pairs, so materializing it necessarily needs quadratic time
and space.

### 3.6 O(2ⁿ): exponential

~~~java
static long fibonacci(int number) {
    if (number <= 1) {
        return number;
    }
    return fibonacci(number - 1) + fibonacci(number - 2);
}
~~~

The naive recursion repeatedly solves the same subproblems. Its precise growth
is closer to \(\varphi^n\), but \(O(2^n)\) is a simple upper bound. Memoization
reduces this problem to linear time because each subproblem is solved once.

### 3.7 O(n!): factorial

~~~java
static <T> java.util.List<java.util.List<T>> permutations(java.util.List<T> values) {
    if (values.size() <= 1) {
        return java.util.List.of(new java.util.ArrayList<>(values));
    }
    java.util.List<java.util.List<T>> result = new java.util.ArrayList<>();
    for (int index = 0; index < values.size(); index++) {
        T selected = values.get(index);
        java.util.List<T> rest = new java.util.ArrayList<>(values);
        rest.remove(index);
        for (java.util.List<T> suffix : permutations(rest)) {
            java.util.List<T> permutation = new java.util.ArrayList<>();
            permutation.add(selected);
            permutation.addAll(suffix);
            result.add(permutation);
        }
    }
    return result;
}
~~~

Ten elements already have \(10!=3{,}628{,}800\) permutations. If all must be
produced, no algorithm can avoid the size of the output. Practical algorithms
therefore prune the search space or generate results lazily where possible.

### 3.8 Orders of magnitude

| \(n\) | \(\log_2n\) | \(n\) | \(n\log_2n\) | \(n^2\) | \(2^n\) |
|---:|---:|---:|---:|---:|---:|
| 10 | 3.3 | 10 | 33 | 100 | 1,024 |
| 100 | 6.6 | 100 | 664 | 10,000 | about \(1.27\cdot10^{30}\) |
| 1,000 | 10.0 | 1,000 | 9,966 | 1,000,000 | astronomical |
| 1,000,000 | 19.9 | 1,000,000 | 19,931,569 | \(10^{12}\) | impractical |

This illustrates why \(O(n\log n)\) eventually beats \(O(n^2)\). Constants move
the crossover point but cannot prevent it.

---

## 4. Formal definitions

### 4.1 Big O: asymptotic upper bound

For functions that are nonnegative for sufficiently large \(n\),

\[
f(n)\in O(g(n))
\]

if positive constants \(c\) and \(n_0\) exist such that

\[
0\le f(n)\le c\,g(n)\quad\text{for every }n\ge n_0.
\]

For \(f(n)=3n^2+20n+400\) and \(n\ge1\),

\[
3n^2+20n+400\le423n^2.
\]

Thus \(c=423\) and \(n_0=1\) establish \(f(n)\in O(n^2)\). Big O is only an
upper bound: a linear function also belongs to \(O(n^2)\). In practice, use the
tightest conventional bound that communicates useful information.

### 4.2 Omega: asymptotic lower bound

\[
f(n)\in\Omega(g(n))
\]

when positive \(c\) and \(n_0\) exist such that
\(f(n)\ge c\,g(n)\) for every \(n\ge n_0\).

### 4.3 Theta: tight asymptotic bound

\(f(n)\in\Theta(g(n))\) when both \(f(n)\in O(g(n))\) and
\(f(n)\in\Omega(g(n))\). Therefore, \(3n^2+20n+400\in\Theta(n^2)\).

### 4.4 Simplifying costs systematically

Three rules handle most introductory analyses:

1. Drop constant factors: \(7n\rightarrow\Theta(n)\).
2. Keep the dominant term: \(n^2+50n+2\rightarrow\Theta(n^2)\).
3. Add sequential phases and multiply independent nested repetitions.

~~~java
static long example(int[] values) {
    long total = 0;
    for (int value : values) {
        total += value;
    }
    for (int left : values) {
        for (int right : values) {
            total += (long) left * right;
        }
    }
    return total;
}
~~~

The cost is \(n+n^2=\Theta(n^2)\). Sequential loops are added, not multiplied.

### 4.5 Keep independent input sizes separate

~~~java
static java.util.List<Integer> commonValues(int[] leftValues, int[] rightValues) {
    java.util.List<Integer> matches = new java.util.ArrayList<>();
    for (int left : leftValues) {
        for (int right : rightValues) {
            if (left == right) {
                matches.add(left);
            }
        }
    }
    return matches;
}
~~~

For lengths \(n\) and \(m\), time is \(\Theta(nm)\), not automatically
\(\Theta(n^2)\). The latter only follows if \(n=m\) is explicitly assumed.

### 4.6 Best, average, and worst case

~~~java
static int linearSearch(int[] values, int target) {
    for (int index = 0; index < values.length; index++) {
        if (values[index] == target) {
            return index;
        }
    }
    return -1;
}
~~~

- Best case: the target is first, so time is \(\Theta(1)\).
- Worst case: it is last or absent, so time is \(\Theta(n)\).
- Average case: under a stated distribution, a linear number of positions is
  typically examined, so expected time is \(\Theta(n)\).

An average-case claim is incomplete without a probability model. Big O and
worst case are also not synonyms: Big O describes a bound, while worst case
selects a scenario among inputs of the same size.

---

## 5. Space complexity

### 5.1 Total space and auxiliary space

State whether an analysis includes input and output or only the additional
memory used by the algorithm.

~~~java
static int[] doubledCopy(int[] values) {
    int[] result = new int[values.length];
    for (int index = 0; index < values.length; index++) {
        result[index] = values[index] * 2;
    }
    return result;
}

static void doubleInPlace(int[] values) {
    for (int index = 0; index < values.length; index++) {
        values[index] *= 2;
    }
}
~~~

Both take \(\Theta(n)\) time. `doubledCopy` needs \(\Theta(n)\) auxiliary
space, while `doubleInPlace` uses \(\Theta(1)\) auxiliary space.

### 5.2 Output size can impose a lower bound

The ordered-pair example returns \(n^2\) pairs. Its materialized result requires
\(\Theta(n^2)\) memory regardless of how cleverly the pairs are computed. A
lazy iterator can reduce simultaneously occupied auxiliary memory, but it does
not reduce the number of produced pairs.

### 5.3 Recursion consumes stack space

Each unfinished recursive call occupies a stack frame. Recursion depth \(n\)
usually implies \(O(n)\) auxiliary stack space. Balanced divide-and-conquer
recursion often has depth \(O(\log n)\).

### 5.4 Time-space trade-offs

The hash-set duplicate detector is faster than pair comparison on average, but
uses \(\Theta(n)\) additional memory. Lookup tables, caches, memoization, and
in-place transformations make similar trade-offs. The best choice depends on
input size, memory limits, latency requirements, and output requirements.

---

## 6. Amortized analysis: why ArrayList.add is O(1)

An `ArrayList` stores elements in a resizable array. When its internal array is
full, it allocates a larger array and copies existing references. One `add` can
therefore cost \(\Theta(n)\), yet appending is amortized \(O(1)\).

With a simplified doubling policy, the total copied elements over \(n\) appends
form a geometric series:

\[
1+2+4+8+\dots<2n.
\]

Together with \(n\) normal writes, the whole sequence costs \(O(n)\), or
\(O(1)\) per append when spread across the sequence.

Amortized analysis is not average-case analysis. Average case assumes a
probability distribution over inputs; amortized analysis bounds every
sufficiently long operation sequence without probability. A particular resize
can still be expensive. The JDK does not promise the simplified doubling policy,
but geometric growth gives the same asymptotic argument.

---

## 7. Common Java analysis mistakes

### 7.1 “Two loops always mean O(n²)”

~~~java
static void twoPasses(int[] values) {
    for (int value : values) {
        System.out.println(value);
    }
    for (int value : values) {
        System.out.println(value);
    }
}
~~~

The costs add to \(n+n=2n=\Theta(n)\).

Even nested loops need not be quadratic:

~~~java
static long shrinkingWork(int size) {
    long count = 0;
    for (int current = size; current > 0; current /= 2) {
        for (int index = 0; index < current; index++) {
            count++;
        }
    }
    return count;
}
~~~

The work is \(n+n/2+n/4+\dots<2n\), so it is \(\Theta(n)\).

### 7.2 Hidden copying costs

`Arrays.copyOfRange`, `String.substring` in modern JDKs, collection copy
constructors, and stream materialization can all create new data. Repeatedly
copying shrinking suffixes can turn an apparently linear recursion into
quadratic work.

~~~java
static int recursiveSum(int[] values, int length) {
    if (length == 0) {
        return 0;
    }
    int[] prefix = java.util.Arrays.copyOf(values, length - 1);
    return values[length - 1] + recursiveSum(prefix, prefix.length);
}
~~~

Passing an index instead avoids the copies and produces linear time.

### 7.3 `contains` depends on the collection

`ArrayList.contains` is \(O(n)\); `HashSet.contains` is expected \(O(1)\) under
ordinary hashing assumptions, with a worse pathological case. The method name
alone does not reveal the cost; the data structure is part of the analysis.

Filtering \(n\) values with repeated membership checks in a list of size \(m\)
costs \(O(nm)\). Converting the allowed values to a hash set once gives expected
\(O(n+m)\) time and \(O(m)\) additional space.

### 7.4 String concatenation in loops

`String` is immutable. Repeated `result += word` can allocate and copy an
increasing prefix, producing quadratic character work. Use `StringBuilder`:

~~~java
static String joinWords(java.util.List<String> words) {
    StringBuilder result = new StringBuilder();
    for (String word : words) {
        result.append(word);
    }
    return result.toString();
}
~~~

If \(L\) is the total character count, this takes \(\Theta(L)\) time. The right
input measure is total text length, not merely the number of words.

### 7.5 Library calls are not constant because they occupy one line

~~~java
int[] ordered = java.util.Arrays.copyOf(values, values.length);
java.util.Arrays.sort(ordered);
int smallest = java.util.Arrays.stream(values).min().orElseThrow();
~~~

Typical costs are \(O(n)\), \(O(n\log n)\), and \(O(n)\). Source-line count is
not a complexity measure.

### 7.6 Early return does not determine worst-case complexity

An early return can make linear search constant in the best case, but the worst
case remains linear. Always name the scenario and input assumptions.

### 7.7 Expected hash-table costs are not absolute guarantees

`HashMap` and `HashSet` normally provide expected constant operations. Collisions,
poor hash functions, and adversarial data can worsen costs. State the hashing
assumption explicitly.

---

## 8. A systematic analysis workflow

1. **Name input sizes.** Define what \(n\), \(m\), or other variables mean.
2. **Mark elementary and hidden costs.** Inspect loops, recursion, copying,
   sorting, membership tests, string construction, boxing, and output size.
3. **Count executions.** Use sums such as
   \(\sum_{i=1}^{n}i=n(n+1)/2\) or geometric series when needed.
4. **Combine costs.** Add sequential phases and multiply independent nested work.
5. **Name the scenario.** Specify best, average, worst, expected, or amortized.
6. **Analyze memory separately.** Include new collections, arrays, copies, and
   recursion stack frames; distinguish total from auxiliary space.
7. **Check empirically.** Measure controlled inputs over several sizes and
   repetitions, but use results only to check the derivation.

---

## 9. Worked examples

### 9.1 Triangular loop

~~~java
static long triangular(int[] values) {
    long count = 0;
    for (int left = 0; left < values.length; left++) {
        for (int right = left; right < values.length; right++) {
            count++;
        }
    }
    return count;
}
~~~

The inner loop runs \(n,n-1,\dots,1\) times. The sum is
\(n(n+1)/2=\Theta(n^2)\).

### 9.2 Halving with a linear scan

~~~java
static long levelsWithScan(int[] values) {
    long total = 0;
    for (int size = values.length; size > 0; size /= 2) {
        for (int index = 0; index < size; index++) {
            total += values[index];
        }
    }
    return total;
}
~~~

Multiplying “logarithmic outer loop” by “linear inner loop” would be too loose.
The concrete total is \(n+n/2+n/4+\dots=\Theta(n)\).

### 9.3 Sort once and search many times

For \(m\) searches against the same \(n\) values:

1. sort once in \(O(n\log n)\),
2. perform each binary search in \(O(\log n)\),
3. obtain total time \(O(n\log n+m\log n)\).

Repeated linear search costs \(O(mn)\). Whether preprocessing is worthwhile
depends on \(m\), \(n\), mutations, and constant factors.

### 9.4 Output-sensitive analysis

If a method finds all matching pairs from two inputs and produces \(k\) results,
an informative bound may be \(O(n+m+k)\). Omitting \(k\) would ignore the cost
of materializing the output.

---

## 10. Self-check

### Exercise 1

Determine time and auxiliary-space complexity:

~~~java
static int[] reverseCopy(int[] values) {
    int[] result = new int[values.length];
    for (int index = 0; index < values.length; index++) {
        result[index] = values[values.length - 1 - index];
    }
    return result;
}
~~~

### Exercise 2

Why is this not \(O(n)\), despite having only one visible loop?

~~~java
static java.util.List<int[]> prefixes(int[] values) {
    java.util.List<int[]> result = new java.util.ArrayList<>();
    for (int length = 1; length <= values.length; length++) {
        result.add(java.util.Arrays.copyOf(values, length));
    }
    return result;
}
~~~

### Exercise 3

Two algorithms use \(50n\log_2n\) and \(n^2\) abstract operations. Explain why
the first wins eventually and why the second may still win for small inputs.

### Exercise 4

Classify these scenarios as best case, worst case, or amortized analysis:

1. one `ArrayList.add` that triggers a resize,
2. average cost per append over a long sequence,
3. linear search with the target at index 0,
4. linear search when the target is absent.

### Solutions

1. `reverseCopy` takes \(\Theta(n)\) time and \(\Theta(n)\) auxiliary space.
2. The copied prefix lengths sum to \(1+2+\dots+n=\Theta(n^2)\). The output
   itself also contains a quadratic total number of elements.
3. The ratio \(n^2/(50n\log_2n)=n/(50\log_2n)\) grows without bound. The
   factor 50 can dominate before the crossover point, but not forever.
4. The resize is an expensive individual case; the sequence is analyzed
   amortized. Index 0 is the best case, and an absent target is the worst case.

---

## 11. Summary

- Complexity describes resource growth in terms of clearly defined input sizes.
- Big O is an upper bound, Omega a lower bound, and Theta a tight bound.
- Best, average, worst, expected, and amortized describe different viewpoints.
- Java costs hidden inside copying, collection operations, strings, boxing,
  streams, and library calls belong in the analysis.
- Time and memory must be analyzed separately and are often traded for each other.
- Doubling \(n\) multiplies \(n\log n\) work by slightly more than two and
  quadratic work by about four.
- Measurements can support a derivation, but only code analysis and mathematical
  reasoning establish an asymptotic class.

If you can derive a class from code, check it with a controlled experiment, and
state your assumptions, you have met the goals of this theory chapter.
