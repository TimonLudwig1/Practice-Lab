# Solution and Interpretation

Before entering a traced body, the thread-local depth equals the number of active
traced frames. Entry is printed at that depth; children see depth plus one; a
`finally` block restores the exact previous depth before return or exception
output. Different traced algorithms therefore nest correctly.

| Method | Time | Maximum stack depth | Call-tree shape |
|---|---:|---:|---|
| `factorial(n)` | \(\Theta(n)\) | \(\Theta(n)\) | chain |
| `fibonacci(n)` | \(O(2^n)\) | \(\Theta(n)\) | branching tree |
| `recursiveSum(values)` | \(\Theta(n)\) | \(\Theta(n)\) | chain |
| `power(base, exponent)` | \(\Theta(|e|)\) | \(\Theta(|e|)\) | chain |

The Fibonacci tree has exponentially many calls but only one root-to-leaf path
is active at once. Indexed summation avoids copying shrinking suffix arrays,
keeping algorithmic time linear. Negative powers recurse on the positive
magnitude and take one reciprocal. Binary exponentiation in the next project
improves power to \(\Theta(\log |e|)\).

Tracing adds formatting and terminal I/O to every call. It is a learning and
diagnostic tool, not suitable for performance measurement.
