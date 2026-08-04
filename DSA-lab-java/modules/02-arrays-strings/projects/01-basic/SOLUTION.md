# Solution — Dynamic Array

## Representation invariants

At all times: \(0\le size\le capacity\); positions 0 through `size - 1`
contain logical elements in order; the `Object[]` has exactly `capacity`
positions; and resizing changes neither size, values, nor order.

Appending with free capacity is \(\Theta(1)\). When full, resizing allocates a
buffer twice as large and copies every active reference, making that individual
append \(\Theta(n)\). Starting at capacity 1, resizes occur on appends 2, 3, 5,
9, 17, and so on. The copies over \(n\) appends satisfy

\[
1+2+4+\dots<2n.
\]

Together with \(n\) new writes, total model cost is below \(3n\), hence append
is amortized \(\Theta(1)\). This is a sequence guarantee, not an assumption
about a probability distribution.

Insertion shifts from right to left so a source value is copied before its
destination is overwritten. Inserting at the front is \(\Theta(n)\); inserting
at the end is \(\Theta(1)\) without resize. Deletion shifts left to close a gap
and clears the former final active position. Deleting the first value is
\(\Theta(n)\), while deleting the last is \(\Theta(1)\).

| Operation | Best | Worst | Amortized | Auxiliary space |
|---|---:|---:|---:|---:|
| indexed access | \(\Theta(1)\) | \(\Theta(1)\) | \(\Theta(1)\) | \(\Theta(1)\) |
| append | \(\Theta(1)\) | \(\Theta(n)\) | \(\Theta(1)\) | up to \(\Theta(n)\) during resize |
| insert at start | \(\Theta(n)\) | \(\Theta(n)\) | \(\Theta(n)\) | up to \(\Theta(n)\) during resize |
| insert at end | \(\Theta(1)\) | \(\Theta(n)\) | \(\Theta(1)\) | up to \(\Theta(n)\) during resize |
| delete at start | \(\Theta(n)\) | \(\Theta(n)\) | \(\Theta(n)\) | \(\Theta(1)\) |
| delete at end | \(\Theta(1)\) | \(\Theta(1)\) | \(\Theta(1)\) | \(\Theta(1)\) |
| resize | \(\Theta(n)\) | \(\Theta(n)\) | — | \(\Theta(n)\) |

Immediate shrinking could cause repeated copying during alternating append and
delete operations. Production arrays use hysteresis if they shrink at all.
This educational implementation omits `ArrayList`'s full API, tuned growth
policy, optimized internals, concurrency considerations, and production error
handling so the core capacity model remains visible.
