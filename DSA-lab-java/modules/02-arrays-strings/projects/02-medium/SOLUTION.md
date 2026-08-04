# Solution — Array and String Pattern Catalog

| Pattern | Time | Auxiliary space | Core invariant |
|---|---:|---:|---|
| rotate right | \(O(n)\) | \(O(1)\) | three reversals place both partitions correctly |
| merge sorted | \(O(n+m)\) | \(O(1)\) | the suffix after `write` is final |
| prefix sum build/query | \(O(n)\) / \(O(1)\) | \(O(n)\) | `prefix[i]` sums values before `i` |
| anagrams | \(O(n)\) expected | \(O(k)\) | counts represent unmatched code points |
| sorted deduplication | \(O(n)\) | \(O(1)\) | the prefix before `write` is unique |
| move zeros | \(O(n)\) | \(O(1)\) | the prefix contains stable nonzero values |
| product except self | \(O(n)\) | \(O(1)\) beyond output | prefix and suffix products exclude the index |
| longest unique substring | \(O(n)\) expected | \(O(k)\) | the current window has unique code points |
| spiral traversal | \(O(rc)\) | \(O(1)\) beyond output | boundaries enclose unvisited cells |
| run compression | \(O(n)\) | \(O(1)\) beyond digit text | `write` follows processed runs |

Backward merging protects unread values in the target buffer. Prefix sums move
work from each query into one preprocessing pass. The sliding-window start only
moves right, so each code point enters and leaves at most once. Stable zero
movement writes nonzero values in encounter order, then fills the suffix with
zeros. In-place compression is safe because encoded output never exceeds the
processed input prefix.
