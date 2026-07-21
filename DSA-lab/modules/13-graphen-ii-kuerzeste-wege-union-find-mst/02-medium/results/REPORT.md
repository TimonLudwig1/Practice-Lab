# Seeded MST Workshop Results

Kruskal and Prim were executed on the exact same immutable edge tuple for every row. Median runtimes use seven repetitions.

| Vertices | Edges | Seed | MST weight | Kruskal (µs) | Prim (µs) | Same edges |
|---:|---:|---:|---:|---:|---:|:---:|
| 25 | 60 | 1301 | 475.000 | 30.417 | 21.709 | yes |
| 75 | 250 | 1302 | 1377.000 | 182.959 | 93.459 | yes |
| 150 | 600 | 1303 | 2506.000 | 469.500 | 221.958 | yes |
| 300 | 1500 | 1304 | 3626.000 | 1428.833 | 618.500 | yes |

## Interpretation

Both algorithms returned valid spanning trees with identical total weight in every case. Equal total weight is the correctness criterion; the selected edge sets may differ when several minimum spanning trees exist due to tied weights.

The timings are a small local experiment, not a universal performance ranking. Kruskal pays for globally sorting all edges, while Prim pays for heap operations along the growing tree frontier. Graph density, input representation, constants, and runtime noise affect the result.
