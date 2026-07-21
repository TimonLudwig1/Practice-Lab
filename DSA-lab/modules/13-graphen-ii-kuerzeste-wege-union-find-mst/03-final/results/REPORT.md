# Synthetic Road Network Routing Report

- Seed: `1313`
- Grid: **12 × 16**
- Nodes: **192**
- Roads: **356**
- Start: `r11c0`
- Target: `r11c15`
- Routing algorithm: **custom Dijkstra implementation**

## Closure impact

| Scenario | Closed roads | Travel time | Delay | Delay | Hops | Settled nodes |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 0 | 47.298 | 0.000 | 0.0 % | 17 | 159 |
| `single_route_road` | 1 | 49.760 | 2.462 | 5.2 % | 19 | 172 |
| `north_gap_barrier` | 11 | 92.846 | 45.548 | 96.3 % | 37 | 192 |

## Interpretation

The single-road closure tests local resilience: the route must leave its previously optimal path but can usually rejoin nearby. The barrier scenario closes every east-west crossing at one column except the northern gap. It forces a structural detour rather than a small local adjustment.

The plot uses matplotlib only for presentation. Route computation, closure filtering, path reconstruction, and all reported metrics come from the custom graph implementation.
