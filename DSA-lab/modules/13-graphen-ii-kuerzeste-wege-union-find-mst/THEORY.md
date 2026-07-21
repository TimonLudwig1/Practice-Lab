# Module 13: Graphs II — Shortest Paths, Union Find & MST

Weighted graphs answer two questions that sound similar, but are fundamentally
different:

1. How do I get from a start to a goal as cheap as possible?
2. How do I connect all nodes with the lowest possible total cost?

The first question leads to **shortest paths**, the second to **minimum spanning
trees**. In between is Union-Find: a small data structure that quickly detects whether
two nodes already belong to the same component.

## Learning objectives

After this module you can:

- explain why BFS is not sufficient for general edge weights;
- Implement Dijkstra with a Priority Queue and simulate by hand,
- justify the non-negativity requirement of Dijkstra,
- Apply Bellman Ford and detect achievable negative cycles,
- Implement Union-Find with Path Compression and Union by Rank,
- Execute and compare Kruskal and Prim on the same graph,
- distinguish between the shortest route and the minimum spanning tree,
- select an appropriate procedure for a scenario;
- the basic idea of A* as a targeted Dijkstra.

---

# Part I — Weighted graphs

## 1. From pedometer to cost model

In an unweighted graph, each edge costs the same amount of thought. The length of a path
is then simply its number of edges. BFS visits nodes according to this number of edges
and therefore finds shortest unweighted paths.

A weighted graph assigns a number to each edge:

- kilometres in a road network,
- travel time or toll costs;
- latency in a computer network,
- energy consumption;
- negative log probability,
- Dissimilarity between data points.

For a path `P = v0 -> v1 -> ... -> vk` the path weight is

```text
w(P) = w(v0, v1) + w(v1, v2) +... + w(v{k-1}, vk)
```

A path with more edges can be cheaper than a direct edge.

## 2. Why BFS is not enough anymore

Consider this directed graph:

```text
S --------10---...---> T
 \                    ^
  1                  1
   \-> A -------...---/
```

BFS looks `T` for an edge and prefers `S -> T`. This path costs 10. The path `S -> A ->
T` needs two edges, but costs only 2.

```python
from collections import deque
from math import inf
import heapq

counterexample = {
    "S": {"T": 10, "A": 1},
    "A": {"T": 1},
    "T": {},
}

def fewest_edges_path(graph, start, target):
    queue = deque([(start, (start,))])
    visited = {start}
    while queue:
        vertex, path = queue.popleft()
        if vertex == target:
            return path
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + (neighbor,)))
    return None

def path_cost(graph, path):
    return sum(graph[a][b] for a, b in zip(path, path[1:]))

bfs_choice = fewest_edges_path(counterexample, "S", "T")
assert bfs_choice == ("S", "T")
assert path_cost(counterexample, bfs_choice) == 10
assert path_cost(counterexample, ("S", "A", "T")) == 2
```

The central lesson is:

> BFS minimizes the number of edges, not any weight.

## 3. Our common example graph

For Dijkstra, Kruskal and Prim we use the same undirected graph:

```text
      4       5       6
  A ----- B ----- D ----- F
   \     /       / \       /
   2\   /1     8/  2\     /3
     \ /       /     \   /
      C ------         E
          10
```

The edges are:

| Edge | Weight |
|---|---:|
| A–B | 4 |
| A–C | 2 |
| B–C | 1 |
| B–D | 5 |
| C–D | 8 |
| C–E | 10 |
| D–E | 2 |
| D–F | 6 |
| E–F | 3 |

```python
weighted_graph = {vertex: {} for vertex in "ABCDEF"}

def add_undirected_edge(graph, source, target, weight):
    if weight < 0:
        raise ValueError("Dieses Hilfsmodell erwartet nichtnegative Gewichte")
    graph[source][target] = weight
    graph[target][source] = weight

for source, target, weight in (
    ("A", "B", 4),
    ("A", "C", 2),
    ("B", "C", 1),
    ("B", "D", 5),
    ("C", "D", 8),
    ("C", "E", 10),
    ("D", "E", 2),
    ("D", "F", 6),
    ("E", "F", 3),
):
    add_undirected_edge(weighted_graph, source, target, weight)

assert weighted_graph["A"] == {"B": 4, "C": 2}
assert weighted_graph["F"] == {"D": 6, "E": 3}
```

An adjacency list stores a pair of neighbors and weight instead of a mere neighbor. The
space requirement remains `O(V + E)`.

---

# Part II — Relaxation: the common basic pattern

## 4. What does relaxation mean?

Suppose the current best known distance to `u` is `dist[u]`. For an edge `u -> v` with
weight `w`, the path via `u` offers a candidate:

```text
Candidate = dist[u] + w
```

If this candidate is smaller than `dist[v]`, we improve distance and predecessors:

```text
if dist[u] + w < dist[v]:
    dist[v] = dist[u] + w
    previous[v] = u
```

This local improvement is called **Relaxation**. Dijkstra and Bellman-Ford use the same
operation, but differ in the order and how often they look at edges.

```python
def relax(distances, previous, source, target, weight):
    candidate = distances[source] + weight
    if candidate < distances[target]:
        distances[target] = candidate
        previous[target] = source
        return True
    return False

distances_demo = {"S": 0, "A": 7, "B": inf}
previous_demo = {"S": None, "A": "S", "B": None}
assert relax(distances_demo, previous_demo, "A", "B", 3)
assert distances_demo["B"] == 10
assert previous_demo["B"] == "A"
assert not relax(distances_demo, previous_demo, "S", "A", 9)
```

## 5. Three states of a distance

- `∞`: No way known yet.
- preliminary finite number: A path is known, but perhaps there is a more favorable one.
- Final number: With the correct Dijkstra, the node with minimal heap entry was taken;
  its distance cannot be improved anymore.

The last point applies only to non-negative weights.

---

# Part III — Dijkstra with Priority Queue

## 6. Intuition

From the start, Dijkstra is growing a region of finally dissolved nodes. The still open
node with the smallest preliminary distance is processed next. A min priority queue
delivers it efficiently.

```text
1. dist[start] = 0, all others = ̄
2. (0, start) in the Min-Heap
3. Smallest pair taken
4. Skip outdated heap entries
5. Relax all outgoing edges
6. Place improved distances in the heap
7. until the heap is empty
```

Python does not have a direct `decrease_key` command in `heapq`. Instead of changing an
old entry, we'll put a new one in it. On later removal we recognize the old entry as
outdated.

## 7. Full implementation

```python
def dijkstra(graph, start):
    if start not in graph:
        raise KeyError(f"Unbekannter Startknoten: {start!r}")
    for source, neighbors in graph.items():
        for target, weight in neighbors.items():
            if target not in graph:
                raise ValueError(f"Unbekannter Zielknoten: {target!r}")
            if weight < 0:
                raise ValueError("Dijkstra erlaubt keine negativen Gewichte")

    distances = {vertex: inf for vertex in graph}
    previous = {vertex: None for vertex in graph}
    distances[start] = 0
    heap = [(0, start)]
    settled = set()
    trace = []

    while heap:
        distance, vertex = heapq.heappop(heap)

        if distance != distances[vertex]:
            trace.append(("veraltet", vertex, distance))
            continue

        settled.add(vertex)
        trace.append(("fixiere", vertex, distance))

        for neighbor, weight in graph[vertex].items():
            candidate = distance + weight
            if candidate < distances[neighbor]:
                old_distance = distances[neighbor]
                distances[neighbor] = candidate
                previous[neighbor] = vertex
                heapq.heappush(heap, (candidate, neighbor))
                trace.append(
                    ("relaxiere", vertex, neighbor, old_distance, candidate)
                )

    return distances, previous, tuple(trace)

dijkstra_distances, dijkstra_previous, dijkstra_trace = dijkstra(
    weighted_graph, "A"
)
assert dijkstra_distances == {
    "A": 0,
    "B": 3,
    "C": 2,
    "D": 8,
    "E": 10,
    "F": 13,
}
```

## 8. Simulation step by step

Start state:

| Fixed | dist(A) | dist(B) | dist(C) | dist(D) | Dist(E) | dist(F) |
|---|---:|---:|---:|---:|---:|---:|
| – | 0 | ∞ | ∞ | ∞ | ∞ | ∞ |

### Step 1: A with distance 0

- `A -> B`: `0 + 4 < ∞`, i.e. `dist(B) = 4`
- `A -> C`: `0 + 2 < ∞`, i.e. `dist(C) = 2`

| Fixed | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 | 4 | 2 | ∞ | ∞ | ∞ |

### Step 2: C with distance 2

- `C -> B`: `2 + 1 = 3 < 4`, improvement of B
- `C -> D`: `2 + 8 = 10`
- `C -> E`: `2 + 10 = 12`

| Fixed | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A, C | 0 | 3 | 2 | 10 | 12 | ∞ |

### Step 3: B with distance 3

- The old heap entry `(4, B)` remains at first.
- `B -> D`: `3 + 5 = 8 < 10`, improvement of D.

| Fixed | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A, C, B | 0 | 3 | 2 | 8 | 12 | ∞ |

### Obsolete entry `(4, B)`

Since `4 != dist(B)`, it is skipped without re-examination of the edges.

### Step 4: D with distance 8

- `D -> E`: `8 + 2 = 10 < 12`, improvement of E.
- `D -> F`: `8 + 6 = 14`.

| Fixed | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A, C, B, D | 0 | 3 | 2 | 8 | 10 | 14 |

### Step 5: E with distance 10

- `E -> F`: `10 + 3 = 13 < 14`, improvement of F.

| Fixed | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A, C, B, D, E | 0 | 3 | 2 | 8 | 10 | 13 |

### Step 6: F with distance 13

No more improvement. Final fixation order:

```text
A, C, B, D, E, F
```

```python
fixed_order = tuple(
    event[1] for event in dijkstra_trace if event[0] == "fixiere"
)
assert fixed_order == ("A", "C", "B", "D", "E", "F")
assert any(event[0] == "veraltet" for event in dijkstra_trace)
```

## 9. Reconstruct Path

The distances answer "how expensive?" The predecessor dictionary answers "about which
nodes?".

```python
def reconstruct_path(previous, start, target):
    if target not in previous:
        raise KeyError(target)

    reversed_path = []
    current = target
    while current is not None:
        reversed_path.append(current)
        if current == start:
            reversed_path.reverse()
            return tuple(reversed_path)
        current = previous[current]
    return None

assert reconstruct_path(dijkstra_previous, "A", "F") == (
    "A", "C", "B", "D", "E", "F"
)
assert path_cost(weighted_graph, ("A", "C", "B", "D", "E", "F")) == 13
```

With equally expensive paths, the chosen predecessor depends on insertion order and Tie-
Breaking. However, the distance remains correct.

## 10. Unattainable nodes

A node in another component remains at distance `∞`; its predecessor remains `None`.

```python
disconnected_graph = {
    "A": {"B": 2},
    "B": {},
    "X": {},
}
dist, prev, _ = dijkstra(disconnected_graph, "A")
assert dist == {"A": 0, "B": 2, "X": inf}
assert reconstruct_path(prev, "A", "X") is None
```

## 11. Why break negative edges Dijkstra

Dijkstra considers a taken node as final. With non-negative edges, any alternative way
through an open node can only be equally expensive or more expensive. A negative edge
can later suddenly improve an already fixed distance and destroy this Greedy argument.

```python
negative_graph = {
    "S": {"A": 2, "B": 5},
    "A": {},
    "B": {"A": -10},
}
try:
    dijkstra(negative_graph, "S")
except ValueError as error:
    assert "keine negativen" in str(error)
else:
    raise AssertionError("Negative edge was not rejected")
```

Important: "Furte" is no problem for Dijkstra. "Poor weight is the problem. Zero weights
are allowed.

## 12. Correctness Idea

If `u` comes out of the heap as the smallest open node, we assume there would be a
cheaper yet unknown way to `u`. This path would have to cross the border from the fixed
to the open area at an edge. Up to this limit, its distance is already known; with a
non-negative edge, the rest of the path cannot fall under `dist(u)`. Objection.

The Greedy invariant is:

> Each node taken with a current minimum distance has its final
> Shortest distance.

## 13. Complexity

With adjacency list and binary heap:

- each edge can trigger an improvement and a heap-push,
- each push/pop costs `O(log V)` or more precisely `O(log E)`,
- Total `O((V + E) log V)`, often shortened to `O(E log V)`,
- Memory `O(V + E)` including graph, distances and heap.

For a dense adjacency matrix and linear minimum search, the classic variant is `O(V²)`.
This can make sense for very dense graphs.

---

# Part IV — Bellman-Ford

## 14. The other strategy

Bellman-Ford doesn't trust premature finality. Instead of choosing the smallest node
locally, it relaxes **repeated all edges**.

A shortest simple path contains at most `V - 1` edges. After the first complete round
all optimal paths with at most one edge are correct, after the second all with at most
two edges and so on. Therefore `V - 1` rounds suffice.

## 15. Implementation

```python
class NegativeCycleError(ValueError):
    pass

def bellman_ford(vertices, edges, start):
    vertex_order = tuple(vertices)
    if start not in vertex_order:
        raise KeyError(start)

    distances = {vertex: inf for vertex in vertex_order}
    previous = {vertex: None for vertex in vertex_order}
    distances[start] = 0
    round_traces = []

    for round_number in range(max(0, len(vertex_order) - 1)):
        changes = []
        for source, target, weight in edges:
            if source not in distances or target not in distances:
                raise ValueError("Edge contains an unknown node")
            if distances[source] == inf:
                continue
            candidate = distances[source] + weight
            if candidate < distances[target]:
                old = distances[target]
                distances[target] = candidate
                previous[target] = source
                changes.append((source, target, old, candidate))
        round_traces.append(tuple(changes))
        if not changes:
            break

    for source, target, weight in edges:
        if (
            distances[source] != inf
            and distances[source] + weight < distances[target]
        ):
            raise NegativeCycleError(
                f"Reachable negative cycle through {source!r}->{target!r}"
            )

    return distances, previous, tuple(round_traces)

bf_vertices = ("S", "A", "B", "C", "D")
bf_edges = (
    ("S", "A", 4),
    ("S", "B", 5),
    ("A", "C", -2),
    ("B", "C", 3),
    ("C", "D", 4),
)
bf_distances, bf_previous, bf_rounds = bellman_ford(
    bf_vertices, bf_edges, "S"
)
assert bf_distances == {"S": 0, "A": 4, "B": 5, "C": 2, "D": 6}
assert reconstruct_path(bf_previous, "S", "D") == ("S", "A", "C", "D")
```

The edge `A -> C` has weight `-2`, but there is no negative cycle. A shortest way is
therefore still well-defined.

## 16. Early termination

If a complete round does not change any distance, all the shortest paths are already
found. Further rounds would be ineffective.

```python
assert len(bf_rounds) < len(bf_vertices) - 1
assert bf_rounds[-1] == ()
```

The number of rounds actually required also depends on the order of the edges. The `V -
1` limit applies regardless of this.

## 17. Detect negative cycles

After `V - 1` rounds, no edge must be relaxable. If an improvement is still possible, a
cycle achievable from the start leads to ever smaller costs. There is then no finite
shortest way for the targets concerned.

```python
cycle_vertices = ("S", "A", "B", "C")
cycle_edges = (
    ("S", "A", 1),
    ("A", "B", 1),
    ("B", "C", -3),
    ("C", "A", 0),
)
try:
    bellman_ford(cycle_vertices, cycle_edges, "S")
except NegativeCycleError as error:
    assert "negative cycle" in str(error)
else:
    raise AssertionError("Negative cycle was not detected")
```

A negative cycle in a **unreachable** component does not affect the single source
response and is not reported by this implementation.

## 18. Dijkstra or Bellman Ford?

| Property | Dijkstra with Heap | Bellman Ford |
|---|---:|---:|
| Negative edges allowed | yes | Yes |
| Detect negative cycles | yes | yes, reachable from start |
| typical runtime | `O(E log V)` | `O(VE)` |
| Core idea | fix the cheapest open nodes | Relax all edges repeatedly |
| preferred | Non-negative weights | Negative weights/cycle test |

If all weights are exactly the same, BFS is easier and faster: `O(V+E)`.

---

# Part V — Union-Find / Disjoint Set Union

## 19. What question does Union-Find answer?

Union-Find manages a disintegration of elements into disjunct quantities. It supports
two key operations:

- `find(x)`: Which representative names the amount of `x`?
- `union(a, b)`: Unify the amounts of `a` and `b`.

This can be answered very quickly:

```text
Are a and b already connected?
find(a) == find(b)
```

Union-Find does not store complete graph paths. It only manages component membership.
This is exactly what Kruskal needs to avoid the cycle.

## 20. Forest from Parents' Houses

Lots of it is a tree that is rooted:

```text
parent[x] = x      means: x is a root
parent[x] = y      means: y is parent node of x
```

The representative is the root. Without optimizations, the trees can degenerate into
long chains.

## 21. Path Compression

While `find(x)` runs to the root, all nodes visited are directly attached to that root.

```text
before: D -> C -> B -> A
find(D)
after: D -> A, C -> A and B -> A
```

This means that future searches will be almost constantly fast.

## 22. Union by Rank

When uniting, the root of the shallower tree is hung under the root of a deeper tree.
Only if both ranks are equal, the rank of the new root grows by 1.

Rank is an upper barrier for the tree height, not necessarily the current height
according to Path Compression.

## 23. Full implementation

```python
class DisjointSet:
    def __init__(self, elements=()):
        self.parent = {}
        self.rank = {}
        self.component_count = 0
        for element in elements:
            self.add(element)

    def add(self, element):
        if element in self.parent:
            return False
        self.parent[element] = element
        self.rank[element] = 0
        self.component_count += 1
        return True

    def find(self, element):
        if element not in self.parent:
            raise KeyError(element)
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def union(self, first, second):
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return False

        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1
        self.component_count -= 1
        return True

    def connected(self, first, second):
        return self.find(first) == self.find(second)

    def components(self):
        groups = {}
        for element in self.parent:
            root = self.find(element)
            groups.setdefault(root, set()).add(element)
        return tuple(frozenset(group) for group in groups.values())
```

## 24. Simulation

Start:

```text
{A} {B} {C} {D} {E} {F}     6 Components
```

Operations:

| Operation | Result | Components |
|---|---|---:|
| `union(A, B)` | `{A,B}` | 5 |
| `union(C, D)` | `{C,D}` | 4 |
| `union(A, C)` | `{A,B,C,D}` | 3 |
| `union(E, F)` | `{E,F}` | 2 |
| `union(B, D)` | already connected | 2 |

```python
dsu_demo = DisjointSet("ABCDEF")
assert dsu_demo.union("A", "B")
assert dsu_demo.union("C", "D")
assert dsu_demo.union("A", "C")
assert dsu_demo.union("E", "F")
assert not dsu_demo.union("B", "D")
assert dsu_demo.component_count == 2
assert dsu_demo.connected("A", "D")
assert not dsu_demo.connected("A", "E")
assert set(dsu_demo.components()) == {
    frozenset("ABCD"),
    frozenset("EF"),
}
```

## 25. Amortised complexity

Both optimizations cost `m` operations to `n` elements in total `O(m α(n))`. `α` is the
inverse Ackermann function and grows so slowly that it is a very small constant for
every practical input.

Therefore, it is often said that `find` and `union` are **amortized almost O(1)**.

"Amortized" means that a single operation may be more expensive, but over a long series
the average is extremely small.

## 26. Other applications

- components in a step-by-step network,
- Recognize whether a new undirected edge creates a cycle,
- Kruskal's minimum spanning tree,
- grouping and equivalence classes,
- Percolation simulation,
- Offline connectivity requests.

Union-Find does not support removal or arbitrary path queries efficiently.

---

# Part VI — Minimum spanning trees

## 27. Tree, spanning tree and minimum spanning tree

For a coherent undirected graph, a **spanning tree** is a subset of edges which:

- connects all `V` nodes,
- does not contain a cycle;
- has exactly `V - 1` edges.

A **minimum spanning tree** (Minimum Spanning Tree, MST) minimizes the sum of these
edges. It does not necessarily minimize the path between two specific nodes.

### Shortest way tree is not the same

Dijkstra minimizes from the selected start to each node. An MST minimizes the total cost
of the entire network. In MST, the path between two nodes can be longer than in the
original graph.

## 28. Edge model

```python
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class Edge:
    weight: float
    source: str
    target: str

mst_vertices = tuple("ABCDEF")
mst_edges = tuple(
    Edge(weight, source, target)
    for source, target, weight in (
        ("A", "B", 4),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 5),
        ("C", "D", 8),
        ("C", "E", 10),
        ("D", "E", 2),
        ("D", "F", 6),
        ("E", "F", 3),
    )
)
assert len(mst_edges) == 9
```

---

# Part VII — Kruskal

## 29. Intuition

Kruskal considers edges globally from the smallest to the greatest weight:

1. Sort all edges.
2. Start with `V` individual components.
3. Take the next cheapest edge.
4. Connect their endpoints only if they are still in different components.
5. Stop after `V - 1` selected edges.

Union-Find answers step 4 efficiently.

## 30. Simulation at the sample graph

Sorted edges:

```text
BC:1, AC:2, DE:2, EF:3, AB:4, BD:5, DF:6, CD:8, CE:10
```

| Edge | Decision | Reasons | Total |
|---|---|---|---:|
| B–C (1) | Take | different components | 1 |
| A–C (2) | Take | A separated from BC | 3 |
| D–E (2) | Take | different components | 5 |
| E–F (3) | Take | F separately from DE | 8 |
| A–B (4) | skip | A and B already connected | 8 |
| B–D (5) | Take | connects ABC to DEF | 13 |

Now all six nodes are connected to five edges.

## 31. Implementation

```python
def kruskal(vertices, edges):
    dsu = DisjointSet(vertices)
    tree = []
    trace = []

    for edge in sorted(edges):
        if dsu.union(edge.source, edge.target):
            tree.append(edge)
            trace.append(("select", edge))
            if len(tree) == len(vertices) - 1:
                break
        else:
            trace.append(("zyklus", edge))

    if vertices and len(tree) != len(vertices) - 1:
        raise ValueError("Graph is disconnected")
    return tuple(tree), sum(edge.weight for edge in tree), tuple(trace)

kruskal_tree, kruskal_cost, kruskal_trace = kruskal(
    mst_vertices, mst_edges
)
assert kruskal_cost == 13
assert kruskal_tree == (
    Edge(1, "B", "C"),
    Edge(2, "A", "C"),
    Edge(2, "D", "E"),
    Edge(3, "E", "F"),
    Edge(5, "B", "D"),
)
assert any(decision == "zyklus" for decision, _ in kruskal_trace)
```

## 32. Why is local choice safe?

**Cut property** provides the idea of correctness:

> For each cut of the node quantity is a lightest edge, which the cut
> a safe choice for any MST.

Kruskal's components form such cuts. The cheapest edge between two different components
can be added without having to install an optimal solution.

**Cyclus property** supplements:

> A clearly heaviest edge on a cycle does not belong to an MST.

## 33. Complexity

- Sort edges: `O(E log E)`
- Union Find Steps: `O(E α(V))`
- Total: `O(E log E)`, equivalent often `O(E log V)`
- Memory: `O(V + E)`

---

# Part VIII — Prim

## 34. Intuition

Prim grows a single tree from a start node. There are candidate edges in the heap that
lead from the tree to the outside. The cheapest edge is always chosen as a node not yet
contained.

The difference in one sentence:

- Kruskal grows many components and unites them.
- Prim grows exactly one contiguous component.

## 35. Simulation

Start at A:

| Tree node of choice | cheapest exit edge | new sum |
|---|---|---:|
| `{A}` | A–C (2) | 2 |
| `{A,C}` | C–B (1) | 3 |
| `{A,B,C}` | B–D (5) | 8 |
| `{A,B,C,D}` | D–E (2) | 10 |
| `{A,B,C,D,E}` | E–F (3) | 13 |

Prim can provide a different MST than Kruskal if there are equally heavy alternatives.
The total weight is always the same minimal.

## 36. Implementation with Heap

```python
def build_undirected_adjacency(vertices, edges):
    graph = {vertex: [] for vertex in vertices}
    for edge in edges:
        if edge.source not in graph or edge.target not in graph:
            raise ValueError("Edge contains an unknown node")
        graph[edge.source].append((edge.weight, edge.target))
        graph[edge.target].append((edge.weight, edge.source))
    return graph

def prim(vertices, edges, start):
    graph = build_undirected_adjacency(vertices, edges)
    if not vertices:
        return (), 0, ()
    if start not in graph:
        raise KeyError(start)

    visited = {start}
    heap = []
    trace = []
    tree = []

    for weight, target in graph[start]:
        heapq.heappush(heap, (weight, start, target))

    while heap and len(visited) < len(vertices):
        weight, source, target = heapq.heappop(heap)
        if target in visited:
            trace.append(("veraltet", source, target, weight))
            continue

        visited.add(target)
        edge = Edge(weight, min(source, target), max(source, target))
        tree.append(edge)
        trace.append(("select", edge))

        for next_weight, neighbor in graph[target]:
            if neighbor not in visited:
                heapq.heappush(heap, (next_weight, target, neighbor))

    if len(visited) != len(vertices):
        raise ValueError("Graph is disconnected")
    return tuple(tree), sum(edge.weight for edge in tree), tuple(trace)

prim_tree, prim_cost, prim_trace = prim(mst_vertices, mst_edges, "A")
assert prim_cost == kruskal_cost == 13
assert len(prim_tree) == len(mst_vertices) - 1
assert {vertex for edge in prim_tree for vertex in (edge.source, edge.target)} == set(
    mst_vertices
)
```

## 37. Prim complexity

With adjacency list and binary heap:

- each edge reaches the heap in a constant number at most;
- `O(E log V)` Time,
- `O(V + E)` Memory.

With adjacency matrix and linear search, Prim is `O(V²)` and can be competitive with
dense graphs.

## 38. Non-contiguous graphs

A single spanning tree exists only if the undirected graph is coherent. Otherwise you
can calculate a **minimal spanning forest**: one MST per component.

```python
disconnected_vertices = ("A", "B", "X", "Y")
disconnected_edges = (Edge(1, "A", "B"), Edge(2, "X", "Y"))
for algorithm_call in (
    lambda: kruskal(disconnected_vertices, disconnected_edges),
    lambda: prim(disconnected_vertices, disconnected_edges, "A"),
):
    try:
        algorithm_call()
    except ValueError as error:
        assert "disconnected" in str(error)
    else:
        raise AssertionError("Disconnected graph was not detected")
```

## 39. Compared to Kruskal and Prim

| Criterion | Kruskal | Prim |
|---|---|---|
| growing | Forest of components | a tree |
| Main structure | sorted edges + Union-Find | Adjacency list + min-heap |
| Cycle avoidance | `find(u) != find(v)` | Target not yet visited |
| Typically good | thin graphs, edge list | Adjacency lists, dense local structure |
| runtime with standard structures | `O(E log E)` | `O(E log V)` |
| Necessary start nodes | yes | yes, result weight independent of |

---

# Part IX — A* as an outlook

## 40. Targeted Search

Dijkstra prioritizes a node `v` after the previously known distance:

```text
Priority = g(v)
```

A* supplements a heuristic `h(v)` which estimates the remaining costs to the goal:

```text
Priority = f(v) = g(v) + h(v)
```

On a map, the airline distance can be a heuristic. A* thus directs the search towards
the goal, rather than growing evenly in all directions.

For a guaranteed optimal way, heuristics must not overestimate the real remaining costs
(**allowable/admissible**). A consistent heuristic additionally meets for each edge:

```text
h(u) ≤ w(u, v) + h(v)
```

Special cases:

- `h(v) = 0`is meant.==External links==
- Perfect Heuristics: A* investigates only the really relevant nodes.
- overestimating heuristics: often faster, but not guaranteed optimality.

The complete A* implementation is not the aim of this module; the classification as a
heuristically prioritized shortest-way search is crucial.

---

# Part X — Choice of procedure

## 41. Decision Tree

```text
Wanted: Path from start to finish(s)?
|
+-- all edges weighted the same? ------- BFS
|
+-- weights all non-negative? -------------------------- Dijkstra
|
+-- negative edges possible? -------...--- Bellman Ford
|                                           + detects negative cycles
|
+-- spatial specific objective + heuristics? ---- A* consider

Wanted: cheapest net for all nodes?
|
+-- undirected + contiguous ------- MST
    +-- Edge list / thin ---------------- Kruskal
    +-- Adjacency list / Tree growing ------- Prim
```

## 42. Scenarios

| Scenario | Procedure | Reasons |
|---|---|---|
| Little change | BFS | every connection counts as one step |
| fastest route with positive times | Dijkstra | Non-negative variable weights |
| Currency arbitrage after log transformation | Bellman Ford | Negative cycles relevant |
| Navigation with crow flies to the destination | A* | Applicable Heuristics |
| Glass fibre network of minimum total length | MST | Connect all locations cheaply |
| huge sortable edge list | Kruskal | Union-Find and global edge selection |
| Graph is available as a neighborhood list | Prim | local growth with heap |

## 43. What MST does not solve

An MST is not automatic:

- the shortest way between all pairs,
- the shortest way from a spring,
- robust against edge failures,
- unambiguously,
- defined on directed graphs.

If redundancy is important, a tree is even particularly fragile: every tree edge removed
separates the net.

---

# Part XI — Typical pitfalls

## 44. Dijkstra despite negative edge

Not "mostly correct", but outside the requirement. Check input.

## 45. Apply BFS to variable weights

BFS minimizes hops. Only equal edge weights justify it.

## 46. View heap entry as automatically up-to-date

Without `decrease_key` there are outdated entries in the heap. Check:

```text
if popped_distance!= distances[vertex]: skip
```

## 47. Forget predecessors while relaxing

Then distances are right, but concrete paths cannot be reconstructed.

## 48. Bellman-Ford only run V-1 rounds

That's enough for distances. The additional test round is decisive for the detection of
negative cycles.

## 49. To confuse Union-Find root with direct parent

`parent[x]` is not necessarily the root. Always use `find(x)`.

## 50. Increase rank at each union

Rank only rises when two roots of the same rank are united.

## 51. Double treat undirected edge as two MST edges

An edge `A–B` is logically identical to `B–A`. The input model should represent it once
or deduplicate it before processing.

## 52. Do not stop after V-1 edges

A spanning tree to `V` node has exactly `V-1` edges. Further edges create cycles.

## 53. Print separate graphs as MST

Without any connection, at most a spanning forest is created. Describe the result
explicitly or cause an error.

## 54. Comparing floating point numbers blindly

Float weights can cause rounding errors. Tests should use tolerances such as
`math.isclose` depending on the domain.

---

# Part XII — Data Science Transfer

## 55. Shortest Paths to Data Problems

- Knowledge graphs: cheapest semantic connection,
- Recommendation systems: paths through user-item relationships,
- Feature dependencies: minimum transformation costs,
- Network and log analysis: latency paths,
- Image processing: pixel costs as weighted grid graph.

## 56. MST in Data Science

- Clustering: Single Linkage Hierarchy can be understood via an MST,
- dimensional reduction and manifold seams;
- cost-effective sensor or communication networks;
- visualization of a thin global data structure,
- Recognizing unusually severe compounds.

An MST reduces `E` edges to `V-1` and preserves connection, but not all local distances.

## 57. Union-Find in Clustering

When edges are sorted by similarity, Union-Find gradually unites clusters. Kruskal can
therefore be read as an agglomeration process: At first each point is a cluster, then
the cheapest permissible edges connect components.

---

# Part XIII — Correctness and Test Thinking

## 58. Useful invariants for Dijkstra

- `dist[start] == 0`
- Distances are only getting smaller, never bigger.
- Each predecessor forms a real graph edge.
- For each edge, `dist[v] <= dist[u] + w(u,v)` applies after completion.
- Unattainable nodes remain `∞`.
- For non-negative weights, the fixation order is not falling.

```python
for source, neighbors in weighted_graph.items():
    for target, weight in neighbors.items():
        assert dijkstra_distances[target] <= dijkstra_distances[source] + weight

settled_distances = [
    event[2] for event in dijkstra_trace if event[0] == "fixiere"
]
assert settled_distances == sorted(settled_distances)
```

## 59. Useful invariants for an MST

- exactly `V-1` edges at `V > 0`,
- all nodes are connected,
- no cycle;
- Total equals Kruskal and Prim,
- Remove a tree edge dismembers the tree,
- for clear edge weights, the MST is unique.

```python
def verify_tree(vertices, edges):
    if not vertices:
        return len(edges) == 0
    if len(edges) != len(vertices) - 1:
        return False
    dsu = DisjointSet(vertices)
    for edge in edges:
        if not dsu.union(edge.source, edge.target):
            return False
    return dsu.component_count == 1

assert verify_tree(mst_vertices, kruskal_tree)
assert verify_tree(mst_vertices, prim_tree)
```

## 60. Reference and Property Tests

For random small graphs, strong tests are possible:

- Dijkstra v. Bellman-Ford on non-negative weights,
- Kruskal total weight versus Prim total weight,
- each output MST meets tree invariants,
- Union-Find connectivity against a slow-calculated component reference;
- path costs against the distance spent,
- Seed-based reproducibility.

Test not only an expected example: edge cases such as empty graph, single nodes, zero
weight, parallel edges, separate components and negative cycles make implementations
robust.

---

# Part XIV — Control questions

## 61. Questions

1. What minimizes BFS, what Dijkstra?
2. What exactly is relaxed with an edge?
3. Why is Dijkstra allowed to permanently fix a removed node?
4. What happens to outdated heap entries?
5. When does a distance remain `∞`?
6. Why does Bellman-Ford need `V-1` main rounds at most?
7. How does the additional round show a negative cycle?
8. What does Path Compression do?
9. When does the rank of a Union Find root rise?
10. What question does Union-Find not answer?
11. Why does Kruskal not create a cycle?
12. How do Kruskals and Prim's growth patterns differ?
13. Why is an MST not a shorter way tree?
14. What does a permissible A* heuristic mean?
15. What procedure would you choose for negative edges?

## 62. Short Answers

1. BFS minimizes edge count; Dijkstra minimizes the sum of non-negative weights.
2. `dist[v]` is compared to `dist[u] + w(u,v)` and improved if necessary.
3. Non-negative residual edges cannot produce a later cheaper way.
4. They are skipped on the basis of the now smaller stored distance.
5. If the node cannot be reached from the start.
6. A shortest simple path has at most `V-1` edges.
7. A further improvement proves achievable cycle with negative sum.
8. Visited nodes then point directly to the root.
9. Only when combining two roots of the same rank.
10. A concrete path between two connected elements.
11. An edge is only chosen between different components.
12. Kruskal unites a forest; Prim expands a tree.
13. It minimizes the global edge sum, not individual path distances.
14. She never overestimates the real remaining costs.
15. Bellman-Ford; with achievable negative cycles there are no finite shortest ways to
    all the affected targets.

---

# Part XV — Compact overview

## 63. Spicking slips

| Procedure | Objective | Prerequisite | Structure | Time |
|---|---|---|---|---:|
| BFS | Less edges | Same cost | Queue | `O(V+E)` |
| Dijkstra | Shortest Paths | Weights `>= 0` | Min-heap | `O(E log V)` |
| Bellman Ford | Shortest Paths | Negative edges allowed | Edge list | `O(VE)` |
| Union Find | Components | Dynamic associations | Parent + Rank | amortised `O(α(V))` |
| Kruskal | MST | Undirected, contiguous | Sorting + DSU | `O(E log E)` |
| Prim | MST | Undirected, contiguous | Adjacency list + heap | `O(E log V)` |
| A* | Towards a goal | suitable heuristics | Heap + `g+h` | Problem-dependent |

## 64. Mental Models

```text
Dijkstra:    "Take the cheapest node available right now."
Bellman-Ford: "Give each edge repeated the chance to improve."
Union-Find: "Do these elements already have the same root?"
Kruskal:     "Take the cheapest edge that connects two components."
Prim:        "Take the cheapest edge out of the tree so far."
A*:          "Like Dijkstra, but with an informed view to the destination."
```

## 65. Outlook for the projects

- **01-basic:** Build Dijkstra and Union-Find itself; test Dijkstra against the complete
  hand calculation of this script.
- **02 medium:** Compare Kruskal and Prim to identical seed random graphs.
- **03-final:** Analyze and visualize routes and barriers in a synthetic road network;
  the algorithms remain self-implementations.

The connecting thought of the module is the controlled local decision: Dijkstra fixes
the cheapest known distance, Union-Find efficiently summarizes components, and MST
methods choose safe cheap edges. It is always crucial to know the prerequisite behind
the local election.

---

# Deutsche Fassung

# Modul 13: Graphen II — Kürzeste Wege, Union-Find & MST

Gewichtete Graphen beantworten zwei Fragen, die ähnlich klingen, aber grundverschieden
sind:

1. Wie komme ich von einem Start zu einem Ziel möglichst günstig?
2. Wie verbinde ich alle Knoten mit möglichst geringen Gesamtkosten?

Die erste Frage führt zu **kürzesten Wegen**, die zweite zu **minimalen
Spannbäumen**. Dazwischen steht Union-Find: eine kleine Datenstruktur, die sehr
schnell erkennt, ob zwei Knoten bereits zur selben Komponente gehören.

## Lernziele

Nach diesem Modul kannst du:

- erklären, weshalb BFS bei allgemeinen Kantengewichten nicht genügt,
- Dijkstra mit einer Priority Queue implementieren und von Hand simulieren,
- die Nichtnegativitäts-Voraussetzung von Dijkstra begründen,
- Bellman-Ford anwenden und erreichbare negative Zyklen erkennen,
- Union-Find mit Path Compression und Union by Rank implementieren,
- Kruskal und Prim auf demselben Graphen ausführen und vergleichen,
- zwischen kürzestem Weg und minimalem Spannbaum unterscheiden,
- für ein Szenario ein geeignetes Verfahren auswählen,
- die Grundidee von A* als zielgerichtetem Dijkstra einordnen.

---

# Teil I — Gewichtete Graphen

## 1. Vom Schrittzähler zum Kostenmodell

In einem ungewichteten Graphen kostet jede Kante gedanklich gleich viel. Die
Länge eines Pfades ist dann einfach seine Kantenzahl. BFS besucht Knoten nach
dieser Kantenzahl und findet deshalb kürzeste ungewichtete Wege.

Ein gewichteter Graph ordnet jeder Kante eine Zahl zu:

- Kilometer in einem Straßennetz,
- Fahrzeit oder Mautkosten,
- Latenz in einem Computernetz,
- Energieverbrauch,
- negative Log-Wahrscheinlichkeit,
- Unähnlichkeit zwischen Datenpunkten.

Für einen Pfad `P = v0 -> v1 -> ... -> vk` ist das Pfadgewicht

```text
w(P) = w(v0, v1) + w(v1, v2) +... + w(v{k-1}, vk)
```

Ein Pfad mit mehr Kanten kann billiger sein als eine direkte Kante.

## 2. Warum BFS nicht mehr reicht

Betrachte diesen gerichteten Graphen:

```text
S --------10---...---> T
 \                    ^
  1                  1
   \-> A -------...---/
```

BFS sieht `T` nach einer Kante und bevorzugt `S -> T`. Dieser Pfad kostet 10.
Der Pfad `S -> A -> T` braucht zwei Kanten, kostet aber nur 2.

```python
from collections import deque
from math import inf
import heapq

counterexample = {
    "S": {"T": 10, "A": 1},
    "A": {"T": 1},
    "T": {},
}

def fewest_edges_path(graph, start, target):
    queue = deque([(start, (start,))])
    visited = {start}
    while queue:
        vertex, path = queue.popleft()
        if vertex == target:
            return path
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + (neighbor,)))
    return None

def path_cost(graph, path):
    return sum(graph[a][b] for a, b in zip(path, path[1:]))

bfs_choice = fewest_edges_path(counterexample, "S", "T")
assert bfs_choice == ("S", "T")
assert path_cost(counterexample, bfs_choice) == 10
assert path_cost(counterexample, ("S", "A", "T")) == 2
```

Die zentrale Lektion lautet:

> BFS minimiert die Anzahl der Kanten, nicht eine beliebige Gewichtssumme.

## 3. Unser gemeinsamer Beispielgraph

Für Dijkstra, Kruskal und Prim verwenden wir denselben ungerichteten Graphen:

```text
      4       5       6
  A ----- B ----- D ----- F
   \     /       / \       /
   2\   /1     8/  2\     /3
     \ /       /     \   /
      C ------         E
          10
```

Die Kanten sind:

| Kante | Gewicht |
|---|---:|
| A–B | 4 |
| A–C | 2 |
| B–C | 1 |
| B–D | 5 |
| C–D | 8 |
| C–E | 10 |
| D–E | 2 |
| D–F | 6 |
| E–F | 3 |

```python
weighted_graph = {vertex: {} for vertex in "ABCDEF"}

def add_undirected_edge(graph, source, target, weight):
    if weight < 0:
        raise ValueError("Dieses Hilfsmodell erwartet nichtnegative Gewichte")
    graph[source][target] = weight
    graph[target][source] = weight

for source, target, weight in (
    ("A", "B", 4),
    ("A", "C", 2),
    ("B", "C", 1),
    ("B", "D", 5),
    ("C", "D", 8),
    ("C", "E", 10),
    ("D", "E", 2),
    ("D", "F", 6),
    ("E", "F", 3),
):
    add_undirected_edge(weighted_graph, source, target, weight)

assert weighted_graph["A"] == {"B": 4, "C": 2}
assert weighted_graph["F"] == {"D": 6, "E": 3}
```

Eine Adjazenzliste speichert hier statt eines bloßen Nachbarn ein Paar aus
Nachbar und Gewicht. Der Speicherbedarf bleibt `O(V + E)`.

---

# Teil II — Relaxierung: das gemeinsame Grundmuster

## 4. Was bedeutet Relaxierung?

Angenommen, die aktuell beste bekannte Distanz zu `u` ist `dist[u]`. Für eine
Kante `u -> v` mit Gewicht `w` bietet der Weg über `u` einen Kandidaten:

```text
Candidate = dist[u] + w
```

Ist dieser Kandidat kleiner als `dist[v]`, verbessern wir Distanz und Vorgänger:

```text
if dist[u] + w < dist[v]:
    dist[v] = dist[u] + w
    previous[v] = u
```

Diese lokale Verbesserung heißt **Relaxierung**. Dijkstra und Bellman-Ford
verwenden dieselbe Operation, unterscheiden sich aber darin, in welcher
Reihenfolge und wie oft sie Kanten betrachten.

```python
def relax(distances, previous, source, target, weight):
    candidate = distances[source] + weight
    if candidate < distances[target]:
        distances[target] = candidate
        previous[target] = source
        return True
    return False

distances_demo = {"S": 0, "A": 7, "B": inf}
previous_demo = {"S": None, "A": "S", "B": None}
assert relax(distances_demo, previous_demo, "A", "B", 3)
assert distances_demo["B"] == 10
assert previous_demo["B"] == "A"
assert not relax(distances_demo, previous_demo, "S", "A", 9)
```

## 5. Drei Zustände einer Distanz

- `∞`: Noch kein Weg bekannt.
- vorläufige endliche Zahl: Ein Weg ist bekannt, aber vielleicht existiert ein
  günstigerer.
- endgültige Zahl: Beim korrekten Dijkstra wurde der Knoten mit minimalem
  Heap-Eintrag entnommen; seine Distanz kann nicht mehr verbessert werden.

Der letzte Punkt gilt nur bei nichtnegativen Gewichten.

---

# Teil III — Dijkstra mit Priority Queue

## 6. Intuition

Dijkstra wächst vom Start aus eine Region endgültig gelöster Knoten. Immer der
noch offene Knoten mit der kleinsten vorläufigen Distanz wird als Nächstes
verarbeitet. Eine Min-Priority-Queue liefert ihn effizient.

```text
1. dist[start] = 0, all others = ̄
2. (0, start) in the Min-Heap
3. Smallest pair taken
4. Skip outdated heap entries
5. Relax all outgoing edges
6. Place improved distances in the heap
7. until the heap is empty
```

Python besitzt keinen direkten `decrease_key`-Befehl in `heapq`. Statt einen
alten Eintrag zu verändern, legen wir einen neuen hinein. Beim späteren
Entnehmen erkennen wir den alten Eintrag als veraltet.

## 7. Vollständige Implementierung

```python
def dijkstra(graph, start):
    if start not in graph:
        raise KeyError(f"Unbekannter Startknoten: {start!r}")
    for source, neighbors in graph.items():
        for target, weight in neighbors.items():
            if target not in graph:
                raise ValueError(f"Unbekannter Zielknoten: {target!r}")
            if weight < 0:
                raise ValueError("Dijkstra erlaubt keine negativen Gewichte")

    distances = {vertex: inf for vertex in graph}
    previous = {vertex: None for vertex in graph}
    distances[start] = 0
    heap = [(0, start)]
    settled = set()
    trace = []

    while heap:
        distance, vertex = heapq.heappop(heap)

        if distance != distances[vertex]:
            trace.append(("veraltet", vertex, distance))
            continue

        settled.add(vertex)
        trace.append(("fixiere", vertex, distance))

        for neighbor, weight in graph[vertex].items():
            candidate = distance + weight
            if candidate < distances[neighbor]:
                old_distance = distances[neighbor]
                distances[neighbor] = candidate
                previous[neighbor] = vertex
                heapq.heappush(heap, (candidate, neighbor))
                trace.append(
                    ("relaxiere", vertex, neighbor, old_distance, candidate)
                )

    return distances, previous, tuple(trace)

dijkstra_distances, dijkstra_previous, dijkstra_trace = dijkstra(
    weighted_graph, "A"
)
assert dijkstra_distances == {
    "A": 0,
    "B": 3,
    "C": 2,
    "D": 8,
    "E": 10,
    "F": 13,
}
```

## 8. Simulation Schritt für Schritt

Startzustand:

| fixiert | dist(A) | dist(B) | dist(C) | dist(D) | dist(E) | dist(F) |
|---|---:|---:|---:|---:|---:|---:|
| – | 0 | ∞ | ∞ | ∞ | ∞ | ∞ |

### Schritt 1: A mit Distanz 0

- `A -> B`: `0 + 4 < ∞`, also `dist(B) = 4`
- `A -> C`: `0 + 2 < ∞`, also `dist(C) = 2`

| fixiert | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 | 4 | 2 | ∞ | ∞ | ∞ |

### Schritt 2: C mit Distanz 2

- `C -> B`: `2 + 1 = 3 < 4`, Verbesserung von B
- `C -> D`: `2 + 8 = 10`
- `C -> E`: `2 + 10 = 12`

| fixiert | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A, C | 0 | 3 | 2 | 10 | 12 | ∞ |

### Schritt 3: B mit Distanz 3

- Der alte Heap-Eintrag `(4, B)` bleibt zunächst liegen.
- `B -> D`: `3 + 5 = 8 < 10`, Verbesserung von D.

| fixiert | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A, C, B | 0 | 3 | 2 | 8 | 12 | ∞ |

### Veralteter Eintrag `(4, B)`

Da `4 != dist(B)`, wird er ohne erneute Kantenprüfung übersprungen.

### Schritt 4: D mit Distanz 8

- `D -> E`: `8 + 2 = 10 < 12`, Verbesserung von E.
- `D -> F`: `8 + 6 = 14`.

| fixiert | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A, C, B, D | 0 | 3 | 2 | 8 | 10 | 14 |

### Schritt 5: E mit Distanz 10

- `E -> F`: `10 + 3 = 13 < 14`, Verbesserung von F.

| fixiert | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| A, C, B, D, E | 0 | 3 | 2 | 8 | 10 | 13 |

### Schritt 6: F mit Distanz 13

Keine Verbesserung mehr. Endgültige Fixierreihenfolge:

```text
A, C, B, D, E, F
```

```python
fixed_order = tuple(
    event[1] for event in dijkstra_trace if event[0] == "fixiere"
)
assert fixed_order == ("A", "C", "B", "D", "E", "F")
assert any(event[0] == "veraltet" for event in dijkstra_trace)
```

## 9. Pfad rekonstruieren

Die Distanzen beantworten „wie teuer?“. Das Vorgänger-Dictionary beantwortet
„über welche Knoten?“.

```python
def reconstruct_path(previous, start, target):
    if target not in previous:
        raise KeyError(target)

    reversed_path = []
    current = target
    while current is not None:
        reversed_path.append(current)
        if current == start:
            reversed_path.reverse()
            return tuple(reversed_path)
        current = previous[current]
    return None

assert reconstruct_path(dijkstra_previous, "A", "F") == (
    "A", "C", "B", "D", "E", "F"
)
assert path_cost(weighted_graph, ("A", "C", "B", "D", "E", "F")) == 13
```

Bei gleich teuren Wegen hängt der gewählte Vorgänger von Einfügereihenfolge und
Tie-Breaking ab. Die Distanz bleibt dennoch korrekt.

## 10. Unerreichbare Knoten

Ein Knoten in einer anderen Komponente bleibt auf Distanz `∞`; sein Vorgänger
bleibt `None`.

```python
disconnected_graph = {
    "A": {"B": 2},
    "B": {},
    "X": {},
}
dist, prev, _ = dijkstra(disconnected_graph, "A")
assert dist == {"A": 0, "B": 2, "X": inf}
assert reconstruct_path(prev, "A", "X") is None
```

## 11. Warum negative Kanten Dijkstra brechen

Dijkstra betrachtet einen entnommenen Knoten als endgültig. Bei nichtnegativen
Kanten kann jeder alternative Weg über einen noch offenen Knoten nur gleich teuer
oder teurer werden. Eine negative Kante kann später plötzlich eine bereits
fixierte Distanz verbessern und zerstört dieses Greedy-Argument.

```python
negative_graph = {
    "S": {"A": 2, "B": 5},
    "A": {},
    "B": {"A": -10},
}
try:
    dijkstra(negative_graph, "S")
except ValueError as error:
    assert "keine negativen" in str(error)
else:
    raise AssertionError("Negative edge was not rejected")
```

Wichtig: „gerichtet“ ist kein Problem für Dijkstra. „Negatives Gewicht“ ist das
Problem. Nullgewichte sind erlaubt.

## 12. Korrektheitsidee

Wenn `u` als kleinster offener Knoten aus dem Heap kommt, nehmen wir an, es gäbe
einen billigeren noch unbekannten Weg zu `u`. Dieser Weg müsste die Grenze vom
fixierten zum offenen Bereich an einer Kante überschreiten. Bis zu dieser Grenze
ist seine Distanz bereits bekannt; mit einer nichtnegativen Kante kann der
restliche Weg nicht unter `dist(u)` fallen. Widerspruch.

Die Greedy-Invariante lautet:

> Jeder mit aktueller Minimaldistanz entnommene Knoten besitzt seine endgültige
> kürzeste Distanz.

## 13. Komplexität

Mit Adjazenzliste und binärem Heap:

- jede Kante kann eine Verbesserung und einen Heap-Push auslösen,
- jeder Push/Pop kostet `O(log V)` beziehungsweise genauer `O(log E)`,
- insgesamt `O((V + E) log V)`, oft verkürzt zu `O(E log V)`,
- Speicher `O(V + E)` einschließlich Graph, Distanzen und Heap.

Bei einer dichten Adjazenzmatrix und linearer Minimumsuche ist die klassische
Variante `O(V²)`. Für sehr dichte Graphen kann das sinnvoll sein.

---

# Teil IV — Bellman-Ford

## 14. Die andere Strategie

Bellman-Ford vertraut keiner vorzeitigen Endgültigkeit. Statt den lokal kleinsten
Knoten zu wählen, relaxiert es **alle Kanten wiederholt**.

Ein kürzester einfacher Pfad enthält höchstens `V - 1` Kanten. Nach der ersten
vollständigen Runde sind alle optimalen Wege mit höchstens einer Kante korrekt,
nach der zweiten alle mit höchstens zwei Kanten und so weiter. Deshalb genügen
`V - 1` Runden.

## 15. Implementierung

```python
class NegativeCycleError(ValueError):
    pass

def bellman_ford(vertices, edges, start):
    vertex_order = tuple(vertices)
    if start not in vertex_order:
        raise KeyError(start)

    distances = {vertex: inf for vertex in vertex_order}
    previous = {vertex: None for vertex in vertex_order}
    distances[start] = 0
    round_traces = []

    for round_number in range(max(0, len(vertex_order) - 1)):
        changes = []
        for source, target, weight in edges:
            if source not in distances or target not in distances:
                raise ValueError("Edge contains an unknown node")
            if distances[source] == inf:
                continue
            candidate = distances[source] + weight
            if candidate < distances[target]:
                old = distances[target]
                distances[target] = candidate
                previous[target] = source
                changes.append((source, target, old, candidate))
        round_traces.append(tuple(changes))
        if not changes:
            break

    for source, target, weight in edges:
        if (
            distances[source] != inf
            and distances[source] + weight < distances[target]
        ):
            raise NegativeCycleError(
                f"Reachable negative cycle through {source!r}->{target!r}"
            )

    return distances, previous, tuple(round_traces)

bf_vertices = ("S", "A", "B", "C", "D")
bf_edges = (
    ("S", "A", 4),
    ("S", "B", 5),
    ("A", "C", -2),
    ("B", "C", 3),
    ("C", "D", 4),
)
bf_distances, bf_previous, bf_rounds = bellman_ford(
    bf_vertices, bf_edges, "S"
)
assert bf_distances == {"S": 0, "A": 4, "B": 5, "C": 2, "D": 6}
assert reconstruct_path(bf_previous, "S", "D") == ("S", "A", "C", "D")
```

Die Kante `A -> C` hat Gewicht `-2`, aber es gibt keinen negativen Zyklus. Ein
kürzester Weg ist deshalb weiterhin wohldefiniert.

## 16. Frühes Beenden

Wenn eine vollständige Runde keine Distanz verändert, sind alle erreichbaren
kürzesten Wege bereits gefunden. Weitere Runden wären wirkungslos.

```python
assert len(bf_rounds) < len(bf_vertices) - 1
assert bf_rounds[-1] == ()
```

Wie viele Runden tatsächlich nötig sind, hängt auch von der Reihenfolge der
Kanten ab. Die `V - 1`-Grenze gilt unabhängig davon.

## 17. Negative Zyklen erkennen

Nach `V - 1` Runden darf keine Kante mehr relaxierbar sein. Ist doch noch eine
Verbesserung möglich, führt ein vom Start erreichbarer Zyklus zu immer kleineren
Kosten. Es existiert dann kein endlicher kürzester Weg für die betroffenen Ziele.

```python
cycle_vertices = ("S", "A", "B", "C")
cycle_edges = (
    ("S", "A", 1),
    ("A", "B", 1),
    ("B", "C", -3),
    ("C", "A", 0),
)
try:
    bellman_ford(cycle_vertices, cycle_edges, "S")
except NegativeCycleError as error:
    assert "negative cycle" in str(error)
else:
    raise AssertionError("Negative cycle was not detected")
```

Ein negative cycle in einer vom Start **nicht erreichbaren** Komponente
beeinflusst die Single-Source-Antwort nicht und wird von dieser Implementierung
nicht gemeldet.

## 18. Dijkstra oder Bellman-Ford?

| Eigenschaft | Dijkstra mit Heap | Bellman-Ford |
|---|---:|---:|
| negative Kanten erlaubt | nein | ja |
| negative Zyklen erkennen | nein | ja, erreichbar vom Start |
| typische Laufzeit | `O(E log V)` | `O(VE)` |
| Kernidee | billigsten offenen Knoten fixieren | alle Kanten wiederholt relaxieren |
| bevorzugt | nichtnegative Gewichte | negative Gewichte / Zyklusprüfung |

Wenn alle Gewichte exakt gleich sind, ist BFS einfacher und schneller: `O(V+E)`.

---

# Teil V — Union-Find / Disjoint Set Union

## 19. Welche Frage beantwortet Union-Find?

Union-Find verwaltet eine Zerlegung von Elementen in disjunkte Mengen. Es
unterstützt zwei zentrale Operationen:

- `find(x)`: Welcher Repräsentant benennt die Menge von `x`?
- `union(a, b)`: Vereinige die Mengen von `a` und `b`.

Damit lässt sich sehr schnell beantworten:

```text
Are a and b already connected?
find(a) == find(b)
```

Union-Find speichert keine vollständigen Graphpfade. Es verwaltet nur
Komponentenmitgliedschaft. Genau das braucht Kruskal zur Zyklusvermeidung.

## 20. Wald aus Elternzeigern

Jede Menge ist ein verwurzelter Baum:

```text
parent[x] = x      means: x is a root
parent[x] = y      means: y is parent node of x
```

Der Repräsentant ist die Wurzel. Ohne Optimierungen können die Bäume zu langen
Ketten entarten.

## 21. Path Compression

Während `find(x)` zur Wurzel läuft, werden alle besuchten Knoten direkt an diese
Wurzel gehängt.

```text
before: D -> C -> B -> A
find(D)
after: D -> A, C -> A and B -> A
```

Künftige Suchen werden dadurch fast konstant schnell.

## 22. Union by Rank

Beim Vereinigen wird die Wurzel des flacheren Baums unter die Wurzel des tieferen
gehängt. Nur wenn beide Ränge gleich sind, wächst der Rang der neuen Wurzel um 1.

Rang ist eine obere Schranke für die Baumhöhe, nicht zwingend die aktuelle Höhe
nach Path Compression.

## 23. Vollständige Implementierung

```python
class DisjointSet:
    def __init__(self, elements=()):
        self.parent = {}
        self.rank = {}
        self.component_count = 0
        for element in elements:
            self.add(element)

    def add(self, element):
        if element in self.parent:
            return False
        self.parent[element] = element
        self.rank[element] = 0
        self.component_count += 1
        return True

    def find(self, element):
        if element not in self.parent:
            raise KeyError(element)
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def union(self, first, second):
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return False

        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1
        self.component_count -= 1
        return True

    def connected(self, first, second):
        return self.find(first) == self.find(second)

    def components(self):
        groups = {}
        for element in self.parent:
            root = self.find(element)
            groups.setdefault(root, set()).add(element)
        return tuple(frozenset(group) for group in groups.values())
```

## 24. Simulation

Start:

```text
{A} {B} {C} {D} {E} {F}     6 Components
```

Operationen:

| Operation | Ergebnis | Komponenten |
|---|---|---:|
| `union(A, B)` | `{A,B}` | 5 |
| `union(C, D)` | `{C,D}` | 4 |
| `union(A, C)` | `{A,B,C,D}` | 3 |
| `union(E, F)` | `{E,F}` | 2 |
| `union(B, D)` | bereits verbunden | 2 |

```python
dsu_demo = DisjointSet("ABCDEF")
assert dsu_demo.union("A", "B")
assert dsu_demo.union("C", "D")
assert dsu_demo.union("A", "C")
assert dsu_demo.union("E", "F")
assert not dsu_demo.union("B", "D")
assert dsu_demo.component_count == 2
assert dsu_demo.connected("A", "D")
assert not dsu_demo.connected("A", "E")
assert set(dsu_demo.components()) == {
    frozenset("ABCD"),
    frozenset("EF"),
}
```

## 25. Amortisierte Komplexität

Mit beiden Optimierungen kosten `m` Operationen auf `n` Elementen insgesamt
`O(m α(n))`. `α` ist die inverse Ackermann-Funktion und wächst so langsam, dass
sie für jede praktische Eingabe höchstens eine sehr kleine Konstante ist.

Man sagt deshalb oft: `find` und `union` sind **amortisiert nahezu O(1)**.

„Amortisiert“ bedeutet: Eine einzelne Operation darf teurer sein, aber über eine
lange Folge verteilt ist der Durchschnitt extrem klein.

## 26. Weitere Anwendungen

- Komponenten in einem schrittweise aufgebauten Netzwerk,
- Erkennen, ob eine neue ungerichtete Kante einen Zyklus erzeugt,
- Kruskals minimaler Spannbaum,
- Gruppenbildung und Äquivalenzklassen,
- Perkolationssimulation,
- Offline-Connectivity-Anfragen.

Union-Find unterstützt Entfernen oder beliebige Pfadabfragen nicht effizient.

---

# Teil VI — Minimale Spannbäume

## 27. Baum, Spannbaum und minimaler Spannbaum

Für einen zusammenhängenden ungerichteten Graphen ist ein **Spannbaum** eine
Teilmenge der Kanten, die:

- alle `V` Knoten verbindet,
- keinen Zyklus enthält,
- genau `V - 1` Kanten besitzt.

Ein **minimaler Spannbaum** (Minimum Spanning Tree, MST) minimiert die Summe
dieser Kanten. Er minimiert nicht notwendigerweise den Weg zwischen zwei
bestimmten Knoten.

### Kürzester-Wege-Baum ist nicht dasselbe

Dijkstra minimiert vom gewählten Start zu jedem Knoten. Ein MST minimiert die
Gesamtkosten des gesamten Verbindungsnetzes. Im MST kann der Weg zwischen zwei
Knoten länger sein als im Originalgraphen.

## 28. Kantenmodell

```python
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class Edge:
    weight: float
    source: str
    target: str

mst_vertices = tuple("ABCDEF")
mst_edges = tuple(
    Edge(weight, source, target)
    for source, target, weight in (
        ("A", "B", 4),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 5),
        ("C", "D", 8),
        ("C", "E", 10),
        ("D", "E", 2),
        ("D", "F", 6),
        ("E", "F", 3),
    )
)
assert len(mst_edges) == 9
```

---

# Teil VII — Kruskal

## 29. Intuition

Kruskal betrachtet Kanten global vom kleinsten zum größten Gewicht:

1. Sortiere alle Kanten.
2. Beginne mit `V` einzelnen Komponenten.
3. Nimm die nächste billigste Kante.
4. Verbinde ihre Endpunkte nur, wenn sie noch in verschiedenen Komponenten sind.
5. Stoppe nach `V - 1` gewählten Kanten.

Union-Find beantwortet Schritt 4 effizient.

## 30. Simulation am Beispielgraphen

Sortierte Kanten:

```text
BC:1, AC:2, DE:2, EF:3, AB:4, BD:5, DF:6, CD:8, CE:10
```

| Kante | Entscheidung | Begründung | Summe |
|---|---|---|---:|
| B–C (1) | nehmen | verschiedene Komponenten | 1 |
| A–C (2) | nehmen | A getrennt von BC | 3 |
| D–E (2) | nehmen | verschiedene Komponenten | 5 |
| E–F (3) | nehmen | F getrennt von DE | 8 |
| A–B (4) | überspringen | A und B bereits verbunden | 8 |
| B–D (5) | nehmen | verbindet ABC mit DEF | 13 |

Jetzt sind alle sechs Knoten mit fünf Kanten verbunden.

## 31. Implementierung

```python
def kruskal(vertices, edges):
    dsu = DisjointSet(vertices)
    tree = []
    trace = []

    for edge in sorted(edges):
        if dsu.union(edge.source, edge.target):
            tree.append(edge)
            trace.append(("select", edge))
            if len(tree) == len(vertices) - 1:
                break
        else:
            trace.append(("zyklus", edge))

    if vertices and len(tree) != len(vertices) - 1:
        raise ValueError("Graph is disconnected")
    return tuple(tree), sum(edge.weight for edge in tree), tuple(trace)

kruskal_tree, kruskal_cost, kruskal_trace = kruskal(
    mst_vertices, mst_edges
)
assert kruskal_cost == 13
assert kruskal_tree == (
    Edge(1, "B", "C"),
    Edge(2, "A", "C"),
    Edge(2, "D", "E"),
    Edge(3, "E", "F"),
    Edge(5, "B", "D"),
)
assert any(decision == "zyklus" for decision, _ in kruskal_trace)
```

## 32. Warum ist die lokale Wahl sicher?

Die **Schnitteigenschaft** liefert die Korrektheitsidee:

> Für jeden Schnitt der Knotenmenge ist eine leichteste Kante, die den Schnitt
> überquert, eine sichere Wahl für irgendeinen MST.

Kruskals Komponenten bilden solche Schnitte. Die billigste Kante zwischen zwei
verschiedenen Komponenten kann aufgenommen werden, ohne eine optimale Lösung zu
verbauen.

Die **Zykluseigenschaft** ergänzt:

> Eine eindeutig schwerste Kante auf einem Zyklus gehört zu keinem MST.

## 33. Komplexität

- Kanten sortieren: `O(E log E)`
- Union-Find-Schritte: `O(E α(V))`
- insgesamt: `O(E log E)`, äquivalent oft `O(E log V)`
- Speicher: `O(V + E)`

---

# Teil VIII — Prim

## 34. Intuition

Prim wächst einen einzigen Baum von einem Startknoten aus. Im Heap liegen
Kandidatenkanten, die vom bisherigen Baum nach außen führen. Es wird stets die
billigste Kante zu einem noch nicht enthaltenen Knoten gewählt.

Der Unterschied in einem Satz:

- Kruskal wächst viele Komponenten und vereinigt sie.
- Prim wächst genau eine zusammenhängende Komponente.

## 35. Simulation

Start bei A:

| Baumknoten vor Wahl | billigste ausgehende Kante | neue Summe |
|---|---|---:|
| `{A}` | A–C (2) | 2 |
| `{A,C}` | C–B (1) | 3 |
| `{A,B,C}` | B–D (5) | 8 |
| `{A,B,C,D}` | D–E (2) | 10 |
| `{A,B,C,D,E}` | E–F (3) | 13 |

Prim kann einen anderen MST als Kruskal liefern, wenn gleich schwere Alternativen
existieren. Das Gesamtgewicht ist immer gleich minimal.

## 36. Implementierung mit Heap

```python
def build_undirected_adjacency(vertices, edges):
    graph = {vertex: [] for vertex in vertices}
    for edge in edges:
        if edge.source not in graph or edge.target not in graph:
            raise ValueError("Edge contains an unknown node")
        graph[edge.source].append((edge.weight, edge.target))
        graph[edge.target].append((edge.weight, edge.source))
    return graph

def prim(vertices, edges, start):
    graph = build_undirected_adjacency(vertices, edges)
    if not vertices:
        return (), 0, ()
    if start not in graph:
        raise KeyError(start)

    visited = {start}
    heap = []
    trace = []
    tree = []

    for weight, target in graph[start]:
        heapq.heappush(heap, (weight, start, target))

    while heap and len(visited) < len(vertices):
        weight, source, target = heapq.heappop(heap)
        if target in visited:
            trace.append(("veraltet", source, target, weight))
            continue

        visited.add(target)
        edge = Edge(weight, min(source, target), max(source, target))
        tree.append(edge)
        trace.append(("select", edge))

        for next_weight, neighbor in graph[target]:
            if neighbor not in visited:
                heapq.heappush(heap, (next_weight, target, neighbor))

    if len(visited) != len(vertices):
        raise ValueError("Graph is disconnected")
    return tuple(tree), sum(edge.weight for edge in tree), tuple(trace)

prim_tree, prim_cost, prim_trace = prim(mst_vertices, mst_edges, "A")
assert prim_cost == kruskal_cost == 13
assert len(prim_tree) == len(mst_vertices) - 1
assert {vertex for edge in prim_tree for vertex in (edge.source, edge.target)} == set(
    mst_vertices
)
```

## 37. Prim-Komplexität

Mit Adjazenzliste und binärem Heap:

- jede Kante gelangt höchstens in konstanter Anzahl in den Heap,
- `O(E log V)` Zeit,
- `O(V + E)` Speicher.

Mit Adjazenzmatrix und linearer Suche ist Prim `O(V²)` und kann bei dichten
Graphen konkurrenzfähig sein.

## 38. Nicht zusammenhängende Graphen

Ein einzelner Spannbaum existiert nur, wenn der ungerichtete Graph
zusammenhängend ist. Andernfalls kann man einen **minimalen Spannwald** berechnen:
einen MST pro Komponente.

```python
disconnected_vertices = ("A", "B", "X", "Y")
disconnected_edges = (Edge(1, "A", "B"), Edge(2, "X", "Y"))
for algorithm_call in (
    lambda: kruskal(disconnected_vertices, disconnected_edges),
    lambda: prim(disconnected_vertices, disconnected_edges, "A"),
):
    try:
        algorithm_call()
    except ValueError as error:
        assert "disconnected" in str(error)
    else:
        raise AssertionError("Disconnected graph was not detected")
```

## 39. Kruskal und Prim gegenübergestellt

| Kriterium | Kruskal | Prim |
|---|---|---|
| wächst | Wald aus Komponenten | einen Baum |
| Hauptstruktur | sortierte Kanten + Union-Find | Adjazenzliste + Min-Heap |
| Zyklusvermeidung | `find(u) != find(v)` | Ziel noch nicht besucht |
| typisch gut | dünne Graphen, Kantenliste | Adjazenzlisten, dichte lokale Struktur |
| Laufzeit mit Standardstrukturen | `O(E log E)` | `O(E log V)` |
| Startknoten nötig | nein | ja, Ergebnisgewicht unabhängig davon |

---

# Teil IX — A* als Ausblick

## 40. Zielgerichtete Suche

Dijkstra priorisiert einen Knoten `v` nach der bisher bekannten Distanz:

```text
Priority = g(v)
```

A* ergänzt eine Heuristik `h(v)`, die die verbleibenden Kosten zum Ziel schätzt:

```text
Priority = f(v) = g(v) + h(v)
```

Auf einer Landkarte kann die Luftlinienentfernung eine Heuristik sein. A* lenkt
die Suche dadurch in Richtung Ziel, statt in alle Richtungen gleichmäßig zu
wachsen.

Für einen garantiert optimalen Weg darf die Heuristik die echten Restkosten
nicht überschätzen (**zulässig/admissible**). Eine konsistente Heuristik erfüllt
zusätzlich für jede Kante:

```text
h(u) ≤ w(u, v) + h(v)
```

Spezialfälle:

- `h(v) = 0`: A* wird zu Dijkstra.
- perfekte Heuristik: A* untersucht nur die wirklich relevanten Knoten.
- überschätzende Heuristik: häufig schneller, aber Optimalität nicht garantiert.

Die vollständige A*-Implementierung ist nicht Ziel dieses Moduls; entscheidend
ist die Einordnung als heuristisch priorisierte Kürzeste-Wege-Suche.

---

# Teil X — Verfahrenswahl

## 41. Entscheidungsbaum

```text
Wanted: Path from start to finish(s)?
|
+-- all edges weighted the same? ------- BFS
|
+-- weights all non-negative? -------------------------- Dijkstra
|
+-- negative edges possible? -------...--- Bellman Ford
|                                           + detects negative cycles
|
+-- spatial specific objective + heuristics? ---- A* consider

Wanted: cheapest net for all nodes?
|
+-- undirected + contiguous ------- MST
    +-- Edge list / thin ---------------- Kruskal
    +-- Adjacency list / Tree growing ------- Prim
```

## 42. Szenarien

| Szenario | Verfahren | Begründung |
|---|---|---|
| wenigste Umstiege | BFS | jede Verbindung zählt als ein Schritt |
| schnellste Route mit positiven Zeiten | Dijkstra | nichtnegative variable Gewichte |
| Währungsarbitrage nach Log-Transformation | Bellman-Ford | negative Zyklen relevant |
| Navigation mit Luftlinie zum Ziel | A* | brauchbare zulässige Heuristik |
| Glasfasernetz minimaler Gesamtlänge | MST | alle Standorte billig verbinden |
| riesige sortierbare Kantenliste | Kruskal | Union-Find und globale Kantenwahl |
| Graph liegt als Nachbarschaftsliste vor | Prim | lokales Wachstum mit Heap |

## 43. Was MST nicht löst

Ein MST ist nicht automatisch:

- der kürzeste Weg zwischen allen Paaren,
- der kürzeste Weg von einer Quelle,
- robust gegen Kantenausfälle,
- eindeutig,
- auf gerichteten Graphen definiert.

Wenn Redundanz wichtig ist, ist ein Baum sogar besonders fragil: Jede entfernte
Baumkante trennt das Netz.

---

# Teil XI — Typische Fallstricke

## 44. Dijkstra trotz negativer Kante

Nicht „meistens korrekt“, sondern außerhalb der Voraussetzung. Eingaben prüfen.

## 45. BFS auf variable Gewichte anwenden

BFS minimiert Hops. Nur gleiche Kantengewichte rechtfertigen es.

## 46. Heap-Eintrag als automatisch aktuell ansehen

Ohne `decrease_key` liegen veraltete Einträge im Heap. Prüfe:

```text
if popped_distance!= distances[vertex]: skip
```

## 47. Vorgänger beim Relaxieren vergessen

Dann stimmen Distanzen, aber konkrete Pfade lassen sich nicht rekonstruieren.

## 48. Bellman-Ford nur V-1 Runden laufen lassen

Für Distanzen reicht das. Zur Erkennung negativer Zyklen ist die zusätzliche
Prüfrunde entscheidend.

## 49. Union-Find-Wurzel mit direktem Parent verwechseln

`parent[x]` ist nicht zwingend die Wurzel. Verwende immer `find(x)`.

## 50. Rank bei jeder Vereinigung erhöhen

Rank steigt nur, wenn zwei Wurzeln gleichen Rangs vereinigt werden.

## 51. Ungerichtete Kante doppelt als zwei MST-Kanten behandeln

Eine Kante `A–B` ist logisch identisch zu `B–A`. Das Eingabemodell sollte sie
einmal repräsentieren oder vor der Verarbeitung deduplizieren.

## 52. Nach V-1 Kanten nicht stoppen

Ein Spannbaum auf `V` Knoten besitzt exakt `V-1` Kanten. Weitere Kanten erzeugen
Zyklen.

## 53. Getrennten Graphen als MST ausgeben

Ohne Zusammenhang entsteht höchstens ein Spannwald. Das Ergebnis explizit
benennen oder einen Fehler auslösen.

## 54. Fließkommazahlen blind vergleichen

Bei Float-Gewichten können Rundungsfehler auftreten. Tests sollten je nach
Domäne eine Toleranz wie `math.isclose` verwenden.

---

# Teil XII — Data-Science-Transfer

## 55. Kürzeste Wege in Datenproblemen

- Wissensgraphen: günstigste semantische Verbindung,
- Empfehlungssysteme: Pfade durch Nutzer-Item-Beziehungen,
- Feature-Abhängigkeiten: minimale Transformationskosten,
- Netzwerk- und Log-Analyse: Latenzpfade,
- Bildverarbeitung: Pixelkosten als gewichteter Grid-Graph.

## 56. MST in Data Science

- Clustering: Single-Linkage-Hierarchie lässt sich über einen MST verstehen,
- Dimensionsreduktion und Manifold-Näherungen,
- kostengünstige Sensor- oder Kommunikationsnetze,
- Visualisierung einer dünnen globalen Datenstruktur,
- Erkennen ungewöhnlich schwerer Verbindungen.

Ein MST reduziert `E` Kanten auf `V-1` und bewahrt Zusammenhang, aber nicht alle
lokalen Distanzen.

## 57. Union-Find in Clustering

Wenn Kanten nach Ähnlichkeit sortiert hinzukommen, vereinigt Union-Find schrittweise
Cluster. Kruskal kann deshalb als Agglomerationsprozess gelesen werden: Anfangs
ist jeder Punkt ein Cluster, danach verbinden die billigsten zulässigen Kanten
Komponenten.

---

# Teil XIII — Korrektheits- und Testdenken

## 58. Nützliche Invarianten für Dijkstra

- `dist[start] == 0`
- Distanzen werden nur kleiner, niemals größer.
- Jeder Vorgänger bildet eine echte Graphkante.
- Für jede Kante gilt nach Abschluss `dist[v] <= dist[u] + w(u,v)`.
- Unerreichbare Knoten bleiben `∞`.
- Bei nichtnegativen Gewichten ist die Fixierreihenfolge nicht fallend.

```python
for source, neighbors in weighted_graph.items():
    for target, weight in neighbors.items():
        assert dijkstra_distances[target] <= dijkstra_distances[source] + weight

settled_distances = [
    event[2] for event in dijkstra_trace if event[0] == "fixiere"
]
assert settled_distances == sorted(settled_distances)
```

## 59. Nützliche Invarianten für einen MST

- exakt `V-1` Kanten bei `V > 0`,
- alle Knoten sind verbunden,
- kein Zyklus,
- Summe entspricht bei Kruskal und Prim,
- Entfernen einer Baumkante zerlegt den Baum,
- bei eindeutigen Kantengewichten ist der MST eindeutig.

```python
def verify_tree(vertices, edges):
    if not vertices:
        return len(edges) == 0
    if len(edges) != len(vertices) - 1:
        return False
    dsu = DisjointSet(vertices)
    for edge in edges:
        if not dsu.union(edge.source, edge.target):
            return False
    return dsu.component_count == 1

assert verify_tree(mst_vertices, kruskal_tree)
assert verify_tree(mst_vertices, prim_tree)
```

## 60. Referenz- und Property-Tests

Für zufällige kleine Graphen sind starke Tests möglich:

- Dijkstra gegen Bellman-Ford auf nichtnegativen Gewichten,
- Kruskal-Gesamtgewicht gegen Prim-Gesamtgewicht,
- jeder ausgegebene MST erfüllt Baum-Invarianten,
- Union-Find-Konnektivität gegen eine langsam berechnete Komponentenreferenz,
- Pfadkosten gegen die ausgegebene Distanz,
- Seed-basierte Reproduzierbarkeit.

Nicht nur ein erwartetes Beispiel testen: Randfälle wie leerer Graph, einzelner
Knoten, Nullgewicht, parallele Kanten, getrennte Komponenten und negative Zyklen
machen Implementierungen robust.

---

# Teil XIV — Kontrollfragen

## 61. Fragen

1. Was minimiert BFS, was Dijkstra?
2. Was genau wird bei einer Kante relaxiert?
3. Warum darf Dijkstra einen entnommenen Knoten endgültig fixieren?
4. Was geschieht mit veralteten Heap-Einträgen?
5. Wann bleibt eine Distanz `∞`?
6. Warum benötigt Bellman-Ford höchstens `V-1` Hauptrunden?
7. Wie weist die zusätzliche Runde einen negativen Zyklus nach?
8. Was bewirkt Path Compression?
9. Wann steigt der Rank einer Union-Find-Wurzel?
10. Welche Frage beantwortet Union-Find nicht?
11. Warum erzeugt Kruskal keinen Zyklus?
12. Wie unterscheiden sich Kruskals und Prims Wachstumsmuster?
13. Warum ist ein MST kein Kürzester-Wege-Baum?
14. Was bedeutet eine zulässige A*-Heuristik?
15. Welches Verfahren würdest du bei negativen Kanten wählen?

## 62. Kurzantworten

1. BFS minimiert Kantenzahl; Dijkstra minimiert die Summe nichtnegativer Gewichte.
2. `dist[v]` wird mit `dist[u] + w(u,v)` verglichen und gegebenenfalls verbessert.
3. Nichtnegative Restkanten können keinen späteren billigeren Weg erzeugen.
4. Sie werden anhand der inzwischen kleineren gespeicherten Distanz übersprungen.
5. Wenn der Knoten vom Start nicht erreichbar ist.
6. Ein kürzester einfacher Pfad hat höchstens `V-1` Kanten.
7. Eine weitere Verbesserung beweist einen erreichbaren Zyklus mit negativer Summe.
8. Besuchte Knoten zeigen danach direkt auf die Wurzel.
9. Nur beim Vereinigen zweier Wurzeln gleichen Rangs.
10. Einen konkreten Pfad zwischen zwei verbundenen Elementen.
11. Eine Kante wird nur zwischen verschiedenen Komponenten gewählt.
12. Kruskal vereinigt einen Wald; Prim erweitert einen Baum.
13. Er minimiert die globale Kantensumme, nicht individuelle Pfaddistanzen.
14. Sie überschätzt die echten Restkosten nie.
15. Bellman-Ford; bei erreichbaren negativen Zyklen existieren keine endlichen
    kürzesten Wege zu allen betroffenen Zielen.

---

# Teil XV — Kompakte Gesamtübersicht

## 63. Spickzettel

| Verfahren | Ziel | Voraussetzung | Struktur | Zeit |
|---|---|---|---|---:|
| BFS | wenigste Kanten | gleiche Kosten | Queue | `O(V+E)` |
| Dijkstra | kürzeste Wege | Gewichte `>= 0` | Min-Heap | `O(E log V)` |
| Bellman-Ford | kürzeste Wege | negative Kanten erlaubt | Kantenliste | `O(VE)` |
| Union-Find | Komponenten | dynamische Vereinigungen | Parent + Rank | amortisiert `O(α(V))` |
| Kruskal | MST | ungerichtet, zusammenhängend | Sortierung + DSU | `O(E log E)` |
| Prim | MST | ungerichtet, zusammenhängend | Adjazenzliste + Heap | `O(E log V)` |
| A* | Weg zu einem Ziel | geeignete Heuristik | Heap + `g+h` | problemabhängig |

## 64. Mentale Modelle

```text
Dijkstra:    "Take the cheapest node available right now."
Bellman-Ford: "Give each edge repeated the chance to improve."
Union-Find: "Do these elements already have the same root?"
Kruskal:     "Take the cheapest edge that connects two components."
Prim:        "Take the cheapest edge out of the tree so far."
A*:          "Like Dijkstra, but with an informed view to the destination."
```

## 65. Ausblick auf die Projekte

- **01-basic:** Dijkstra und Union-Find selbst bauen; Dijkstra gegen die
  vollständige Handrechnung dieses Skripts testen.
- **02-medium:** Kruskal und Prim auf identischen Seed-Zufallsgraphen vergleichen.
- **03-final:** Routen und Sperrungen in einem synthetischen Straßennetz analysieren
  und visualisieren; die Algorithmen bleiben Eigenimplementierungen.

Der verbindende Gedanke des Moduls ist die kontrollierte lokale Entscheidung:
Dijkstra fixiert die billigste bekannte Distanz, Union-Find fasst Komponenten
effizient zusammen, und MST-Verfahren wählen sichere billige Kanten. Entscheidend
ist immer, die Voraussetzung hinter der lokalen Wahl zu kennen.
