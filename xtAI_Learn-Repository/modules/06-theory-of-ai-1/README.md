# Module 06 — Theory of Artificial Intelligence 1

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

**What is this about?** This module deals with the classical, *symbolic* foundation of AI: how do you formulate a problem so that a machine can solve it by **search**, and how do you represent knowledge so formally that a computer can **draw logically correct conclusions** from it? It is the theoretical counterpart to statistical learning (modules 04/05): instead of learning from data, here we construct procedures whose correctness and optimality can be *proved*.

**Helpful prior knowledge.** Basic discrete mathematics (sets, relations, functions, proof by induction), some graph theory (nodes, edges, paths) and familiarity with Landau notation ($O$, $\Theta$). Programming skills (Python) for the projects. No prior knowledge from the ML modules is needed — this module stands on its own.

**Recommended earlier modules.** None is mandatory. Whoever has done module 01 "Introduction in AI" already knows A\* and minimax informally; here they are *proved*.

**Following module.** Module 07 "Theory of AI 2" builds on this: planning, acting under uncertainty (probabilistic reasoning, Bayesian networks), non-monotonic reasoning and description logics.

---

## Learning objectives

After this module you should be able to

- formalize a real problem as a **search problem** (state space, actions, transition model, goal test, path costs) and estimate the size of the state space;
- implement the **uninformed search procedures** (BFS, UCS/Dijkstra, DFS, IDDFS) and **prove** their completeness, optimality, time and space complexity;
- design **heuristics**, distinguish the notions *admissible* and *consistent* precisely and prove the **optimality of A\***;
- place **local search** (hill climbing, simulated annealing, genetic algorithms) and **adversarial search** (minimax, $\alpha$-$\beta$ pruning);
- model a **constraint satisfaction problem (CSP)** and solve it with backtracking, forward checking and arc consistency (AC-3) plus ordering heuristics;
- master **propositional logic** completely: syntax, semantics, satisfiability, entailment, normal forms, the **resolution calculus** and the **DPLL algorithm**, including soundness and refutation completeness;
- understand **first-order logic (FOL)**: syntax and semantics (interpretations, models), **unification** (most general unifier), **FOL resolution**, the **Herbrand theorem** and the **(semi-)decidability** of validity;
- explain *why* these procedures give the right answer — not only *that* they do.

---

## Part 1 — Foundations: problem solving as search

### 1.1 The agent and the problem formulation

A **goal-based agent** thinks before it acts: it imagines a sequence of actions, checks in its head where they lead, and picks one that leads to the goal. For a computer to do that, the problem has to be decomposed into five components. A **search problem** is the tuple

$$
\mathcal{P} = (S, s_0, A, \mathrm{Result}, \mathrm{Goal}, c)
$$

with

- $S$: the set of all **states** (the *state space*),
- $s_0 \in S$: the **initial state**,
- $A$: the set of **actions**; $\mathrm{Actions}(s) \subseteq A$ gives the actions applicable in $s$,
- $\mathrm{Result}: S \times A \to S$: the **transition model** (deterministic), $\mathrm{Result}(s,a) = s'$,
- $\mathrm{Goal}: S \to \{\text{true}, \text{false}\}$: the **goal test** (which may be a set $S_g \subseteq S$ or a property),
- $c(s,a,s') \ge 0$: the **step cost** of an action. We require $c(s,a,s') \ge \varepsilon > 0$ for a fixed $\varepsilon$ (costs are bounded away from 0 from below) — we need that later for the termination and optimality proofs.

A **solution** is a sequence of actions $a_1, \dots, a_n$ that carries $s_0$ via $s_i = \mathrm{Result}(s_{i-1}, a_i)$ into a goal state $s_n$. The **path cost** is $g = \sum_{i=1}^n c(s_{i-1}, a_i, s_i)$. An **optimal solution** has minimal path cost $C^\ast$.

> **An important abstraction.** The search problem is a *model* of reality. The art of formalization lies in leaving out just enough that the problem stays solvable without making the solution useless. The state space is a directed graph: nodes = states, edges = actions (weighted with step costs).

**Two running examples:**

*Romania* (route planning): states = cities, actions = "drive to a neighbouring city", costs = road length in km. Start Arad, goal Bucharest. The state space is small and given explicitly as a map.

*The 8-puzzle*: a $3\times 3$ sliding puzzle with tiles 1–8 and one gap. State = the arrangement of the tiles, actions = slide the gap up/down/left/right, cost = 1 per move. The state space has $9!/2 = 181\,440$ reachable states (half of all $9!$ permutations are unreachable from a given starting position — the *parity invariant*). For the 15-puzzle it is already $16!/2 \approx 10^{13}$ — too large to store the state space explicitly. That is why we generate states **on demand** via the transition model.

### 1.2 Search tree, search graph and the generic algorithm

Search explores the state space by building a **search tree**. A **node** $n$ of the search tree is a bookkeeping structure with:

- `n.state` — the associated state,
- `n.parent` — the parent node (for reconstructing the path),
- `n.action` — the action that led here from the parent state,
- `n.g` — the path cost from $s_0$ to `n.state` along this path.

**Careful, a subtle difference:** a *state* is a configuration of the world; a *node* is a path to it. Two different nodes can carry the same state (reached via different paths). **Expanding** a node creates one child node per applicable action. The **frontier** (the open list) is the set of nodes that have been generated but not yet expanded. The **explored set** (the closed list) records already expanded states in order to avoid cycles and redundancy.

```
function GRAPH-SEARCH(problem) returns a solution or failure:
    frontier  ← {Node(s0)}            # priority structure depending on the strategy
    explored  ← ∅
    while frontier ≠ ∅:
        n ← REMOVE(frontier)          # which node? -> that IS the strategy
        if Goal(n.state): return SOLUTION(n)
        add n.state to explored
        for each a in Actions(n.state):
            s' ← Result(n.state, a)
            child ← Node(state=s', parent=n, action=a, g = n.g + c(...))
            if s' ∉ explored and s' not already in frontier (with ≤ cost):
                frontier ← INSERT(child, frontier)
    return failure
```

The only adjustable screw is **which node `REMOVE` picks next**. That single decision generates all of the procedures that follow.

### 1.3 Evaluation criteria

We judge a search procedure by four criteria:

1. **Completeness:** does it *guarantee* to find a solution if one exists?
2. **Optimality:** does it guarantee to find a *cost-minimal* solution?
3. **Time complexity:** how many nodes are generated/expanded?
4. **Space complexity:** how many nodes have to be held in memory simultaneously?

We express the complexity in three parameters:

- $b$ — the **branching factor**, the maximum number of successors of a node,
- $d$ — the **depth of the shallowest solution**,
- $m$ — the **maximum depth** of the state space (which may be $\infty$).

### 1.4 Uninformed (blind) search

"Uninformed" means: the procedure uses **no** problem-specific information about how close a state is to the goal. It only knows the problem definition.

**Breadth-first search (BFS).** The `frontier` is a FIFO queue → expand nodes in the order of their generation, that is layer by layer: first all nodes at depth 0, then depth 1, and so on. The goal test happens at *generation* time (not only at expansion), which saves one layer.

- *Complete:* yes, if $b$ is finite (the shallowest solution at depth $d$ is found after finitely many nodes).
- *Optimal:* yes, **if all step costs are equal** (then the shallowest solution is also the cheapest). With unequal costs, in general **no**.
- *Time:* $1 + b + b^2 + \dots + b^d = O(b^d)$.
- *Space:* $O(b^d)$ — every generated node stays in memory (frontier + explored). **Space is the real killer criterion**: with $b=10$ and $d=12$ that is about $10^{12}$ nodes.

> **Proof of the optimality of BFS with unit costs (induction over the layer).** BFS expands nodes in non-decreasing order of depth. Claim: if a goal node at depth $d$ is generated for the first time, then there is no goal node at depth $< d$. Suppose there were one at depth $d' < d$. Since BFS proceeds layer by layer, all nodes at depth $d'$ would have been generated *before* any node at depth $d$ is generated — so the goal node at $d'$ would have been found first, a contradiction. Since with unit costs depth is proportional to cost, the shallowest solution is optimal. $\qquad\blacksquare$

**Uniform-cost search (UCS) = Dijkstra for search problems.** The `frontier` is a **priority queue on $g(n)$** (the path cost so far) → always expand the cheapest open node. Two decisive details compared with BFS: (a) the goal test happens at **expansion** time, not at generation (otherwise it is not optimal, because an expensive path to the goal could be generated earlier than a cheaper one). (b) If a cheaper path to a frontier state is found, it replaces the more expensive one.

- *Complete:* yes, if the step costs are $\ge \varepsilon > 0$ (otherwise infinitely many zero-cost steps could form an infinite chain).
- *Optimal:* yes, always (proof below).
- *Complexity:* let $C^\ast$ be the cost of the optimal solution. UCS expands all nodes with path cost $< C^\ast$. That is up to $O\!\left(b^{1 + \lfloor C^\ast / \varepsilon \rfloor}\right)$ — with unit costs ($\varepsilon = 1$, $C^\ast = d$) somewhat more than BFS, namely $O(b^{d+1})$, because UCS also examines nodes at the goal depth before it picks the goal as the cheapest.

> **Proof of the optimality of UCS.** We show: when UCS selects a node $n$ for expansion, `n.g` is already the *optimal* path cost $g^\ast(n.\text{state})$ from $s_0$ to that state. **(a) Non-decreasing expansion costs:** since the cheapest frontier node is always chosen and children have costs $\ge$ their parent (step costs $\ge 0$), nodes are expanded in non-decreasing order of $g$. **(b) Optimality at expansion:** suppose $n$ is expanded with `n.g` $> g^\ast(n.\text{state})$. Then there is an optimal path with smaller cost; on it there is a first node $n'$ that is still in the frontier. It satisfies `n'.g` $\le g^\ast(n.\text{state}) <$ `n.g`, so $n'$ would have been chosen before $n$ — a contradiction. When the goal is expanded as the cheapest frontier node, its path cost is therefore optimal. $\qquad\blacksquare$

**Depth-first search (DFS).** The `frontier` is a LIFO stack → always expand the most recently generated node, that is, go as deep as possible before going back (backtracking).

- *Complete:* **no** in general (it can get stuck in an infinite branch or a cycle); yes in finite state spaces with cycle detection.
- *Optimal:* no.
- *Time:* $O(b^m)$ — bad when $m \gg d$.
- *Space:* $O(bm)$ — **that is the advantage**: DFS only has to store the current path plus the sibling nodes, not the whole tree. Linear instead of exponential.

**Depth-limited search and iterative deepening (IDDFS).** Depth-limited search is DFS with a hard depth limit $\ell$ (it cuts off deeper branches). IDDFS calls it with $\ell = 0, 1, 2, \dots$ until a solution is found.

- It combines **the best of both worlds**: space $O(bd)$ like DFS, completeness and (with unit costs) optimality like BFS.
- *Time:* $O(b^d)$. The apparent extra work of repeatedly searching the upper layers is asymptotically negligible: the bottom layer ($b^d$ nodes) is generated only once, the second to last twice, …, the root $d{+}1$ times. Sum: $\sum_{i=0}^{d}(d{+}1-i)\,b^i = O(b^d)$, dominated by the last term. With $b=10$ the overhead is only about 11 %.

> **IDDFS is the standard workhorse of uninformed search** when the state space is large and the solution depth is unknown — precisely because it avoids the exponential memory bound of BFS.

**Bidirectional search.** Search forwards from $s_0$ and backwards from the goal simultaneously; stop when the two fronts meet. Time and space $O(b^{d/2})$ — dramatically better, but only applicable if the transition model is invertible and an *explicit* goal state is available (not merely a goal predicate test).

**Overview (finite $b$; unit costs for "optimal"):**

| Criterion | BFS | UCS | DFS | IDDFS |
|---|---|---|---|---|
| Complete? | yes | yes ($c\ge\varepsilon$) | no\* | yes |
| Optimal? | yes (unit costs) | yes (always) | no | yes (unit costs) |
| Time | $O(b^d)$ | $O(b^{1+\lfloor C^\ast/\varepsilon\rfloor})$ | $O(b^m)$ | $O(b^d)$ |
| Space | $O(b^d)$ | $O(b^{1+\lfloor C^\ast/\varepsilon\rfloor})$ | $O(bm)$ | $O(bd)$ |

\* complete in finite spaces with a cycle check.

---

## Part 2 — Building up: informed and advanced search

### 2.1 Heuristics

A **heuristic** $h(n) \ge 0$ estimates the cost of the *cheapest path from the state `n.state` to a goal state*. By convention $h(n) = 0$ for goal states. It is the problem-specific "hunch" that uninformed search lacks. Two properties are central:

**Admissibility.** $h$ is *admissible* if it **never overestimates** the true remaining cost $h^\ast(n)$:
$$
0 \le h(n) \le h^\ast(n) \quad \text{for all } n.
$$
An admissible heuristic is *optimistic* — it believes the goal to be at least as close as it really is.

**Consistency / monotonicity.** $h$ is *consistent* if it satisfies the **triangle inequality** for every action $a$ from $s$ to $s'$:
$$
h(s) \le c(s, a, s') + h(s').
$$
Intuitively: the estimated remaining cost may fall by at most the real step cost in one step.

> **Theorem: consistency $\Rightarrow$ admissibility** (but not conversely). *Proof* by induction over the number $k$ of steps on the optimal remaining path from $s$ to the nearest goal. **Base** $k=0$: $s$ is a goal state, $h(s)=0=h^\ast(s)$. **Step:** let the optimal remaining path be $s \to s' \to \dots \to \text{goal}$ with $k$ steps, the first step being $a$. By the induction hypothesis $h(s') \le h^\ast(s')$. Consistency gives $h(s) \le c(s,a,s') + h(s') \le c(s,a,s') + h^\ast(s') = h^\ast(s)$, because $s'$ lies on the optimal remaining path. $\qquad\blacksquare$

**Heuristics for the 8-puzzle** (both admissible):
- $h_1$ = the number of misplaced tiles (Hamming). Admissible, because every misplaced tile needs at least one move.
- $h_2$ = the sum of the **Manhattan distances** of every tile to its goal position. Admissible, because one move moves a tile by exactly 1 in the Manhattan metric and tiles cannot move "through each other". We always have $h_2(n) \ge h_1(n)$: $h_2$ **dominates** $h_1$.

### 2.2 A\* — informed search

A\* is UCS, but with a different priority function. Instead of considering only the cost so far $g(n)$, A\* uses the **estimated total cost** of a path through $n$:
$$
f(n) = g(n) + h(n).
$$
$g(n)$ = the known cost from $s_0$ to `n.state`, $h(n)$ = the estimated remaining cost to the goal. The `frontier` is a priority queue on $f$. A\* always expands the node with the smallest $f$.

**A\* generalizes the other procedures:** $h \equiv 0$ gives UCS. "Greedy best-first search" ($f = h$, ignoring $g$) is the other extreme — fast, but **neither optimal nor complete** (it can run off in the wrong direction).

> **Theorem (optimality of A\*, tree search).** If $h$ is admissible, then A\* without an explored set (pure tree search) returns an optimal solution. *Proof.* Let $C^\ast$ be the optimal solution cost and $G_2$ a *suboptimal* goal node in the frontier with $g(G_2) > C^\ast$. Since $h(G_2)=0$, we have $f(G_2) = g(G_2) > C^\ast$. On the optimal path there is always a frontier node $n$ (the path has been expanded up to somewhere). For it, admissibility gives $f(n) = g(n) + h(n) \le g(n) + h^\ast(n) = C^\ast$. Hence $f(n) \le C^\ast < f(G_2)$: A\* chooses $n$ before $G_2$. A suboptimal goal node is therefore never expanded before the optimal one is reached. $\qquad\blacksquare$

> **Theorem (optimality of A\*, graph search).** If $h$ is **consistent**, then A\* with an explored set is optimal. The reason: under consistency the $f$ values along every path are **non-decreasing** ($f(s') = g(s')+h(s') = g(s)+c(s,a,s')+h(s') \ge g(s)+h(s) = f(s)$). It follows that A\* expands states in non-decreasing order of $f$ and reaches a state at its *first* expansion already with the optimal $g$ — exactly as with UCS. Mere admissibility is **not** enough for graph search, unless one allows re-opening already closed nodes when a cheaper path is found.

**Optimal efficiency.** Among all procedures that use the same admissible heuristic, A\* expands (apart from nodes with exactly $f=C^\ast$) *no node* that it would not have to: every node with $f(n) < C^\ast$ **must** be expanded by every optimal, complete procedure (otherwise a better solution could be hidden there). In this sense A\* is **optimally efficient**.

**Dominance.** If $h_a, h_b$ are both admissible and $h_a(n) \ge h_b(n)$ for all $n$, then A\* with $h_a$ never expands more nodes than with $h_b$ (apart from ties). **A higher admissible heuristic is better.** That is why $h_2$ (Manhattan) is preferable to $h_1$ (Hamming) on the 8-puzzle. From several heuristics one can always form $h(n) = \max\{h_a(n), h_b(n)\}$ — again admissible and at least as good.

**Constructing heuristics systematically.** Where do good admissible heuristics come from?
- **Relaxation:** one solves a *simplified* problem exactly. If in the 8-puzzle you drop the rule "a tile can only move onto the empty square", every tile may "jump" straight to its goal → cost = the Manhattan distance. The exact cost of a relaxed problem is **always** an admissible (and even consistent) heuristic for the original, because the original only has *more* restrictions and is therefore never cheaper.
- **Pattern databases:** one solves subproblems (e.g. only tiles 1–4) completely in advance and stores their exact costs in a table. At runtime the remaining cost is looked up. For the 15-puzzle, *disjoint* pattern databases lower the search time by orders of magnitude.

**The memory problem of A\*.** Like UCS, A\* potentially holds exponentially many nodes. Remedies: **IDA\*** (iterative-deepening A\*, DFS with an $f$ threshold instead of a depth threshold) and **SMA\*** (memory-bounded A\*) trade memory for time.

### 2.3 Local search

When only the **goal state** matters and not the way there (e.g. with $n$-queens, scheduling, layout optimization), the path is irrelevant. Then one works with **local search**: one keeps only *one* current state and tries to improve it step by step. Memory $O(1)$, huge state spaces become manageable. One imagines a **landscape of the objective function** (height = quality); the global peak is sought.

**Hill climbing.** Always move to the best neighbour that is better than the current state; stop when no neighbour is better. Simple and memory efficient, but it gets stuck in **local maxima**, on **plateaus** and on **ridges**. Variants: *stochastic* HC (a random choice among the improving neighbours), *first-choice* HC, *random-restart* HC (restart from a random state when stuck — surprisingly effective).

**Simulated annealing.** Combines HC with occasional "downhill" steps in order to escape local maxima. A worsening step by $\Delta E < 0$ is accepted with probability $e^{\Delta E / T}$; the "temperature" $T$ decreases slowly towards 0 according to a cooling schedule. At the start (high $T$) it is almost a random walk, at the end (low $T$) almost pure HC. **A theoretical result:** if $T$ decreases sufficiently slowly ($T_k \ge c/\log k$), SA converges to the global optimum with probability 1 — but that is too slow in practice; one uses faster, heuristic schedules.

**Genetic algorithms.** These keep a *population* of states. New states arise through **selection** (fit individuals are preferred), **crossover** (two parents combine partial solutions) and **mutation** (a random small change). Crossover is only effective if the encoding is chosen so that related partial solutions lie next to each other (the schema theorem). GAs are robust but coarse optimizers.

### 2.4 Adversarial search (games)

In **two-player zero-sum games with perfect information** (chess, checkers, tic-tac-toe) an opponent plans *against* us. The state space is a **game tree** in which the moves of MAX (us, maximizing) and MIN (the opponent, minimizing) alternate. A **utility function** evaluates terminal states (+1 win, 0 draw, −1 loss).

**Minimax.** The value of a node is:
$$
\mathrm{Minimax}(s) =
\begin{cases}
\mathrm{Utility}(s) & \text{if } s \text{ is terminal}\\
\max_{a}\mathrm{Minimax}(\mathrm{Result}(s,a)) & \text{if MAX is to move}\\
\min_{a}\mathrm{Minimax}(\mathrm{Result}(s,a)) & \text{if MIN is to move}
\end{cases}
$$
One computes it by depth-first search from the leaves back up. At the root MAX chooses the move to the child with the maximal minimax value. The result is the **optimal move under the assumption that the opponent also plays optimally**. Time $O(b^m)$, space $O(bm)$ — unaffordable down to the leaves for real games.

**$\alpha$-$\beta$ pruning.** Prunes subtrees that can no longer change the result, without falsifying the minimax value. One carries two bounds: $\alpha$ = the best (highest) value MAX can guarantee so far; $\beta$ = the best (lowest) value MIN can guarantee. As soon as $\alpha \ge \beta$ at a node, the remaining subtree can be cut off (the opponent would never allow this branch).

- It gives **exactly the same** minimax value as unpruned minimax — it is not an approximation.
- With an **optimal move ordering** the time falls from $O(b^m)$ to $O(b^{m/2})$ — that *doubles* the searchable depth at the same effort. This is why good **move ordering heuristics** are decisive.

**In practice:** since one can rarely compute down to the leaves, one stops at a depth limit and replaces `Utility` by a heuristic **evaluation function** that estimates the winning chance of a non-terminal position (e.g. the material value in chess). Modern engines (AlphaZero) replace it by neural networks plus Monte Carlo tree search — the transition to statistical AI.

### 2.5 Constraint satisfaction problems (CSP)

A **CSP** is a search problem with a *factored* state representation: a state is an assignment of **variables**, and a goal is any assignment that satisfies all **constraints**. Formally $(X, D, C)$:

- $X = \{X_1, \dots, X_n\}$ — the variables,
- $D = \{D_1, \dots, D_n\}$ — the domains, $X_i$ takes values from $D_i$,
- $C$ — the constraints; a constraint $\langle \mathrm{scope}, \mathrm{rel}\rangle$ names the variables involved and the permitted combinations of values.

A **consistent** assignment violates no constraint; a **complete** one assigns all variables; a **solution** is both. Classics: **map colouring** (colour adjacent regions differently), **Sudoku**, **$n$-queens**, timetables.

**Why a theory of its own?** Because the factored structure permits powerful, *generic* techniques that speed up naive search ($d^n$ leaves) drastically — without a problem-specific heuristic.

**Backtracking search.** A depth-first search that assigns variables one after the other and backtracks when a constraint is violated. The core of the efficiency lies in three ideas:

**(1) Constraint propagation and consistency.** One excludes values *before* branching. An arc $X_i \to X_j$ is called **arc-consistent** if for *every* value in $D_i$ there is at least one permitted partner value in $D_j$. The **AC-3 algorithm** establishes arc consistency across the whole network:

```
function AC-3(csp) returns false (if an inconsistency is detected) otherwise true:
    queue ← all directed arcs (Xi, Xj)
    while queue ≠ ∅:
        (Xi, Xj) ← REMOVE(queue)
        if REVISE(csp, Xi, Xj):                  # removes from Di values with no partner in Dj
            if Di = ∅: return false              # empty domain -> no solution on this branch
            for each Xk in neighbours(Xi) \ {Xj}: # a change to Di can affect the neighbours
                add (Xk, Xi) to queue
    return true
```
`REVISE` deletes from $D_i$ every value that finds no partner in $D_j$. **Complexity:** a CSP with $c$ binary constraints and domain size $\le d$ runs in $O(c\,d^3)$: each of the $c$ arcs enters the queue at most $d$ times (once per removed value of a neighbour), and `REVISE` costs $O(d^2)$.

**(2) Backtracking and inference combined.** *Forward checking* establishes arc consistency after every assignment, but only for the neighbours of the variable just assigned; *MAC* (maintaining arc consistency) calls AC-3 fully on the affected arcs after every assignment.

**(3) Ordering heuristics.**
- **MRV (minimum remaining values / "most constrained variable"):** assign next the variable with the *fewest* remaining legal values — it leads to failure fastest (fail-fast) and prunes the tree.
- **The degree heuristic:** on an MRV tie, take the variable with the most constraints to still unassigned variables.
- **LCV (least constraining value):** try first the value that leaves the neighbours the *most* options.

**Exploiting structure.** If the constraint graph decomposes into independent components, one solves them separately (a multiplicative instead of an exponential gain). If the constraint graph is a **tree**, the CSP is solvable in **$O(n\,d^2)$** — that is, *polynomially* (order topologically, make it arc-consistent backwards, then assign greedily forwards). **Cutset conditioning** and **tree decomposition** carry this advantage approximately over to almost-tree-like graphs. This is the deep insight of CSP theory: **the graph structure determines the complexity.**
---

## Part 3 — Building up: propositional logic

Search finds *solutions*; logic makes it possible to **represent knowledge** and to **derive new facts correctly** from it. A **knowledge-based agent** keeps a *knowledge base* (KB) of sentences and answers questions by checking what *follows* from the KB.

### 3.1 Syntax and semantics

**Syntax.** Formulas of propositional logic are built from **atomic propositions** (proposition symbols $P, Q, R, \dots$, each true or false) and the **connectives** $\lnot$ (not), $\land$ (and), $\lor$ (or), $\Rightarrow$ (implies), $\Leftrightarrow$ (if and only if). Formally (BNF):
$$
\varphi ::= \top \mid \bot \mid P \mid \lnot\varphi \mid (\varphi \land \varphi) \mid (\varphi \lor \varphi) \mid (\varphi \Rightarrow \varphi) \mid (\varphi \Leftrightarrow \varphi).
$$

**Semantics.** A **model** (or *interpretation*) $m$ is an assignment of true/false to every proposition symbol. The truth of a formula under $m$ follows recursively from the **truth tables** of the connectives (in particular: $A \Rightarrow B$ is false only when $A$ is true and $B$ is false — "anything follows from a falsehood"). With $n$ symbols there are $2^n$ models.

**Central semantic notions:**
- $\varphi$ is **satisfiable** if *at least one* model makes $\varphi$ true.
- $\varphi$ is **valid / a tautology** if *every* model makes $\varphi$ true (e.g. $P \lor \lnot P$).
- $\varphi$ is **unsatisfiable** if *no* model makes $\varphi$ true.
- **Duality:** $\varphi$ is valid $\iff$ $\lnot\varphi$ is unsatisfiable.

### 3.2 Logical entailment

The central notion. A KB **entails** a sentence $\alpha$, written
$$
\mathrm{KB} \models \alpha,
$$
exactly when $\alpha$ is true in **every** model in which the KB is true. So: "$\alpha$ is an unavoidable consequence of the KB." Formally $M(\mathrm{KB}) \subseteq M(\alpha)$, where $M(\cdot)$ denotes the set of models.

**The bridging theorem (deduction theorem / refutation):**
$$
\mathrm{KB} \models \alpha \quad\iff\quad (\mathrm{KB} \land \lnot\alpha) \text{ is unsatisfiable.}
$$
This is the workhorse of machine theorem proving: instead of "does $\alpha$ follow?" (a statement about *all* models) one checks "is $\mathrm{KB} \land \lnot\alpha$ unsatisfiable?" — one assumes the *opposite* of the claim and derives a contradiction (**proof by refutation**).

**Model checking** solves entailment by enumerating all $2^n$ models (the truth table method) — sound and complete, but exponential. Propositional satisfiability (**SAT**) is the canonical **NP-complete** problem (the Cook–Levin theorem); entailment is correspondingly **co-NP-complete**. So we look for procedures that are faster *on average* without enumerating all models.

### 3.3 Inference rules, soundness and completeness

An **inference procedure** $i$ derives sentences from a KB syntactically: $\mathrm{KB} \vdash_i \alpha$ means "$i$ derives $\alpha$ from the KB". Two quality criteria connect this *syntactic* derivability with *semantic* entailment:

- **Soundness:** $\mathrm{KB} \vdash_i \alpha \implies \mathrm{KB} \models \alpha$. (It derives only truths — no false conclusions.)
- **Completeness:** $\mathrm{KB} \models \alpha \implies \mathrm{KB} \vdash_i \alpha$. (It derives everything that follows — it misses nothing.)

A procedure that is sound *and* complete reproduces semantic entailment exactly, syntactically. Well-known sound rules: **modus ponens** ($\alpha \Rightarrow \beta,\ \alpha \ \vdash\ \beta$), **and-elimination** ($\alpha \land \beta \vdash \alpha$).

### 3.4 Normal forms and the resolution calculus

**Conjunctive normal form (CNF).** A formula is in CNF if it is a **conjunction of clauses**, where a **clause** is a **disjunction of literals** (a *literal* being an atom or its negation). **Every** propositional formula can be converted into an equivalent CNF:

1. Eliminate $\Leftrightarrow$: $A \Leftrightarrow B$ becomes $(A \Rightarrow B) \land (B \Rightarrow A)$.
2. Eliminate $\Rightarrow$: $A \Rightarrow B$ becomes $\lnot A \lor B$.
3. Move negations inwards (De Morgan): $\lnot(A\land B)\equiv \lnot A\lor\lnot B$, $\lnot(A\lor B)\equiv \lnot A\land\lnot B$, $\lnot\lnot A\equiv A$.
4. Distribute $\lor$ over $\land$: $A\lor(B\land C)\equiv (A\lor B)\land(A\lor C)$.

**The resolution rule.** From two clauses that contain a complementary pair of literals $\ell$ / $\lnot\ell$, one derives their **resolvent** — the union of the remaining literals:
$$
\frac{(\ell \lor a_1 \lor \dots \lor a_k)\qquad(\lnot\ell \lor b_1 \lor \dots \lor b_m)}{(a_1 \lor \dots \lor a_k \lor b_1 \lor \dots \lor b_m)}.
$$
If the result contains a literal *and* its negation, it is a tautology and is discarded. If two complementary single-literal clauses ($\ell$ and $\lnot\ell$) are resolved, the **empty clause** $\square$ arises — by definition it is **unsatisfiable** and signals the contradiction we were looking for. The rule is **sound**: every model that satisfies both premises also satisfies the resolvent (case distinction on the truth value of $\ell$).

**The resolution algorithm (a refutation procedure).** To show $\mathrm{KB} \models \alpha$:
1. Form $\mathrm{KB} \land \lnot\alpha$ and convert it to CNF → a set of clauses.
2. Apply resolution repeatedly to all pairs of clauses, adding new resolvents.
3. If the **empty clause** $\square$ arises → $\mathrm{KB}\land\lnot\alpha$ is unsatisfiable → $\mathrm{KB}\models\alpha$. **Proved.**
4. If no new clauses can be generated and $\square$ is not among them → $\alpha$ does **not** follow.

> **Theorem (refutation completeness of resolution).** If a propositional set of clauses is unsatisfiable, then resolution derives the empty clause $\square$ in finitely many steps. *Proof idea:* via the **ground resolution theorem** — one shows by induction over the number of symbols that $\square$ can always be derived from an unsatisfiable set of clauses (semantic trees / the construction of a model if $\square$ is not derivable, which contradicts unsatisfiability). Since there are only finitely many clauses over $n$ symbols, the procedure terminates. Important: resolution is *refutation* complete — it proves unsatisfiability, not directly every entailment; but that suffices because of the bridging theorem. $\qquad\blacksquare$

### 3.5 Horn clauses and efficient inference

The general SAT case is NP-hard. For a practically important **subclass** it works *linearly*. A **Horn clause** is a clause with **at most one positive literal**. Written as an implication: $(\lnot P_1 \lor \dots \lor \lnot P_k \lor Q) \equiv (P_1 \land \dots \land P_k \Rightarrow Q)$ — a *definite* clause (exactly one positive literal). Facts are definite clauses without a premise.

For Horn KBs, **forward chaining** and **backward chaining** are sound and complete and run in **linear time** in the size of the KB. Forward chaining applies modus ponens in a data-driven way until nothing new follows (this is the theoretical basis of modules 32/33 — deductive databases and logic programming/Prolog). Backward chaining starts goal-driven from the query. The satisfiability of Horn formulas (**HORNSAT**) is therefore in P.

### 3.6 DPLL — the engine of modern SAT solvers

Instead of enumerating models completely (a truth table), **DPLL** (Davis–Putnam–Logemann–Loveland) searches the assignment tree with backtracking and three accelerators. DPLL decides the **satisfiability** of a set of CNF clauses:

1. **Early termination.** If every clause is already satisfied by a true literal → *satisfiable* (the rest does not matter). If a clause is completely false → this branch is *unsatisfiable*, backtrack.
2. **Unit propagation (the one-literal rule).** If a clause contains only *one* unassigned literal left (all others false), that one must be **true**. Set it and propagate — this can trigger a cascade of further unit clauses. (This is by far the most effective step.)
3. **The pure literal rule.** If a symbol occurs in all remaining clauses with only *one* polarity (purely positive or purely negative), assign it accordingly — that satisfies clauses without ever doing harm.

Only when none of these rules applies does DPLL **branch** on a still free symbol (true/false) and recurse. Modern **CDCL solvers** (conflict-driven clause learning) extend DPLL by *clause learning* from conflicts, non-chronological backtracking (backjumping) and activity heuristics (VSIDS). They solve industrial SAT instances with *millions* of variables — the backbone of verification, planning and configuration.

---

## Part 4 — Advanced: first-order logic

Propositional logic is weak in expressive power: "all humans are mortal" can only be represented by *one proposition per human*. **First-order logic (FOL)** introduces **objects**, **relations/predicates**, **functions** and **quantifiers** and can thus talk about *whole classes* of objects.

### 4.1 Syntax

The building blocks:
- **Terms** denote objects: **constants** ($a, b, \mathrm{Socrates}$), **variables** ($x, y$), **function applications** ($f(x)$, $\mathrm{Father}(\mathrm{Socrates})$).
- **Atomic formulas** are **predicates** over terms: $\mathrm{Human}(\mathrm{Socrates})$, $\mathrm{Greater}(x, y)$, as well as equality $t_1 = t_2$.
- **Connectives** as in propositional logic plus **quantifiers**: the **universal quantifier** $\forall x\,\varphi$ ("for all $x$") and the **existential quantifier** $\exists x\,\varphi$ ("there is an $x$").

Examples:
$$
\forall x\,\big(\mathrm{Human}(x) \Rightarrow \mathrm{Mortal}(x)\big), \qquad
\exists x\,\big(\mathrm{Cat}(x) \land \mathrm{Black}(x)\big).
$$
**Rule of thumb:** $\forall$ usually goes with $\Rightarrow$ (not $\land$!), $\exists$ usually with $\land$ (not $\Rightarrow$!). $\forall x\,(\mathrm{Human}(x)\land\mathrm{Mortal}(x))$ would mean "everything is a mortal human" — too strong.

**Duality of the quantifiers:** $\lnot\forall x\,\varphi \equiv \exists x\,\lnot\varphi$ and $\lnot\exists x\,\varphi \equiv \forall x\,\lnot\varphi$.

### 4.2 Semantics

An **interpretation** (structure) $\mathcal{I} = (\mathcal{D}, \cdot^\mathcal{I})$ consists of a non-empty **domain** $\mathcal{D}$ (the *universe of discourse* — the objects that exist) and an **interpretation function** that assigns
- an object in $\mathcal{D}$ to every constant,
- a function $\mathcal{D}^k \to \mathcal{D}$ to every $k$-ary function symbol,
- a relation $\subseteq \mathcal{D}^k$ to every $k$-ary predicate symbol.

Together with a variable assignment this fixes the truth value of every formula: $\forall x\,\varphi$ is true if $\varphi$ is true for *every* assignment of an object from $\mathcal{D}$ to $x$; $\exists x\,\varphi$ if it is true for *at least one*. An $\mathcal{I}$ that makes $\varphi$ true is a **model** of $\varphi$. The notions satisfiable / valid / entailment ($\mathrm{KB}\models\alpha$) are defined as in part 3 — only now over the in general **infinite** set of all interpretations.

### 4.3 Unification

To lift inference rules to FOL, one needs a mechanism that assigns (**substitutes**) values to variables so that two terms/atoms become *syntactically equal*. A **substitution** $\theta$ is a finite map from variables to terms, written $\{x/t, \dots\}$; $\varphi\theta$ applies it. Two atoms $p, q$ are **unifiable** if a $\theta$ with $p\theta = q\theta$ exists; such a $\theta$ is called a **unifier**.

Example: $\mathrm{Knows}(\mathrm{John}, x)$ and $\mathrm{Knows}(y, \mathrm{Mary})$ unify with $\theta = \{y/\mathrm{John},\ x/\mathrm{Mary}\}$ to $\mathrm{Knows}(\mathrm{John}, \mathrm{Mary})$.

The **most general unifier (MGU)** is the unifier that *commits to the least* — every other unifier is a specialization of it. The MGU is **unique** up to the renaming of variables. The unification algorithm runs recursively over the term structure:

```
function UNIFY(x, y, θ) returns a substitution or failure:
    if θ = failure: return failure
    if x = y: return θ
    if VARIABLE?(x): return UNIFY-VAR(x, y, θ)
    if VARIABLE?(y): return UNIFY-VAR(y, x, θ)
    if COMPOUND?(x) and COMPOUND?(y):        # same function/predicate, arguments pairwise
        return UNIFY(ARGS(x), ARGS(y), UNIFY(OP(x), OP(y), θ))
    if LIST?(x) and LIST?(y):
        return UNIFY(REST(x), REST(y), UNIFY(FIRST(x), FIRST(y), θ))
    return failure
```

**The occurs check.** `UNIFY-VAR` may only bind $x/t$ if $x$ **does not occur in $t$** — otherwise one would try to unify $x$ with $f(x)$, which leads to an infinite term. The naive occurs check costs time; many Prolog systems omit it for efficiency (unsound, but rarely a problem in practice). With suitable data structures, unification is possible in **linear time**.

### 4.4 FOL resolution and the Herbrand theorem

The resolution calculus carries over to FOL — with two extensions: **Skolemization** (to get rid of existential quantifiers) and **unification** (instead of exactly complementary literals, *unifiable* ones suffice).

**Conjunctive normal form in FOL.** In addition to the steps from 3.4:
- **Standardize the variables apart:** rename every quantified variable uniquely (no name collisions).
- **Skolemization:** eliminate existential quantifiers. An $\exists y$ that lies in the scope of no $\forall$ is replaced by a new **Skolem constant**; if $\exists y$ lies in the scope of $\forall x_1\dots\forall x_k$, one replaces $y$ by a new **Skolem function** $g(x_1,\dots,x_k)$ (the witness depends on the outer variables). Skolemization preserves *satisfiability* (not logical equivalence) — which is exactly what the refutation procedure needs.
- **Drop the universal quantifiers** (they are assumed implicitly) and extract the clauses.

**Generalized resolution.** Two clauses with literals $\ell_i$ and $\lnot m_j$ such that $\mathrm{UNIFY}(\ell_i, m_j) = \theta$ exists resolve to the resolvent, to which $\theta$ is applied:
$$
\frac{(\ell \lor \mathbf{a}) \qquad (\lnot m \lor \mathbf{b})}{(\mathbf{a} \lor \mathbf{b})\theta}, \qquad \theta = \mathrm{MGU}(\ell, m).
$$
The rest works as in propositional logic: KB $\land\ \lnot$query in CNF, resolve until $\square$ arises.

**The Herbrand theorem — why this works.** The key that reduces FOL inference to the propositional case. The **Herbrand universe** of a set of clauses is the set of *all* ground (variable-free) terms that can be formed from its constants and function symbols (*infinite* when function symbols are present). A **Herbrand interpretation** assigns values only to these ground terms.

> **Herbrand's theorem (1930).** A set of FOL clauses is unsatisfiable if and only if a **finite** set of **ground instances** (clauses substituted with terms of the Herbrand universe) is *propositionally* unsatisfiable.

In principle this reduces FOL unsatisfiability to finitely many propositional unsatisfiability checks — one would only have to find the right finite set of ground instances. The **lifting lemma** shows: instead of blindly generating ground instances, one can resolve directly at the first-order level with unification and obtains the same power — that is the basis of the **refutation completeness of FOL resolution** (Robinson, 1965): *if a set of FOL clauses is unsatisfiable, then resolution with unification derives the empty clause.*

### 4.5 Decidability: the fundamental limit

A fundamental difference from propositional logic:

> **FOL validity is *undecidable* (Church & Turing, 1936), but *semi-decidable*.**

- **Semi-decidable (recursively enumerable):** if $\alpha$ is *valid* (or follows from the KB), then resolution **finds** that after finitely many steps (completeness) — the proof terminates with "yes".
- **Undecidable:** if $\alpha$ is **not** valid, the procedure can **run forever** without ever stopping (the Herbrand universe is infinite when function symbols are present; there is no algorithm that terminates with "no" in *all* cases). There is provably **no** Turing machine that decides "valid / not valid" correctly for every FOL formula — by reduction to the halting problem.

This is not the weakness of one particular algorithm but a **fundamental limit** of all machine FOL inference. The contrast shapes the whole of symbolic AI:

| | Propositional logic | First-order logic (FOL) |
|---|---|---|
| Expressive power | objects individually | objects, relations, functions, quantifiers |
| Satisfiability (SAT/validity) | **decidable** (NP- resp. co-NP-complete) | **semi-decidable**, undecidable |
| Inference | DPLL/CDCL, resolution | resolution with unification |
| Models | finitely many ($2^n$) | in general infinitely many |

This is exactly why one often restricts things in practice: **Datalog** (deductive databases, module 32) and **description logics** (ontologies, the semantic web) are carefully tailored FOL fragments that remain *decidable* — the price of computability is reduced expressive power. This conflict between **expressive power** and **decidability/efficiency** runs through module 07 and the whole of knowledge representation.
---

## Summary / cheat sheet

**Search — formalism and criteria**

| Notion | Core |
|---|---|
| Search problem | $(S, s_0, A, \mathrm{Result}, \mathrm{Goal}, c)$; a solution = a sequence of actions $s_0 \to$ goal; optimal = minimal $\sum c$ |
| Parameters | $b$ branching, $d$ solution depth, $m$ maximum depth |
| Criteria | complete · optimal · time · space |

**Uninformed search**

| Procedure | Frontier | complete | optimal | Time | Space |
|---|---|---|---|---|---|
| BFS | FIFO | yes | yes (unit costs) | $O(b^d)$ | $O(b^d)$ |
| UCS | prio. $g$ | yes | yes | $O(b^{1+\lfloor C^\ast/\varepsilon\rfloor})$ | same |
| DFS | LIFO | no | no | $O(b^m)$ | $O(bm)$ |
| IDDFS | iterated $\ell$ | yes | yes (unit costs) | $O(b^d)$ | $O(bd)$ |

**Informed search**

| Notion | Core |
|---|---|
| Heuristic | $h(n)\ge 0$ estimates the remaining cost $h^\ast(n)$ |
| admissible | $h(n) \le h^\ast(n)$ (never overestimate) |
| consistent | $h(s) \le c(s,a,s') + h(s')$; $\Rightarrow$ admissible |
| A\* | $f(n)=g(n)+h(n)$; optimal if $h$ is admissible (tree) / consistent (graph) |
| Dominance | $h_a \ge h_b$ admissible $\Rightarrow$ A\* with $h_a$ is more efficient; the $\max$ of several heuristics is admissible |
| Origin | relaxation, pattern databases |

**Adversarial / local / CSP**

| Notion | Core |
|---|---|
| Minimax | $\max$/$\min$ backwards; optimal against an optimal opponent; $O(b^m)$ |
| $\alpha$-$\beta$ | the same result, with a good ordering $O(b^{m/2})$ |
| Local search | only one state; hill climbing, simulated annealing ($e^{\Delta E/T}$), GA |
| CSP | $(X,D,C)$; AC-3 in $O(cd^3)$; MRV/degree/LCV; a tree CSP in $O(nd^2)$ |

**Logic**

| Notion | Core |
|---|---|
| Entailment | $\mathrm{KB}\models\alpha$: $\alpha$ is true in *every* model of the KB |
| Refutation | $\mathrm{KB}\models\alpha \iff \mathrm{KB}\land\lnot\alpha$ unsatisfiable |
| sound/complete | $\vdash\Rightarrow\models$ / $\models\Rightarrow\vdash$ |
| CNF | a conjunction of clauses (disjunctions of literals) |
| Resolution | a complementary pair → the resolvent; the empty clause $\square$ = a contradiction; refutation complete |
| Horn | $\le 1$ positive literal; forward/backward chaining, HORNSAT in P |
| DPLL | backtracking + unit propagation + pure literal + early termination |
| Unification | $\mathrm{MGU}(p,q)$: $p\theta=q\theta$, most general; the occurs check |
| FOL resolution | Skolemization + unification; the Herbrand theorem as its foundation |
| Decidability | propositional logic decidable (NP); FOL only **semi**-decidable |

---

## Self-test

<details><summary><b>1. Why is the memory requirement, not the time, the main problem of breadth-first search? How does IDDFS solve it?</b></summary>

BFS has to keep *all* generated nodes in memory (frontier + explored), which is $O(b^d)$. With $b=10, d=12$ that is about $10^{12}$ nodes — terabytes, whereas the time (about $10^{12}$ operations) would still be feasible on modern machines. Memory is the hard limit. IDDFS runs as a repeated depth-limited DFS and needs only $O(bd)$ memory (just the current path), yet achieves the same $O(b^d)$ time and the same completeness/optimality as BFS. The repetition overhead is asymptotically negligible (about 11 % at $b=10$), because the bottom, largest layer is generated only once.
</details>

<details><summary><b>2. Define admissible and consistent. Prove: consistent ⇒ admissible. Does the converse hold?</b></summary>

*Admissible:* $h(n) \le h^\ast(n)$ (it never overestimates the remaining cost). *Consistent:* $h(s) \le c(s,a,s')+h(s')$ for every step (the triangle inequality). Proof that consistent ⇒ admissible, by induction over the number of steps $k$ of the optimal remaining path: $k=0$ ⇒ a goal, $h=0=h^\ast$. Step: $h(s)\le c(s,a,s')+h(s')\le c(s,a,s')+h^\ast(s')=h^\ast(s)$ (the middle inequality from the induction hypothesis, $s'$ on the optimal remaining path). The **converse does not hold**: there are admissible but inconsistent heuristics. In practice almost all natural admissible heuristics (relaxations) are also consistent.
</details>

<details><summary><b>3. Prove the optimality of A\* with an admissible heuristic (tree search).</b></summary>

Let $C^\ast$ be the optimal solution cost and $G_2$ a suboptimal goal node ($g(G_2)>C^\ast$) in the frontier. Then $f(G_2)=g(G_2)+0>C^\ast$. On the optimal path there is always a frontier node $n$; for it $f(n)=g(n)+h(n)\le g(n)+h^\ast(n)=C^\ast$ (admissibility). Hence $f(n)\le C^\ast<f(G_2)$ — A\* expands $n$ before $G_2$. So a suboptimal goal is never chosen before the optimal one. For *graph* search one needs consistency (or re-opening), so that $f$ does not fall along paths.
</details>

<details><summary><b>4. Why is a higher admissible heuristic better? How do you combine two heuristics?</b></summary>

A\* expands all nodes with $f(n)<C^\ast$, that is $g(n)+h(n)<C^\ast$, i.e. $h(n)<C^\ast-g(n)$. A *larger* (but still admissible) $h$ makes this condition true for fewer nodes → fewer expansions. Formally: if $h_a\ge h_b$ (both admissible), A\* with $h_a$ expands a subset (apart from $f=C^\ast$ ties) of the nodes of $h_b$ — $h_a$ **dominates**. Combination: $h(n)=\max\{h_a(n),h_b(n)\}$ is again admissible and at least as good as both.
</details>

<details><summary><b>5. What does AC-3 do, and what is its complexity? Why is a tree-structured CSP "easy"?</b></summary>

AC-3 establishes **arc consistency**: it deletes from every domain $D_i$ the values that have no permitted partner at a neighbouring variable $X_j$, and propagates the changes. Complexity $O(c\,d^3)$ ($c$ binary constraints, domain size $d$): every arc enters the queue at most $d$ times, and `REVISE` costs $O(d^2)$. A **tree CSP** is solvable in $O(n\,d^2)$: order topologically, make it arc-consistent from the leaves to the root (directional arc consistency), then assign greedily forwards from the root — no backtracking is needed, because every variable has only one parent. The graph structure determines the complexity.
</details>

<details><summary><b>6. Explain the bridging theorem $\mathrm{KB}\models\alpha \iff \mathrm{KB}\land\lnot\alpha$ unsatisfiable. Why is it so important for machine theorem proving?</b></summary>

$\mathrm{KB}\models\alpha$ means: in every model of the KB, $\alpha$ is true, i.e. there is *no* model with the KB true and $\alpha$ false — so $\mathrm{KB}\land\lnot\alpha$ is unsatisfiable. Its significance: instead of checking a statement about *all infinitely many* models (entailment), one looks for *one single* contradiction. That permits **proof by refutation**: negate the claim, add it to the KB, derive the empty clause with resolution. Resolution is only *refutation* complete — which is precisely why this theorem is the bridge that nevertheless turns it into a complete entailment procedure.
</details>

<details><summary><b>7. What is unit propagation in DPLL and why is it so effective?</b></summary>

If only *one* literal in a clause is still unassigned and all the others are false, that literal must be true for the clause (and hence the whole CNF) to remain satisfiable. DPLL sets it by force, without branching — and that can trigger a chain reaction of further unit clauses. The effect: unit propagation eliminates branchings without the cost of guessing; empirically SAT solvers spend most of their time here. CDCL solvers build *clause learning* and backjumping on top of it and scale to millions of variables.
</details>

<details><summary><b>8. What is an MGU and what is the occurs check for?</b></summary>

A **most general unifier** $\theta$ of two atoms $p,q$ makes them equal ($p\theta=q\theta$) while committing to as *little* as possible — every other unifier is an instance of $\theta$. It is unique up to renaming. The **occurs check** prevents binding a variable $x$ to a term that contains $x$ itself (e.g. $x$ with $f(x)$) — that would give an infinite term. Without it, unification becomes unsound; many Prolog systems omit it for performance reasons.
</details>

<details><summary><b>9. What is Skolemization for? Does it preserve logical equivalence?</b></summary>

Skolemization eliminates **existential quantifiers** when forming the CNF in FOL: an $\exists y$ with no surrounding $\forall$ → a new **Skolem constant**; an $\exists y$ in the scope of $\forall x_1\dots x_k$ → a **Skolem function** $g(x_1,\dots,x_k)$ (the "witness" depends on the outer variables). It does **not** preserve logical equivalence, but it does preserve **satisfiability** — and that is exactly enough for the refutation procedure, which only has to establish the unsatisfiability of $\mathrm{KB}\land\lnot\alpha$.
</details>

<details><summary><b>10. FOL validity is semi-decidable but undecidable. What does that mean concretely for a theorem prover?</b></summary>

*Semi-decidable:* if the formula is valid (follows from the KB), a complete procedure (resolution) finds the proof in finitely many steps and halts with "yes". *Undecidable:* if it is **not** valid, the same prover can run forever without ever outputting "no" — there is provably no algorithm that terminates for *all* FOL formulas and decides correctly (a reduction to the halting problem, Church/Turing 1936). In practice: a FOL prover can confirm validity, but it can never *guarantee* to establish invalidity. That is why one uses decidable fragments (Datalog, description logics) when termination is required.
</details>

---

## Literature and sources

**Textbooks**
- **Russell & Norvig, *Artificial Intelligence: A Modern Approach* (AIMA), 4th ed.** — *the* standard reference. For this module: ch. 3 (search), 4 (informed/local), 5 (games), 6 (CSP), 7 (propositional logic), 8–9 (FOL and inference). Beginner friendly and nevertheless complete. *The primary recommendation.*
- **Ertel, *Grundkurs Künstliche Intelligenz*, Springer Vieweg** — in German, compact, good for logic and search. *Beginner friendly.*
- **Nilsson, *Artificial Intelligence: A New Synthesis*** — classic, an elegant presentation of search and logic. *Advanced.*
- **Chang & Lee, *Symbolic Logic and Mechanical Theorem Proving*** — the reference for resolution, unification and the Herbrand theorem. *Advanced, mathematical.*

**Freely available courses and materials** (free)
- **UC Berkeley CS188 "Introduction to AI"** — videos, slides, the famous *Pac-Man projects* (search, CSP, games in Python). `inst.eecs.berkeley.edu/~cs188`. *Beginner friendly, highly practical.*
- **The AIMA companion site** with pseudocode and reference implementations (`aima-python` on GitHub). *Straight to reimplementing.*
- **Stanford CS221 "AI: Principles and Techniques"** — notes and exercises free online. *Advanced.*
- **MIT 6.034 "Artificial Intelligence"** (OpenCourseWare) — video lectures by Patrick Winston on search, constraints and logic. *Beginner friendly.*

**Interactive / visualizations** (free)
- **PathFinding.js** (`qiao.github.io/PathFinding.js/visual`) — watch A\*, Dijkstra and IDA\* live on a grid. *Very beginner friendly.*
- **Red Blob Games — "Introduction to A\*"** (`redblobgames.com/pathfinding/a-star/introduction.html`) — the best interactive explanation of A\* that exists. *Beginner friendly.*
- **Sudoku and SAT solver visualizations** and the **MiniSat** source code for a real, lean CDCL solver. *Advanced.*

**Classical papers** (free, advanced)
- Hart, Nilsson & Raphael (1968): *A Formal Basis for the Heuristic Determination of Minimum Cost Paths* — the original A\* paper.
- Robinson (1965): *A Machine-Oriented Logic Based on the Resolution Principle* — FOL resolution.
- Davis, Logemann & Loveland (1962): *A Machine Program for Theorem-Proving* — the DPLL algorithm.

---

## The three projects

This module is theory-heavy, but the core procedures only *come alive* once you see them run. That is why all three projects are **implementations** (Python), each with a theoretical part for reflection. Rising difficulty and decreasing amounts of given code:

- **01 – basic** (`projects/01-basic/`): **search algorithms on the 8-puzzle and Romania.** A guided notebook: implement BFS/UCS/IDDFS/A\*, compare the heuristics $h_1,h_2$, check admissibility/dominance empirically (by counting expanded nodes). Plenty of guidance.
- **02 – medium** (`projects/02-medium/`): **a CSP solver + a DPLL SAT solver.** A structured Python project: backtracking with MRV/AC-3 for Sudoku *and* a DPLL solver with unit propagation; one Sudoku is additionally encoded as a SAT instance and solved by both routes. Only occasional inspiration.
- **03 – final** (`projects/03-final/`): **a resolution theorem prover for propositional and first-order logic.** No given code. CNF conversion, unification (with the occurs check), resolution with refutation; applied to a realistic knowledge base scenario. Master's level: it consolidates logic, unification and the refutation principle.

Details, setup and reference solutions are in the `README.md` of each project folder.

---
---

# Modul 06 — Theorie der Künstlichen Intelligenz 1 (deutsche Fassung)

**Worum geht es?** Dieses Modul behandelt das klassische, *symbolische* Fundament der KI: Wie formuliert man ein Problem so, dass eine Maschine es durch **Suche** lösen kann, und wie repräsentiert man Wissen so formal, dass ein Rechner daraus **logisch korrekt schließen** kann. Es ist die theoretische Gegenseite zum statistischen Lernen (Module 04/05): Statt aus Daten zu lernen, konstruieren wir hier Verfahren, deren Korrektheit und Optimalität sich *beweisen* lässt.

**Hilfreiche Vorkenntnisse.** Grundlegende Diskrete Mathematik (Mengen, Relationen, Funktionen, Beweis durch Induktion), etwas Graphentheorie (Knoten, Kanten, Pfade) und Vertrautheit mit Landau-Notation ($O$, $\Theta$). Programmierkenntnisse (Python) für die Projekte. Kein Vorwissen aus den ML-Modulen nötig — dieses Modul steht eigenständig.

**Empfohlene Vormodule.** Keines zwingend. Wer Modul 01 „Introduction in AI" gemacht hat, kennt A\* und Minimax bereits informell; hier werden sie *bewiesen*.

**Folgemodul.** Modul 07 „Theorie der KI 2" baut hierauf auf: Planung, Handeln unter Unsicherheit (probabilistisches Schließen, Bayes-Netze), nichtmonotones Schließen und Beschreibungslogiken.

---

## Lernziele

Nach diesem Modul solltest du in der Lage sein,

- ein reales Problem als **Suchproblem** zu formalisieren (Zustandsraum, Aktionen, Übergangsmodell, Zielprüfung, Pfadkosten) und die Größe des Zustandsraums abzuschätzen;
- die **uninformierten Suchverfahren** (BFS, UCS/Dijkstra, DFS, IDDFS) zu implementieren und ihre Vollständigkeit, Optimalität, Zeit- und Speicherkomplexität **zu beweisen**;
- **Heuristiken** zu entwerfen, die Begriffe *zulässig* (admissible) und *konsistent* (consistent) präzise zu unterscheiden und die **Optimalität von A\*** zu beweisen;
- **lokale Suche** (Hill Climbing, Simulated Annealing, genetische Algorithmen) und **adversariale Suche** (Minimax, $\alpha$-$\beta$-Pruning) einzuordnen;
- ein **Constraint-Satisfaction-Problem (CSP)** zu modellieren und mit Backtracking, Forward Checking und Kantenkonsistenz (AC-3) plus Ordnungsheuristiken zu lösen;
- die **Aussagenlogik** vollständig zu beherrschen: Syntax, Semantik, Erfüllbarkeit, Folgerung, Normalformen, den **Resolutionskalkül** und den **DPLL-Algorithmus**, inklusive Korrektheit und Widerlegungsvollständigkeit;
- die **Prädikatenlogik erster Stufe (FOL)** zu verstehen: Syntax und Semantik (Interpretationen, Modelle), **Unifikation** (allgemeinster Unifikator), **FOL-Resolution**, das **Herbrand-Theorem** und die **(Halb-)Entscheidbarkeit** der Gültigkeit;
- zu erklären, *warum* diese Verfahren die richtige Antwort liefern — nicht nur *dass* sie es tun.

---

## Teil 1 — Grundlagen: Problemlösen als Suche

### 1.1 Der Agent und die Problemformulierung

Ein **zielbasierter Agent** überlegt vor dem Handeln: Er stellt sich eine Folge von Aktionen vor, prüft im Kopf, wohin sie führt, und wählt eine, die ins Ziel führt. Damit ein Rechner das kann, muss das Problem in fünf Komponenten zerlegt werden. Ein **Suchproblem** ist das Tupel

$$
\mathcal{P} = (S, s_0, A, \mathrm{Result}, \mathrm{Goal}, c)
$$

mit

- $S$: die Menge aller **Zustände** (der *Zustandsraum*),
- $s_0 \in S$: der **Startzustand**,
- $A$: die Menge der **Aktionen**; $\mathrm{Actions}(s) \subseteq A$ liefert die in $s$ anwendbaren Aktionen,
- $\mathrm{Result}: S \times A \to S$: das **Übergangsmodell** (deterministisch), $\mathrm{Result}(s,a) = s'$,
- $\mathrm{Goal}: S \to \{\text{wahr}, \text{falsch}\}$: die **Zielprüfung** (kann eine Menge $S_g \subseteq S$ oder eine Eigenschaft sein),
- $c(s,a,s') \ge 0$: die **Schrittkosten** einer Aktion. Wir setzen $c(s,a,s') \ge \varepsilon > 0$ für ein festes $\varepsilon$ voraus (Kosten sind nach unten weg von 0 beschränkt) — das brauchen wir später für Terminierungs- und Optimalitätsbeweise.

Eine **Lösung** ist eine Aktionsfolge $a_1, \dots, a_n$, die $s_0$ über $s_i = \mathrm{Result}(s_{i-1}, a_i)$ in einen Zielzustand $s_n$ überführt. Die **Pfadkosten** sind $g = \sum_{i=1}^n c(s_{i-1}, a_i, s_i)$. Eine **optimale Lösung** hat minimale Pfadkosten $C^\ast$.

> **Wichtige Abstraktion.** Das Suchproblem ist ein *Modell* der Realität. Die Kunst der Formalisierung liegt darin, genau so viel wegzulassen, dass das Problem lösbar bleibt, ohne die Lösung unbrauchbar zu machen. Der Zustandsraum ist ein gerichteter Graph: Knoten = Zustände, Kanten = Aktionen (gewichtet mit Schrittkosten).

**Zwei laufende Beispiele:**

*Romania* (Wegplanung): Zustände = Städte, Aktionen = „fahre nach Nachbarstadt", Kosten = Straßenlänge in km. Start Arad, Ziel Bukarest. Der Zustandsraum ist klein und explizit als Landkarte gegeben.

*8-Puzzle*: Ein $3\times 3$-Schiebepuzzle mit Kacheln 1–8 und einer Lücke. Zustand = Anordnung der Kacheln, Aktionen = Lücke nach oben/unten/links/rechts schieben, Kosten = 1 pro Zug. Der Zustandsraum hat $9!/2 = 181\,440$ erreichbare Zustände (die Hälfte aller $9!$ Permutationen ist von einer gegebenen Startstellung aus unerreichbar — die *Paritäts-Invariante*). Beim 15-Puzzle sind es bereits $16!/2 \approx 10^{13}$ — zu groß, um den Zustandsraum explizit zu speichern. Deshalb erzeugen wir Zustände **bedarfsgesteuert** über das Übergangsmodell.

### 1.2 Suchbaum, Suchgraph und der generische Algorithmus

Suche exploriert den Zustandsraum, indem sie einen **Suchbaum** aufbaut. Ein **Knoten** $n$ des Suchbaums ist eine Buchhaltungsstruktur mit:

- `n.state` — der zugehörige Zustand,
- `n.parent` — der Elternknoten (zur Pfadrekonstruktion),
- `n.action` — die Aktion, die vom Elternzustand hierher führte,
- `n.g` — die Pfadkosten von $s_0$ bis `n.state` entlang dieses Pfades.

**Achtung, feiner Unterschied:** Ein *Zustand* ist eine Konfiguration der Welt; ein *Knoten* ist ein Pfad dorthin. Zwei verschiedene Knoten können denselben Zustand tragen (über verschiedene Pfade erreicht). Das **Expandieren** eines Knotens erzeugt für jede anwendbare Aktion einen Kindknoten. Die **Frontier** (Rand, offene Liste) ist die Menge der erzeugten, aber noch nicht expandierten Knoten. Die **Explored Set** (geschlossene Liste) hält bereits expandierte Zustände fest, um Zyklen und Redundanz zu vermeiden.

```
function GRAPH-SEARCH(problem) returns Lösung oder Fehlschlag:
    frontier  ← {Knoten(s0)}          # Prioritätsstruktur je nach Strategie
    explored  ← ∅
    while frontier ≠ ∅:
        n ← REMOVE(frontier)          # welchen Knoten? -> das IST die Strategie
        if Goal(n.state): return SOLUTION(n)
        add n.state to explored
        for each a in Actions(n.state):
            s' ← Result(n.state, a)
            child ← Knoten(state=s', parent=n, action=a, g = n.g + c(...))
            if s' ∉ explored and s' nicht schon in frontier (mit ≤ Kosten):
                frontier ← INSERT(child, frontier)
    return Fehlschlag
```

Die einzige Stellschraube ist, **welchen Knoten `REMOVE` als Nächstes wählt**. Diese eine Entscheidung erzeugt alle folgenden Verfahren.

### 1.3 Bewertungskriterien

Ein Suchverfahren beurteilen wir nach vier Kriterien:

1. **Vollständigkeit (completeness):** Findet es *garantiert* eine Lösung, wenn eine existiert?
2. **Optimalität (optimality):** Findet es garantiert eine *kostenminimale* Lösung?
3. **Zeitkomplexität:** Wie viele Knoten werden erzeugt/expandiert?
4. **Speicherkomplexität:** Wie viele Knoten müssen gleichzeitig im Speicher gehalten werden?

Die Komplexität drücken wir in drei Parametern aus:

- $b$ — der **Verzweigungsfaktor** (branching factor), maximale Anzahl Nachfolger eines Knotens,
- $d$ — die **Tiefe der flachsten Lösung** (depth),
- $m$ — die **maximale Tiefe** des Zustandsraums (kann $\infty$ sein).

### 1.4 Uninformierte (blinde) Suche

„Uninformiert" heißt: Das Verfahren nutzt **keine** problemspezifische Information darüber, wie nah ein Zustand am Ziel ist. Es kennt nur die Problemdefinition.

**Breitensuche (BFS).** `frontier` ist eine FIFO-Queue → expandiere Knoten in der Reihenfolge ihrer Erzeugung, also schichtweise: erst alle Knoten der Tiefe 0, dann Tiefe 1, usw. Zieltest beim *Erzeugen* (nicht erst beim Expandieren), das spart eine Schicht.

- *Vollständig:* ja, falls $b$ endlich (die flachste Lösung in Tiefe $d$ wird nach endlich vielen Knoten gefunden).
- *Optimal:* ja, **falls alle Schrittkosten gleich sind** (dann ist die flachste Lösung auch die billigste). Bei ungleichen Kosten i.\,A. **nicht**.
- *Zeit:* $1 + b + b^2 + \dots + b^d = O(b^d)$.
- *Speicher:* $O(b^d)$ — jeder erzeugte Knoten bleibt im Speicher (Frontier + Explored). **Der Speicher ist das eigentliche Killerkriterium**: Bei $b=10$ und $d=12$ sind das $\sim 10^{12}$ Knoten.

> **Beweis der Optimalität von BFS bei Einheitskosten (Induktion über die Schicht).** BFS expandiert Knoten in nichtfallender Tiefenreihenfolge. Behauptung: Wird ein Zielknoten in Tiefe $d$ erstmals erzeugt, so gibt es keinen Zielknoten in Tiefe $< d$. Angenommen doch, es gäbe einen Zielknoten in Tiefe $d' < d$. Da BFS Schicht für Schicht vorgeht, wären alle Knoten der Tiefe $d'$ erzeugt worden, *bevor* irgendein Knoten der Tiefe $d$ erzeugt wird — der Zielknoten in $d'$ wäre also zuerst gefunden worden, Widerspruch. Da bei Einheitskosten Tiefe $\propto$ Kosten, ist die flachste Lösung optimal. $\qquad\blacksquare$

**Uniforme-Kosten-Suche (UCS) = Dijkstra für Suchprobleme.** `frontier` ist eine **Prioritätsschlange nach $g(n)$** (bisherige Pfadkosten) → expandiere immer den billigsten offenen Knoten. Zwei entscheidende Details gegenüber BFS: (a) Zieltest beim **Expandieren**, nicht beim Erzeugen (sonst nicht optimal, weil ein teurer Weg zum Ziel früher erzeugt sein könnte als ein billigerer). (b) Findet man einen billigeren Weg zu einem Frontier-Zustand, ersetzt man den teureren.

- *Vollständig:* ja, falls Schrittkosten $\ge \varepsilon > 0$ (sonst könnten unendlich viele Null-Kosten-Schritte eine unendliche Kette bilden).
- *Optimal:* ja, immer (Beweis unten).
- *Komplexität:* Sei $C^\ast$ die Kosten der optimalen Lösung. UCS expandiert alle Knoten mit Pfadkosten $< C^\ast$. Das sind bis zu $O\!\left(b^{1 + \lfloor C^\ast / \varepsilon \rfloor}\right)$ — bei Einheitskosten ($\varepsilon = 1$, $C^\ast = d$) etwas mehr als BFS, nämlich $O(b^{d+1})$, weil UCS auch noch Knoten der Zieltiefe untersucht, bevor es das Ziel als billigsten wählt.

> **Beweis der Optimalität von UCS.** Wir zeigen: Wenn UCS einen Knoten $n$ zum Expandieren wählt, ist `n.g` bereits die *optimale* Pfadkostensumme $g^\ast(n.\text{state})$ von $s_0$ zu diesem Zustand. **(a) Nichtfallende Expansionskosten:** Da immer der billigste Frontier-Knoten gewählt wird und Kinder Kosten $\ge$ ihrem Elter haben (Schrittkosten $\ge 0$), werden Knoten in nichtfallender $g$-Reihenfolge expandiert. **(b) Optimalität beim Expandieren:** Angenommen, $n$ wird mit `n.g` $> g^\ast(n.\text{state})$ expandiert. Dann existiert ein optimaler Pfad mit kleineren Kosten; auf ihm gibt es einen ersten Knoten $n'$, der noch in der Frontier liegt. Es gilt `n'.g` $\le g^\ast(n.\text{state}) <$ `n.g`, also wäre $n'$ vor $n$ gewählt worden — Widerspruch. Wird das Ziel als billigster Frontier-Knoten expandiert, ist folglich seine Pfadkostensumme optimal. $\qquad\blacksquare$

**Tiefensuche (DFS).** `frontier` ist ein LIFO-Stack → expandiere immer den zuletzt erzeugten Knoten, gehe also so tief wie möglich, bevor du zurückgehst (Backtracking).

- *Vollständig:* **nein** im Allgemeinen (kann in einem unendlichen Zweig oder Zyklus hängen bleiben); ja in endlichen Zustandsräumen mit Zyklenerkennung.
- *Optimal:* nein.
- *Zeit:* $O(b^m)$ — schlecht, wenn $m \gg d$.
- *Speicher:* $O(bm)$ — **das ist der Vorteil**: DFS muss nur den aktuellen Pfad plus die Geschwisterknoten speichern, nicht den ganzen Baum. Linear statt exponentiell.

**Tiefenbeschränkte Suche & Iterative Vertiefung (IDDFS).** Depth-Limited Search ist DFS mit hartem Tiefenlimit $\ell$ (schneidet tiefere Zweige ab). IDDFS ruft sie mit $\ell = 0, 1, 2, \dots$ auf, bis eine Lösung gefunden wird.

- Kombiniert **das Beste beider Welten**: Speicher $O(bd)$ wie DFS, Vollständigkeit und (bei Einheitskosten) Optimalität wie BFS.
- *Zeit:* $O(b^d)$. Der scheinbare Mehraufwand durch wiederholtes Durchsuchen der oberen Schichten ist asymptotisch vernachlässigbar: Die unterste Schicht ($b^d$ Knoten) wird nur einmal erzeugt, die vorletzte zweimal, …, die Wurzel $d{+}1$-mal. Summe: $\sum_{i=0}^{d}(d{+}1-i)\,b^i = O(b^d)$, dominiert vom letzten Term. Bei $b=10$ beträgt der Overhead nur ~11\,%.

> **IDDFS ist das Standard-Arbeitspferd der uninformierten Suche**, wenn der Zustandsraum groß und die Lösungstiefe unbekannt ist — genau weil es die exponentielle Speichergrenze von BFS umgeht.

**Bidirektionale Suche.** Suche gleichzeitig vorwärts von $s_0$ und rückwärts vom Ziel; brich ab, wenn sich die Fronten treffen. Zeit/Speicher $O(b^{d/2})$ — dramatisch besser, aber nur anwendbar, wenn das Übergangsmodell invertierbar ist und ein *expliziter* Zielzustand vorliegt (kein reiner Zielprädikat-Test).

**Übersicht (endlicher $b$; Einheitskosten für „optimal"):**

| Kriterium | BFS | UCS | DFS | IDDFS |
|---|---|---|---|---|
| Vollständig? | ja | ja ($c\ge\varepsilon$) | nein\* | ja |
| Optimal? | ja (Einheitsk.) | ja (immer) | nein | ja (Einheitsk.) |
| Zeit | $O(b^d)$ | $O(b^{1+\lfloor C^\ast/\varepsilon\rfloor})$ | $O(b^m)$ | $O(b^d)$ |
| Speicher | $O(b^d)$ | $O(b^{1+\lfloor C^\ast/\varepsilon\rfloor})$ | $O(bm)$ | $O(bd)$ |

\* in endlichen Räumen mit Zyklencheck vollständig.

---

## Teil 2 — Aufbau: Informierte und fortgeschrittene Suche

### 2.1 Heuristiken

Eine **Heuristik** $h(n) \ge 0$ schätzt die Kosten des *billigsten Pfades vom Zustand `n.state` zu einem Zielzustand*. Per Konvention $h(n) = 0$ für Zielzustände. Sie ist die problemspezifische „Ahnung", die uninformierte Suche fehlt. Zwei Eigenschaften sind zentral:

**Zulässigkeit (admissibility).** $h$ ist *zulässig*, wenn sie die wahren Restkosten $h^\ast(n)$ **nie überschätzt**:
$$
0 \le h(n) \le h^\ast(n) \quad \text{für alle } n.
$$
Eine zulässige Heuristik ist *optimistisch* — sie glaubt, das Ziel sei mindestens so nah, wie es wirklich ist.

**Konsistenz / Monotonie (consistency).** $h$ ist *konsistent*, wenn sie für jede Aktion $a$ von $s$ nach $s'$ die **Dreiecksungleichung** erfüllt:
$$
h(s) \le c(s, a, s') + h(s').
$$
Anschaulich: Die geschätzten Restkosten dürfen durch einen Schritt höchstens um die realen Schrittkosten sinken.

> **Satz: Konsistenz $\Rightarrow$ Zulässigkeit** (aber nicht umgekehrt). *Beweis* durch Induktion über die Anzahl $k$ der Schritte auf dem optimalen Restpfad von $s$ zum nächsten Ziel. **Basis** $k=0$: $s$ ist Zielzustand, $h(s)=0=h^\ast(s)$. **Schritt:** Sei der optimale Restpfad $s \to s' \to \dots \to \text{Ziel}$ mit $k$ Schritten, erster Schritt $a$. Nach Induktionsannahme $h(s') \le h^\ast(s')$. Konsistenz gibt $h(s) \le c(s,a,s') + h(s') \le c(s,a,s') + h^\ast(s') = h^\ast(s)$, denn $s'$ liegt auf dem optimalen Restpfad. $\qquad\blacksquare$

**Heuristiken für das 8-Puzzle** (beide zulässig):
- $h_1$ = Zahl der fehlplatzierten Kacheln (Hamming). Zulässig, weil jede fehlplatzierte Kachel mindestens einen Zug braucht.
- $h_2$ = Summe der **Manhattan-Distanzen** jeder Kachel zu ihrer Zielposition. Zulässig, weil ein Zug eine Kachel um genau 1 in Manhattan-Metrik bewegt und Kacheln sich nicht „durcheinander" bewegen. Es gilt stets $h_2(n) \ge h_1(n)$: $h_2$ **dominiert** $h_1$.

### 2.2 A\* — die informierte Suche

A\* ist UCS, aber mit einer anderen Prioritätsfunktion. Statt nur die bisherigen Kosten $g(n)$ zu betrachten, nutzt A\* die **geschätzten Gesamtkosten** eines Pfades durch $n$:
$$
f(n) = g(n) + h(n).
$$
$g(n)$ = bekannte Kosten von $s_0$ bis `n.state`, $h(n)$ = geschätzte Restkosten zum Ziel. `frontier` ist eine Prioritätsschlange nach $f$. A\* expandiert immer den Knoten mit kleinstem $f$.

**A\* verallgemeinert die anderen Verfahren:** $h \equiv 0$ ergibt UCS. „Greedy Best-First Search" ($f = h$, ignoriert $g$) ist der andere Extremfall — schnell, aber **nicht optimal und nicht vollständig** (kann in die Irre laufen).

> **Satz (Optimalität von A\*, Baumsuche).** Ist $h$ zulässig, so liefert A\* ohne Explored-Set (reine Baumsuche) eine optimale Lösung. *Beweis.* Sei $C^\ast$ die optimale Lösungskosten und $G_2$ ein *suboptimaler* Zielknoten in der Frontier mit $g(G_2) > C^\ast$. Da $h(G_2)=0$, ist $f(G_2) = g(G_2) > C^\ast$. Auf dem optimalen Pfad liegt stets ein Frontier-Knoten $n$ (der Pfad wurde bis irgendwohin expandiert). Für ihn gilt wegen Zulässigkeit $f(n) = g(n) + h(n) \le g(n) + h^\ast(n) = C^\ast$. Also $f(n) \le C^\ast < f(G_2)$: A\* wählt $n$ vor $G_2$. Ein suboptimaler Zielknoten wird nie vor Erreichen des optimalen expandiert. $\qquad\blacksquare$

> **Satz (Optimalität von A\*, Graphsuche).** Ist $h$ **konsistent**, so ist A\* mit Explored-Set optimal. Der Grund: Bei Konsistenz sind die $f$-Werte entlang jedes Pfades **nichtfallend** ($f(s') = g(s')+h(s') = g(s)+c(s,a,s')+h(s') \ge g(s)+h(s) = f(s)$). Daraus folgt, dass A\* Zustände in nichtfallender $f$-Reihenfolge expandiert und einen Zustand beim *ersten* Expandieren bereits mit optimalem $g$ erreicht — genau wie bei UCS. Bloße Zulässigkeit reicht für die Graphsuche **nicht**, es sei denn, man erlaubt das Wiedereröffnen (re-opening) bereits geschlossener Knoten bei Fund eines billigeren Pfades.

**Optimale Effizienz.** Unter allen Verfahren, die dieselbe zulässige Heuristik nutzen, expandiert A\* (bis auf Knoten mit exakt $f=C^\ast$) *keinen Knoten*, den es nicht müsste: Jeder Knoten mit $f(n) < C^\ast$ **muss** von jedem optimalen, vollständigen Verfahren expandiert werden (sonst könnte dort eine bessere Lösung versteckt sein). A\* ist in diesem Sinne **optimal effizient**.

**Dominanz.** Sind $h_a, h_b$ beide zulässig und gilt $h_a(n) \ge h_b(n)$ für alle $n$, so expandiert A\* mit $h_a$ nie mehr Knoten als mit $h_b$ (bis auf Gleichstände). **Höhere zulässige Heuristik = besser.** Deshalb ist $h_2$ (Manhattan) dem $h_1$ (Hamming) beim 8-Puzzle vorzuziehen. Aus mehreren Heuristiken kann man stets $h(n) = \max\{h_a(n), h_b(n)\}$ bilden — wieder zulässig und mindestens so gut.

**Heuristiken systematisch konstruieren.** Woher kommen gute zulässige Heuristiken?
- **Relaxation:** Man löst ein *vereinfachtes* Problem exakt. Lässt man beim 8-Puzzle die Regel „Kachel kann nur auf leeres Feld" fallen, darf jede Kachel direkt zum Ziel „springen" → Kosten = Manhattan-Distanz. Die exakten Kosten eines relaxierten Problems sind **immer** eine zulässige (und sogar konsistente) Heuristik für das Original, weil das Original nur *mehr* Einschränkungen hat, also nie billiger ist.
- **Musterdatenbanken (pattern databases):** Man löst Teilprobleme (z.\,B. nur Kacheln 1–4) vollständig vor und speichert deren exakte Kosten in einer Tabelle. Zur Laufzeit liest man die Restkosten nach. Für das 15-Puzzle senken *disjunkte* Musterdatenbanken die Suchzeit um Größenordnungen.

**Speicherproblem von A\*.** A\* hält wie UCS potenziell exponentiell viele Knoten. Abhilfen: **IDA\*** (Iterative-Deepening-A\*, DFS mit $f$-Schwelle statt Tiefenschwelle) und **SMA\*** (memory-bounded A\*) tauschen Speicher gegen Zeit.

### 2.3 Lokale Suche

Wenn nur der **Zielzustand** interessiert, nicht der Weg dorthin (z.\,B. bei $n$-Damen, Scheduling, Layout-Optimierung), ist der Pfad irrelevant. Dann arbeitet man mit **lokaler Suche**: Man hält nur *einen* aktuellen Zustand und versucht, ihn schrittweise zu verbessern. Speicher $O(1)$, riesige Zustandsräume handhabbar. Man stellt sich eine **Zielfunktion-Landschaft** vor (Höhe = Güte); gesucht wird der globale Gipfel.

**Hill Climbing (Bergsteigen).** Gehe immer zum besten Nachbarn, der besser ist als der aktuelle Zustand; stoppe, wenn kein Nachbar besser ist. Einfach und speichereffizient, aber bleibt in **lokalen Maxima**, auf **Plateaus** und an **Graten** hängen. Varianten: *stochastisches* HC (zufällige Auswahl unter den Verbesserern), *first-choice* HC, *random-restart* HC (bei Steckenbleiben von zufälligem Zustand neu starten — überraschend wirksam).

**Simulated Annealing (simuliertes Abkühlen).** Kombiniert HC mit gelegentlichen „bergab"-Schritten, um lokalen Maxima zu entkommen. Ein Verschlechterungsschritt um $\Delta E < 0$ wird mit Wahrscheinlichkeit $e^{\Delta E / T}$ akzeptiert; die „Temperatur" $T$ sinkt nach einem Abkühlplan langsam gegen 0. Anfangs (hohes $T$) fast Random Walk, am Ende (niedriges $T$) fast reines HC. **Theoretisches Resultat:** Sinkt $T$ hinreichend langsam ($T_k \ge c/\log k$), konvergiert SA mit Wahrscheinlichkeit 1 zum globalen Optimum — praktisch aber zu langsam; man nimmt schnellere, heuristische Pläne.

**Genetische Algorithmen.** Halten eine *Population* von Zuständen. Neue Zustände entstehen durch **Selektion** (fitte Individuen bevorzugt), **Crossover** (zwei Eltern kombinieren Teillösungen) und **Mutation** (zufällige kleine Änderung). Crossover ist nur dann wirksam, wenn die Kodierung so gewählt ist, dass zusammengehörige Teillösungen benachbart liegen (Schema-Theorem). GAs sind robuste, aber grobe Optimierer.

### 2.4 Adversariale Suche (Spiele)

Bei **Zwei-Personen-Nullsummenspielen mit perfekter Information** (Schach, Dame, Tic-Tac-Toe) plant ein Gegner *gegen* uns. Der Zustandsraum ist ein **Spielbaum**, in dem sich die Züge von MAX (wir, maximieren) und MIN (Gegner, minimiert) abwechseln. Eine **Nutzenfunktion** (utility) bewertet Endzustände (+1 Sieg, 0 Remis, −1 Niederlage).

**Minimax.** Der Wert eines Knotens ist:
$$
\mathrm{Minimax}(s) =
\begin{cases}
\mathrm{Utility}(s) & \text{wenn } s \text{ Endzustand}\\
\max_{a}\mathrm{Minimax}(\mathrm{Result}(s,a)) & \text{wenn MAX am Zug}\\
\min_{a}\mathrm{Minimax}(\mathrm{Result}(s,a)) & \text{wenn MIN am Zug}
\end{cases}
$$
Man rechnet per Tiefensuche vom Blatt zurück nach oben. MAX wählt an der Wurzel den Zug zum Kind mit maximalem Minimax-Wert. Das Ergebnis ist der **optimale Zug unter der Annahme, dass der Gegner ebenfalls optimal spielt**. Zeit $O(b^m)$, Speicher $O(bm)$ — für reale Spiele bis zum Blatt unbezahlbar.

**$\alpha$-$\beta$-Pruning.** Beschneidet Teilbäume, die das Ergebnis nicht mehr ändern können, ohne den Minimax-Wert zu verfälschen. Man führt zwei Schranken mit: $\alpha$ = bester (höchster) Wert, den MAX bisher garantieren kann; $\beta$ = bester (niedrigster) Wert, den MIN garantieren kann. Sobald an einem Knoten $\alpha \ge \beta$, kann der restliche Teilbaum abgeschnitten werden (der Gegner würde diesen Zweig nie zulassen).

- Liefert **exakt denselben** Minimax-Wert wie unbeschnittenes Minimax — es ist keine Approximation.
- Bei **optimaler Zugreihenfolge** sinkt die Zeit von $O(b^m)$ auf $O(b^{m/2})$ — das *verdoppelt* die durchsuchbare Tiefe bei gleichem Aufwand. Deshalb sind gute **Zug-Ordnungsheuristiken** entscheidend.

**Praxis:** Da man selten bis zum Blatt rechnen kann, bricht man bei einer Tiefengrenze ab und ersetzt `Utility` durch eine heuristische **Bewertungsfunktion** (evaluation function), die die Gewinnchance einer Nicht-Endstellung schätzt (z.\,B. Materialwert im Schach). Moderne Engines (AlphaZero) ersetzen diese durch neuronale Netze plus Monte-Carlo Tree Search — der Übergang zur statistischen KI.

### 2.5 Constraint-Satisfaction-Probleme (CSP)

Ein **CSP** ist ein Suchproblem mit *faktorisierter* Zustandsdarstellung: Ein Zustand ist eine Belegung von **Variablen**, und ein Ziel ist jede Belegung, die alle **Constraints** erfüllt. Formal $(X, D, C)$:

- $X = \{X_1, \dots, X_n\}$ — Variablen,
- $D = \{D_1, \dots, D_n\}$ — Wertebereiche (Domänen), $X_i$ nimmt Werte aus $D_i$,
- $C$ — Constraints; ein Constraint $\langle \mathrm{scope}, \mathrm{rel}\rangle$ benennt beteiligte Variablen und die erlaubten Wertekombinationen.

Eine **konsistente** Belegung verletzt kein Constraint; eine **vollständige** belegt alle Variablen; eine **Lösung** ist beides. Klassiker: **Landkartenfärbung** (benachbarte Regionen verschieden färben), **Sudoku**, **$n$-Damen**, **Stundenpläne**.

**Warum eine eigene Theorie?** Weil die faktorisierte Struktur mächtige, *generische* Techniken erlaubt, die die naive Suche ($d^n$ Blätter) drastisch beschleunigen — ohne problemspezifische Heuristik.

**Backtracking-Suche.** Tiefensuche, die Variablen der Reihe nach belegt und bei Verletzung eines Constraints zurücksetzt. Der Kern der Effizienz liegt in drei Ideen:

**(1) Constraint-Propagation & Konsistenz.** Man schließt Werte *vor* dem Verzweigen aus. Eine Kante $X_i \to X_j$ heißt **kantenkonsistent (arc-consistent)**, wenn es zu *jedem* Wert in $D_i$ mindestens einen zulässigen Partnerwert in $D_j$ gibt. Der **AC-3-Algorithmus** stellt Kantenkonsistenz im gesamten Netz her:

```
function AC-3(csp) returns false (wenn Inkonsistenz erkannt) sonst true:
    queue ← alle gerichteten Kanten (Xi, Xj)
    while queue ≠ ∅:
        (Xi, Xj) ← REMOVE(queue)
        if REVISE(csp, Xi, Xj):                 # entfernt aus Di Werte ohne Partner in Dj
            if Di = ∅: return false              # Domäne leer -> keine Lösung auf diesem Zweig
            for each Xk in Nachbarn(Xi) \ {Xj}:  # Änderung an Di kann Nachbarn betreffen
                add (Xk, Xi) to queue
    return true
```
`REVISE` streicht aus $D_i$ jeden Wert, der in $D_j$ keinen Partner findet. **Komplexität:** Ein CSP mit $c$ binären Constraints und Domänengröße $\le d$ läuft in $O(c\,d^3)$: Jede der $c$ Kanten kommt höchstens $d$-mal in die Queue (einmal pro entferntem Wert eines Nachbarn), und `REVISE` kostet $O(d^2)$.

**(2) Backtracking + Inferenz kombiniert.** *Forward Checking* stellt nach jeder Belegung Kantenkonsistenz nur für die Nachbarn der eben belegten Variable her; *MAC* (Maintaining Arc Consistency) ruft nach jeder Belegung vollständig AC-3 auf den betroffenen Kanten auf.

**(3) Ordnungsheuristiken.**
- **MRV (Minimum Remaining Values / „most constrained variable"):** Belege als Nächstes die Variable mit den *wenigsten* verbleibenden legalen Werten — sie führt am schnellsten zum Fehlschlag (fail-fast) und beschneidet den Baum.
- **Degree-Heuristik:** Bei MRV-Gleichstand die Variable mit den meisten Constraints zu noch unbelegten Variablen.
- **LCV (Least Constraining Value):** Probiere den Wert zuerst, der den Nachbarn die *meisten* Optionen lässt.

**Struktur nutzen.** Zerfällt der Constraint-Graph in unabhängige Komponenten, löst man sie getrennt (multiplikativer statt exponentieller Gewinn). Ist der Constraint-Graph ein **Baum**, ist das CSP in **$O(n\,d^2)$** — also *polynomiell* — lösbar (topologisch ordnen, rückwärts kantenkonsistent machen, dann vorwärts greedy belegen). **Cutset Conditioning** und **Tree Decomposition** übertragen diesen Vorteil näherungsweise auf fast-baumartige Graphen. Dies ist die tiefe Einsicht der CSP-Theorie: **Die Graphstruktur bestimmt die Komplexität.**

---

## Teil 3 — Aufbau: Aussagenlogik (Propositional Logic)

Suche findet *Lösungen*; Logik erlaubt es, **Wissen zu repräsentieren** und daraus **neue Fakten korrekt abzuleiten**. Ein **wissensbasierter Agent** hält eine *Wissensbasis* (KB) aus Sätzen und beantwortet Fragen, indem er prüft, was aus der KB *folgt*.

### 3.1 Syntax und Semantik

**Syntax.** Formeln der Aussagenlogik werden aus **atomaren Aussagen** (Aussagensymbolen $P, Q, R, \dots$, jede wahr oder falsch) und den **Junktoren** $\lnot$ (nicht), $\land$ (und), $\lor$ (oder), $\Rightarrow$ (impliziert), $\Leftrightarrow$ (genau dann) aufgebaut. Formal (BNF):
$$
\varphi ::= \top \mid \bot \mid P \mid \lnot\varphi \mid (\varphi \land \varphi) \mid (\varphi \lor \varphi) \mid (\varphi \Rightarrow \varphi) \mid (\varphi \Leftrightarrow \varphi).
$$

**Semantik.** Ein **Modell** (bzw. eine *Interpretation*) $m$ ist eine Belegung jedes Aussagensymbols mit wahr/falsch. Die Wahrheit einer Formel unter $m$ ergibt sich rekursiv aus den **Wahrheitstafeln** der Junktoren (insbesondere: $A \Rightarrow B$ ist nur falsch, wenn $A$ wahr und $B$ falsch — „aus Falschem folgt Beliebiges"). Bei $n$ Symbolen gibt es $2^n$ Modelle.

**Zentrale semantische Begriffe:**
- $\varphi$ ist **erfüllbar (satisfiable)**, wenn *mindestens ein* Modell $\varphi$ wahr macht.
- $\varphi$ ist **gültig / eine Tautologie**, wenn *jedes* Modell $\varphi$ wahr macht (z.\,B. $P \lor \lnot P$).
- $\varphi$ ist **unerfüllbar (unsatisfiable)**, wenn *kein* Modell $\varphi$ wahr macht.
- **Dualität:** $\varphi$ ist gültig $\iff$ $\lnot\varphi$ ist unerfüllbar.

### 3.2 Logische Folgerung (Entailment)

Der zentrale Begriff. Eine KB **folgert** einen Satz $\alpha$, geschrieben
$$
\mathrm{KB} \models \alpha,
$$
genau dann, wenn $\alpha$ in **jedem** Modell wahr ist, in dem die KB wahr ist. Also: „$\alpha$ ist eine unausweichliche Konsequenz der KB." Formal $M(\mathrm{KB}) \subseteq M(\alpha)$, wobei $M(\cdot)$ die Modellmenge bezeichnet.

**Der Brückensatz (Deduktionstheorem / Refutation):**
$$
\mathrm{KB} \models \alpha \quad\iff\quad (\mathrm{KB} \land \lnot\alpha) \text{ ist unerfüllbar.}
$$
Das ist das Arbeitspferd des maschinellen Beweisens: Statt „folgt $\alpha$?" (Aussage über *alle* Modelle) prüft man „ist $\mathrm{KB} \land \lnot\alpha$ unerfüllbar?" — man nimmt das *Gegenteil* der Behauptung an und leitet einen Widerspruch her (**Beweis durch Widerlegung / proof by refutation**).

**Model Checking** löst Folgerung durch Aufzählen aller $2^n$ Modelle (Wahrheitstafelmethode) — korrekt und vollständig, aber exponentiell. Aussagenlogische Erfüllbarkeit (**SAT**) ist das kanonische **NP-vollständige** Problem (Satz von Cook–Levin); Folgerung ist entsprechend **co-NP-vollständig**. Wir suchen also Verfahren, die *im Mittel* schneller sind, ohne alle Modelle aufzuzählen.

### 3.3 Inferenzregeln, Korrektheit und Vollständigkeit

Ein **Inferenzverfahren** $i$ leitet aus einer KB syntaktisch Sätze ab: $\mathrm{KB} \vdash_i \alpha$ heißt „$i$ leitet $\alpha$ aus KB ab". Zwei Gütekriterien verbinden diese *syntaktische* Ableitbarkeit mit der *semantischen* Folgerung:

- **Korrektheit / soundness:** $\mathrm{KB} \vdash_i \alpha \implies \mathrm{KB} \models \alpha$. (Leitet nur Wahres ab — keine falschen Schlüsse.)
- **Vollständigkeit / completeness:** $\mathrm{KB} \models \alpha \implies \mathrm{KB} \vdash_i \alpha$. (Leitet alles Folgende ab — verpasst nichts.)

Ein korrektes *und* vollständiges Verfahren bildet die semantische Folgerung exakt syntaktisch nach. Bekannte korrekte Regeln: **Modus Ponens** ($\alpha \Rightarrow \beta,\ \alpha \ \vdash\ \beta$), **Und-Elimination** ($\alpha \land \beta \vdash \alpha$).

### 3.4 Normalformen und der Resolutionskalkül

**Konjunktive Normalform (KNF/CNF).** Eine Formel ist in KNF, wenn sie eine **Konjunktion von Klauseln** ist, wobei eine **Klausel** eine **Disjunktion von Literalen** ist (ein *Literal* ist ein Atom oder dessen Negation). **Jede** aussagenlogische Formel lässt sich in eine äquivalente KNF überführen:

1. $\Leftrightarrow$ eliminieren: $A \Leftrightarrow B$ wird zu $(A \Rightarrow B) \land (B \Rightarrow A)$.
2. $\Rightarrow$ eliminieren: $A \Rightarrow B$ wird zu $\lnot A \lor B$.
3. Negationen nach innen ziehen (De Morgan): $\lnot(A\land B)\equiv \lnot A\lor\lnot B$, $\lnot(A\lor B)\equiv \lnot A\land\lnot B$, $\lnot\lnot A\equiv A$.
4. $\lor$ über $\land$ distribuieren: $A\lor(B\land C)\equiv (A\lor B)\land(A\lor C)$.

**Die Resolutionsregel.** Aus zwei Klauseln, die ein komplementäres Literalpaar $\ell$ / $\lnot\ell$ enthalten, leitet man ihre **Resolvente** ab — die Vereinigung der übrigen Literale:
$$
\frac{(\ell \lor a_1 \lor \dots \lor a_k)\qquad(\lnot\ell \lor b_1 \lor \dots \lor b_m)}{(a_1 \lor \dots \lor a_k \lor b_1 \lor \dots \lor b_m)}.
$$
Enthält das Ergebnis ein Literal *und* seine Negation, ist es eine Tautologie und wird verworfen. Resolvieren sich zwei komplementäre Ein-Literal-Klauseln ($\ell$ und $\lnot\ell$), entsteht die **leere Klausel** $\square$ — sie ist per Definition **unerfüllbar** und signalisiert den gesuchten Widerspruch. Die Regel ist **korrekt**: Jedes Modell, das beide Prämissen erfüllt, erfüllt auch die Resolvente (Fallunterscheidung nach dem Wahrheitswert von $\ell$).

**Resolutionsalgorithmus (Refutationsverfahren).** Um $\mathrm{KB} \models \alpha$ zu zeigen:
1. Bilde $\mathrm{KB} \land \lnot\alpha$ und wandle in KNF um → eine Klauselmenge.
2. Wende Resolution wiederholt auf alle Klauselpaare an, füge neue Resolventen hinzu.
3. Entsteht die **leere Klausel** $\square$ → $\mathrm{KB}\land\lnot\alpha$ ist unerfüllbar → $\mathrm{KB}\models\alpha$. **bewiesen.**
4. Können keine neuen Klauseln mehr erzeugt werden und $\square$ ist nicht dabei → $\alpha$ folgt **nicht**.

> **Satz (Widerlegungsvollständigkeit der Resolution).** Ist eine aussagenlogische Klauselmenge unerfüllbar, so leitet die Resolution die leere Klausel $\square$ in endlich vielen Schritten ab. *Beweisidee:* Über den **Ground Resolution Theorem** — man zeigt per Induktion über die Zahl der Symbole, dass sich aus einer unerfüllbaren Klauselmenge stets $\square$ ableiten lässt (semantische Bäume / Konstruktion eines Modells, falls $\square$ nicht ableitbar ist, was der Unerfüllbarkeit widerspricht). Da nur endlich viele Klauseln über $n$ Symbolen existieren, terminiert das Verfahren. Wichtig: Resolution ist *widerlegungs*vollständig — sie beweist Unerfüllbarkeit, nicht direkt jede Folgerung; das genügt aber wegen des Brückensatzes. $\qquad\blacksquare$

### 3.5 Horn-Klauseln und effiziente Inferenz

Der allgemeine SAT-Fall ist NP-hart. Für eine praktisch wichtige **Teilklasse** geht es *linear*. Eine **Horn-Klausel** ist eine Klausel mit **höchstens einem positiven Literal**. Als Implikation geschrieben: $(\lnot P_1 \lor \dots \lor \lnot P_k \lor Q) \equiv (P_1 \land \dots \land P_k \Rightarrow Q)$ — eine *definite* Klausel (genau ein positives Literal). Fakten sind definite Klauseln ohne Prämisse.

Für Horn-KBs sind **Vorwärtsverkettung (forward chaining)** und **Rückwärtsverkettung (backward chaining)** korrekt und vollständig und laufen in **Linearzeit** in der Größe der KB. Vorwärtsverkettung wendet Modus Ponens datengetrieben an, bis nichts Neues mehr folgt (das ist die theoretische Grundlage von Modul 32/33 — deduktive Datenbanken und Logikprogrammierung/Prolog). Rückwärtsverkettung startet zielgetrieben von der Anfrage. Die Erfüllbarkeit von Horn-Formeln (**HORNSAT**) ist damit in P.

### 3.6 DPLL — der Motor moderner SAT-Solver

Statt Modelle vollständig aufzuzählen (Wahrheitstafel), durchsucht **DPLL** (Davis–Putnam–Logemann–Loveland) den Belegungsbaum mit Backtracking und drei Beschleunigern. DPLL entscheidet **Erfüllbarkeit** einer KNF-Klauselmenge:

1. **Early Termination.** Ist jede Klausel schon durch ein wahres Literal erfüllt → *erfüllbar* (Rest egal). Ist eine Klausel komplett falsch → dieser Zweig *unerfüllbar*, backtracke.
2. **Unit Propagation (Ein-Literal-Regel).** Enthält eine Klausel nur noch *ein* unbelegtes Literal (alle anderen falsch), muss dieses **wahr** sein. Setze es und propagiere — das kann eine Kaskade weiterer Unit-Klauseln auslösen. (Dies ist der weitaus wirkungsvollste Schritt.)
3. **Pure-Literal-Regel.** Kommt ein Symbol in allen verbleibenden Klauseln nur mit *einer* Polarität vor (rein positiv oder rein negativ), belege es passend — das erfüllt Klauseln, ohne je zu schaden.

Erst wenn keine dieser Regeln greift, **verzweigt** DPLL über ein noch freies Symbol (wahr/falsch) und rekursiert. Moderne **CDCL-Solver** (Conflict-Driven Clause Learning) erweitern DPLL um *Klausellernen* aus Konflikten, nicht-chronologisches Backtracking (Backjumping) und Aktivitäts-Heuristiken (VSIDS). Sie lösen industrielle SAT-Instanzen mit *Millionen* Variablen — das Rückgrat von Verifikation, Planung und Konfiguration.

---

## Teil 4 — Advanced: Prädikatenlogik erster Stufe (First-Order Logic)

Die Aussagenlogik ist ausdrucksschwach: „Alle Menschen sind sterblich" lässt sich nur durch *eine Aussage pro Mensch* darstellen. Die **Prädikatenlogik erster Stufe (FOL)** führt **Objekte**, **Relationen/Prädikate**, **Funktionen** und **Quantoren** ein und kann so über *ganze Klassen* von Objekten sprechen.

### 4.1 Syntax

Bausteine:
- **Terme** bezeichnen Objekte: **Konstanten** ($a, b, \mathrm{Sokrates}$), **Variablen** ($x, y$), **Funktionsanwendungen** ($f(x)$, $\mathrm{Vater}(\mathrm{Sokrates})$).
- **Atomare Formeln** sind **Prädikate** über Termen: $\mathrm{Mensch}(\mathrm{Sokrates})$, $\mathrm{Größer}(x, y)$, sowie Gleichheit $t_1 = t_2$.
- **Junktoren** wie in der Aussagenlogik plus **Quantoren**: der **Allquantor** $\forall x\,\varphi$ („für alle $x$") und der **Existenzquantor** $\exists x\,\varphi$ („es gibt ein $x$").

Beispiele:
$$
\forall x\,\big(\mathrm{Mensch}(x) \Rightarrow \mathrm{Sterblich}(x)\big), \qquad
\exists x\,\big(\mathrm{Katze}(x) \land \mathrm{Schwarz}(x)\big).
$$
**Faustregel:** $\forall$ steht meist mit $\Rightarrow$ (nicht $\land$!), $\exists$ meist mit $\land$ (nicht $\Rightarrow$!). $\forall x\,(\mathrm{Mensch}(x)\land\mathrm{Sterblich}(x))$ hieße „alles ist ein sterblicher Mensch" — zu stark.

**Dualität der Quantoren:** $\lnot\forall x\,\varphi \equiv \exists x\,\lnot\varphi$ und $\lnot\exists x\,\varphi \equiv \forall x\,\lnot\varphi$.

### 4.2 Semantik

Eine **Interpretation** (Struktur) $\mathcal{I} = (\mathcal{D}, \cdot^\mathcal{I})$ besteht aus einer nichtleeren **Domäne** $\mathcal{D}$ (dem *Diskursuniversum* — die existierenden Objekte) und einer **Deutung**, die
- jeder Konstanten ein Objekt in $\mathcal{D}$,
- jedem $k$-stelligen Funktionssymbol eine Funktion $\mathcal{D}^k \to \mathcal{D}$,
- jedem $k$-stelligen Prädikatssymbol eine Relation $\subseteq \mathcal{D}^k$

zuordnet. Zusammen mit einer Variablenbelegung legt das den Wahrheitswert jeder Formel fest: $\forall x\,\varphi$ ist wahr, wenn $\varphi$ für *jede* Belegung von $x$ mit einem Objekt aus $\mathcal{D}$ wahr ist; $\exists x\,\varphi$, wenn für *mindestens eine*. Ein $\mathcal{I}$, das $\varphi$ wahr macht, ist ein **Modell** von $\varphi$. Die Begriffe erfüllbar / gültig / Folgerung ($\mathrm{KB}\models\alpha$) sind wie in Teil 3 definiert — nur über der jetzt i.\,A. **unendlichen** Menge aller Interpretationen.

### 4.3 Unifikation

Um Inferenzregeln auf FOL zu heben, braucht man einen Mechanismus, der Variablen so belegt (**substituiert**), dass zwei Terme/Atome *syntaktisch gleich* werden. Eine **Substitution** $\theta$ ist eine endliche Abbildung Variablen → Terme, geschrieben $\{x/t, \dots\}$; $\varphi\theta$ wendet sie an. Zwei Atome $p, q$ sind **unifizierbar**, wenn ein $\theta$ mit $p\theta = q\theta$ existiert; ein solches $\theta$ heißt **Unifikator**.

Beispiel: $\mathrm{Kennt}(\mathrm{Hans}, x)$ und $\mathrm{Kennt}(y, \mathrm{Maria})$ unifizieren mit $\theta = \{y/\mathrm{Hans},\ x/\mathrm{Maria}\}$ zu $\mathrm{Kennt}(\mathrm{Hans}, \mathrm{Maria})$.

Der **allgemeinste Unifikator (most general unifier, MGU)** ist der Unifikator, der *am wenigsten festlegt* — jeder andere Unifikator ist eine Spezialisierung von ihm. Der MGU ist bis auf Variablenumbenennung **eindeutig**. Der Unifikationsalgorithmus läuft rekursiv über die Termstruktur:

```
function UNIFY(x, y, θ) returns Substitution oder Fehlschlag:
    if θ = Fehlschlag: return Fehlschlag
    if x = y: return θ
    if VARIABLE?(x): return UNIFY-VAR(x, y, θ)
    if VARIABLE?(y): return UNIFY-VAR(y, x, θ)
    if COMPOUND?(x) and COMPOUND?(y):        # gleiche Funktion/Prädikat, Argumente paarweise
        return UNIFY(ARGS(x), ARGS(y), UNIFY(OP(x), OP(y), θ))
    if LIST?(x) and LIST?(y):
        return UNIFY(REST(x), REST(y), UNIFY(FIRST(x), FIRST(y), θ))
    return Fehlschlag
```

**Der Occurs-Check.** `UNIFY-VAR` darf $x/t$ nur binden, wenn $x$ **nicht in $t$ vorkommt** — sonst würde man $x$ mit $f(x)$ unifizieren wollen, was auf einen unendlichen Term führt. Der naive Occurs-Check kostet Zeit; viele Prolog-Systeme lassen ihn aus Effizienzgründen weg (unsound, aber praktisch selten problematisch). Mit geeigneten Datenstrukturen ist Unifikation in **linearer Zeit** möglich.

### 4.4 FOL-Resolution und das Herbrand-Theorem

Der Resolutionskalkül überträgt sich auf FOL — mit zwei Erweiterungen: **Skolemisierung** (um Existenzquantoren loszuwerden) und **Unifikation** (statt exakt komplementärer Literale genügen *unifizierbare*).

**Konjunktive Normalform in FOL.** Zusätzlich zu den Schritten aus 3.4:
- **Variablen standardisieren:** Jede quantifizierte Variable eindeutig umbenennen (keine Namenskollisionen).
- **Skolemisierung:** Existenzquantoren eliminieren. Ein $\exists y$, das im Bereich keines $\forall$ steht, wird durch eine neue **Skolem-Konstante** ersetzt; steht $\exists y$ im Bereich von $\forall x_1\dots\forall x_k$, ersetzt man $y$ durch eine neue **Skolem-Funktion** $g(x_1,\dots,x_k)$ (der Zeuge hängt von den äußeren Variablen ab). Skolemisierung erhält die *Erfüllbarkeit* (nicht die logische Äquivalenz) — genau das, was das Refutationsverfahren braucht.
- **All-Quantoren weglassen** (implizit angenommen), Klauseln extrahieren.

**Verallgemeinerte Resolution.** Zwei Klauseln mit Literalen $\ell_i$ und $\lnot m_j$, sodass $\mathrm{UNIFY}(\ell_i, m_j) = \theta$ existiert, resolvieren zur Resolvente, auf die $\theta$ angewandt wird:
$$
\frac{(\ell \lor \mathbf{a}) \qquad (\lnot m \lor \mathbf{b})}{(\mathbf{a} \lor \mathbf{b})\theta}, \qquad \theta = \mathrm{MGU}(\ell, m).
$$
Der Rest läuft wie in der Aussagenlogik: KB $\land\ \lnot$Anfrage in KNF, resolvieren bis $\square$ entsteht.

**Das Herbrand-Theorem — warum das funktioniert.** Der Schlüssel, der FOL-Inferenz auf den aussagenlogischen Fall zurückführt. Das **Herbrand-Universum** einer Klauselmenge ist die Menge *aller* grundtermigen (variablenfreien) Terme, die man aus ihren Konstanten und Funktionssymbolen bilden kann (bei Funktionssymbolen *unendlich*). Eine **Herbrand-Interpretation** belegt nur diese Grundterme.

> **Satz von Herbrand (1930).** Eine FOL-Klauselmenge ist genau dann unerfüllbar, wenn eine **endliche** Menge von **Grundinstanzen** (durch Terme des Herbrand-Universums substituierte Klauseln) *aussagenlogisch* unerfüllbar ist.

Das reduziert FOL-Unerfüllbarkeit prinzipiell auf endlich viele aussagenlogische Unerfüllbarkeitsprüfungen — man müsste nur die richtige endliche Grundinstanzmenge finden. Das **Lifting-Lemma** zeigt: Statt blind Grundinstanzen zu erzeugen, kann man direkt auf der First-Order-Ebene mit Unifikation resolvieren und erhält dieselbe Kraft — das ist die Grundlage der **Widerlegungsvollständigkeit der FOL-Resolution** (Robinson, 1965): *Ist eine FOL-Klauselmenge unerfüllbar, so leitet die Resolution mit Unifikation die leere Klausel ab.*

### 4.5 Entscheidbarkeit: die prinzipielle Grenze

Ein fundamentaler Unterschied zur Aussagenlogik:

> **FOL-Gültigkeit ist *unentscheidbar* (Church & Turing, 1936), aber *semi-entscheidbar*.**

- **Semi-entscheidbar (rekursiv aufzählbar):** Ist $\alpha$ *gültig* (bzw. folgt aus KB), so **findet** die Resolution das nach endlich vielen Schritten (Vollständigkeit) — der Beweis terminiert mit „ja".
- **Unentscheidbar:** Ist $\alpha$ **nicht** gültig, kann das Verfahren **ewig weiterlaufen**, ohne je zu stoppen (das Herbrand-Universum ist bei Funktionssymbolen unendlich; es gibt keinen Algorithmus, der in *allen* Fällen mit „nein" terminiert). Es gibt beweisbar **keine** Turingmaschine, die für jede FOL-Formel korrekt „gültig / nicht gültig" entscheidet — Reduktion auf das Halteproblem.

Das ist keine Schwäche eines konkreten Algorithmus, sondern eine **prinzipielle Grenze** jeder maschinellen FOL-Inferenz. Der Kontrast prägt die ganze symbolische KI:

| | Aussagenlogik | Prädikatenlogik (FOL) |
|---|---|---|
| Ausdrucksstärke | Objekte einzeln | Objekte, Relationen, Funktionen, Quantoren |
| Erfüllbarkeit (SAT/Gültigkeit) | **entscheidbar** (NP- bzw. co-NP-vollständig) | **semi-entscheidbar**, unentscheidbar |
| Inferenz | DPLL/CDCL, Resolution | Resolution mit Unifikation |
| Modelle | endlich ($2^n$) | i.\,A. unendlich viele |

Genau deshalb schränkt man in der Praxis oft ein: **Datalog** (deduktive Datenbanken, Modul 32) und **Description Logics** (Ontologien, Semantic Web) sind sorgfältig zugeschnittene FOL-Fragmente, die *entscheidbar* bleiben — der Preis für Berechenbarkeit ist reduzierte Ausdrucksstärke. Dieser Zielkonflikt zwischen **Ausdrucksstärke** und **Entscheidbarkeit/Effizienz** zieht sich durch Modul 07 und die gesamte Wissensrepräsentation.

---

## Zusammenfassung / Cheat-Sheet

**Suche — Formalismus & Kriterien**

| Begriff | Kern |
|---|---|
| Suchproblem | $(S, s_0, A, \mathrm{Result}, \mathrm{Goal}, c)$; Lösung = Aktionsfolge $s_0 \to$ Ziel; optimal = min. $\sum c$ |
| Parameter | $b$ Verzweigung, $d$ Lösungstiefe, $m$ max. Tiefe |
| Kriterien | vollständig · optimal · Zeit · Speicher |

**Uninformierte Suche**

| Verfahren | Frontier | vollst. | optimal | Zeit | Speicher |
|---|---|---|---|---|---|
| BFS | FIFO | ja | ja (Einheitsk.) | $O(b^d)$ | $O(b^d)$ |
| UCS | Prio $g$ | ja | ja | $O(b^{1+\lfloor C^\ast/\varepsilon\rfloor})$ | dito |
| DFS | LIFO | nein | nein | $O(b^m)$ | $O(bm)$ |
| IDDFS | iterat. $\ell$ | ja | ja (Einheitsk.) | $O(b^d)$ | $O(bd)$ |

**Informierte Suche**

| Begriff | Kern |
|---|---|
| Heuristik | $h(n)\ge 0$ schätzt Restkosten $h^\ast(n)$ |
| zulässig | $h(n) \le h^\ast(n)$ (nie überschätzen) |
| konsistent | $h(s) \le c(s,a,s') + h(s')$; $\Rightarrow$ zulässig |
| A\* | $f(n)=g(n)+h(n)$; optimal, wenn $h$ zulässig (Baum) / konsistent (Graph) |
| Dominanz | $h_a \ge h_b$ zulässig $\Rightarrow$ A\* mit $h_a$ effizienter; $\max$ mehrerer Heur. ist zulässig |
| Herkunft | Relaxation, Musterdatenbanken |

**Adversarial / Lokal / CSP**

| Begriff | Kern |
|---|---|
| Minimax | $\max$/$\min$ rückwärts; optimal ggü. optimalem Gegner; $O(b^m)$ |
| $\alpha$-$\beta$ | gleiches Ergebnis, bei guter Ordnung $O(b^{m/2})$ |
| lokale Suche | nur ein Zustand; Hill Climbing, Simulated Annealing ($e^{\Delta E/T}$), GA |
| CSP | $(X,D,C)$; AC-3 in $O(cd^3)$; MRV/Degree/LCV; Baum-CSP in $O(nd^2)$ |

**Logik**

| Begriff | Kern |
|---|---|
| Folgerung | $\mathrm{KB}\models\alpha$: $\alpha$ wahr in *jedem* Modell von KB |
| Refutation | $\mathrm{KB}\models\alpha \iff \mathrm{KB}\land\lnot\alpha$ unerfüllbar |
| korrekt/vollst. | $\vdash\Rightarrow\models$ / $\models\Rightarrow\vdash$ |
| KNF | Konjunktion von Klauseln (Disjunktionen von Literalen) |
| Resolution | komplementäres Paar → Resolvente; leere Klausel $\square$ = Widerspruch; widerlegungsvollständig |
| Horn | $\le 1$ positives Literal; forward/backward chaining, HORNSAT in P |
| DPLL | Backtracking + Unit Propagation + Pure Literal + Early Termination |
| Unifikation | $\mathrm{MGU}(p,q)$: $p\theta=q\theta$, allgemeinst; Occurs-Check |
| FOL-Resolution | Skolemisierung + Unifikation; Herbrand-Theorem als Fundament |
| Entscheidbarkeit | Aussagenlogik entscheidbar (NP); FOL nur **semi**-entscheidbar |

---

## Selbsttest

<details><summary><b>1. Warum ist der Speicherbedarf, nicht die Zeit, das Hauptproblem der Breitensuche? Wie löst IDDFS das?</b></summary>

BFS muss *alle* erzeugten Knoten im Speicher halten (Frontier + Explored), das sind $O(b^d)$. Bei $b=10, d=12$ sind das $\sim 10^{12}$ Knoten — Terabytes, während die Zeit ($\sim 10^{12}$ Operationen) bei modernen Rechnern noch machbar wäre. Der Speicher ist die harte Grenze. IDDFS läuft als wiederholte tiefenbeschränkte DFS und braucht nur $O(bd)$ Speicher (nur der aktuelle Pfad), erreicht aber dieselbe $O(b^d)$-Zeit und dieselbe Vollständigkeit/Optimalität wie BFS. Der Wiederholungs-Overhead ist asymptotisch vernachlässigbar (~11 % bei $b=10$), weil die unterste, größte Schicht nur einmal erzeugt wird.
</details>

<details><summary><b>2. Definiere zulässig und konsistent. Beweise: konsistent ⇒ zulässig. Gilt die Umkehrung?</b></summary>

*Zulässig:* $h(n) \le h^\ast(n)$ (überschätzt Restkosten nie). *Konsistent:* $h(s) \le c(s,a,s')+h(s')$ für jeden Schritt (Dreiecksungleichung). Beweis konsistent ⇒ zulässig per Induktion über die Schrittzahl $k$ des optimalen Restpfads: $k=0$ ⇒ Ziel, $h=0=h^\ast$. Schritt: $h(s)\le c(s,a,s')+h(s')\le c(s,a,s')+h^\ast(s')=h^\ast(s)$ (mittlere Ungl. aus IA, $s'$ auf opt. Restpfad). Die **Umkehrung gilt nicht**: Es gibt zulässige, aber inkonsistente Heuristiken. In der Praxis sind fast alle natürlichen zulässigen Heuristiken (Relaxationen) auch konsistent.
</details>

<details><summary><b>3. Beweise die Optimalität von A\* mit zulässiger Heuristik (Baumsuche).</b></summary>

Sei $C^\ast$ die optimale Lösungskosten, $G_2$ ein suboptimaler Zielknoten ($g(G_2)>C^\ast$) in der Frontier. Dann $f(G_2)=g(G_2)+0>C^\ast$. Auf dem optimalen Pfad liegt stets ein Frontier-Knoten $n$; für ihn gilt $f(n)=g(n)+h(n)\le g(n)+h^\ast(n)=C^\ast$ (Zulässigkeit). Also $f(n)\le C^\ast<f(G_2)$ — A\* expandiert $n$ vor $G_2$. Somit wird nie ein suboptimales Ziel vor dem optimalen gewählt. Für die *Graph*suche braucht man Konsistenz (oder Re-Opening), damit $f$ entlang Pfaden nicht fällt.
</details>

<details><summary><b>4. Warum ist eine höhere zulässige Heuristik besser? Wie kombiniert man zwei Heuristiken?</b></summary>

A\* expandiert alle Knoten mit $f(n)<C^\ast$, also $g(n)+h(n)<C^\ast$, d.\,h. $h(n)<C^\ast-g(n)$. Eine *größere* (aber weiterhin zulässige) $h$ macht diese Bedingung für weniger Knoten wahr → weniger Expansionen. Formal: Ist $h_a\ge h_b$ (beide zulässig), expandiert A\* mit $h_a$ eine Teilmenge (bis auf $f=C^\ast$-Gleichstände) der Knoten von $h_b$ — $h_a$ **dominiert**. Kombination: $h(n)=\max\{h_a(n),h_b(n)\}$ ist wieder zulässig und mindestens so gut wie beide.
</details>

<details><summary><b>5. Was macht AC-3, und was ist seine Komplexität? Warum ist ein baumstrukturiertes CSP „einfach"?</b></summary>

AC-3 stellt **Kantenkonsistenz** her: streicht aus jeder Domäne $D_i$ Werte, die zu einer Nachbarvariable $X_j$ keinen zulässigen Partner haben, und propagiert Änderungen. Komplexität $O(c\,d^3)$ ($c$ binäre Constraints, Domänengröße $d$): jede Kante kommt $\le d$-mal in die Queue, `REVISE` kostet $O(d^2)$. Ein **Baum-CSP** ist in $O(n\,d^2)$ lösbar: topologisch ordnen, von den Blättern zur Wurzel kantenkonsistent machen (Directional Arc Consistency), dann von der Wurzel greedy vorwärts belegen — kein Backtracking nötig, weil jede Variable nur einen Elter hat. Die Graphstruktur bestimmt die Komplexität.
</details>

<details><summary><b>6. Erkläre den Brückensatz $\mathrm{KB}\models\alpha \iff \mathrm{KB}\land\lnot\alpha$ unerfüllbar. Warum ist er für maschinelles Beweisen so wichtig?</b></summary>

$\mathrm{KB}\models\alpha$ heißt: In jedem Modell von KB ist $\alpha$ wahr, d.\,h. es gibt *kein* Modell mit KB wahr und $\alpha$ falsch — also ist $\mathrm{KB}\land\lnot\alpha$ unerfüllbar. Bedeutung: Statt eine Aussage über *alle unendlich vielen* Modelle zu prüfen (Folgerung), sucht man *einen einzigen* Widerspruch. Das erlaubt **Beweis durch Widerlegung**: Negiere die Behauptung, füge sie zur KB hinzu, leite mit Resolution die leere Klausel ab. Resolution ist nur *widerlegungs*vollständig — genau darum ist dieser Satz die Brücke, die das trotzdem zu einem vollständigen Folgerungsverfahren macht.
</details>

<details><summary><b>7. Was ist Unit Propagation in DPLL und warum ist sie so wirkungsvoll?</b></summary>

Ist in einer Klausel nur noch *ein* Literal unbelegt und alle anderen falsch, muss dieses Literal wahr sein, damit die Klausel (und damit die ganze KNF) erfüllbar bleibt. DPLL setzt es zwangsweise, ohne zu verzweigen — das kann eine Kettenreaktion weiterer Unit-Klauseln auslösen. Wirkung: Unit Propagation eliminiert Verzweigungen ohne Ratekosten; empirisch verbringen SAT-Solver den Großteil ihrer Zeit hier. CDCL-Solver bauen darauf *Klausellernen* und Backjumping auf und skalieren zu Millionen Variablen.
</details>

<details><summary><b>8. Was ist ein MGU und wofür der Occurs-Check?</b></summary>

Ein **allgemeinster Unifikator** $\theta$ zweier Atome $p,q$ macht sie gleich ($p\theta=q\theta$) und legt dabei so *wenig* wie möglich fest — jeder andere Unifikator ist eine Instanz von $\theta$. Er ist bis auf Umbenennung eindeutig. Der **Occurs-Check** verhindert, eine Variable $x$ an einen Term zu binden, der $x$ selbst enthält (z.\,B. $x$ mit $f(x)$) — das ergäbe einen unendlichen Term. Ohne ihn wird Unifikation unsound; viele Prolog-Systeme lassen ihn aus Performancegründen weg.
</details>

<details><summary><b>9. Wozu Skolemisierung? Erhält sie die logische Äquivalenz?</b></summary>

Skolemisierung eliminiert **Existenzquantoren** bei der KNF-Bildung in FOL: $\exists y$ ohne umgebendes $\forall$ → neue **Skolem-Konstante**; $\exists y$ im Bereich von $\forall x_1\dots x_k$ → **Skolem-Funktion** $g(x_1,\dots,x_k)$ (der „Zeuge" hängt von den äußeren Variablen ab). Sie erhält **nicht** die logische Äquivalenz, wohl aber die **Erfüllbarkeit** — und genau das genügt dem Refutationsverfahren, das nur Unerfüllbarkeit von $\mathrm{KB}\land\lnot\alpha$ nachweisen muss.
</details>

<details><summary><b>10. FOL-Gültigkeit ist semi-entscheidbar, aber unentscheidbar. Was heißt das konkret für einen Theorembeweiser?</b></summary>

*Semi-entscheidbar:* Ist die Formel gültig (folgt aus KB), findet ein vollständiges Verfahren (Resolution) den Beweis in endlich vielen Schritten und hält mit „ja". *Unentscheidbar:* Ist sie **nicht** gültig, kann derselbe Beweiser ewig weiterlaufen, ohne je „nein" auszugeben — es gibt beweisbar keinen Algorithmus, der für *alle* FOL-Formeln terminiert und korrekt entscheidet (Reduktion auf das Halteproblem, Church/Turing 1936). Praktisch: Ein FOL-Beweiser kann Gültigkeit bestätigen, aber Ungültigkeit nie *garantiert* feststellen. Deshalb nutzt man entscheidbare Fragmente (Datalog, Description Logics), wenn Terminierung gebraucht wird.
</details>

---

## Literatur & Quellen

**Lehrbücher**
- **Russell & Norvig, *Artificial Intelligence: A Modern Approach* (AIMA), 4. Aufl.** — *das* Standardwerk. Für dieses Modul: Kap. 3 (Suche), 4 (informiert/lokal), 5 (Spiele), 6 (CSP), 7 (Aussagenlogik), 8–9 (FOL & Inferenz). Einsteigerfreundlich und trotzdem vollständig. *Die primäre Empfehlung.*
- **Ertel, *Grundkurs Künstliche Intelligenz*, Springer Vieweg** — deutschsprachig, kompakt, gut für Logik und Suche. *Einsteigerfreundlich.*
- **Nilsson, *Artificial Intelligence: A New Synthesis*** — klassisch, elegante Darstellung von Suche und Logik. *Vertiefend.*
- **Chang & Lee, *Symbolic Logic and Mechanical Theorem Proving*** — die Referenz für Resolution, Unifikation, Herbrand-Theorem. *Vertiefend, mathematisch.*

**Frei verfügbare Kurse & Materialien** (kostenlos)
- **UC Berkeley CS188 „Introduction to AI"** — Videos, Folien, die berühmten *Pac-Man-Projekte* (Suche, CSP, Spiele in Python). `inst.eecs.berkeley.edu/~cs188`. *Einsteigerfreundlich, hochpraktisch.*
- **AIMA-Begleitseite** mit Pseudocode und Referenzimplementierungen (`aima-python` auf GitHub). *Direkt zum Nachprogrammieren.*
- **Stanford CS221 „AI: Principles and Techniques"** — Skripte und Aufgaben frei online. *Vertiefend.*
- **MIT 6.034 „Artificial Intelligence"** (OpenCourseWare) — Videovorlesungen von Patrick Winston zu Suche, Constraints, Logik. *Einsteigerfreundlich.*

**Interaktiv / Visualisierungen** (kostenlos)
- **PathFinding.js** (`qiao.github.io/PathFinding.js/visual`) — A\*, Dijkstra, IDA\* live auf einem Gitter beobachten. *Sehr einsteigerfreundlich.*
- **Red Blob Games — „Introduction to A\*"** (`redblobgames.com/pathfinding/a-star/introduction.html`) — die beste interaktive Erklärung von A\*, die es gibt. *Einsteigerfreundlich.*
- **Sudoku-/SAT-Solver-Visualisierungen** und der **MiniSat**-Quellcode für einen realen, schlanken CDCL-Solver. *Vertiefend.*

**Klassische Papers** (kostenlos, vertiefend)
- Hart, Nilsson & Raphael (1968): *A Formal Basis for the Heuristic Determination of Minimum Cost Paths* — die A\*-Originalarbeit.
- Robinson (1965): *A Machine-Oriented Logic Based on the Resolution Principle* — die FOL-Resolution.
- Davis, Logemann & Loveland (1962): *A Machine Program for Theorem-Proving* — der DPLL-Algorithmus.

---

## Die drei Projekte

Dieses Modul ist theorielastig, aber die Kernverfahren *leben* erst, wenn man sie laufen sieht. Deshalb sind alle drei Projekte **implementierend** (Python), jeweils mit einem theoretischen Reflexionsteil. Aufsteigende Schwierigkeit und abnehmende Code-Vorgabe:

- **01 – basic** (`projects/01-basic/`): **Suchalgorithmen auf dem 8-Puzzle & Romania.** Geführtes Notebook: BFS/UCS/IDDFS/A\* implementieren, Heuristiken $h_1,h_2$ vergleichen, Zulässigkeit/Dominanz empirisch nachprüfen (expandierte Knoten zählen). Viel Anleitung.
- **02 – medium** (`projects/02-medium/`): **CSP-Solver + DPLL-SAT-Solver.** Strukturiertes Python-Projekt: Backtracking mit MRV/AC-3 für Sudoku *und* ein DPLL-Solver mit Unit Propagation; ein Sudoku wird zusätzlich als SAT-Instanz kodiert und mit beiden Wegen gelöst. Nur vereinzelte Inspiration.
- **03 – final** (`projects/03-final/`): **Ein Resolutions-Theorembeweiser für die Aussagen- und Prädikatenlogik.** Keine Code-Vorgabe. KNF-Umwandlung, Unifikation (mit Occurs-Check), Resolution mit Widerlegung; anwenden auf ein realistisches Wissensbasis-Szenario. Master-Niveau: konsolidiert Logik, Unifikation und das Refutationsprinzip.

Details, Setup und Musterlösungen jeweils in der `README.md` des Projektordners.
