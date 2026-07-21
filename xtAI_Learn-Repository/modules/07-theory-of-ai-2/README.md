# Module 07 — Theory of Artificial Intelligence 2

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

**What is this about?** Module 06 showed how an agent acts in a *known,
deterministic* world through search and logic. This module goes three steps
further: (1) **planning** — acting in structured but still deterministic worlds
with a compact, logic-like representation of actions; (2) **reasoning under
uncertainty** — when the world is no longer known for certain, we replace
"true/false" by **probabilities** and compute with **Bayesian networks**; (3)
**rational decision making** — when actions have uncertain outcomes, the agent
maximizes the **expected utility**, in the sequential case via **Markov
decision processes (MDPs)**. That is the transition from *symbolic* to
*probabilistic* and *decision-theoretic* AI — and the theoretical bridge to
reinforcement learning (modules 13/14).

**Helpful prior knowledge.** Module 06 (search, logic — planning builds
directly on it). The basics of probability and linear algebra (modules 02/03
are enough). Some calculus for the convergence proofs (contraction).

**Recommended earlier modules.** Module 06 "Theory of AI 1" (mandatory for the
planning chapter), Data Science 1/2 for the probability basics.

**Following modules.** Reinforcement learning (13) and deep RL (14) build
directly on the MDP part. Bayesian networks reappear in ML, NLP and
bioinformatics.

---

## Learning objectives

After this module you should be able to

- formalize a planning problem in **STRIPS/PDDL** and explain the difference
  between **forward (progression), backward (regression) and plan-space search**;
- derive **domain-independent heuristics** from **relaxation** (ignoring the
  delete lists, $h_{\text{add}}$, $h_{\max}$, $h_{\text{FF}}$);
- place **GraphPlan** (the planning graph, mutex relations) and **SATPlan** as
  alternative planning paradigms;
- apply the **axioms of probability**, conditional probability, **Bayes'
  theorem** and **(conditional) independence** confidently;
- construct a **Bayesian network**, justify its **factorization of the joint
  distribution** and read off independences with **d-separation**;
- carry out **exact inference** (enumeration, **variable elimination**) and
  **approximate inference** (likelihood weighting, **Gibbs sampling/MCMC**);
- understand **temporal models** (Markov chains, **HMMs**) and derive
  **filtering, smoothing** and the **Viterbi algorithm**;
- make **rational decisions** via the **maximum expected utility (MEU)**
  principle and compute the **value of information (VPI)**;
- define **MDPs**, set up the **Bellman equations** and understand **value
  iteration** and **policy iteration** including the **convergence proof
  (contraction)**;
- explain the connection between planning, probabilistic reasoning and
  sequential decision making.

---

## Part 1 — Foundations: classical planning

### 1.1 Why planning instead of plain search?

One *could* treat every planning problem as a search problem (module 06). The
problem: in a world with many objects and predicates the state space is
astronomical, and pure state search does not "see" the **structure** of the
actions. **Planning** uses a **factored, logic-like representation** of states
and actions. That makes it possible to (a) derive **domain-independent
heuristics** automatically from the action description and (b) exploit actions
that are independent of each other.

### 1.2 STRIPS and PDDL

In the **STRIPS** formalism (Stanford Research Institute Problem Solver) a
**state** is a set of **ground fluents** (variable-free, true atomic statements)
under the **closed-world assumption**: whatever is not in the set counts as
false. A **planning problem** is $(\mathcal{F}, s_0, g, \mathcal{A})$ with

- $\mathcal{F}$: the set of all fluents,
- $s_0 \subseteq \mathcal{F}$: the **initial state**,
- $g \subseteq \mathcal{F}$: the **goal condition** (a state $s$ satisfies the goal if $g \subseteq s$),
- $\mathcal{A}$: the set of **action schemas**. A ground action $a$ consists of
  - $\mathrm{PRE}(a)$ — the **preconditions** (fluents that must hold),
  - $\mathrm{ADD}(a)$ — the **add list** (fluents that $a$ makes true),
  - $\mathrm{DEL}(a)$ — the **delete list** (fluents that $a$ makes false).

An action $a$ is **applicable** in $s$ if $\mathrm{PRE}(a) \subseteq s$. The
**transition model** (progression) is purely set-theoretic:
$$
\mathrm{Result}(s, a) = (s \setminus \mathrm{DEL}(a)) \cup \mathrm{ADD}(a).
$$

**PDDL** (Planning Domain Definition Language) is the syntax commonly used for
this today; it separates a **domain file** (predicates, action schemas with
variables) from a **problem file** (objects, initial state, goal). Action
schemas with variables are instantiated into ground actions before the search
(*grounding*).

> **Example (blocks world).** Fluents: `On(x,y)`, `OnTable(x)`, `Clear(x)`,
> `Holding(x)`, `ArmEmpty`. The action `PickUp(x)`:
> $\mathrm{PRE} = \{\mathrm{Clear}(x), \mathrm{OnTable}(x), \mathrm{ArmEmpty}\}$,
> $\mathrm{ADD} = \{\mathrm{Holding}(x)\}$,
> $\mathrm{DEL} = \{\mathrm{Clear}(x), \mathrm{OnTable}(x), \mathrm{ArmEmpty}\}$.

The charm of it: the **frame problem** (what does *not* change?) is solved
elegantly — everything that is not in ADD/DEL stays unchanged.

### 1.3 Forward, backward and plan-space search

**Progression (forward search)** searches the state space from $s_0$ towards the
goal with the procedures from module 06 (usually A\* with one of the heuristics
below). The advantage: simple states; the disadvantage: a high branching factor
(many applicable actions).

**Regression (backward search)** starts at the goal $g$ and works backwards.
The regression step: in order to reach the state description $g'$ after
applying $a$, the following must hold beforehand
$$
\mathrm{Regress}(g', a) = (g' \setminus \mathrm{ADD}(a)) \cup \mathrm{PRE}(a),
\quad \text{provided } a \text{ is relevant } (\mathrm{ADD}(a)\cap g' \neq \emptyset)
\text{ and consistent } (\mathrm{DEL}(a) \cap g' = \emptyset).
$$
The advantage: only **relevant** actions are considered (a small branching
factor); the disadvantage: states are *partial descriptions* (sets of
conditions), and good heuristic design is harder.

**Plan-space search / partial-order planning (POP).** Instead of searching the
state space, one searches the space of *partial plans*. A partial plan consists
of (i) a set of actions, (ii) **ordering constraints** $a \prec b$ ("$a$ before
$b$"), (iii) **causal links** $a \xrightarrow{p} b$ ("$a$ provides the
precondition $p$ of $b$") and (iv) open preconditions. One refines the plan
until no open precondition remains and all **threats** are resolved. An action
$c$ **threatens** a link $a \xrightarrow{p} b$ if $c$ deletes $p$ and could lie
between $a$ and $b$; one resolves that by **promotion** ($c \prec a$) or
**demotion** ($b \prec c$). POP produces **partially ordered plans** — it does
not commit prematurely to an ordering of independent actions (**least
commitment**).

### 1.4 Planning heuristics from relaxation

The breakthrough of modern planning: **domain-independent** heuristics that
arise automatically from the STRIPS description — analogous to relaxation in
module 06, only systematic.

**Delete relaxation.** One deletes *all* delete lists: $\mathrm{DEL}(a) =
\emptyset$. In the relaxed world a property that has once been achieved can
never be lost again (the set of fluents grows monotonically), and the relaxed
problem is solvable in **polynomial time**. From it one obtains:

- $h_{\text{add}}(s) = \sum_{p \in g}\, \Delta(p)$ — the sum of the estimated
  costs of achieving every goal fluent individually (it overestimates when they
  interact → **not admissible**, but informative).
- $h_{\max}(s) = \max_{p \in g}\, \Delta(p)$ — the most expensive individual
  goal fluent (**admissible**, but often weak).
- $h_{\text{FF}}$ (Fast Forward): it extracts a concrete **relaxed plan** from
  the relaxed planning graph and takes its length — usually the best heuristic
  in practice (not admissible, but very accurate).

Here $\Delta(p)$ is defined recursively: $\Delta(p)=0$ if $p\in s$, otherwise
$\Delta(p) = \min_{a:\, p\in \mathrm{ADD}(a)} \big(\mathrm{cost}(a) + \text{combine}_{q\in \mathrm{PRE}(a)}\Delta(q)\big)$,
where $\text{combine}=\sum$ for $h_{\text{add}}$ and $\text{combine}=\max$ for $h_{\max}$.

### 1.5 GraphPlan and SATPlan

**GraphPlan** builds a layered **planning graph** out of alternating **state
levels** ($S_0, S_1, \dots$) and **action levels** ($A_0, A_1, \dots$). Every
level contains *all* fluents/actions that are *possibly* reachable, plus
**mutex relations** (mutual exclusion):
- Two **actions** are mutex if one deletes the precondition/effect of the other
  (*inconsistent effects*, *interference*) or if their preconditions are mutex
  (*competing needs*).
- Two **fluents** are mutex if every action producing them is mutex
  (*inconsistent support*).

The graph grows until all goal fluents appear non-mutex in one level (*levelling
off*); then a backward search extracts a plan. At the same time the graph
provides the admissible heuristic "the first level in which a goal fluent
appears".

**SATPlan.** Encode "does a plan of length $\le k$ exist?" as a **propositional
formula** (fluents and actions with a time index, precondition/effect and frame
axioms) and throw a **SAT solver** (module 06, DPLL/CDCL) at it; increase $k$
iteratively. It shows impressively how the technique of module 06 is reused here.

---

## Part 2 — Building up: reasoning under uncertainty

### 2.1 Why probability?

Real agents do not know the world for certain: sensors are noisy, actions fail,
knowledge is incomplete. Purely logical reasoning fails because one would have
to enumerate all the exceptions (the "qualification problem"). **Probability**
captures this uncertainty in *one number* per statement — as a **degree of
belief** (the Bayesian view), not necessarily as a frequency. De Finetti's
theorem shows: anyone who takes bets whose odds do *not* obey the axioms of
probability is guaranteed to be exploitable by a **Dutch book** — rationality
*forces* the probability calculus.

### 2.2 Foundations: axioms, conditioning, Bayes

For events the **Kolmogorov axioms** hold:
$$
0 \le P(a) \le 1, \quad P(\text{true}) = 1, \quad
P(a \lor b) = P(a) + P(b) - P(a \land b).
$$
The **joint distribution** $P(X_1, \dots, X_n)$ over all variables determines
*everything*. From it one obtains every query through:
- **marginalization (summing out):** $P(\mathbf{Y}) = \sum_{\mathbf{z}} P(\mathbf{Y}, \mathbf{z})$,
- **conditioning:** $\displaystyle P(a \mid b) = \frac{P(a \land b)}{P(b)}$ (for $P(b) > 0$).

Rearranging the definition gives the **product rule** $P(a\land b) = P(a\mid b)\,P(b)$
and, by equating, **Bayes' theorem**:
$$
\boxed{\,P(h \mid e) = \frac{P(e \mid h)\,P(h)}{P(e)}\,}
$$
the workhorse: from the **likelihood** $P(e\mid h)$ (how probable is the
evidence under the hypothesis) and the **prior** $P(h)$ one obtains the
**posterior** $P(h\mid e)$. The denominator $P(e) = \sum_{h'} P(e\mid h')P(h')$
is the **normalization constant**; one often writes $P(h\mid e) = \alpha\, P(e\mid h)P(h)$.

> **Worked through (a medical test).** A disease with prevalence $P(D)=0.01$;
> a test with sensitivity $P(+\mid D)=0.9$ and false positive rate $P(+\mid \lnot D)=0.09$.
> A positive test — how probable is the disease?
> $$P(D\mid +) = \frac{0.9 \cdot 0.01}{0.9\cdot 0.01 + 0.09\cdot 0.99}
> = \frac{0.009}{0.009 + 0.0891} \approx 0.092.$$
> Only about 9 %! The low **base rate** dominates — the classical *base rate
> fallacy*. A model example of why one must not ignore priors.

**(Conditional) independence.** $X$ and $Y$ are **independent** if
$P(X,Y)=P(X)P(Y)$. More important still is **conditional independence**:
$X \perp Y \mid Z$ iff $P(X,Y\mid Z) = P(X\mid Z)\,P(Y\mid Z)$. It is the key
that makes the exponentially large joint distribution **compactly factorizable**.

### 2.3 Bayesian networks: structure and semantics

A full joint distribution over $n$ boolean variables needs $2^n - 1$ numbers —
unmanageable. A **Bayesian network** uses conditional independences to represent
the same distribution *compactly*:

- a **directed acyclic graph (DAG)**, nodes = random variables, an edge
  $X \to Y$ = "$X$ influences $Y$ directly",
- one **conditional probability table (CPT)** per node, $P(X_i \mid \mathrm{Parents}(X_i))$.

**The central semantics (the chain rule for Bayesian networks):** the network
represents the joint distribution as the **product of the local CPTs**:
$$
\boxed{\,P(x_1, \dots, x_n) = \prod_{i=1}^{n} P\big(x_i \mid \mathrm{parents}(x_i)\big)\,}
$$
That holds exactly when every variable is **conditionally independent of its
non-descendants given its parents** (the *local Markov condition*). With a
bounded number of parents $k$ the memory shrinks from $2^n$ to $n\cdot 2^k$ —
often from astronomical to manageable.

> **The classic (the alarm network, Pearl).** A burglary ($B$) or an earthquake
> ($E$) can trigger an alarm ($A$); John ($J$) and Mary ($M$) call depending on
> whether they hear the alarm. The structure: $B\to A \leftarrow E$, $A\to J$,
> $A\to M$. The joint: $P(B,E,A,J,M) = P(B)P(E)P(A\mid B,E)P(J\mid A)P(M\mid A)$
> — five small CPTs instead of one 32-row table.

### 2.4 d-separation — reading independence off the graph structure

When does $X \perp Y \mid Z$ hold *purely because of the structure*? That is
answered by **d-separation**. Consider every undirected path between $X$ and
$Y$; the path is **blocked** if it contains a node $n$ of the following kind:

1. a **chain** $\to n \to$ or a **fork** $\leftarrow n \to$, and $n \in Z$
   (observed) → blocked.
2. a **collider** $\to n \leftarrow$ (a "v-structure"), and **neither $n$ nor a
   descendant of $n$** is in $Z$ → blocked.

If *all* paths are blocked, then $X \perp Y \mid Z$. The collider rule is the
subtle point: an unobserved collider *blocks*, but **observing the collider (or
a descendant) opens** it — that is **"explaining away"**: an earthquake and a
burglary are a priori independent, but *given the alarm* they become dependent
(hearing about the earthquake lowers the probability of a burglary).

### 2.5 Exact inference

The **inference task**: compute $P(\mathbf{X}_{\text{query}} \mid \mathbf{e})$
for query variables given the evidence $\mathbf{e}$.

**Inference by enumeration.** Directly from the joint factorization:
$$
P(X \mid \mathbf{e}) = \alpha \sum_{\mathbf{y}} P(X, \mathbf{e}, \mathbf{y}),
$$
where $\mathbf{y}$ are the *hidden* variables and the product of the CPTs is
substituted. Correct, but $O(2^n)$ — the naive sum repeats subproducts.

**Variable elimination (VE).** Speeds the enumeration up by **factoring out**
(the distributive law) and **caching**. One works with **factors**
(multidimensional tables). Two operations:
- the **pointwise product** of two factors $f_1 \times f_2$,
- **summing out** a variable: $\sum_x f(\dots, x, \dots)$.

The algorithm: choose an **elimination order** for the hidden variables; for
each one, multiply all the factors containing it and sum it out.
$$
P(B\mid j,m) = \alpha\, P(B) \sum_e P(e) \sum_a P(a\mid B,e)\,P(j\mid a)\,P(m\mid a).
$$
VE is dramatically faster than enumeration, but the cost depends strongly on
the order (the largest intermediate factor determines it — the **treewidth** of
the graph). **Bayesian network inference is NP-hard in general**; for networks
of small treewidth (polytrees, for instance) it is polynomial.

### 2.6 Approximate inference by sampling

When exact inference is too expensive, one **estimates**
$P(\mathbf{X}\mid\mathbf{e})$ from samples.

- **Direct/prior sampling:** draw values in topological order according to the
  CPTs → samples from the joint distribution.
- **Rejection sampling:** as above, but discard every sample that contradicts
  $\mathbf{e}$. Correct, but wasteful when the evidence is rare.
- **Likelihood weighting:** fix the evidence variables at their observed values
  and **weight** every sample by the product of the evidence likelihoods
  $\prod_{e_i} P(e_i \mid \mathrm{parents}(e_i))$. It discards nothing and is
  more efficient.
- **Gibbs sampling (MCMC):** a **Markov chain Monte Carlo** procedure. Fix the
  evidence, initialize the remaining variables arbitrarily; then repeatedly
  resample *one* non-evidence variable from its distribution given its **Markov
  blanket** (parents, children, co-parents). The chain generated this way has
  the posterior as its **stationary distribution**; the sample means converge to
  $P(\mathbf{X}\mid\mathbf{e})$.

All sampling procedures are **consistent** (the error $\to 0$ as $N\to\infty$,
at rate $O(1/\sqrt N)$), so they trade exactness for computation time — the
usual deal with difficult networks.

### 2.7 Temporal models: Markov chains and HMMs

The world changes over time. A **discrete-time** model has state variables
$\mathbf{X}_t$ and evidence variables $\mathbf{E}_t$ per step $t$. Two
assumptions make it tractable:

- **The (first-order) Markov assumption:** $P(\mathbf{X}_t \mid \mathbf{X}_{0:t-1}) =
  P(\mathbf{X}_t \mid \mathbf{X}_{t-1})$ — the future depends on the past only
  through the **current** state. → the **transition model**.
- **The sensor Markov assumption:** $P(\mathbf{E}_t \mid \mathbf{X}_{0:t}, \mathbf{E}_{0:t-1})
  = P(\mathbf{E}_t \mid \mathbf{X}_t)$. → the **sensor model**.

A **hidden Markov model (HMM)** has a single discrete state variable that one
cannot see directly (*hidden*), only through the evidence. Four standard tasks:

**Filtering** — $P(\mathbf{X}_t \mid \mathbf{e}_{1:t})$ (the current state given
all observations so far). Recursively (the **forward algorithm**):
$$
P(\mathbf{X}_{t+1}\mid \mathbf{e}_{1:t+1}) = \alpha\, \underbrace{P(\mathbf{e}_{t+1}\mid \mathbf{X}_{t+1})}_{\text{update (sensor)}}
\sum_{\mathbf{x}_t} \underbrace{P(\mathbf{X}_{t+1}\mid \mathbf{x}_t)}_{\text{predict (transition)}} P(\mathbf{x}_t\mid \mathbf{e}_{1:t}).
$$
A "predict-update" loop — the same idea is inside the **Kalman filter** (the
continuous Gaussian special case) and in the localization of robots (module 21).

**Prediction** — $P(\mathbf{X}_{t+k}\mid\mathbf{e}_{1:t})$ (the future without new evidence).

**Smoothing** — $P(\mathbf{X}_k\mid\mathbf{e}_{1:t})$ for $k<t$ (the past with
hindsight). The **forward-backward algorithm** combines the forward message with
a backward message $b_{k+1:t} = P(\mathbf{e}_{k+1:t}\mid\mathbf{X}_k)$.

**The most likely explanation** — $\arg\max_{\mathbf{x}_{1:t}} P(\mathbf{x}_{1:t}\mid\mathbf{e}_{1:t})$.
The **Viterbi algorithm** is dynamic programming: in the forward recursion it
replaces the sum by a **maximum** and remembers back pointers in order to
reconstruct the best path. The basis of speech recognition, POS tagging (module
08) and sequence analysis in bioinformatics (module 28).
---

## Part 3 — Advanced: rational decisions and MDPs

### 3.1 Utility theory and maximum expected utility

So far we have *reasoned*, now we *act*. **Utility theory** (von Neumann &
Morgenstern) shows: if a preference relation over uncertain outcomes
("lotteries") satisfies six **rationality axioms** (completeness, transitivity,
continuity, substitutability, monotonicity, decomposability), then a **utility
function** $U$ **exists** such that the agent prefers lottery $L_1$ to $L_2$
exactly when $\mathrm{EU}(L_1) > \mathrm{EU}(L_2)$, with the **expected utility**
$$
\mathrm{EU}(a\mid \mathbf{e}) = \sum_{s'} P(\mathrm{Result}(a) = s' \mid a, \mathbf{e})\; U(s').
$$
The **MEU principle (maximum expected utility):** a rational agent chooses the
action that maximizes the expected utility:
$a^\ast = \arg\max_a \mathrm{EU}(a\mid\mathbf{e})$. Important: utility is **not**
the same as money — the typically **concave** utility function for money
explains **risk aversion** (a certain 100 euros can have more utility than a
50/50 chance at 0/220 euros).

### 3.2 Decision networks and the value of information

**Decision networks (influence diagrams)** extend Bayesian networks by
**decision nodes** (actions the agent chooses) and a **utility node**. The
evaluation picks the actions with the maximum expected utility.

**Value of perfect information (VPI).** Is it worth measuring a variable $E_j$
*before* the decision? The value of the information is the expected increase in
utility:
$$
\mathrm{VPI}_{\mathbf{e}}(E_j) = \Big(\sum_{e_{j}} P(e_{j}\mid\mathbf{e})\;
\mathrm{EU}(a^\ast_{e_{j}} \mid \mathbf{e}, e_{j})\Big) - \mathrm{EU}(a^\ast \mid \mathbf{e}).
$$
VPI is **never negative** (in expectation, more knowledge cannot hurt) and it is
**not additive**. It provides the theoretical foundation for *rational
information gathering* — which sensor or test to reach for.

### 3.3 Markov decision processes (MDPs)

Now the **sequential** case: decisions over many steps, with uncertain
outcomes. An **MDP** is $(S, A, P, R, \gamma)$:

- $S$ states, $A$ actions,
- $P(s' \mid s, a)$ the **transition model** (stochastic! — this is where the
  break with classical planning happens),
- $R(s)$ (or $R(s,a,s')$) the **reward**,
- $\gamma \in [0,1)$ the **discount factor** (later rewards count less; it also
  secures convergence for an infinite horizon).

We are looking for a **policy** $\pi: S \to A$ that maximizes the expected
**discounted return** $\mathbb{E}\big[\sum_{t=0}^{\infty}\gamma^t R(s_t)\big]$.
The **value** of a state under $\pi$ is $V^\pi(s) = \mathbb{E}\big[\sum_t \gamma^t
R(s_t) \mid s_0=s, \pi\big]$.

**The Bellman equation** characterizes $V^\pi$ self-consistently:
$$
V^\pi(s) = R(s) + \gamma \sum_{s'} P(s'\mid s, \pi(s))\, V^\pi(s').
$$
For the **optimal** policy $\pi^\ast$ the **Bellman optimality equation** holds:
$$
\boxed{\,V^\ast(s) = R(s) + \gamma \max_{a} \sum_{s'} P(s'\mid s, a)\, V^\ast(s')\,}
$$
and one reads the optimal policy off greedily:
$\pi^\ast(s) = \arg\max_a \sum_{s'} P(s'\mid s,a)\,V^\ast(s')$.

### 3.4 Value iteration and policy iteration

**Value iteration.** Read the Bellman optimality equation as an **update** and
iterate to convergence:
$$
V_{k+1}(s) \leftarrow R(s) + \gamma \max_a \sum_{s'} P(s'\mid s,a)\, V_k(s).
$$

> **Why does that converge? (The contraction proof.)** The **Bellman optimality
> operator** $B$, defined by $(BV)(s) = R(s) + \gamma\max_a\sum_{s'}P(s'\mid s,a)V(s')$,
> is a **contraction** with respect to the maximum norm $\lVert V\rVert_\infty = \max_s|V(s)|$
> with factor $\gamma$: for arbitrary $V, V'$ we have
> $$\lVert BV - BV'\rVert_\infty \le \gamma\,\lVert V - V'\rVert_\infty.$$
> *Proof sketch:* $|\max_a f(a) - \max_a g(a)| \le \max_a|f(a)-g(a)|$, and
> $\gamma\sum_{s'}P(s'\mid s,a)|V(s')-V'(s')| \le \gamma\lVert V-V'\rVert_\infty$, since
> $\sum_{s'}P=1$. By the **Banach fixed point theorem**, $B$ has a **unique
> fixed point** $V^\ast$, and $V_k \to V^\ast$ **geometrically**: $\lVert V_k - V^\ast\rVert_\infty
> \le \gamma^k \lVert V_0 - V^\ast\rVert_\infty$. For $\gamma<1$ that guarantees
> convergence — the closer $\gamma$ is to 1, the slower it is. $\quad\blacksquare$

**Policy iteration.** It alternates two steps until the policy is stable:
1. **Policy evaluation:** solve $V^{\pi}(s) = R(s) + \gamma\sum_{s'}P(s'\mid s,\pi(s))V^\pi(s')$
   — a **linear system of equations** in $|S|$ unknowns (solvable exactly or
   approximated iteratively).
2. **Policy improvement:** set
   $\pi'(s) \leftarrow \arg\max_a \sum_{s'}P(s'\mid s,a)V^\pi(s')$.

Once the policy no longer changes, it is optimal. Policy iteration converges in
**finitely many** steps (there are only finitely many policies, and every step
either strictly improves or terminates) — often in *very few* iterations, but
each one is more expensive than a value iteration step. Both are special cases
of **generalized policy iteration**, the conceptual core of reinforcement
learning (module 13): the only difference there is that $P$ and $R$ are
**unknown** and have to be learned from experience.

### 3.5 Partial observability (POMDPs) — an outlook

If the agent cannot observe the state directly (only noisy sensors), the MDP
becomes a **POMDP**. The trick: the agent keeps a **belief state** $b(s)$ (a
probability distribution over states, updated by the filtering of section 2.7)
and solves an MDP in the *continuous* belief space. POMDPs are theoretically
elegant, but solving them exactly is **PSPACE-hard** — in practice one uses
approximations. They connect filtering (part 2) with MDPs (part 3) into the
complete picture of the rational agent under uncertainty.

### 3.6 An outlook: non-monotonic reasoning and description logics

Two further answers to uncertainty — not probabilistic but **qualitative**:

- **Non-monotonic reasoning.** Classical logic is *monotonic*: more premises →
  never fewer conclusions. Everyday inferences, however, are **defeasible**
  ("birds fly — but not penguins"). **Default logic**, **circumscription** and
  **answer set programming** formalize such default assumptions, which new
  information can *retract*. The basis of knowledge representation and of logic
  programming with negation (module 33).
- **Description logics (DL).** The decidable FOL fragment behind **ontologies**
  and the **semantic web** (OWL). A **TBox** defines concepts/roles
  ($\text{Father} \equiv \text{Man} \sqcap \exists\text{hasChild}.\top$), an
  **ABox** contains the instance facts. The core inferences (subsumption,
  instance checking) are decidable — the deliberate trade of expressive power
  for decidability that module 06 (section 4.5) already announced. DL reasoners
  (e.g. via tableau procedures) are the practical continuation of the theorem
  proving from module 06.

---

## Summary / cheat sheet

**Planning**

| Notion | Core |
|---|---|
| A STRIPS action | $\langle\mathrm{PRE},\mathrm{ADD},\mathrm{DEL}\rangle$; applicable if $\mathrm{PRE}\subseteq s$ |
| Progression | $\mathrm{Result}(s,a) = (s\setminus\mathrm{DEL})\cup\mathrm{ADD}$ |
| Regression | $(g'\setminus\mathrm{ADD})\cup\mathrm{PRE}$, relevant + consistent |
| POP | a partial order + causal links; threats via promotion/demotion |
| Relaxation | delete the delete lists → $h_{\max}$ (admissible), $h_{\text{add}}$, $h_{\text{FF}}$ |
| GraphPlan | a planning graph + mutexes; SATPlan: encode the plan as SAT |

**Probability and Bayesian networks**

| Notion | Core |
|---|---|
| Bayes | $P(h\mid e) = \dfrac{P(e\mid h)P(h)}{P(e)} = \alpha\,P(e\mid h)P(h)$ |
| conditional indep. | $X\perp Y\mid Z \iff P(X,Y\mid Z)=P(X\mid Z)P(Y\mid Z)$ |
| BN factorization | $P(x_1..x_n) = \prod_i P(x_i\mid\mathrm{parents}(x_i))$ |
| d-separation | a chain/fork blocks when observed; a collider blocks when **un**observed |
| explaining away | a collider opens on observation → the parents become dependent |
| Enumeration | $P(X\mid\mathbf e)=\alpha\sum_{\mathbf y}\prod_i P(x_i\mid\mathrm{parents})$ |
| Variable elim. | factors: pointwise product + summing out; cost ~ treewidth; BN inference NP-hard |
| Sampling | prior/rejection/**likelihood weighting**/**Gibbs (MCMC)**; consistent, $O(1/\sqrt N)$ |

**Temporal models**

| Notion | Core |
|---|---|
| 1st-order Markov | $P(\mathbf X_t\mid\mathbf X_{0:t-1})=P(\mathbf X_t\mid\mathbf X_{t-1})$ |
| Filtering (forward) | $\alpha\,P(\mathbf e_{t+1}\mid\mathbf X_{t+1})\sum_{\mathbf x_t}P(\mathbf X_{t+1}\mid\mathbf x_t)f_t$ |
| Smoothing | forward-backward (the forward × the backward message) |
| Viterbi | like filtering, but $\max$ instead of $\sum$ + back pointers; the best state sequence |

**Decisions and MDPs**

| Notion | Core |
|---|---|
| MEU | $a^\ast=\arg\max_a\sum_{s'}P(s'\mid a,\mathbf e)U(s')$ |
| VPI | the expected increase in utility from a measurement; $\ge 0$, not additive |
| MDP | $(S,A,P,R,\gamma)$; maximize $\mathbb E[\sum_t\gamma^t R]$ |
| Bellman opt. | $V^\ast(s)=R(s)+\gamma\max_a\sum_{s'}P(s'\mid s,a)V^\ast(s')$ |
| Value iteration | iterate the Bellman update; $B$ is a $\gamma$-contraction → $V_k\to V^\ast$ geometrically |
| Policy iteration | evaluation (a linear system) + improvement; terminates in finitely many steps |
---

## Self-test

<details><summary><b>1. What is the difference between progression and regression in STRIPS, and when does one prefer which?</b></summary>

*Progression* searches forwards from $s_0$: states are complete sets of fluents,
$\mathrm{Result}(s,a)=(s\setminus\mathrm{DEL})\cup\mathrm{ADD}$. The branching
factor is high (all applicable actions), but the states are concrete and can be
evaluated heuristically well — which is why in practice (with $h_{\text{FF}}$ or
similar) it is usually the winner. *Regression* searches backwards from the goal:
states are partial descriptions,
$\mathrm{Regress}(g',a)=(g'\setminus\mathrm{ADD})\cup\mathrm{PRE}$ for relevant,
consistent $a$. The branching factor is small (only relevant actions), but
heuristics are harder. Regression pays off when few actions are relevant to the goal.
</details>

<details><summary><b>2. Why is the delete relaxation useful even though $h_{\text{add}}$ is not admissible?</b></summary>

If one deletes all delete lists, the set of fluents grows monotonically and the
relaxed problem is solvable in polynomial time — so one gets an estimate
*cheaply*. $h_{\max}$ (the maximum) is even admissible, but weak. $h_{\text{add}}$
(the sum) overestimates, because it treats the goal fluents as independent and
counts shared subplans twice — in exchange it is far more informative and guides
the search well. $h_{\text{FF}}$ extracts a real relaxed plan and is usually the
best. In practice informativeness often counts for more than strict admissibility
(as long as you do not have to guarantee optimality).
</details>

<details><summary><b>3. A test is 90 % sensitive and has 9 % false positives; the disease has a prevalence of 1 %. Why is $P(\text{ill}\mid+)$ only about 9 %?</b></summary>

Bayes: $P(D\mid+)=\frac{0.9\cdot0.01}{0.9\cdot0.01+0.09\cdot0.99}\approx0.092$.
The reason is the low **base rate**: there are 99 times as many healthy people
as ill ones. Even at only 9 % false positives, the many healthy people
($0.09\cdot0.99\approx0.089$) produce almost ten times as many positive tests as
the few genuinely ill ones ($0.9\cdot0.01=0.009$). The prior must never be
ignored (*base rate fallacy*).
</details>

<details><summary><b>4. Explain the Bayesian network factorization and why it saves memory.</b></summary>

A BN asserts $P(x_1,\dots,x_n)=\prod_i P(x_i\mid\mathrm{parents}(x_i))$. That
follows from the local Markov condition (every variable is conditionally
independent of its non-descendants given its parents). Instead of the full joint
table with $2^n-1$ entries one stores only $2^k$ numbers per node ($k$ = the
number of parents), $n\cdot2^k$ in total. With a bounded $k$ that is linear
instead of exponential in $n$ — the entire point of Bayesian networks.
</details>

<details><summary><b>5. What is "explaining away"? State it in terms of d-separation.</b></summary>

At a collider $A\to C\leftarrow B$, $A$ and $B$ are **a priori independent** (the
unobserved collider blocks the path). If one observes $C$ (or a descendant), the
path **opens**: $A$ and $B$ become *conditionally dependent*. If $C$ has occurred
and one learns that $A$ explains it, the probability of $B$ falls — the one cause
"explains the other away". An example: the alarm ($C$) caused by a burglary ($A$)
or an earthquake ($B$); an earthquake report lowers
$P(\text{burglary}\mid\text{alarm})$.
</details>

<details><summary><b>6. Why is variable elimination faster than inference by enumeration, and what do the costs depend on?</b></summary>

Enumeration computes the same subproducts over and over again (the naive double
sum has exponentially many repeated factors). VE factors them out via the
distributive law and **stores intermediate factors**, so that every subproduct is
computed only once. The cost is dominated by the **largest intermediate factor**,
whose size depends on the **elimination order** and ultimately on the
**treewidth** of the graph. With a small treewidth (polytrees) it is polynomial;
in general BN inference is NP-hard.
</details>

<details><summary><b>7. When does one use likelihood weighting instead of rejection sampling?</b></summary>

Rejection sampling discards all samples that contradict the evidence — with
*rare* evidence almost everything ends up in the bin (exponentially
inefficient). Likelihood weighting instead fixes the evidence variables at their
values and **weights** every sample by $\prod_{e_i}P(e_i\mid\mathrm{parents}(e_i))$;
no sample is discarded. It is consistent and considerably more efficient, but it
can also have high variance when the evidence sits "far down" in the network —
then use Gibbs/MCMC.
</details>

<details><summary><b>8. Derive the filtering recursion of the HMM (predict/update).</b></summary>

We want $f_{1:t+1}=P(\mathbf X_{t+1}\mid\mathbf e_{1:t+1})$. Bayes with respect
to the new evidence: $\propto P(\mathbf e_{t+1}\mid\mathbf X_{t+1},\mathbf e_{1:t})\,P(\mathbf X_{t+1}\mid\mathbf e_{1:t})$.
By the sensor Markov assumption the first factor $=P(\mathbf e_{t+1}\mid\mathbf X_{t+1})$
(**update**). The second is the **prediction**: marginalize over $\mathbf X_t$,
$P(\mathbf X_{t+1}\mid\mathbf e_{1:t})=\sum_{\mathbf x_t}P(\mathbf X_{t+1}\mid\mathbf x_t)P(\mathbf x_t\mid\mathbf e_{1:t})$
(the Markov transition × the previous filter message). Together:
$f_{1:t+1}=\alpha\,P(\mathbf e_{t+1}\mid\mathbf X_{t+1})\sum_{\mathbf x_t}P(\mathbf X_{t+1}\mid\mathbf x_t)f_{1:t}$.
</details>

<details><summary><b>9. Prove that value iteration converges.</b></summary>

The Bellman optimality operator $B$ with $(BV)(s)=R(s)+\gamma\max_a\sum_{s'}P(s'\mid s,a)V(s')$
is a $\gamma$-contraction in the maximum norm: for arbitrary $V,V'$ we have
$\lVert BV-BV'\rVert_\infty\le\gamma\lVert V-V'\rVert_\infty$ (use
$|\max_a f-\max_a g|\le\max_a|f-g|$ and $\sum_{s'}P=1$). By the Banach fixed
point theorem, $B$ has a unique fixed point $V^\ast$, and the iteration
$V_{k+1}=BV_k$ converges geometrically: $\lVert V_k-V^\ast\rVert_\infty\le\gamma^k\lVert V_0-V^\ast\rVert_\infty$.
For $\gamma<1$ convergence follows; $\gamma\to1$ makes it arbitrarily slow.
</details>

<details><summary><b>10. Value iteration vs. policy iteration — advantages and disadvantages?</b></summary>

*Value iteration*: one cheap Bellman update over all states per step, but many
steps until convergence (geometrically in $\gamma$), and the policy often
stabilizes *before* the values have converged. *Policy iteration*: more expensive
per step (the policy evaluation solves an $|S|\times|S|$ system of equations),
but very few steps — it terminates exactly, in finitely many iterations, since
there are only finitely many policies and every iteration improves strictly. A
compromise: *modified* policy iteration (evaluating only approximately). Both are
instances of generalized policy iteration — the core of RL (module 13).
</details>

---

## Literature and sources

**Textbooks**
- **Russell & Norvig, *AIMA*, 4th ed.** — ch. 11 (classical planning), 12
  (planning in the real world), 13 (quantifying uncertainty), 14 (probabilistic
  reasoning/Bayesian networks), 15 (temporal models), 16 (simple decisions),
  17 (complex decisions/MDPs). *The primary source for this module.*
- **Koller & Friedman, *Probabilistic Graphical Models*, MIT Press** — the
  exhaustive reference on Bayesian networks, inference and learning. *Advanced, demanding.*
- **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.** — ch. 3–4
  (MDPs, dynamic programming) as the perfect deepening of the MDP part and a
  bridge to module 13. **Free** at `incompleteideas.net/book/the-book.html`. *Highly recommended.*
- **Ghallab, Nau & Traverso, *Automated Planning and Acting***, for the planning part. *Advanced.*

**Freely available courses and materials** (free)
- **UC Berkeley CS188** — the units on Bayesian networks, HMMs and MDPs with the
  Pac-Man projects (`inst.eecs.berkeley.edu/~cs188`). *Beginner friendly, practical.*
- **Stanford CS228 "Probabilistic Graphical Models"** — notes online, `ermongroup.github.io/cs228-notes`. *Advanced.*
- **David Silver, *RL Course* (DeepMind/UCL)** — lecture videos; lectures 2–3 on
  MDPs and dynamic programming. *Very good for the MDP part.*
- **Fast Downward / the PDDL editor** (`editor.planning.domains`) — write PDDL in
  the browser and run a real planner. *Practical.*

**Interactive / visualizations** (free)
- **"Seeing Theory"** (`seeing-theory.brown.edu`) — interactive probability and Bayes. *Beginner friendly.*
- **Bayesian network demos** (e.g. `github.com/mbilalzonjy/BayesNetVisualization`) and the **SamIam** reasoner to play with.
- **Gridworld MDP visualizations** (value/policy iteration step by step), e.g. Andrej Karpathy's `reinforcejs`.

**Classical papers** (free, advanced)
- Pearl (1988): *Probabilistic Reasoning in Intelligent Systems* — the birth of Bayesian networks.
- Blum & Furst (1997): *Fast Planning Through Planning Graph Analysis* — GraphPlan.
- Hoffmann & Nebel (2001): *The FF Planning System* — the $h_{\text{FF}}$ heuristic.

---

## The three projects

The three projects mirror the three parts of the module — planning,
probabilistic reasoning, sequential decision making — and increase in difficulty
and in the amount of your own work:

- **01 – basic** (`projects/01-basic/`): **a STRIPS forward planner.** A guided
  notebook: STRIPS states/actions, forward search with BFS *and* with a
  relaxation heuristic ($h_{\text{add}}$) via A\*; applied to the blocks world.
  Plenty of guidance, it ties in directly with module 06.
- **02 – medium** (`projects/02-medium/`): **a Bayesian network with exact and
  approximate inference.** A Python project: the network structure + CPTs,
  inference by enumeration *and* variable elimination *and* likelihood
  weighting; validated on the alarm network and a diagnostic scenario. Little
  guidance.
- **03 – final** (`projects/03-final/`): **a decision-theoretic MDP agent.** No
  given code: value iteration *and* policy iteration on a gridworld with
  stochastic movement, checking convergence empirically, visualizing the optimal
  policy, a $\gamma$ study. Master's level, a bridge to RL.

Details, setup and reference solutions are in the `README.md` of each project folder.

---
---

# Modul 07 — Theorie der Künstlichen Intelligenz 2 (deutsche Fassung)

**Worum geht es?** Modul 06 hat gezeigt, wie ein Agent in einer *bekannten,
deterministischen* Welt durch Suche und Logik handelt. Dieses Modul macht drei
Schritte weiter: (1) **Planung** — Handeln in strukturierten, aber weiterhin
deterministischen Welten mit einer kompakten, logik-nahen Repräsentation von
Aktionen; (2) **Schließen unter Unsicherheit** — wenn die Welt nicht mehr
sicher bekannt ist, ersetzen wir „wahr/falsch" durch **Wahrscheinlichkeiten**
und rechnen mit **Bayes-Netzen**; (3) **rationales Entscheiden** — wenn Aktionen
unsichere Ausgänge haben, maximiert der Agent den **erwarteten Nutzen**, im
sequenziellen Fall über **Markov-Entscheidungsprozesse (MDPs)**. Das ist der
Übergang von der *symbolischen* zur *probabilistischen* und
*entscheidungstheoretischen* KI — und die theoretische Brücke zu Reinforcement
Learning (Module 13/14).

**Hilfreiche Vorkenntnisse.** Modul 06 (Suche, Logik — Planung baut direkt
darauf auf). Grundlagen der Wahrscheinlichkeitsrechnung und lineare Algebra
(Module 02/03 genügen). Etwas Analysis für die Konvergenzbeweise (Kontraktion).

**Empfohlene Vormodule.** Modul 06 „Theorie der KI 1" (zwingend fürs Planungs­kapitel),
Data Science 1/2 für die Wahrscheinlichkeitsbasics.

**Folgemodule.** Reinforcement Learning (13) und Deep RL (14) bauen unmittelbar
auf dem MDP-Teil auf. Bayes-Netze tauchen in ML, NLP und Bioinformatik wieder auf.

---

## Lernziele

Nach diesem Modul solltest du in der Lage sein,

- ein Planungsproblem in **STRIPS/PDDL** zu formalisieren und den Unterschied
  zwischen **Vorwärts-(Progression)-, Rückwärts-(Regression)- und Plan-Raum-Suche**
  zu erklären;
- **domänenunabhängige Heuristiken** aus der **Relaxation** (Ignorieren der
  Delete-Listen, $h_{\text{add}}$, $h_{\max}$, $h_{\text{FF}}$) herzuleiten;
- **GraphPlan** (Planungsgraph, Mutex-Relationen) und **SATPlan** als alternative
  Planungsparadigmen einzuordnen;
- die **Axiome der Wahrscheinlichkeit**, bedingte Wahrscheinlichkeit, den **Satz
  von Bayes** und **(bedingte) Unabhängigkeit** sicher anzuwenden;
- ein **Bayes-Netz** zu konstruieren, seine **Faktorisierung der Verbundverteilung**
  zu begründen und mit **d-Separation** Unabhängigkeiten abzulesen;
- **exakte Inferenz** (Aufzählung, **Variable Elimination**) und **approximative
  Inferenz** (Likelihood Weighting, **Gibbs-Sampling/MCMC**) durchzuführen;
- **temporale Modelle** (Markov-Kette, **HMM**) zu verstehen und **Filtering,
  Smoothing** und den **Viterbi-Algorithmus** herzuleiten;
- **rationale Entscheidungen** über den **Maximum-Expected-Utility (MEU)**-Grundsatz
  zu treffen, den **Wert von Information (VPI)** zu berechnen;
- **MDPs** zu definieren, die **Bellman-Gleichungen** aufzustellen und **Value
  Iteration** und **Policy Iteration** samt **Konvergenzbeweis (Kontraktion)** zu
  verstehen;
- den Zusammenhang zwischen Planung, probabilistischem Schließen und
  sequenzieller Entscheidung zu erklären.

---

## Teil 1 — Grundlagen: Klassische Planung

### 1.1 Warum Planung statt reiner Suche?

Man *könnte* jedes Planungsproblem als Suchproblem (Modul 06) auffassen. Das
Problem: Bei einer Welt mit vielen Objekten und Prädikaten ist der Zustandsraum
astronomisch, und eine reine Zustandssuche „sieht" die **Struktur** der Aktionen
nicht. **Planung** nutzt eine **faktorisierte, logik-nahe Repräsentation** von
Zuständen und Aktionen. Dadurch kann man (a) **domänenunabhängige Heuristiken**
automatisch aus der Aktionsbeschreibung ableiten und (b) Aktionen ausnutzen, die
unabhängig voneinander sind.

### 1.2 STRIPS und PDDL

Im **STRIPS**-Formalismus (Stanford Research Institute Problem Solver) ist ein
**Zustand** eine Menge von **Grundfluenten** (variablenfreie, wahre atomare
Aussagen) unter der **Closed-World-Assumption**: Was nicht in der Menge steht,
gilt als falsch. Ein **Planungsproblem** ist $(\mathcal{F}, s_0, g, \mathcal{A})$ mit

- $\mathcal{F}$: Menge aller Fluenten,
- $s_0 \subseteq \mathcal{F}$: **Startzustand**,
- $g \subseteq \mathcal{F}$: **Zielbedingung** (ein Zustand $s$ erfüllt das Ziel, wenn $g \subseteq s$),
- $\mathcal{A}$: Menge der **Aktionsschemata**. Eine Grundaktion $a$ besteht aus
  - $\mathrm{PRE}(a)$ — **Vorbedingungen** (Fluenten, die gelten müssen),
  - $\mathrm{ADD}(a)$ — **Add-Liste** (Fluenten, die $a$ wahr macht),
  - $\mathrm{DEL}(a)$ — **Delete-Liste** (Fluenten, die $a$ falsch macht).

Eine Aktion $a$ ist in $s$ **anwendbar**, wenn $\mathrm{PRE}(a) \subseteq s$. Das
**Übergangsmodell** (Progression) ist rein mengentheoretisch:
$$
\mathrm{Result}(s, a) = (s \setminus \mathrm{DEL}(a)) \cup \mathrm{ADD}(a).
$$

**PDDL** (Planning Domain Definition Language) ist die heute übliche Syntax dafür;
sie trennt eine **Domänendatei** (Prädikate, Aktionsschemata mit Variablen) von
einer **Problemdatei** (Objekte, Start, Ziel). Aktionsschemata mit Variablen
werden vor der Suche zu Grundaktionen instanziiert (*grounding*).

> **Beispiel (Blocksworld).** Fluenten: `On(x,y)`, `OnTable(x)`, `Clear(x)`,
> `Holding(x)`, `ArmEmpty`. Aktion `PickUp(x)`:
> $\mathrm{PRE} = \{\mathrm{Clear}(x), \mathrm{OnTable}(x), \mathrm{ArmEmpty}\}$,
> $\mathrm{ADD} = \{\mathrm{Holding}(x)\}$,
> $\mathrm{DEL} = \{\mathrm{Clear}(x), \mathrm{OnTable}(x), \mathrm{ArmEmpty}\}$.

Der Charme: Die **Frame-Problematik** (was ändert sich *nicht*?) ist elegant
gelöst — alles, was nicht in ADD/DEL steht, bleibt unverändert.

### 1.3 Vorwärts-, Rückwärts- und Plan-Raum-Suche

**Progression (Vorwärtssuche)** durchsucht den Zustandsraum von $s_0$ Richtung
Ziel mit den Verfahren aus Modul 06 (meist A\* mit einer der Heuristiken unten).
Vorteil: einfache Zustände; Nachteil: hoher Verzweigungsfaktor (viele anwendbare
Aktionen).

**Regression (Rückwärtssuche)** startet beim Ziel $g$ und arbeitet rückwärts.
Der Regressionsschritt: Um Zustandsbeschreibung $g'$ nach Anwendung von $a$ zu
erreichen, muss vorher gelten
$$
\mathrm{Regress}(g', a) = (g' \setminus \mathrm{ADD}(a)) \cup \mathrm{PRE}(a),
\quad \text{sofern } a \text{ relevant ist } (\mathrm{ADD}(a)\cap g' \neq \emptyset)
\text{ und konsistent } (\mathrm{DEL}(a) \cap g' = \emptyset).
$$
Vorteil: nur **relevante** Aktionen werden betrachtet (kleiner Verzweigungsfaktor);
Nachteil: Zustände sind *Teilbeschreibungen* (Mengen von Bedingungen), gutes
Heuristik-Design ist schwerer.

**Plan-Raum-Suche / Partial-Order Planning (POP).** Statt im Zustandsraum sucht
man im Raum *partieller Pläne*. Ein partieller Plan besteht aus (i) einer Menge
von Aktionen, (ii) **Ordnungsbeschränkungen** $a \prec b$ („$a$ vor $b$"),
(iii) **kausalen Links** $a \xrightarrow{p} b$ („$a$ stellt Vorbedingung $p$ von
$b$ bereit") und (iv) offenen Vorbedingungen. Man verfeinert den Plan, bis keine
offene Vorbedingung mehr existiert und alle **Bedrohungen (threats)** aufgelöst
sind. Eine Aktion $c$ **bedroht** einen Link $a \xrightarrow{p} b$, wenn $c$ das
$p$ löscht und zwischen $a$ und $b$ liegen könnte; man löst das durch
**Promotion** ($c \prec a$) oder **Demotion** ($b \prec c$). POP erzeugt **partiell
geordnete Pläne** — es committet sich nicht vorschnell auf eine Reihenfolge
unabhängiger Aktionen (**least commitment**).

### 1.4 Planungsheuristiken aus Relaxation

Der Durchbruch der modernen Planung: **domänenunabhängige** Heuristiken, die
automatisch aus der STRIPS-Beschreibung entstehen — analog zur Relaxation in
Modul 06, nur systematisch.

**Delete-Relaxation.** Man streicht *alle* Delete-Listen: $\mathrm{DEL}(a) =
\emptyset$. In der relaxierten Welt kann eine einmal erreichte Eigenschaft nie
wieder verloren gehen (monotoner Fluenten-Zuwachs), das relaxierte Problem ist in
**Polynomzeit** lösbar. Aus ihm gewinnt man:

- $h_{\text{add}}(s) = \sum_{p \in g}\, \Delta(p)$ — Summe der geschätzten Kosten,
  jedes Ziel-Fluent einzeln zu erreichen (überschätzt bei Interaktion → **nicht
  zulässig**, aber informativ).
- $h_{\max}(s) = \max_{p \in g}\, \Delta(p)$ — teuerstes einzelnes Ziel-Fluent
  (**zulässig**, aber oft schwach).
- $h_{\text{FF}}$ (Fast Forward): extrahiert einen konkreten **relaxierten Plan**
  aus dem relaxierten Planungsgraphen und nimmt dessen Länge — meist die beste
  praktische Heuristik (nicht zulässig, aber sehr treffsicher).

Dabei ist $\Delta(p)$ rekursiv definiert: $\Delta(p)=0$, wenn $p\in s$, sonst
$\Delta(p) = \min_{a:\, p\in \mathrm{ADD}(a)} \big(\mathrm{cost}(a) + \text{Kombi}_{q\in \mathrm{PRE}(a)}\Delta(q)\big)$,
wobei $\text{Kombi}=\sum$ für $h_{\text{add}}$ und $\text{Kombi}=\max$ für $h_{\max}$.

### 1.5 GraphPlan und SATPlan

**GraphPlan** baut einen geschichteten **Planungsgraphen** aus abwechselnden
**Zustandsebenen** ($S_0, S_1, \dots$) und **Aktionsebenen** ($A_0, A_1, \dots$)
auf. Jede Ebene enthält *alle* Fluenten/Aktionen, die *möglicherweise* erreichbar
sind, plus **Mutex-Relationen** (mutual exclusion):
- Zwei **Aktionen** sind mutex, wenn eine die Vorbedingung/Wirkung der anderen
  löscht (*inconsistent effects*, *interference*) oder ihre Vorbedingungen mutex
  sind (*competing needs*).
- Zwei **Fluenten** sind mutex, wenn jede sie erzeugende Aktion mutex ist
  (*inconsistent support*).

Der Graph wächst, bis alle Ziel-Fluenten nicht-mutex in einer Ebene erscheinen
(*level off*); dann extrahiert eine Rückwärts-Suche einen Plan. Der Graph liefert
zugleich die zulässige Heuristik „erste Ebene, in der ein Ziel-Fluent auftaucht".

**SATPlan.** Kodiere „existiert ein Plan der Länge $\le k$?" als **aussagenlogische
Formel** (Fluenten und Aktionen mit Zeitindex, Vorbedingungs-/Wirkungs- und
Frame-Axiome) und wirf einen **SAT-Solver** (Modul 06, DPLL/CDCL) darauf; erhöhe
$k$ iterativ. Zeigt eindrucksvoll, wie Modul-06-Technik hier wiederverwendet wird.

---

## Teil 2 — Aufbau: Schließen unter Unsicherheit

### 2.1 Warum Wahrscheinlichkeit?

Reale Agenten kennen die Welt nicht sicher: Sensoren rauschen, Aktionen misslingen,
Wissen ist unvollständig. Rein logisches Schließen scheitert, weil man alle
Ausnahmen aufzählen müsste („qualification problem"). **Wahrscheinlichkeit** fasst
diese Unsicherheit in *einer Zahl* pro Aussage — als **Grad der Überzeugung**
(bayesianische Sicht), nicht notwendig als Häufigkeit. Der Satz von de Finetti
zeigt: Wer Wetten abschließt, deren Quoten *nicht* den Wahrscheinlichkeitsaxiomen
gehorchen, ist durch ein **Dutch Book** garantiert ausbeutbar — Rationalität
*erzwingt* die Wahrscheinlichkeitsrechnung.

### 2.2 Grundlagen: Axiome, Bedingtheit, Bayes

Für Ereignisse gelten die **Kolmogorov-Axiome**:
$$
0 \le P(a) \le 1, \quad P(\text{wahr}) = 1, \quad
P(a \lor b) = P(a) + P(b) - P(a \land b).
$$
Die **Verbundverteilung (joint distribution)** $P(X_1, \dots, X_n)$ über alle
Variablen legt *alles* fest. Aus ihr gewinnt man jede Frage durch:
- **Marginalisierung (summing out):** $P(\mathbf{Y}) = \sum_{\mathbf{z}} P(\mathbf{Y}, \mathbf{z})$,
- **Konditionierung:** $\displaystyle P(a \mid b) = \frac{P(a \land b)}{P(b)}$ (für $P(b) > 0$).

Umstellen der Definition liefert die **Produktregel** $P(a\land b) = P(a\mid b)\,P(b)$
und, durch Gleichsetzen, den **Satz von Bayes**:
$$
\boxed{\,P(h \mid e) = \frac{P(e \mid h)\,P(h)}{P(e)}\,}
$$
das Arbeitspferd: aus dem **Likelihood** $P(e\mid h)$ (wie wahrscheinlich ist die
Evidenz unter der Hypothese) und dem **Prior** $P(h)$ wird der **Posterior**
$P(h\mid e)$. Der Nenner $P(e) = \sum_{h'} P(e\mid h')P(h')$ ist die
**Normierungskonstante**; oft schreibt man $P(h\mid e) = \alpha\, P(e\mid h)P(h)$.

> **Durchgerechnet (medizinischer Test).** Krankheit mit Prävalenz $P(D)=0{,}01$;
> Test mit Sensitivität $P(+\mid D)=0{,}9$ und Falsch-Positiv-Rate $P(+\mid \lnot D)=0{,}09$.
> Positiver Test — wie wahrscheinlich krank?
> $$P(D\mid +) = \frac{0{,}9 \cdot 0{,}01}{0{,}9\cdot 0{,}01 + 0{,}09\cdot 0{,}99}
> = \frac{0{,}009}{0{,}009 + 0{,}0891} \approx 0{,}092.$$
> Nur ~9\,%! Die niedrige **Basisrate** dominiert — der klassische *base rate
> fallacy*. Ein Musterbeispiel, warum man Priors nicht ignorieren darf.

**(Bedingte) Unabhängigkeit.** $X$ und $Y$ sind **unabhängig**, wenn
$P(X,Y)=P(X)P(Y)$. Wichtiger noch ist **bedingte Unabhängigkeit**: $X \perp Y \mid Z$
gdw. $P(X,Y\mid Z) = P(X\mid Z)\,P(Y\mid Z)$. Sie ist der Schlüssel, der die
exponentiell große Verbundverteilung **kompakt faktorisierbar** macht.

### 2.3 Bayes-Netze: Struktur und Semantik

Eine volle Verbundverteilung über $n$ booleschen Variablen braucht $2^n - 1$
Zahlen — unhandhabbar. Ein **Bayes-Netz (Bayesian Network)** nutzt bedingte
Unabhängigkeiten, um dieselbe Verteilung *kompakt* darzustellen:

- ein **gerichteter azyklischer Graph (DAG)**, Knoten = Zufallsvariablen, Kante
  $X \to Y$ = „$X$ beeinflusst $Y$ direkt",
- pro Knoten eine **bedingte Wahrscheinlichkeitstabelle (CPT)** $P(X_i \mid \mathrm{Parents}(X_i))$.

**Die zentrale Semantik (Kettenregel für Bayes-Netze):** Das Netz repräsentiert
die Verbundverteilung als **Produkt der lokalen CPTs**:
$$
\boxed{\,P(x_1, \dots, x_n) = \prod_{i=1}^{n} P\big(x_i \mid \mathrm{parents}(x_i)\big)\,}
$$
Das gilt genau dann, wenn jede Variable **bedingt unabhängig von ihren
Nicht-Nachkommen gegeben ihre Eltern** ist (die *lokale Markov-Bedingung*). Bei
begrenzter Elternzahl $k$ schrumpft der Speicher von $2^n$ auf $n\cdot 2^k$ —
oft von astronomisch auf handhabbar.

> **Klassiker (Alarm-Netz, Pearl).** Ein Einbruch ($B$) oder ein Erdbeben ($E$)
> kann einen Alarm ($A$) auslösen; John ($J$) und Mary ($M$) rufen an, je nachdem,
> ob sie den Alarm hören. Struktur: $B\to A \leftarrow E$, $A\to J$, $A\to M$.
> Verbund: $P(B,E,A,J,M) = P(B)P(E)P(A\mid B,E)P(J\mid A)P(M\mid A)$ — fünf kleine
> CPTs statt einer 32-Zeilen-Tabelle.

### 2.4 d-Separation — Unabhängigkeit aus der Graphstruktur ablesen

Wann gilt $X \perp Y \mid Z$ *allein aufgrund der Struktur*? Das beantwortet
**d-Separation**. Betrachte jeden ungerichteten Pfad zwischen $X$ und $Y$; der
Pfad ist **blockiert**, wenn er einen Knoten $n$ folgender Art enthält:

1. **Kette** $\to n \to$ oder **Gabel** $\leftarrow n \to$, und $n \in Z$
   (beobachtet) → blockiert.
2. **Kollider** $\to n \leftarrow$ („v-Struktur"), und **weder $n$ noch ein
   Nachkomme von $n$** ist in $Z$ → blockiert.

Sind *alle* Pfade blockiert, gilt $X \perp Y \mid Z$. Die Kollider-Regel ist der
subtile Punkt: Ein unbeobachteter Kollider *blockiert*, aber **Beobachtung des
Kolliders (oder eines Nachkommen) öffnet** ihn — das ist das **„explaining away"**:
Erdbeben und Einbruch sind a priori unabhängig, aber *gegeben den Alarm* werden
sie abhängig (hört man vom Erdbeben, sinkt die Einbruchswahrscheinlichkeit).

### 2.5 Exakte Inferenz

Die **Inferenzaufgabe**: Berechne $P(\mathbf{X}_{\text{query}} \mid \mathbf{e})$
für Anfragevariablen gegeben Evidenz $\mathbf{e}$.

**Inferenz durch Aufzählung.** Direkt aus der Verbund-Faktorisierung:
$$
P(X \mid \mathbf{e}) = \alpha \sum_{\mathbf{y}} P(X, \mathbf{e}, \mathbf{y}),
$$
wobei $\mathbf{y}$ die *versteckten* Variablen sind und das Produkt der CPTs
eingesetzt wird. Korrekt, aber $O(2^n)$ — die naive Summe wiederholt Teilprodukte.

**Variable Elimination (VE).** Beschleunigt die Aufzählung durch **Ausklammern**
(Distributivgesetz) und **Zwischenspeichern**. Man arbeitet mit **Faktoren**
(mehrdimensionale Tabellen). Zwei Operationen:
- **punktweises Produkt** zweier Faktoren $f_1 \times f_2$,
- **Ausmarginalisieren (summing out)** einer Variablen: $\sum_x f(\dots, x, \dots)$.

Algorithmus: Wähle eine **Eliminationsreihenfolge** der versteckten Variablen; für
jede: multipliziere alle Faktoren, die sie enthalten, und summiere sie heraus.
$$
P(B\mid j,m) = \alpha\, P(B) \sum_e P(e) \sum_a P(a\mid B,e)\,P(j\mid a)\,P(m\mid a).
$$
VE ist dramatisch schneller als Aufzählung, aber die Kosten hängen stark von der
Reihenfolge ab (der größte Zwischenfaktor bestimmt sie — die **Baumweite** des
Graphen). **Allgemeine Bayes-Netz-Inferenz ist NP-schwer**; für Netze mit kleiner
Baumweite (z. B. Polybäume) ist sie polynomiell.

### 2.6 Approximative Inferenz durch Sampling

Wenn exakte Inferenz zu teuer ist, **schätzt** man $P(\mathbf{X}\mid\mathbf{e})$ aus
Stichproben.

- **Direct/Prior Sampling:** Ziehe Werte topologisch geordnet gemäß den CPTs →
  Stichproben aus der Verbundverteilung.
- **Rejection Sampling:** Wie oben, aber verwirf alle Stichproben, die $\mathbf{e}$
  widersprechen. Korrekt, aber verschwenderisch bei seltener Evidenz.
- **Likelihood Weighting:** Fixiere die Evidenzvariablen auf ihre beobachteten
  Werte und **gewichte** jede Stichprobe mit dem Produkt der Evidenz-Likelihoods
  $\prod_{e_i} P(e_i \mid \mathrm{parents}(e_i))$. Verwirft nichts, effizienter.
- **Gibbs-Sampling (MCMC):** Ein **Markov-Chain-Monte-Carlo**-Verfahren. Fixiere
  Evidenz, initialisiere die übrigen Variablen beliebig; resample dann wiederholt
  *eine* Nicht-Evidenz-Variable aus ihrer Verteilung gegeben ihre **Markov-Decke**
  (Eltern, Kinder, Ko-Eltern). Die so erzeugte Kette hat die Posterior-Verteilung
  als **stationäre Verteilung**; Stichprobenmittel konvergieren gegen $P(\mathbf{X}\mid\mathbf{e})$.

Alle Sampling-Verfahren sind **konsistent** (Fehler $\to 0$ mit $N\to\infty$, Rate
$O(1/\sqrt N)$), tauschen also Exaktheit gegen Rechenzeit — der übliche Deal bei
schwierigen Netzen.

### 2.7 Temporale Modelle: Markov-Ketten und HMMs

Die Welt ändert sich über die Zeit. Ein **zeitdiskretes** Modell hat pro Schritt
$t$ Zustandsvariablen $\mathbf{X}_t$ und Evidenzvariablen $\mathbf{E}_t$. Zwei
Annahmen machen es handhabbar:

- **Markov-Annahme (erster Ordnung):** $P(\mathbf{X}_t \mid \mathbf{X}_{0:t-1}) =
  P(\mathbf{X}_t \mid \mathbf{X}_{t-1})$ — die Zukunft hängt nur über den
  **aktuellen** Zustand von der Vergangenheit ab. → **Übergangsmodell**.
- **Sensor-Markov-Annahme:** $P(\mathbf{E}_t \mid \mathbf{X}_{0:t}, \mathbf{E}_{0:t-1})
  = P(\mathbf{E}_t \mid \mathbf{X}_t)$. → **Sensormodell**.

Ein **Hidden Markov Model (HMM)** hat eine einzelne diskrete Zustandsvariable, die
man nicht direkt sieht (*hidden*), nur über Evidenz. Vier Standardaufgaben:

**Filtering** — $P(\mathbf{X}_t \mid \mathbf{e}_{1:t})$ (aktueller Zustand gegeben
alle bisherigen Beobachtungen). Rekursiv (**Forward-Algorithmus**):
$$
P(\mathbf{X}_{t+1}\mid \mathbf{e}_{1:t+1}) = \alpha\, \underbrace{P(\mathbf{e}_{t+1}\mid \mathbf{X}_{t+1})}_{\text{Update (Sensor)}}
\sum_{\mathbf{x}_t} \underbrace{P(\mathbf{X}_{t+1}\mid \mathbf{x}_t)}_{\text{Predict (Transition)}} P(\mathbf{x}_t\mid \mathbf{e}_{1:t}).
$$
„Predict-Update"-Schleife — dieselbe Idee steckt im **Kalman-Filter** (der
stetig-gaußsche Spezialfall) und in der Lokalisierung von Robotern (Modul 21).

**Prediction** — $P(\mathbf{X}_{t+k}\mid\mathbf{e}_{1:t})$ (Zukunft ohne neue Evidenz).

**Smoothing** — $P(\mathbf{X}_k\mid\mathbf{e}_{1:t})$ für $k<t$ (Vergangenheit mit
Rückschau). Der **Forward-Backward-Algorithmus** kombiniert die Vorwärtsnachricht
mit einer rückwärts laufenden Nachricht $b_{k+1:t} = P(\mathbf{e}_{k+1:t}\mid\mathbf{X}_k)$.

**Wahrscheinlichste Erklärung** — $\arg\max_{\mathbf{x}_{1:t}} P(\mathbf{x}_{1:t}\mid\mathbf{e}_{1:t})$.
Der **Viterbi-Algorithmus** ist dynamische Programmierung: Er ersetzt in der
Vorwärtsrekursion die Summe durch ein **Maximum** und merkt sich Rückzeiger, um den
besten Pfad zu rekonstruieren. Basis von Spracherkennung, POS-Tagging (Modul 08),
Bioinformatik-Sequenzanalyse (Modul 28).

---

## Teil 3 — Advanced: Rationale Entscheidungen und MDPs

### 3.1 Nutzentheorie und Maximum Expected Utility

Bisher haben wir *geschlossen*, jetzt *handeln* wir. Die **Nutzentheorie** (von
Neumann & Morgenstern) zeigt: Erfüllt eine Präferenzrelation über unsichere
Ausgänge („Lotterien") sechs **Rationalitätsaxiome** (Vollständigkeit,
Transitivität, Stetigkeit, Substituierbarkeit, Monotonie, Zerlegbarkeit), dann
**existiert eine Nutzenfunktion** $U$, sodass der Agent Lotterie $L_1$ genau dann
$L_2$ vorzieht, wenn $\mathrm{EU}(L_1) > \mathrm{EU}(L_2)$, mit dem **erwarteten
Nutzen**
$$
\mathrm{EU}(a\mid \mathbf{e}) = \sum_{s'} P(\mathrm{Result}(a) = s' \mid a, \mathbf{e})\; U(s').
$$
Das **MEU-Prinzip (Maximum Expected Utility):** Ein rationaler Agent wählt die
Aktion, die den erwarteten Nutzen maximiert:
$a^\ast = \arg\max_a \mathrm{EU}(a\mid\mathbf{e})$. Wichtig: Nutzen ist **nicht**
gleich Geld — die typischerweise **konkave** Nutzenfunktion für Geld erklärt
**Risikoaversion** (eine sichere 100 € kann mehr Nutzen haben als eine 50/50-Chance
auf 0/220 €).

### 3.2 Entscheidungsnetze und der Wert von Information

**Entscheidungsnetze (Influence Diagrams)** erweitern Bayes-Netze um
**Entscheidungsknoten** (Aktionen, die der Agent wählt) und einen **Nutzenknoten**.
Die Auswertung wählt die Aktionen mit maximalem erwarteten Nutzen.

**Value of Perfect Information (VPI).** Lohnt es sich, *vor* der Entscheidung eine
Variable $E_j$ zu messen? Der Informationswert ist die erwartete Nutzensteigerung:
$$
\mathrm{VPI}_{\mathbf{e}}(E_j) = \Big(\sum_{e_{j}} P(e_{j}\mid\mathbf{e})\;
\mathrm{EU}(a^\ast_{e_{j}} \mid \mathbf{e}, e_{j})\Big) - \mathrm{EU}(a^\ast \mid \mathbf{e}).
$$
VPI ist **nie negativ** (mehr Wissen kann im Erwartungswert nicht schaden) und
**nicht additiv**. Es liefert die theoretische Grundlage für *rationale
Informationsbeschaffung* — welchen Sensor/Test man ansteuert.

### 3.3 Markov-Entscheidungsprozesse (MDPs)

Jetzt der **sequenzielle** Fall: Entscheidungen über viele Schritte, mit
unsicheren Ausgängen. Ein **MDP** ist $(S, A, P, R, \gamma)$:

- $S$ Zustände, $A$ Aktionen,
- $P(s' \mid s, a)$ **Übergangsmodell** (stochastisch! — hier der Bruch mit der
  klassischen Planung),
- $R(s)$ (oder $R(s,a,s')$) **Belohnung**,
- $\gamma \in [0,1)$ **Diskontfaktor** (spätere Belohnungen zählen weniger; sichert
  auch Konvergenz bei unendlichem Horizont).

Gesucht ist eine **Policy** $\pi: S \to A$, die den erwarteten **diskontierten
Return** $\mathbb{E}\big[\sum_{t=0}^{\infty}\gamma^t R(s_t)\big]$ maximiert. Der
**Wert** eines Zustands unter $\pi$ ist $V^\pi(s) = \mathbb{E}\big[\sum_t \gamma^t
R(s_t) \mid s_0=s, \pi\big]$.

**Die Bellman-Gleichung** charakterisiert $V^\pi$ selbstkonsistent:
$$
V^\pi(s) = R(s) + \gamma \sum_{s'} P(s'\mid s, \pi(s))\, V^\pi(s').
$$
Für die **optimale** Policy $\pi^\ast$ gilt die **Bellman-Optimalitätsgleichung**:
$$
\boxed{\,V^\ast(s) = R(s) + \gamma \max_{a} \sum_{s'} P(s'\mid s, a)\, V^\ast(s')\,}
$$
und die optimale Policy liest man greedy ab:
$\pi^\ast(s) = \arg\max_a \sum_{s'} P(s'\mid s,a)\,V^\ast(s')$.

### 3.4 Value Iteration und Policy Iteration

**Value Iteration.** Fasse die Bellman-Optimalitätsgleichung als **Update** auf und
iteriere bis zur Konvergenz:
$$
V_{k+1}(s) \leftarrow R(s) + \gamma \max_a \sum_{s'} P(s'\mid s,a)\, V_k(s).
$$

> **Warum konvergiert das? (Kontraktionsbeweis.)** Der **Bellman-Optimalitäts­operator**
> $B$, definiert durch $(BV)(s) = R(s) + \gamma\max_a\sum_{s'}P(s'\mid s,a)V(s')$,
> ist eine **Kontraktion** bzgl. der Maximumsnorm $\lVert V\rVert_\infty = \max_s|V(s)|$
> mit Faktor $\gamma$: für beliebige $V, V'$ gilt
> $$\lVert BV - BV'\rVert_\infty \le \gamma\,\lVert V - V'\rVert_\infty.$$
> *Beweisskizze:* $|\max_a f(a) - \max_a g(a)| \le \max_a|f(a)-g(a)|$, und
> $\gamma\sum_{s'}P(s'\mid s,a)|V(s')-V'(s')| \le \gamma\lVert V-V'\rVert_\infty$, da
> $\sum_{s'}P=1$. Nach dem **Banachschen Fixpunktsatz** hat $B$ einen **eindeutigen
> Fixpunkt** $V^\ast$, und $V_k \to V^\ast$ **geometrisch**: $\lVert V_k - V^\ast\rVert_\infty
> \le \gamma^k \lVert V_0 - V^\ast\rVert_\infty$. Für $\gamma<1$ garantiert das
> Konvergenz — je näher $\gamma$ an 1, desto langsamer. $\quad\blacksquare$

**Policy Iteration.** Alterniert zwei Schritte bis zur Stabilität der Policy:
1. **Policy Evaluation:** Löse $V^{\pi}(s) = R(s) + \gamma\sum_{s'}P(s'\mid s,\pi(s))V^\pi(s')$
   — ein **lineares Gleichungssystem** in $|S|$ Unbekannten (exakt lösbar oder
   iterativ genähert).
2. **Policy Improvement:** Setze
   $\pi'(s) \leftarrow \arg\max_a \sum_{s'}P(s'\mid s,a)V^\pi(s')$.

Ändert sich die Policy nicht mehr, ist sie optimal. Policy Iteration konvergiert in
**endlich vielen** Schritten (es gibt nur endlich viele Policies, und jeder Schritt
verbessert strikt oder terminiert) — oft in *sehr wenigen* Iterationen, dafür ist
jede teurer als ein Value-Iteration-Schritt. Beide sind Spezialfälle der
**generalisierten Policy-Iteration**, dem konzeptionellen Kern des
Reinforcement Learning (Modul 13): Der Unterschied dort ist nur, dass $P$ und $R$
**unbekannt** sind und aus Erfahrung gelernt werden müssen.

### 3.5 Partielle Beobachtbarkeit (POMDPs) — Ausblick

Kann der Agent den Zustand nicht direkt beobachten (nur verrauschte Sensoren),
wird das MDP zum **POMDP**. Der Trick: Der Agent hält einen **belief state** $b(s)$
(eine Wahrscheinlichkeitsverteilung über Zustände, per Filtering aus Abschnitt 2.7
aktualisiert) und löst ein MDP im *kontinuierlichen* Belief-Raum. POMDPs sind
theoretisch elegant, aber exakte Lösung ist **PSPACE-hart** — praktisch nutzt man
Approximationen. Sie verbinden Filtering (Teil 2) mit MDPs (Teil 3) zum
vollständigen Bild des rationalen Agenten unter Unsicherheit.

### 3.6 Ausblick: Nichtmonotones Schließen und Beschreibungslogiken

Zwei weitere Antworten auf Unsicherheit — nicht probabilistisch, sondern
**qualitativ**:

- **Nichtmonotones Schließen.** Klassische Logik ist *monoton*: Mehr Prämissen →
  nie weniger Schlüsse. Alltagsschlüsse sind aber **defeasible** („Vögel fliegen —
  aber nicht Pinguine"). **Default-Logik**, **Circumscription** und
  **Answer-Set-Programming** formalisieren solche Standardannahmen, die neue
  Information *zurücknehmen* kann. Grundlage der Wissensrepräsentation und von
  Logikprogrammierung mit Negation (Modul 33).
- **Beschreibungslogiken (Description Logics, DL).** Das entscheidbare
  FOL-Fragment hinter **Ontologien** und dem **Semantic Web** (OWL). Eine
  **TBox** definiert Konzepte/Rollen ($\text{Vater} \equiv \text{Mann} \sqcap
  \exists\text{hatKind}.\top$), eine **ABox** enthält Instanzfakten. Kern-Inferenzen
  (Subsumption, Instanz-Check) sind entscheidbar — der bewusste Tausch
  Ausdrucksstärke gegen Entscheidbarkeit, den Modul 06 (Abschnitt 4.5) schon
  angekündigt hat. DL-Reasoner (z. B. via Tableau-Verfahren) sind die praktische
  Fortsetzung des Theorembeweisens aus Modul 06.

---

## Zusammenfassung / Cheat-Sheet

**Planung**

| Begriff | Kern |
|---|---|
| STRIPS-Aktion | $\langle\mathrm{PRE},\mathrm{ADD},\mathrm{DEL}\rangle$; anwendbar wenn $\mathrm{PRE}\subseteq s$ |
| Progression | $\mathrm{Result}(s,a) = (s\setminus\mathrm{DEL})\cup\mathrm{ADD}$ |
| Regression | $(g'\setminus\mathrm{ADD})\cup\mathrm{PRE}$, relevant + konsistent |
| POP | partielle Ordnung + kausale Links; Bedrohungen via Promotion/Demotion |
| Relaxation | Delete-Listen streichen → $h_{\max}$ (zulässig), $h_{\text{add}}$, $h_{\text{FF}}$ |
| GraphPlan | Planungsgraph + Mutex; SATPlan: Plan als SAT kodieren |

**Wahrscheinlichkeit & Bayes-Netze**

| Begriff | Kern |
|---|---|
| Bayes | $P(h\mid e) = \dfrac{P(e\mid h)P(h)}{P(e)} = \alpha\,P(e\mid h)P(h)$ |
| bedingte Unabh. | $X\perp Y\mid Z \iff P(X,Y\mid Z)=P(X\mid Z)P(Y\mid Z)$ |
| BN-Faktorisierung | $P(x_1..x_n) = \prod_i P(x_i\mid\mathrm{parents}(x_i))$ |
| d-Separation | Kette/Gabel blockiert wenn beobachtet; Kollider blockiert wenn **un**beobachtet |
| explaining away | Kollider öffnet bei Beobachtung → Eltern werden abhängig |
| Enumeration | $P(X\mid\mathbf e)=\alpha\sum_{\mathbf y}\prod_i P(x_i\mid\mathrm{parents})$ |
| Variable Elim. | Faktoren: punktweises Produkt + summing out; Kosten ~ Baumweite; BN-Inferenz NP-schwer |
| Sampling | prior/rejection/**likelihood weighting**/**Gibbs (MCMC)**; konsistent, $O(1/\sqrt N)$ |

**Temporale Modelle**

| Begriff | Kern |
|---|---|
| Markov 1. Ordn. | $P(\mathbf X_t\mid\mathbf X_{0:t-1})=P(\mathbf X_t\mid\mathbf X_{t-1})$ |
| Filtering (Forward) | $\alpha\,P(\mathbf e_{t+1}\mid\mathbf X_{t+1})\sum_{\mathbf x_t}P(\mathbf X_{t+1}\mid\mathbf x_t)f_t$ |
| Smoothing | Forward-Backward (Vorwärts- × Rückwärtsnachricht) |
| Viterbi | wie Filtering, aber $\max$ statt $\sum$ + Rückzeiger; beste Zustandsfolge |

**Entscheidungen & MDPs**

| Begriff | Kern |
|---|---|
| MEU | $a^\ast=\arg\max_a\sum_{s'}P(s'\mid a,\mathbf e)U(s')$ |
| VPI | erwartete Nutzensteigerung durch Messung; $\ge 0$, nicht additiv |
| MDP | $(S,A,P,R,\gamma)$; maximiere $\mathbb E[\sum_t\gamma^t R]$ |
| Bellman opt. | $V^\ast(s)=R(s)+\gamma\max_a\sum_{s'}P(s'\mid s,a)V^\ast(s')$ |
| Value Iteration | Bellman-Update iterieren; $B$ ist $\gamma$-Kontraktion → $V_k\to V^\ast$ geometrisch |
| Policy Iteration | Eval (lin. System) + Improvement; terminiert in endlich vielen Schritten |

---

## Selbsttest

<details><summary><b>1. Was ist der Unterschied zwischen Progression und Regression in STRIPS, und wann bevorzugt man welche?</b></summary>

*Progression* sucht vorwärts von $s_0$: Zustände sind vollständige Fluentenmengen,
$\mathrm{Result}(s,a)=(s\setminus\mathrm{DEL})\cup\mathrm{ADD}$. Verzweigungsfaktor
hoch (alle anwendbaren Aktionen), aber Zustände konkret und gut heuristisch
bewertbar — deshalb in der Praxis (mit $h_{\text{FF}}$ o. Ä.) meist der Sieger.
*Regression* sucht rückwärts vom Ziel: Zustände sind Teilbeschreibungen,
$\mathrm{Regress}(g',a)=(g'\setminus\mathrm{ADD})\cup\mathrm{PRE}$ für relevante,
konsistente $a$. Kleiner Verzweigungsfaktor (nur relevante Aktionen), aber
Heuristiken schwerer. Regression lohnt bei wenigen zielrelevanten Aktionen.
</details>

<details><summary><b>2. Warum ist die Delete-Relaxation nützlich, obwohl $h_{\text{add}}$ nicht zulässig ist?</b></summary>

Streicht man alle Delete-Listen, wächst die Fluentenmenge monoton, und das
relaxierte Problem ist in Polynomzeit lösbar — man bekommt also *billig* eine
Schätzung. $h_{\max}$ (Maximum) ist sogar zulässig, aber schwach. $h_{\text{add}}$
(Summe) überschätzt, weil es Ziel-Fluenten als unabhängig behandelt und geteilte
Teilpläne doppelt zählt — dafür ist es viel informativer und lenkt die Suche gut.
$h_{\text{FF}}$ extrahiert einen echten relaxierten Plan und ist meist am besten.
In der Praxis zählt Informativität oft mehr als strikte Zulässigkeit (solange man
nicht Optimalität garantieren muss).
</details>

<details><summary><b>3. Ein Test ist zu 90 % sensitiv und hat 9 % Falsch-Positive; die Krankheit hat 1 % Prävalenz. Warum ist $P(\text{krank}\mid+)$ nur ~9 %?</b></summary>

Bayes: $P(D\mid+)=\frac{0{,}9\cdot0{,}01}{0{,}9\cdot0{,}01+0{,}09\cdot0{,}99}\approx0{,}092$.
Der Grund ist die niedrige **Basisrate**: Es gibt 99-mal so viele Gesunde wie
Kranke. Selbst bei nur 9 % Falsch-Positiven erzeugen die vielen Gesunden
($0{,}09\cdot0{,}99\approx0{,}089$) fast zehnmal so viele positive Tests wie die
wenigen echten Kranken ($0{,}9\cdot0{,}01=0{,}009$). Der Prior darf nie ignoriert
werden (*base rate fallacy*).
</details>

<details><summary><b>4. Erkläre die Bayes-Netz-Faktorisierung und warum sie Speicher spart.</b></summary>

Ein BN behauptet $P(x_1,\dots,x_n)=\prod_i P(x_i\mid\mathrm{parents}(x_i))$. Das
folgt aus der lokalen Markov-Bedingung (jede Variable ist bedingt unabhängig von
ihren Nicht-Nachkommen gegeben ihre Eltern). Statt der vollen Verbundtabelle mit
$2^n-1$ Einträgen speichert man pro Knoten nur $2^k$ Zahlen ($k$ = Elternzahl),
insgesamt $n\cdot2^k$. Bei begrenztem $k$ ist das linear statt exponentiell in $n$
— der ganze Sinn von Bayes-Netzen.
</details>

<details><summary><b>5. Was ist „explaining away"? Formuliere es mit d-Separation.</b></summary>

An einem Kollider $A\to C\leftarrow B$ sind $A$ und $B$ **a priori unabhängig**
(der unbeobachtete Kollider blockiert den Pfad). Beobachtet man $C$ (oder einen
Nachkommen), **öffnet** sich der Pfad: $A$ und $B$ werden *bedingt abhängig*. Wenn
$C$ eingetreten ist und man erfährt, dass $A$ es erklärt, sinkt die
Wahrscheinlichkeit von $B$ — die eine Ursache „erklärt die andere weg". Beispiel:
Alarm ($C$) durch Einbruch ($A$) oder Erdbeben ($B$); Erdbebenmeldung senkt
$P(\text{Einbruch}\mid\text{Alarm})$.
</details>

<details><summary><b>6. Wieso ist Variable Elimination schneller als Inferenz durch Aufzählung, und wovon hängen die Kosten ab?</b></summary>

Aufzählung berechnet dieselben Teilprodukte immer wieder (die naive Doppelsumme
hat exponentiell viele wiederholte Faktoren). VE klammert per Distributivgesetz
aus und **speichert Zwischenfaktoren**, sodass jedes Teilprodukt nur einmal
berechnet wird. Die Kosten werden vom **größten Zwischenfaktor** dominiert, dessen
Größe von der **Eliminationsreihenfolge** und letztlich der **Baumweite** des
Graphen abhängt. Bei kleiner Baumweite (Polybäume) polynomiell; allgemein ist
BN-Inferenz NP-schwer.
</details>

<details><summary><b>7. Wann nutzt man Likelihood Weighting statt Rejection Sampling?</b></summary>

Rejection Sampling verwirft alle Stichproben, die der Evidenz widersprechen — bei
*seltener* Evidenz landet fast alles im Müll (exponentiell ineffizient).
Likelihood Weighting fixiert stattdessen die Evidenzvariablen auf ihre Werte und
**gewichtet** jede Stichprobe mit $\prod_{e_i}P(e_i\mid\mathrm{parents}(e_i))$;
keine Stichprobe wird verworfen. Es ist konsistent und deutlich effizienter, kann
aber bei Evidenz „weit unten" im Netz ebenfalls hohe Varianz haben — dann Gibbs/MCMC.
</details>

<details><summary><b>8. Leite die Filtering-Rekursion des HMM her (Predict/Update).</b></summary>

Gesucht $f_{1:t+1}=P(\mathbf X_{t+1}\mid\mathbf e_{1:t+1})$. Bayes bzgl. der neuen
Evidenz: $\propto P(\mathbf e_{t+1}\mid\mathbf X_{t+1},\mathbf e_{1:t})\,P(\mathbf X_{t+1}\mid\mathbf e_{1:t})$.
Sensor-Markov: erster Faktor $=P(\mathbf e_{t+1}\mid\mathbf X_{t+1})$ (**Update**).
Der zweite ist die **Prediction**: marginalisieren über $\mathbf X_t$,
$P(\mathbf X_{t+1}\mid\mathbf e_{1:t})=\sum_{\mathbf x_t}P(\mathbf X_{t+1}\mid\mathbf x_t)P(\mathbf x_t\mid\mathbf e_{1:t})$
(Markov-Übergang × vorherige Filter-Nachricht). Zusammen:
$f_{1:t+1}=\alpha\,P(\mathbf e_{t+1}\mid\mathbf X_{t+1})\sum_{\mathbf x_t}P(\mathbf X_{t+1}\mid\mathbf x_t)f_{1:t}$.
</details>

<details><summary><b>9. Beweise, dass Value Iteration konvergiert.</b></summary>

Der Bellman-Optimalitätsoperator $B$ mit $(BV)(s)=R(s)+\gamma\max_a\sum_{s'}P(s'\mid s,a)V(s')$
ist eine $\gamma$-Kontraktion in der Maximumsnorm: Für beliebige $V,V'$ gilt
$\lVert BV-BV'\rVert_\infty\le\gamma\lVert V-V'\rVert_\infty$ (nutze
$|\max_a f-\max_a g|\le\max_a|f-g|$ und $\sum_{s'}P=1$). Nach dem Banachschen
Fixpunktsatz hat $B$ einen eindeutigen Fixpunkt $V^\ast$, und die Iteration
$V_{k+1}=BV_k$ konvergiert geometrisch: $\lVert V_k-V^\ast\rVert_\infty\le\gamma^k\lVert V_0-V^\ast\rVert_\infty$.
Für $\gamma<1$ folgt Konvergenz; $\gamma\to1$ macht sie beliebig langsam.
</details>

<details><summary><b>10. Value Iteration vs. Policy Iteration — Vor- und Nachteile?</b></summary>

*Value Iteration*: pro Schritt ein billiges Bellman-Update über alle Zustände,
aber viele Schritte bis zur Konvergenz (geometrisch mit $\gamma$), und die Policy
stabilisiert sich oft *bevor* die Werte konvergiert sind. *Policy Iteration*: pro
Schritt teurer (Policy Evaluation löst ein $|S|\times|S|$-Gleichungssystem), dafür
sehr wenige Schritte — terminiert exakt in endlich vielen Iterationen, da es nur
endlich viele Policies gibt und jede Iteration strikt verbessert. Kompromiss:
*modifizierte* Policy Iteration (Evaluation nur näherungsweise). Beide sind
Instanzen der generalisierten Policy-Iteration — dem Kern des RL (Modul 13).
</details>

---

## Literatur & Quellen

**Lehrbücher**
- **Russell & Norvig, *AIMA*, 4. Aufl.** — Kap. 11 (klassische Planung), 12 (Planung
  in der realen Welt), 13 (Quantifizierung von Unsicherheit), 14 (probabilistisches
  Schließen/Bayes-Netze), 15 (temporale Modelle), 16 (einfache Entscheidungen),
  17 (komplexe Entscheidungen/MDPs). *Die primäre Quelle für dieses Modul.*
- **Koller & Friedman, *Probabilistic Graphical Models*, MIT Press** — die
  erschöpfende Referenz zu Bayes-Netzen, Inferenz und Lernen. *Vertiefend, anspruchsvoll.*
- **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2. Aufl.** — Kap. 3–4
  (MDPs, dynamische Programmierung) als perfekte Vertiefung des MDP-Teils und Brücke
  zu Modul 13. **Kostenlos** unter `incompleteideas.net/book/the-book.html`. *Sehr empfohlen.*
- **Ghallab, Nau & Traverso, *Automated Planning and Acting***, für den Planungsteil. *Vertiefend.*

**Frei verfügbare Kurse & Materialien** (kostenlos)
- **UC Berkeley CS188** — die Einheiten zu Bayes-Netzen, HMMs und MDPs mit den
  Pac-Man-Projekten (`inst.eecs.berkeley.edu/~cs188`). *Einsteigerfreundlich, praktisch.*
- **Stanford CS228 „Probabilistic Graphical Models"** — Notes online, `ermongroup.github.io/cs228-notes`. *Vertiefend.*
- **David Silver, *RL Course* (DeepMind/UCL)** — Vorlesungsvideos; Lecture 2–3 zu
  MDPs und dynamischer Programmierung. *Sehr gut für den MDP-Teil.*
- **Fast Downward / PDDL-Editor** (`editor.planning.domains`) — PDDL im Browser
  schreiben und einen echten Planer laufen lassen. *Praktisch.*

**Interaktiv / Visualisierungen** (kostenlos)
- **„Seeing Theory"** (`seeing-theory.brown.edu`) — interaktive Wahrscheinlichkeit & Bayes. *Einsteigerfreundlich.*
- **Bayes-Netz-Demos** (z. B. `github.com/mbilalzonjy/BayesNetVisualization`) und der **SamIam**-Reasoner zum Herumspielen.
- **Gridworld-MDP-Visualisierungen** (Value/Policy Iteration Schritt für Schritt), z. B. Andrej Karpathys `reinforcejs`.

**Klassische Papers** (kostenlos, vertiefend)
- Pearl (1988): *Probabilistic Reasoning in Intelligent Systems* — die Geburt der Bayes-Netze.
- Blum & Furst (1997): *Fast Planning Through Planning Graph Analysis* — GraphPlan.
- Hoffmann & Nebel (2001): *The FF Planning System* — die $h_{\text{FF}}$-Heuristik.

---

## Die drei Projekte

Die drei Projekte spiegeln die drei Modulteile — Planung, probabilistisches
Schließen, sequenzielle Entscheidung — und steigern sich in Schwierigkeit und
Eigenleistung:

- **01 – basic** (`projects/01-basic/`): **Ein STRIPS-Vorwärtsplaner.** Geführtes
  Notebook: STRIPS-Zustände/Aktionen, Vorwärtssuche mit BFS *und* mit einer
  Relaxations-Heuristik ($h_{\text{add}}$) via A\*; angewandt auf Blocksworld. Viel
  Anleitung, knüpft direkt an Modul 06 an.
- **02 – medium** (`projects/02-medium/`): **Bayes-Netz mit exakter und
  approximativer Inferenz.** Python-Projekt: Netzstruktur + CPTs, Inferenz durch
  Aufzählung *und* Variable Elimination *und* Likelihood Weighting; validiert am
  Alarm-Netz und einem Diagnose-Szenario. Wenig Anleitung.
- **03 – final** (`projects/03-final/`): **Ein entscheidungstheoretischer
  MDP-Agent.** Keine Code-Vorgabe: Value Iteration *und* Policy Iteration auf einer
  Gridworld mit stochastischer Bewegung, Konvergenz empirisch prüfen, optimale
  Policy visualisieren, $\gamma$-Studie. Master-Niveau, Brücke zu RL.

Details, Setup und Musterlösungen jeweils in der `README.md` des Projektordners.
