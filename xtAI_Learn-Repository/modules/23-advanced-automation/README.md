# Module 23 — Advanced Automation

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** A robot (modules 21–22) is one machine. **Automation** is the discipline of making *whole systems* run by themselves — a production line, a chemical process, a building, a logistics cell — deciding **what to do, when, and how much**, safely and efficiently, with little or no human intervention. This module covers the *decision and control mathematics* of automation across its three classic layers: **discrete logic** (finite automata — what state is the plant in, which event triggers which action), **concurrency and resources** (Petri nets and *supervisory control* — many processes sharing machines without deadlock), and **constrained continuous optimisation** (**Model Predictive Control** — repeatedly optimising the next moves subject to hard limits). One arc, three layers of the automation stack.
>
> **Prior knowledge**: discrete mathematics (sets, graphs, relations), linear algebra, basic optimisation, and control basics. From this repo the following build in: **module 06/07** (automata, search, logic — a finite automaton is a labelled transition system and reachability is graph search), **module 14** (LQR/optimal control — MPC is its constrained, receding-horizon generalisation), **module 13** (MDPs — supervisory control is a close cousin of safe policy synthesis), **module 21** (control of a single system). No single preceding module is strictly mandatory, but 14 makes the MPC part much easier.

> **Note on the scope.** As with modules 15–22 no official module description is available; I scoped the content myself along the standard automation-engineering canon (Cassandras & Lafortune for discrete-event systems and supervisory control, Rawlings/Maciejowski for MPC) and as the natural "systems" layer above the single-robot control of modules 21–22. The three layers deliberately span **discrete** and **continuous** automation, because real plants are both: PLC logic *and* process control live in the same factory. Everything is built **from scratch** in `numpy`/`scipy` — automata as transition maps, Petri nets as incidence matrices, MPC as a condensed quadratic program — CPU seconds, no industrial tooling.

---

## Contents

1. [Learning objectives](#learning-objectives)
2. [Basics](#basics)
3. [Intermediate](#intermediate)
4. [Advanced topics](#advanced-topics)
5. [Summary / cheat sheet](#summary--cheat-sheet)
6. [Self-test](#self-test)
7. [Literature & sources](#literature--sources)

---

## Learning objectives

After this module you should be able to …

- explain what **automation** is, the **automation pyramid** (field → control → SCADA → MES → ERP), and the difference between **discrete-event** and **time-driven (continuous)** dynamics.
- model an automated process as a **deterministic finite automaton (DFA)**, define its **language**, compute its **reachable** states, and combine components by **parallel composition** (the synchronous product).
- model concurrency and shared resources with a **Petri net**: places, transitions, marking, the **incidence matrix** and the **state (marking) equation** $\mathbf M' = \mathbf M + \mathbf C\,\mathbf u$; build the **reachability graph**; and detect **deadlock** and check **boundedness**.
- state the **supervisory control** problem (Ramadge–Wonham): a plant, a specification, **controllable vs. uncontrollable events**, and the **supremal controllable sublanguage**; and **synthesise a maximally permissive safe supervisor** by forbidden-state avoidance (a backward fixed-point over uncontrollable transitions).
- derive **Model Predictive Control (MPC)**: the finite-horizon constrained optimal-control problem, its **condensation** into a **quadratic program** in the input sequence, the **receding-horizon** principle, its relation to **LQR** (module 14), and how it enforces **input and state constraints** and rejects disturbances.

---

## Basics

### 1. What is automation? The automation pyramid

**Automation** is making a technical system perform its task with minimal human intervention — sensing its state, deciding the next action, and actuating it, in a closed loop, continuously. The classic reference structure is the **automation pyramid**, five layers from the physical process up to the business:

```
     ERP        (enterprise: orders, planning)          slow, aggregate
     MES        (manufacturing execution: scheduling)
     SCADA      (supervision, monitoring, HMI)
     Control    (PLC / controllers: the real-time loop)
     Field      (sensors & actuators: the physical plant) fast, detailed
```

The lower you go, the **faster** and more **detailed** the decisions (a controller closes a loop every millisecond); the higher you go, the **slower** and more **aggregate** (an ERP plans a week). This module is about the **decision mathematics** at the control and supervision layers — and it turns out those decisions come in two fundamentally different flavours.

### 2. Two kinds of dynamics: discrete-event vs. time-driven

- **Time-driven (continuous) systems** evolve *continuously in time*, described by differential/difference equations $\dot{\mathbf x}=f(\mathbf x,\mathbf u)$ — a temperature rising, a tank filling, a motor spinning (modules 14, 21). The state is real-valued and changes at every instant.
- **Discrete-event systems (DES)** evolve by **discrete events at discrete instants**: a part *arrives*, a machine *finishes*, a valve *opens*, a fault *occurs*. Between events nothing changes; at an event the state jumps. The state is **symbolic** (idle / working / blocked), and *time is not the driver — events are*.

A real factory is both: a conveyor's speed is time-driven, but "part detected at station 3" is an event that triggers a discrete decision. Advanced automation needs **both toolkits**, and this module builds one for each: automata and Petri nets for the discrete layer, MPC for the continuous layer.

### 3. Discrete-event systems and finite automata

The workhorse model of the discrete layer is the **deterministic finite automaton (DFA)**, a 5-tuple

$$G = (Q,\;\Sigma,\;\delta,\;q_0,\;Q_m),$$

- $Q$ — a finite set of **states** (e.g. *idle, loading, processing, unloading, fault*),
- $\Sigma$ — a finite **event alphabet** (e.g. *start, done, remove, error, reset*),
- $\delta: Q\times\Sigma \rightharpoonup Q$ — the (partial) **transition function**: $\delta(q,\sigma)$ is the state reached from $q$ on event $\sigma$ (undefined if $\sigma$ cannot occur in $q$),
- $q_0\in Q$ — the **initial** state,
- $Q_m\subseteq Q$ — the **marked** (accepting / "task-complete") states.

The **language generated** by $G$, written $\mathcal L(G)$, is the set of all event strings $s\in\Sigma^\ast$ for which $\delta(q_0,s)$ is defined — every sequence the plant *can* produce. The **marked language** $\mathcal L_m(G)$ additionally requires ending in a marked state — the *completed* tasks. Automata are how you specify and reason about the **legal behaviour** of an automated system.

Two computations do most of the work:

- **Reachability.** Which states can actually occur? A breadth-first search from $q_0$ over $\delta$ gives the **reachable set** $\mathrm{Reach}(G)$. A **safety property** — "the fault-with-door-open state never occurs" — is verified by checking it is **not reachable**. (This is exactly graph search from module 06, applied to the transition graph.)
- **Parallel composition.** Real plants are many interacting components. The **synchronous product** $G_1\,\|\,G_2$ builds the joint automaton on the state set $Q_1\times Q_2$: a **shared event** (in both alphabets) may occur only if **both** components can take it (they *synchronise*); a **private event** is taken by its owner alone. This is how you compose a machine, a buffer and a robot into one model — and how the state space explodes (the reason the field also uses Petri nets, next).

---

## Intermediate

### 4. Petri nets: modelling concurrency and resources

Automata list *global* states, so modelling $k$ concurrent components multiplies their state counts — the classic **state explosion**. **Petri nets** model **concurrency and resources compactly and locally**. A Petri net is a 4-tuple

$$N = (P,\;T,\;F,\;\mathbf M_0),$$

- $P=\{p_1,\dots,p_n\}$ — **places** (conditions / resources / buffers), drawn as circles,
- $T=\{t_1,\dots,t_m\}$ — **transitions** (events / actions), drawn as bars,
- $F$ — **arcs** connecting places to transitions and vice versa (with weights),
- $\mathbf M_0\in\mathbb N^n$ — the **initial marking**: how many **tokens** sit in each place.

A **marking** $\mathbf M\in\mathbb N^n$ is the state — the token count per place (e.g. "2 parts in the input buffer, robot free"). A transition $t$ is **enabled** if each of its input places holds at least the arc weight of tokens; firing it **removes** tokens from input places and **adds** them to output places. This local rule captures **mutual exclusion** (a resource place with one token that both machines must grab), **synchronisation**, and **producer–consumer** flow naturally.

**The algebra.** Encode the arcs in two $n\times m$ matrices: $\mathbf{Pre}$ (tokens a transition *consumes*) and $\mathbf{Post}$ (tokens it *produces*). The **incidence matrix** is

$$\mathbf C = \mathbf{Post} - \mathbf{Pre}\;\in\mathbb Z^{n\times m}.$$

Firing a transition given by the unit vector $\mathbf u$ updates the marking by the **state (marking) equation**

$$\boxed{\;\mathbf M' = \mathbf M + \mathbf C\,\mathbf u\;}$$

and a whole firing sequence with count vector $\boldsymbol\sigma$ gives $\mathbf M' = \mathbf M_0 + \mathbf C\,\boldsymbol\sigma$. This linear-algebraic form is what makes Petri nets computable: **place invariants** (left null-space of $\mathbf C$, $\mathbf y^\top\mathbf C=\mathbf 0$) give conserved token sums — e.g. "robot is free XOR in use" — that prove properties without enumerating states.

**Reachability graph, boundedness, deadlock.** Enumerating all markings reachable from $\mathbf M_0$ (BFS, firing every enabled transition) builds the **reachability graph**. From it you read off the key automation properties: **boundedness** (no place accumulates unboundedly — buffers don't overflow), **liveness** (every transition can always eventually fire again — no action dies out), and crucially **deadlock**: a reachable marking where **no transition is enabled** — the plant is stuck forever. Deadlock detection and *prevention* is the central safety question of resource-sharing automation, and it leads directly to supervisory control.

### 5. Supervisory control (Ramadge–Wonham)

The Petri net or automaton is the **plant** — everything it *can* do, including reaching bad states. **Supervisory Control Theory (SCT)** answers: *how do we restrict it to only good behaviour, minimally?* The setup:

- The plant $G$ generates a language $\mathcal L(G)$.
- A **specification** carves out the **legal** (admissible) sublanguage $K\subseteq\mathcal L(G)$ — e.g. "never deadlock", "never both machines on the shared track at once".
- Events split into **controllable** $\Sigma_c$ (the supervisor can *disable* them — e.g. "don't start machine 2 yet") and **uncontrollable** $\Sigma_{uc}$ (the supervisor **cannot** prevent them — a part arriving, a fault occurring, a motion finishing). This asymmetry is the whole difficulty.

A **supervisor** $S$ observes the string so far and, at each step, outputs the set of events it **allows**; it may never disable an uncontrollable event. The closed loop $S/G$ must stay inside $K$. The catch: you cannot simply forbid every path into a bad state, because the *last* controllable event before the badness might be far upstream, and an **uncontrollable** event could carry you into the bad region with no controllable event to stop it. The largest achievable legal behaviour is the **supremal controllable sublanguage** $\sup\mathcal C(K)$ — the biggest sublanguage of $K$ that is **controllable** (closed under uncontrollable events) and **non-blocking** (can always still complete a task).

**Forbidden-state synthesis (the constructive core).** For the common case of avoiding a set of **forbidden states** $Q_x$ (e.g. all deadlocks and specification-violating states), the maximally permissive safe supervisor is computed by a **backward fixed-point**:

1. Start with the **bad set** $B \leftarrow Q_x$.
2. Add to $B$ any state that has an **uncontrollable** transition into $B$ (from there, badness is unavoidable) — repeat until $B$ stops growing.
3. The supervisor: from any safe state, **disable exactly the controllable transitions** that lead into $B$, and leave everything else enabled.

This yields the **least restrictive** (maximally permissive) supervisor that keeps the plant out of $B$ forever using only controllable events. It is a graph fixed-point — the discrete-automation analogue of computing a safe region for a controller. The **medium project** builds exactly this: model a resource-sharing cell as a Petri net, find the deadlocks in its reachability graph, and synthesise the supervisor that provably prevents them while allowing as much concurrency as safely possible.

---

## Advanced topics

### 6. Model Predictive Control: automation of the continuous layer

The continuous layer needs a controller that (unlike LQR) respects **hard constraints** — actuators saturate, tanks overflow, temperatures must stay in band. **Model Predictive Control (MPC)** is the dominant modern answer and the workhorse of process automation. The idea, at every sampling instant:

1. Use the model to **predict** the plant's response over a finite **horizon** $N$ for a candidate input sequence.
2. **Optimise** that input sequence to minimise a cost, **subject to the constraints**.
3. **Apply only the first input**, then discard the rest, move one step, and **re-optimise** with fresh measurements — the **receding horizon**.

Re-solving every step is what turns an open-loop plan into **feedback** and gives robustness to disturbance and model error.

**The optimisation problem.** For a linear plant $\mathbf x_{k+1}=\mathbf A\mathbf x_k+\mathbf B\mathbf u_k$, at the current state $\mathbf x_0$ solve

$$\min_{\mathbf u_0,\dots,\mathbf u_{N-1}}\;\; \mathbf x_N^\top\mathbf P\mathbf x_N + \sum_{k=0}^{N-1}\big(\mathbf x_k^\top\mathbf Q\mathbf x_k + \mathbf u_k^\top\mathbf R\mathbf u_k\big)$$
$$\text{s.t.}\quad \mathbf x_{k+1}=\mathbf A\mathbf x_k+\mathbf B\mathbf u_k,\qquad \mathbf u_{\min}\le\mathbf u_k\le\mathbf u_{\max},\qquad \mathbf x_{\min}\le\mathbf x_k\le\mathbf x_{\max}.$$

with $\mathbf Q\succeq0$, $\mathbf R\succ0$ the state/input weights and $\mathbf P$ a terminal weight.

**Condensation into a QP.** Because the dynamics are linear, every predicted state is an affine function of the initial state and the stacked input vector $\mathbf U=(\mathbf u_0^\top,\dots,\mathbf u_{N-1}^\top)^\top$:

$$\mathbf X = \mathbf S_x\,\mathbf x_0 + \mathbf S_u\,\mathbf U,$$

where $\mathbf S_x,\mathbf S_u$ are built from powers of $\mathbf A$ and $\mathbf B$ (the **prediction matrices**). Substituting into the cost turns it into a **convex quadratic program** in $\mathbf U$ alone:

$$\boxed{\;\min_{\mathbf U}\;\tfrac12\mathbf U^\top\mathbf H\mathbf U + \mathbf x_0^\top\mathbf F^\top\mathbf U \quad\text{s.t.}\quad \mathbf G\,\mathbf U\le\mathbf w(\mathbf x_0)\;}$$

with $\mathbf H = \mathbf S_u^\top\bar{\mathbf Q}\mathbf S_u + \bar{\mathbf R}\succ0$ (a positive-definite Hessian ⇒ a unique optimum), and $\mathbf G,\mathbf w$ encoding the input and (state-dependent) state bounds. Solve it (`scipy.optimize`), apply $\mathbf u_0$, and repeat. The **final project** implements this end to end.

**The bridge to LQR (module 14).** Drop the constraints and take $N\to\infty$: the MPC problem *becomes* the LQR problem, and its solution is the same constant gain $\mathbf u_0=-\mathbf K\mathbf x_0$ with $\mathbf K$ from the discrete Riccati equation. So **unconstrained MPC = LQR** — a numerical fact the final project verifies to machine precision. MPC's value is everything LQR cannot do: **honour hard limits**. When a disturbance would push the state past a bound, LQR (which knows no bounds) drives the input past its saturation and the real actuator clips, degrading control; MPC *plans within the limits* and keeps the state feasible. Choosing the **terminal cost** $\mathbf P$ as the LQR cost-to-go (the Riccati solution) and adding a terminal constraint is the standard route to a **stability guarantee** for the receding-horizon controller.

### 7. Scheduling, hybrid systems and the wider picture (brief)

Two threads round out advanced automation. **Scheduling** — deciding the *order and timing* of operations on shared machines (job-shop, flow-shop) — is the higher (MES) layer's optimisation; it is combinatorial (often NP-hard) and solved with MILP, constraint programming, or heuristics, and it sits directly on top of the resource models of section 4. **Hybrid systems** mix the discrete and continuous worlds in one model (a thermostat: continuous temperature + discrete on/off), formalised as **hybrid automata**; their verification and control unify the two halves of this module. Modern automation increasingly adds a **learning** layer — RL for scheduling and control (modules 13/14), learned anomaly detection on process data (module 16), and **digital twins** that run the models of this module in parallel with the real plant for prediction and monitoring — but the models being learned, twinned and optimised are exactly the automata, nets and dynamics built here.

---

## Summary / cheat sheet

**Automation pyramid**: Field → Control → SCADA → MES → ERP (fast/detailed at the bottom, slow/aggregate at the top). Two dynamics: **discrete-event** (events drive state jumps) vs. **time-driven** (differential equations).

**Finite automaton**: $G=(Q,\Sigma,\delta,q_0,Q_m)$. Language $\mathcal L(G)$ = strings with $\delta(q_0,s)$ defined; marked $\mathcal L_m$ = ending in $Q_m$. **Reachability** = BFS from $q_0$; **safety** = bad state not reachable. **Parallel composition** $G_1\|G_2$ on $Q_1\times Q_2$: shared events synchronise, private events interleave.

**Petri net**: $N=(P,T,F,\mathbf M_0)$. Marking $\mathbf M\in\mathbb N^n$ (tokens). $t$ enabled if inputs hold enough tokens; firing moves tokens. **Incidence** $\mathbf C=\mathbf{Post}-\mathbf{Pre}$; **state equation** $\mathbf M'=\mathbf M+\mathbf C\mathbf u$. **Place invariants** $\mathbf y^\top\mathbf C=\mathbf 0$ (conserved token sums). **Reachability graph** → boundedness, liveness, **deadlock** (a reachable marking with no enabled transition).

**Supervisory control (RW)**: plant language $\mathcal L(G)$, spec/legal $K$, **controllable $\Sigma_c$** (can disable) vs. **uncontrollable $\Sigma_{uc}$** (cannot). Goal: **supremal controllable sublanguage** $\sup\mathcal C(K)$ — largest legal + controllable + non-blocking behaviour. **Forbidden-state synthesis**: grow the bad set $B$ backward over *uncontrollable* transitions to a fixed point; disable the controllable transitions into $B$ ⇒ maximally permissive safe supervisor.

**MPC**: at each step solve $\min\;\mathbf x_N^\top\mathbf P\mathbf x_N+\sum(\mathbf x_k^\top\mathbf Q\mathbf x_k+\mathbf u_k^\top\mathbf R\mathbf u_k)$ s.t. dynamics + $\mathbf u,\mathbf x$ bounds; apply $\mathbf u_0$; recede. **Condense**: $\mathbf X=\mathbf S_x\mathbf x_0+\mathbf S_u\mathbf U$ → QP $\min\tfrac12\mathbf U^\top\mathbf H\mathbf U+\mathbf x_0^\top\mathbf F^\top\mathbf U$ s.t. $\mathbf G\mathbf U\le\mathbf w$, $\mathbf H\succ0$. **Unconstrained, $N\to\infty$ ⇒ LQR** (Riccati gain). MPC's edge: enforces hard constraints where LQR saturates.

---

## Self-test

<details>
<summary><b>1.</b> What is the automation pyramid, and how do the decisions differ between its top and bottom?</summary>

The automation pyramid is the five-layer reference structure Field → Control → SCADA → MES → ERP. At the **bottom** (field/control) the decisions are **fast and detailed** — a controller closes a real-time loop every millisecond on raw sensor/actuator signals. At the **top** (MES/ERP) they are **slow and aggregate** — planning and scheduling over hours to weeks on summarised data. Advanced automation supplies the decision mathematics for the control and supervision layers in the middle.
</details>

<details>
<summary><b>2.</b> Distinguish discrete-event and time-driven dynamics with an example of each.</summary>

**Time-driven (continuous)** systems evolve continuously per differential/difference equations — e.g. a tank's level $\dot h = (q_{in}-q_{out})/A$, changing at every instant. **Discrete-event** systems change only at **discrete events** and are otherwise static — e.g. "part arrives", "machine finishes", "valve opens"; the state is symbolic (idle/working) and jumps at events, with time not the driver. A real plant contains both, which is why automation needs both automata/Petri nets and MPC.
</details>

<details>
<summary><b>3.</b> Define a DFA and its language, and explain how you verify a safety property.</summary>

A DFA is $G=(Q,\Sigma,\delta,q_0,Q_m)$: states, event alphabet, (partial) transition function, initial state, marked states. Its **generated language** $\mathcal L(G)$ is all event strings $s$ for which $\delta(q_0,s)$ is defined (everything the plant can do). A **safety property** of the form "bad state $q_{bad}$ never occurs" is verified by computing the **reachable set** from $q_0$ (BFS over $\delta$) and checking that $q_{bad}\notin\mathrm{Reach}(G)$ — i.e. it is unreachable.
</details>

<details>
<summary><b>4.</b> What is parallel composition and why does it matter?</summary>

Parallel composition (synchronous product) $G_1\|G_2$ builds the joint automaton on $Q_1\times Q_2$: a **shared** event (in both alphabets) may fire only when **both** components can take it (they synchronise), while a **private** event is taken by its owner alone. It matters because real automated systems are many interacting components (machine + buffer + robot), and composition builds their combined model — but it also causes **state explosion** (the product of state counts), which motivates Petri nets.
</details>

<details>
<summary><b>5.</b> Write the Petri-net state equation and explain each part.</summary>

$\mathbf M' = \mathbf M + \mathbf C\,\mathbf u$, where $\mathbf M,\mathbf M'\in\mathbb N^n$ are the token markings before/after, $\mathbf u$ is the firing vector (which transition(s) fire), and $\mathbf C=\mathbf{Post}-\mathbf{Pre}$ is the incidence matrix (tokens produced minus consumed per transition). It says firing a transition moves tokens according to its column of $\mathbf C$. For a firing sequence with count vector $\boldsymbol\sigma$, $\mathbf M'=\mathbf M_0+\mathbf C\boldsymbol\sigma$. This linear form enables invariant analysis ($\mathbf y^\top\mathbf C=\mathbf 0$ gives conserved token sums).
</details>

<details>
<summary><b>6.</b> What is a deadlock in a Petri net, and how do you find one?</summary>

A **deadlock** is a reachable marking at which **no transition is enabled** — no input place has enough tokens for any transition, so the system is stuck forever. You find deadlocks by building the **reachability graph** (BFS from $\mathbf M_0$, firing every enabled transition to enumerate reachable markings) and listing the markings with **no successors** (no enabled transition). In resource-sharing systems these are typically circular-wait situations (each process holds a resource the other needs).
</details>

<details>
<summary><b>7.</b> Why does the controllable/uncontrollable split make supervisory control non-trivial?</summary>

A supervisor may **disable controllable events** but can **never** disable **uncontrollable** ones (a fault, a part arriving, a motion finishing happen whether you like it or not). So it is not enough to block the last step into a bad state: if the only transitions from some state into the bad region are **uncontrollable**, that state is *itself* effectively bad (badness is unavoidable from it), and you must prevent reaching *it* — recursively. Hence the largest safe behaviour is the **supremal controllable sublanguage**, not simply "the legal language".
</details>

<details>
<summary><b>8.</b> Describe the forbidden-state supervisor synthesis as a fixed-point.</summary>

Start with the bad set $B=Q_x$ (forbidden states, e.g. all deadlocks). Repeatedly **add to $B$ any state that has an uncontrollable transition into $B$** (from such a state, entering $B$ cannot be prevented) until $B$ no longer grows — a backward fixed-point over uncontrollable edges. The resulting supervisor, from any safe state, **disables exactly the controllable transitions that lead into $B$** and enables everything else. This is the **maximally permissive** (least restrictive) supervisor that keeps the plant out of $B$ forever.
</details>

<details>
<summary><b>9.</b> State the MPC optimisation and how it is condensed into a QP.</summary>

At the current state $\mathbf x_0$, MPC solves $\min\;\mathbf x_N^\top\mathbf P\mathbf x_N+\sum_{k=0}^{N-1}(\mathbf x_k^\top\mathbf Q\mathbf x_k+\mathbf u_k^\top\mathbf R\mathbf u_k)$ subject to the dynamics $\mathbf x_{k+1}=\mathbf A\mathbf x_k+\mathbf B\mathbf u_k$ and input/state bounds, applies $\mathbf u_0$, then recedes. **Condensation** uses the linearity to write $\mathbf X=\mathbf S_x\mathbf x_0+\mathbf S_u\mathbf U$ (predicted states as an affine function of the stacked inputs $\mathbf U$), turning the cost into a convex QP $\min\tfrac12\mathbf U^\top\mathbf H\mathbf U+\mathbf x_0^\top\mathbf F^\top\mathbf U$ s.t. $\mathbf G\mathbf U\le\mathbf w(\mathbf x_0)$, with $\mathbf H=\mathbf S_u^\top\bar{\mathbf Q}\mathbf S_u+\bar{\mathbf R}\succ0$ (unique optimum).
</details>

<details>
<summary><b>10.</b> How is MPC related to LQR, and what does MPC add?</summary>

If you drop the constraints and let the horizon $N\to\infty$, the MPC problem *is* the infinite-horizon LQR problem, and the optimal first move is the constant LQR gain $\mathbf u_0=-\mathbf K\mathbf x_0$ from the discrete Riccati equation — so **unconstrained MPC = LQR**. What MPC adds is **hard constraint handling**: it optimises within input and state limits, so when a disturbance would push the state toward a bound, MPC plans a feasible input sequence and stays inside the limits, whereas LQR (which ignores bounds) commands inputs the actuator must clip, degrading performance and possibly violating state limits.
</details>

---

## Literature & sources

**Textbooks — discrete-event systems & supervisory control**
- **Cassandras & Lafortune, *Introduction to Discrete Event Systems*** (Springer). The definitive text on automata, languages, Petri nets and Ramadge–Wonham supervisory control — chapters 2–4 cover sections 3–5 here. *In-depth, the standard reference.*
- **Wonham & Cai, *Supervisory Control of Discrete-Event Systems*** — free lecture notes/book online. The rigorous RW theory. *Free, advanced.*
- **Murata, "Petri Nets: Properties, Analysis and Applications"**, *Proc. IEEE 1989*. The classic survey of Petri-net theory (incidence matrix, invariants, reachability). *Free, in-depth.*

**Textbooks — MPC**
- **Rawlings, Mayne & Diehl, *Model Predictive Control: Theory, Computation, and Design*** — free PDF. The modern reference (stability, constraints, condensation). *Free, in-depth.*
- **Maciejowski, *Predictive Control with Constraints***. Very readable introduction with the QP condensation worked out. *Beginner- to intermediate-friendly.*
- **Borrelli, Bemporad & Morari, *Predictive Control for Linear and Hybrid Systems*** — free PDF. Connects MPC, QP and hybrid systems. *Free, advanced.*

**Key papers**
- **Ramadge & Wonham, "Supervisory Control of a Class of Discrete Event Processes"**, *SIAM J. Control 1987*. The origin of SCT. *In-depth.*
- **Mayne, Rawlings, Rao & Scokaert, "Constrained model predictive control: Stability and optimality"**, *Automatica 2000*. The stability reference for MPC. *In-depth.*

**Freely available courses**
- **Cassandras / Lafortune DES lectures** and various university DES courses on YouTube. *Free.*
- **MPC lectures** (e.g. Borrelli at Berkeley, del Re, or the free MPC book's companion material). *Free.*

**For hands-on practice**
- The **three projects** build a finite-automaton model + reachability/composition (basic), a Petri-net cell with deadlock detection and supervisor synthesis (medium) and a constrained **MPC** controller with the LQR comparison (final) — all from scratch, the best way to make the three layers concrete.

---

> **Next module:** Module 24 "Self-aware Computing" — systems that monitor and adapt themselves at runtime (self-optimisation, self-healing), building on the feedback-control and decision ideas of this module and the estimation of modules 21–22.

---
---

# Modul 23 — Advanced Automation (deutsche Fassung)

> **Worum geht es?** Ein Roboter (Module 21–22) ist eine Maschine. **Automatisierung** ist die Disziplin, *ganze Systeme* von selbst laufen zu lassen — eine Fertigungslinie, einen chemischen Prozess, ein Gebäude, eine Logistikzelle — und zu entscheiden, **was zu tun ist, wann und wie viel**, sicher und effizient, mit wenig oder keinem menschlichen Eingriff. Dieses Modul behandelt die *Entscheidungs- und Regelungsmathematik* der Automatisierung über ihre drei klassischen Ebenen: **diskrete Logik** (endliche Automaten — in welchem Zustand ist die Anlage, welches Ereignis löst welche Aktion aus), **Nebenläufigkeit und Ressourcen** (Petri-Netze und *Supervisory Control* — viele Prozesse, die Maschinen teilen, ohne Deadlock), und **beschränkte kontinuierliche Optimierung** (**Model Predictive Control** — wiederholtes Optimieren der nächsten Züge unter harten Grenzen). Ein Bogen, drei Ebenen des Automatisierungs-Stacks.
>
> **Vorkenntnisse**: diskrete Mathematik (Mengen, Graphen, Relationen), lineare Algebra, Optimierungsgrundlagen, Regelungsgrundlagen. Aus diesem Repo bauen ein: **Modul 06/07** (Automaten, Suche, Logik — ein endlicher Automat ist ein beschriftetes Transitionssystem, Erreichbarkeit ist Graphsuche), **Modul 14** (LQR/Optimalregelung — MPC ist dessen beschränkte, receding-horizon-Verallgemeinerung), **Modul 13** (MDPs — Supervisory Control ist ein naher Verwandter der sicheren Policy-Synthese), **Modul 21** (Regelung eines einzelnen Systems). Kein einzelnes Vormodul ist streng Pflicht, aber 14 erleichtert den MPC-Teil erheblich.

> **Hinweis zum Zuschnitt.** Wie bei den Modulen 15–22 liegt keine offizielle Modulbeschreibung vor; ich habe den Inhalt selbst zugeschnitten, entlang des Standard-Kanons der Automatisierungstechnik (Cassandras & Lafortune für diskrete-Ereignis-Systeme und Supervisory Control, Rawlings/Maciejowski für MPC) und als die natürliche „Systeme"-Ebene über der Einzelroboter-Regelung der Module 21–22. Die drei Ebenen spannen bewusst **diskrete** und **kontinuierliche** Automatisierung auf, weil reale Anlagen beides sind: SPS-Logik *und* Prozessregelung leben in derselben Fabrik. Alles ist **from scratch** in `numpy`/`scipy` gebaut — Automaten als Transitionsabbildungen, Petri-Netze als Inzidenzmatrizen, MPC als kondensiertes quadratisches Programm — CPU-Sekunden, kein Industrie-Tooling.

---

## Inhalt

1. [Lernziele](#lernziele)
2. [Grundlagen (Basics)](#grundlagen-basics)
3. [Aufbau (Intermediate)](#aufbau-intermediate)
4. [Advanced-Themen](#advanced-themen)
5. [Zusammenfassung / Cheat-Sheet](#zusammenfassung--cheat-sheet)
6. [Selbsttest](#selbsttest)
7. [Literatur & Quellen](#literatur--quellen)

---

## Lernziele

Nach diesem Modul solltest du …

- erklären können, was **Automatisierung** ist, die **Automatisierungspyramide** (Feld → Steuerung → SCADA → MES → ERP) und den Unterschied zwischen **diskreter-Ereignis-** und **zeitgetriebener (kontinuierlicher)** Dynamik.
- einen automatisierten Prozess als **deterministischen endlichen Automaten (DFA)** modellieren, seine **Sprache** definieren, seine **erreichbaren** Zustände berechnen und Komponenten per **Parallelkomposition** (synchrones Produkt) kombinieren.
- Nebenläufigkeit und geteilte Ressourcen mit einem **Petri-Netz** modellieren: Plätze, Transitionen, Markierung, die **Inzidenzmatrix** und die **Zustands- (Markierungs-) Gleichung** $\mathbf M' = \mathbf M + \mathbf C\,\mathbf u$; den **Erreichbarkeitsgraphen** bauen; und **Deadlock** erkennen sowie **Beschränktheit** prüfen.
- das **Supervisory-Control**-Problem (Ramadge–Wonham) formulieren: eine Anlage, eine Spezifikation, **steuerbare vs. nicht-steuerbare Ereignisse**, und die **supremale steuerbare Untersprache**; und einen **maximal permissiven sicheren Supervisor** per Forbidden-State-Avoidance synthetisieren (ein Rückwärts-Fixpunkt über nicht-steuerbare Transitionen).
- **Model Predictive Control (MPC)** herleiten: das beschränkte Optimalregelungsproblem über endlichem Horizont, seine **Kondensierung** in ein **quadratisches Programm** in der Eingangsfolge, das **Receding-Horizon**-Prinzip, seine Beziehung zum **LQR** (Modul 14) und wie es **Eingangs- und Zustands-Constraints** durchsetzt und Störungen ausregelt.

---

## Grundlagen (Basics)

### 1. Was ist Automatisierung? Die Automatisierungspyramide

**Automatisierung** heißt, ein technisches System seine Aufgabe mit minimalem menschlichem Eingriff erfüllen zu lassen — seinen Zustand erfassen, die nächste Aktion entscheiden und sie stellen, in geschlossener Schleife, fortlaufend. Die klassische Referenzstruktur ist die **Automatisierungspyramide**, fünf Ebenen vom physischen Prozess bis zum Geschäft:

```
     ERP        (Unternehmen: Auftraege, Planung)         langsam, aggregiert
     MES        (Manufacturing Execution: Scheduling)
     SCADA      (Ueberwachung, Monitoring, HMI)
     Steuerung  (SPS / Regler: die Echtzeitschleife)
     Feld       (Sensoren & Aktoren: die physische Anlage) schnell, detailliert
```

Je tiefer man geht, desto **schneller** und detaillierter die Entscheidungen (ein Regler schließt eine Schleife jede Millisekunde); je höher, desto **langsamer** und aggregierter (ein ERP plant eine Woche). Dieses Modul behandelt die **Entscheidungsmathematik** auf der Steuerungs- und Überwachungsebene — und diese Entscheidungen kommen in zwei grundverschiedenen Ausprägungen.

### 2. Zwei Arten von Dynamik: diskrete-Ereignis- vs. zeitgetrieben

- **Zeitgetriebene (kontinuierliche) Systeme** entwickeln sich *kontinuierlich in der Zeit*, beschrieben durch Differential-/Differenzengleichungen $\dot{\mathbf x}=f(\mathbf x,\mathbf u)$ — eine steigende Temperatur, ein füllender Tank, ein drehender Motor (Module 14, 21). Der Zustand ist reellwertig und ändert sich in jedem Augenblick.
- **Diskrete-Ereignis-Systeme (DES)** entwickeln sich durch **diskrete Ereignisse zu diskreten Zeitpunkten**: ein Teil *kommt an*, eine Maschine *fertigt fertig*, ein Ventil *öffnet*, ein Fehler *tritt auf*. Zwischen Ereignissen ändert sich nichts; bei einem Ereignis springt der Zustand. Der Zustand ist **symbolisch** (idle / working / blocked), und *nicht die Zeit treibt, sondern die Ereignisse*.

Eine reale Fabrik ist beides: die Geschwindigkeit eines Förderbands ist zeitgetrieben, aber „Teil an Station 3 erkannt" ist ein Ereignis, das eine diskrete Entscheidung auslöst. Advanced Automation braucht **beide Werkzeugkästen**, und dieses Modul baut für jeden einen: Automaten und Petri-Netze für die diskrete Ebene, MPC für die kontinuierliche.

### 3. Diskrete-Ereignis-Systeme und endliche Automaten

Das Arbeitspferd-Modell der diskreten Ebene ist der **deterministische endliche Automat (DFA)**, ein 5-Tupel

$$G = (Q,\;\Sigma,\;\delta,\;q_0,\;Q_m),$$

- $Q$ — eine endliche Menge von **Zuständen** (z. B. *idle, loading, processing, unloading, fault*),
- $\Sigma$ — ein endliches **Ereignisalphabet** (z. B. *start, done, remove, error, reset*),
- $\delta: Q\times\Sigma \rightharpoonup Q$ — die (partielle) **Transitionsfunktion**: $\delta(q,\sigma)$ ist der von $q$ bei Ereignis $\sigma$ erreichte Zustand (undefiniert, wenn $\sigma$ in $q$ nicht auftreten kann),
- $q_0\in Q$ — der **Anfangszustand**,
- $Q_m\subseteq Q$ — die **markierten** (akzeptierenden / „Aufgabe fertig") Zustände.

Die von $G$ **generierte Sprache** $\mathcal L(G)$ ist die Menge aller Ereignisketten $s\in\Sigma^\ast$, für die $\delta(q_0,s)$ definiert ist — jede Folge, die die Anlage erzeugen *kann*. Die **markierte Sprache** $\mathcal L_m(G)$ verlangt zusätzlich, in einem markierten Zustand zu enden — die *abgeschlossenen* Aufgaben. Automaten sind, wie man das **legale Verhalten** eines automatisierten Systems spezifiziert und begründet.

Zwei Berechnungen leisten die meiste Arbeit:

- **Erreichbarkeit.** Welche Zustände können tatsächlich auftreten? Eine Breitensuche von $q_0$ über $\delta$ liefert die **erreichbare Menge** $\mathrm{Reach}(G)$. Eine **Sicherheits-Property** — „der Zustand Fehler-bei-offener-Tür tritt nie auf" — wird verifiziert, indem man prüft, dass er **nicht erreichbar** ist. (Das ist genau die Graphsuche aus Modul 06, angewandt auf den Transitionsgraphen.)
- **Parallelkomposition.** Reale Anlagen sind viele interagierende Komponenten. Das **synchrone Produkt** $G_1\,\|\,G_2$ baut den gemeinsamen Automaten auf der Zustandsmenge $Q_1\times Q_2$: ein **geteiltes Ereignis** (in beiden Alphabeten) kann nur auftreten, wenn **beide** Komponenten es nehmen können (sie *synchronisieren*); ein **privates Ereignis** wird von seinem Besitzer allein genommen. So komponiert man eine Maschine, einen Puffer und einen Roboter zu einem Modell — und so explodiert der Zustandsraum (der Grund, warum das Feld auch Petri-Netze nutzt, nächster Abschnitt).

---

## Aufbau (Intermediate)

### 4. Petri-Netze: Nebenläufigkeit und Ressourcen modellieren

Automaten listen *globale* Zustände, sodass das Modellieren von $k$ nebenläufigen Komponenten ihre Zustandszahlen multipliziert — die klassische **Zustandsexplosion**. **Petri-Netze** modellieren **Nebenläufigkeit und Ressourcen kompakt und lokal**. Ein Petri-Netz ist ein 4-Tupel

$$N = (P,\;T,\;F,\;\mathbf M_0),$$

- $P=\{p_1,\dots,p_n\}$ — **Plätze** (Bedingungen / Ressourcen / Puffer), als Kreise gezeichnet,
- $T=\{t_1,\dots,t_m\}$ — **Transitionen** (Ereignisse / Aktionen), als Balken gezeichnet,
- $F$ — **Kanten**, die Plätze mit Transitionen und umgekehrt verbinden (mit Gewichten),
- $\mathbf M_0\in\mathbb N^n$ — die **Anfangsmarkierung**: wie viele **Token** in jedem Platz liegen.

Eine **Markierung** $\mathbf M\in\mathbb N^n$ ist der Zustand — die Token-Zahl pro Platz (z. B. „2 Teile im Eingangspuffer, Roboter frei"). Eine Transition $t$ ist **aktiviert**, wenn jeder ihrer Eingangsplätze mindestens das Kantengewicht an Token hält; ihr Feuern **entfernt** Token aus den Eingangsplätzen und **fügt** sie den Ausgangsplätzen hinzu. Diese lokale Regel erfasst **wechselseitigen Ausschluss** (ein Ressourcenplatz mit einem Token, den beide Maschinen greifen müssen), **Synchronisation** und **Producer-Consumer**-Fluss natürlich.

**Die Algebra.** Kodiere die Kanten in zwei $n\times m$-Matrizen: $\mathbf{Pre}$ (Token, die eine Transition *verbraucht*) und $\mathbf{Post}$ (Token, die sie *produziert*). Die **Inzidenzmatrix** ist

$$\mathbf C = \mathbf{Post} - \mathbf{Pre}\;\in\mathbb Z^{n\times m}.$$

Das Feuern einer durch den Einheitsvektor $\mathbf u$ gegebenen Transition aktualisiert die Markierung per **Zustands- (Markierungs-) Gleichung**

$$\boxed{\;\mathbf M' = \mathbf M + \mathbf C\,\mathbf u\;}$$

und eine ganze Feuersequenz mit Zählvektor $\boldsymbol\sigma$ ergibt $\mathbf M' = \mathbf M_0 + \mathbf C\,\boldsymbol\sigma$. Diese linear-algebraische Form macht Petri-Netze berechenbar: **Platz-Invarianten** (linker Nullraum von $\mathbf C$, $\mathbf y^\top\mathbf C=\mathbf 0$) liefern erhaltene Token-Summen — z. B. „Roboter ist frei XOR in Benutzung" — die Eigenschaften ohne Zustands-Enumeration beweisen.

**Erreichbarkeitsgraph, Beschränktheit, Deadlock.** Das Enumerieren aller von $\mathbf M_0$ erreichbaren Markierungen (BFS, jede aktivierte Transition feuern) baut den **Erreichbarkeitsgraphen**. Aus ihm liest man die zentralen Automatisierungseigenschaften ab: **Beschränktheit** (kein Platz akkumuliert unbeschränkt — Puffer laufen nicht über), **Lebendigkeit** (jede Transition kann immer irgendwann wieder feuern — keine Aktion stirbt aus), und entscheidend **Deadlock**: eine erreichbare Markierung, bei der **keine Transition aktiviert** ist — die Anlage steht für immer. Deadlock-Erkennung und -*Verhinderung* ist die zentrale Sicherheitsfrage ressourcenteilender Automatisierung und führt direkt zur Supervisory Control.

### 5. Supervisory Control (Ramadge–Wonham)

Das Petri-Netz oder der Automat ist die **Anlage** — alles, was sie tun *kann*, einschließlich des Erreichens schlechter Zustände. **Supervisory Control Theory (SCT)** beantwortet: *Wie beschränken wir sie auf nur gutes Verhalten, und zwar minimal?* Der Aufbau:

- Die Anlage $G$ generiert eine Sprache $\mathcal L(G)$.
- Eine **Spezifikation** schneidet die **legale** (zulässige) Untersprache $K\subseteq\mathcal L(G)$ heraus — z. B. „nie Deadlock", „nie beide Maschinen gleichzeitig auf der geteilten Bahn".
- Ereignisse teilen sich in **steuerbare** $\Sigma_c$ (der Supervisor kann sie *deaktivieren* — z. B. „starte Maschine 2 noch nicht") und **nicht-steuerbare** $\Sigma_{uc}$ (der Supervisor **kann sie nicht** verhindern — ein ankommendes Teil, ein auftretender Fehler, eine endende Bewegung). Diese Asymmetrie ist die ganze Schwierigkeit.

Ein **Supervisor** $S$ beobachtet die bisherige Kette und gibt bei jedem Schritt die Menge der **erlaubten** Ereignisse aus; er darf nie ein nicht-steuerbares Ereignis deaktivieren. Der geschlossene Kreis $S/G$ muss innerhalb von $K$ bleiben. Der Haken: Man kann nicht einfach jeden Pfad in einen schlechten Zustand verbieten, denn das *letzte* steuerbare Ereignis vor der Schlechtigkeit könnte weit stromaufwärts liegen, und ein **nicht-steuerbares** Ereignis könnte einen in den schlechten Bereich tragen, ohne dass ein steuerbares Ereignis stoppen könnte. Das größte erreichbare legale Verhalten ist die **supremale steuerbare Untersprache** $\sup\mathcal C(K)$ — die größte Untersprache von $K$, die **steuerbar** (abgeschlossen unter nicht-steuerbaren Ereignissen) und **nicht-blockierend** (kann immer noch eine Aufgabe abschließen) ist.

**Forbidden-State-Synthese (der konstruktive Kern).** Für den häufigen Fall, eine Menge **verbotener Zustände** $Q_x$ zu vermeiden (z. B. alle Deadlocks und Spezifikations-verletzenden Zustände), wird der maximal permissive sichere Supervisor per **Rückwärts-Fixpunkt** berechnet:

1. Starte mit der **schlechten Menge** $B \leftarrow Q_x$.
2. Füge zu $B$ jeden Zustand hinzu, der eine **nicht-steuerbare** Transition nach $B$ hat (von dort ist die Schlechtigkeit unvermeidbar) — wiederhole, bis $B$ nicht mehr wächst.
3. Der Supervisor: Von jedem sicheren Zustand aus **deaktiviere genau die steuerbaren Transitionen**, die nach $B$ führen, und lasse alles andere aktiviert.

Das ergibt den **am wenigsten restriktiven** (maximal permissiven) Supervisor, der die Anlage mit nur steuerbaren Ereignissen für immer aus $B$ hält. Es ist ein Graph-Fixpunkt — das diskrete-Automatisierungs-Analogon zur Berechnung einer sicheren Region für einen Regler. Das **Medium-Projekt** baut genau das: eine ressourcenteilende Zelle als Petri-Netz modellieren, die Deadlocks in ihrem Erreichbarkeitsgraphen finden und den Supervisor synthetisieren, der sie beweisbar verhindert und dabei so viel Nebenläufigkeit wie sicher möglich erlaubt.

---

## Advanced-Themen

### 6. Model Predictive Control: Automatisierung der kontinuierlichen Ebene

Die kontinuierliche Ebene braucht einen Regler, der (anders als LQR) **harte Constraints** respektiert — Aktoren sättigen, Tanks laufen über, Temperaturen müssen im Band bleiben. **Model Predictive Control (MPC)** ist die dominierende moderne Antwort und das Arbeitspferd der Prozessautomatisierung. Die Idee, bei jedem Abtastzeitpunkt:

1. Nutze das Modell, um die Antwort der Anlage über einen endlichen **Horizont** $N$ für eine Kandidaten-Eingangsfolge zu **prädizieren**.
2. **Optimiere** diese Eingangsfolge, um eine Kostenfunktion zu minimieren, **unter den Constraints**.
3. **Wende nur den ersten Eingang an**, verwirf den Rest, gehe einen Schritt weiter und **optimiere neu** mit frischen Messungen — der **Receding Horizon**.

Das Neu-Lösen jeden Schritt ist es, was einen Open-Loop-Plan in **Feedback** verwandelt und Robustheit gegen Störung und Modellfehler gibt.

**Das Optimierungsproblem.** Für eine lineare Anlage $\mathbf x_{k+1}=\mathbf A\mathbf x_k+\mathbf B\mathbf u_k$ löse am aktuellen Zustand $\mathbf x_0$

$$\min_{\mathbf u_0,\dots,\mathbf u_{N-1}}\;\; \mathbf x_N^\top\mathbf P\mathbf x_N + \sum_{k=0}^{N-1}\big(\mathbf x_k^\top\mathbf Q\mathbf x_k + \mathbf u_k^\top\mathbf R\mathbf u_k\big)$$
$$\text{u. d. N.}\quad \mathbf x_{k+1}=\mathbf A\mathbf x_k+\mathbf B\mathbf u_k,\qquad \mathbf u_{\min}\le\mathbf u_k\le\mathbf u_{\max},\qquad \mathbf x_{\min}\le\mathbf x_k\le\mathbf x_{\max}.$$

mit $\mathbf Q\succeq0$, $\mathbf R\succ0$ den Zustands-/Eingangsgewichten und $\mathbf P$ einem Endgewicht.

**Kondensierung in ein QP.** Weil die Dynamik linear ist, ist jeder prädizierte Zustand eine affine Funktion des Anfangszustands und des gestapelten Eingangsvektors $\mathbf U=(\mathbf u_0^\top,\dots,\mathbf u_{N-1}^\top)^\top$:

$$\mathbf X = \mathbf S_x\,\mathbf x_0 + \mathbf S_u\,\mathbf U,$$

wobei $\mathbf S_x,\mathbf S_u$ aus Potenzen von $\mathbf A$ und $\mathbf B$ gebaut sind (die **Prädiktionsmatrizen**). Einsetzen in die Kosten macht sie zu einem **konvexen quadratischen Programm** in $\mathbf U$ allein:

$$\boxed{\;\min_{\mathbf U}\;\tfrac12\mathbf U^\top\mathbf H\mathbf U + \mathbf x_0^\top\mathbf F^\top\mathbf U \quad\text{u. d. N.}\quad \mathbf G\,\mathbf U\le\mathbf w(\mathbf x_0)\;}$$

mit $\mathbf H = \mathbf S_u^\top\bar{\mathbf Q}\mathbf S_u + \bar{\mathbf R}\succ0$ (eine positiv-definite Hesse-Matrix ⇒ ein eindeutiges Optimum) und $\mathbf G,\mathbf w$, die die Eingangs- und (zustandsabhängigen) Zustands-Grenzen kodieren. Löse es (`scipy.optimize`), wende $\mathbf u_0$ an und wiederhole. Das **Final-Projekt** implementiert das von Anfang bis Ende.

**Die Brücke zum LQR (Modul 14).** Lässt man die Constraints weg und nimmt $N\to\infty$: Das MPC-Problem *wird* zum LQR-Problem, und seine Lösung ist dieselbe konstante Verstärkung $\mathbf u_0=-\mathbf K\mathbf x_0$ mit $\mathbf K$ aus der diskreten Riccati-Gleichung. Also **unbeschränktes MPC = LQR** — eine numerische Tatsache, die das Final-Projekt bis auf Maschinengenauigkeit verifiziert. Der Wert von MPC ist alles, was LQR nicht kann: **harte Grenzen einhalten**. Wenn eine Störung den Zustand über eine Grenze drücken würde, treibt LQR (das keine Grenzen kennt) den Eingang über seine Sättigung, und der reale Aktor klippt, was die Regelung verschlechtert; MPC *plant innerhalb der Grenzen* und hält den Zustand zulässig. Die **Endkosten** $\mathbf P$ als LQR-Cost-to-go (die Riccati-Lösung) zu wählen und eine Endbedingung hinzuzufügen, ist der Standardweg zu einer **Stabilitätsgarantie** für den Receding-Horizon-Regler.

### 7. Scheduling, hybride Systeme und das größere Bild (kurz)

Zwei Stränge runden Advanced Automation ab. **Scheduling** — die *Reihenfolge und das Timing* von Operationen auf geteilten Maschinen zu entscheiden (Job-Shop, Flow-Shop) — ist die Optimierung der höheren (MES-) Ebene; sie ist kombinatorisch (oft NP-schwer) und wird mit MILP, Constraint Programming oder Heuristiken gelöst, und sie sitzt direkt auf den Ressourcenmodellen aus Abschnitt 4. **Hybride Systeme** mischen die diskrete und kontinuierliche Welt in einem Modell (ein Thermostat: kontinuierliche Temperatur + diskretes Ein/Aus), formalisiert als **hybride Automaten**; ihre Verifikation und Regelung vereint die zwei Hälften dieses Moduls. Moderne Automatisierung ergänzt zunehmend eine **Lern**-Ebene — RL für Scheduling und Regelung (Module 13/14), gelernte Anomalieerkennung auf Prozessdaten (Modul 16), und **digitale Zwillinge**, die die Modelle dieses Moduls parallel zur realen Anlage laufen lassen zur Prädiktion und Überwachung — aber die Modelle, die gelernt, gezwillingt und optimiert werden, sind genau die Automaten, Netze und Dynamiken, die hier gebaut werden.

---

## Zusammenfassung / Cheat-Sheet

**Automatisierungspyramide**: Feld → Steuerung → SCADA → MES → ERP (unten schnell/detailliert, oben langsam/aggregiert). Zwei Dynamiken: **diskrete-Ereignis** (Ereignisse treiben Zustandssprünge) vs. **zeitgetrieben** (Differentialgleichungen).

**Endlicher Automat**: $G=(Q,\Sigma,\delta,q_0,Q_m)$. Sprache $\mathcal L(G)$ = Ketten mit definiertem $\delta(q_0,s)$; markiert $\mathcal L_m$ = enden in $Q_m$. **Erreichbarkeit** = BFS von $q_0$; **Sicherheit** = schlechter Zustand nicht erreichbar. **Parallelkomposition** $G_1\|G_2$ auf $Q_1\times Q_2$: geteilte Ereignisse synchronisieren, private verschränken.

**Petri-Netz**: $N=(P,T,F,\mathbf M_0)$. Markierung $\mathbf M\in\mathbb N^n$ (Token). $t$ aktiviert, wenn Eingänge genug Token halten; Feuern bewegt Token. **Inzidenz** $\mathbf C=\mathbf{Post}-\mathbf{Pre}$; **Zustandsgleichung** $\mathbf M'=\mathbf M+\mathbf C\mathbf u$. **Platz-Invarianten** $\mathbf y^\top\mathbf C=\mathbf 0$ (erhaltene Token-Summen). **Erreichbarkeitsgraph** → Beschränktheit, Lebendigkeit, **Deadlock** (erreichbare Markierung ohne aktivierte Transition).

**Supervisory Control (RW)**: Anlagensprache $\mathcal L(G)$, Spec/legal $K$, **steuerbar $\Sigma_c$** (deaktivierbar) vs. **nicht-steuerbar $\Sigma_{uc}$** (nicht). Ziel: **supremale steuerbare Untersprache** $\sup\mathcal C(K)$ — größtes legales + steuerbares + nicht-blockierendes Verhalten. **Forbidden-State-Synthese**: schlechte Menge $B$ rückwärts über *nicht-steuerbare* Transitionen bis zum Fixpunkt wachsen; steuerbare Transitionen nach $B$ deaktivieren ⇒ maximal permissiver sicherer Supervisor.

**MPC**: bei jedem Schritt löse $\min\;\mathbf x_N^\top\mathbf P\mathbf x_N+\sum(\mathbf x_k^\top\mathbf Q\mathbf x_k+\mathbf u_k^\top\mathbf R\mathbf u_k)$ u. d. N. Dynamik + $\mathbf u,\mathbf x$-Grenzen; wende $\mathbf u_0$ an; recede. **Kondensieren**: $\mathbf X=\mathbf S_x\mathbf x_0+\mathbf S_u\mathbf U$ → QP $\min\tfrac12\mathbf U^\top\mathbf H\mathbf U+\mathbf x_0^\top\mathbf F^\top\mathbf U$ u. d. N. $\mathbf G\mathbf U\le\mathbf w$, $\mathbf H\succ0$. **Unbeschränkt, $N\to\infty$ ⇒ LQR** (Riccati-Verstärkung). MPCs Vorteil: setzt harte Constraints durch, wo LQR sättigt.

---

## Selbsttest

<details>
<summary><b>1.</b> Was ist die Automatisierungspyramide, und wie unterscheiden sich die Entscheidungen zwischen oben und unten?</summary>

Die Automatisierungspyramide ist die fünfschichtige Referenzstruktur Feld → Steuerung → SCADA → MES → ERP. **Unten** (Feld/Steuerung) sind die Entscheidungen **schnell und detailliert** — ein Regler schließt eine Echtzeitschleife jede Millisekunde auf rohen Sensor-/Aktorsignalen. **Oben** (MES/ERP) sind sie **langsam und aggregiert** — Planung und Scheduling über Stunden bis Wochen auf zusammengefassten Daten. Advanced Automation liefert die Entscheidungsmathematik für die Steuerungs- und Überwachungsebenen in der Mitte.
</details>

<details>
<summary><b>2.</b> Unterscheide diskrete-Ereignis- und zeitgetriebene Dynamik mit je einem Beispiel.</summary>

**Zeitgetriebene (kontinuierliche)** Systeme entwickeln sich kontinuierlich per Differential-/Differenzengleichungen — z. B. der Füllstand eines Tanks $\dot h = (q_{in}-q_{out})/A$, der sich in jedem Augenblick ändert. **Diskrete-Ereignis-**Systeme ändern sich nur bei **diskreten Ereignissen** und sind sonst statisch — z. B. „Teil kommt an", „Maschine fertig", „Ventil öffnet"; der Zustand ist symbolisch (idle/working) und springt bei Ereignissen, wobei nicht die Zeit treibt. Eine reale Anlage enthält beides, weshalb Automatisierung sowohl Automaten/Petri-Netze als auch MPC braucht.
</details>

<details>
<summary><b>3.</b> Definiere einen DFA und seine Sprache, und erkläre, wie man eine Sicherheits-Property verifiziert.</summary>

Ein DFA ist $G=(Q,\Sigma,\delta,q_0,Q_m)$: Zustände, Ereignisalphabet, (partielle) Transitionsfunktion, Anfangszustand, markierte Zustände. Seine **generierte Sprache** $\mathcal L(G)$ sind alle Ereignisketten $s$, für die $\delta(q_0,s)$ definiert ist (alles, was die Anlage tun kann). Eine **Sicherheits-Property** der Form „schlechter Zustand $q_{bad}$ tritt nie auf" wird verifiziert, indem man die **erreichbare Menge** von $q_0$ berechnet (BFS über $\delta$) und prüft, dass $q_{bad}\notin\mathrm{Reach}(G)$ — also unerreichbar ist.
</details>

<details>
<summary><b>4.</b> Was ist Parallelkomposition und warum ist sie wichtig?</summary>

Die Parallelkomposition (synchrones Produkt) $G_1\|G_2$ baut den gemeinsamen Automaten auf $Q_1\times Q_2$: ein **geteiltes** Ereignis (in beiden Alphabeten) kann nur feuern, wenn **beide** Komponenten es nehmen können (sie synchronisieren), während ein **privates** Ereignis von seinem Besitzer allein genommen wird. Sie ist wichtig, weil reale automatisierte Systeme viele interagierende Komponenten sind (Maschine + Puffer + Roboter), und Komposition ihr kombiniertes Modell baut — aber sie verursacht auch **Zustandsexplosion** (das Produkt der Zustandszahlen), was Petri-Netze motiviert.
</details>

<details>
<summary><b>5.</b> Schreibe die Petri-Netz-Zustandsgleichung auf und erkläre jeden Teil.</summary>

$\mathbf M' = \mathbf M + \mathbf C\,\mathbf u$, wobei $\mathbf M,\mathbf M'\in\mathbb N^n$ die Token-Markierungen vor/nach sind, $\mathbf u$ der Feuervektor (welche Transition(en) feuern), und $\mathbf C=\mathbf{Post}-\mathbf{Pre}$ die Inzidenzmatrix (produzierte minus verbrauchte Token pro Transition). Sie sagt, dass das Feuern einer Transition Token gemäß ihrer Spalte von $\mathbf C$ bewegt. Für eine Feuersequenz mit Zählvektor $\boldsymbol\sigma$ gilt $\mathbf M'=\mathbf M_0+\mathbf C\boldsymbol\sigma$. Diese lineare Form ermöglicht Invariantenanalyse ($\mathbf y^\top\mathbf C=\mathbf 0$ gibt erhaltene Token-Summen).
</details>

<details>
<summary><b>6.</b> Was ist ein Deadlock in einem Petri-Netz, und wie findet man einen?</summary>

Ein **Deadlock** ist eine erreichbare Markierung, bei der **keine Transition aktiviert** ist — kein Eingangsplatz hat genug Token für irgendeine Transition, also steckt das System für immer fest. Man findet Deadlocks, indem man den **Erreichbarkeitsgraphen** baut (BFS von $\mathbf M_0$, jede aktivierte Transition feuern, um erreichbare Markierungen zu enumerieren) und die Markierungen ohne **Nachfolger** (keine aktivierte Transition) auflistet. In ressourcenteilenden Systemen sind das typischerweise zirkuläre-Warte-Situationen (jeder Prozess hält eine Ressource, die der andere braucht).
</details>

<details>
<summary><b>7.</b> Warum macht die steuerbar/nicht-steuerbar-Teilung Supervisory Control nicht-trivial?</summary>

Ein Supervisor darf **steuerbare Ereignisse deaktivieren**, aber **nie** **nicht-steuerbare** (ein Fehler, ein ankommendes Teil, eine endende Bewegung passieren, ob man will oder nicht). Es genügt also nicht, den letzten Schritt in einen schlechten Zustand zu blockieren: Wenn die einzigen Transitionen von einem Zustand in den schlechten Bereich **nicht-steuerbar** sind, ist dieser Zustand *selbst* effektiv schlecht (Schlechtigkeit ist von ihm aus unvermeidbar), und man muss verhindern, *ihn* zu erreichen — rekursiv. Daher ist das größte sichere Verhalten die **supremale steuerbare Untersprache**, nicht einfach „die legale Sprache".
</details>

<details>
<summary><b>8.</b> Beschreibe die Forbidden-State-Supervisor-Synthese als Fixpunkt.</summary>

Starte mit der schlechten Menge $B=Q_x$ (verbotene Zustände, z. B. alle Deadlocks). Füge wiederholt **jeden Zustand hinzu, der eine nicht-steuerbare Transition nach $B$ hat** (von einem solchen Zustand kann der Eintritt in $B$ nicht verhindert werden), bis $B$ nicht mehr wächst — ein Rückwärts-Fixpunkt über nicht-steuerbare Kanten. Der resultierende Supervisor **deaktiviert von jedem sicheren Zustand aus genau die steuerbaren Transitionen, die nach $B$ führen**, und aktiviert alles andere. Das ist der **maximal permissive** (am wenigsten restriktive) Supervisor, der die Anlage für immer aus $B$ hält.
</details>

<details>
<summary><b>9.</b> Formuliere das MPC-Optimierungsproblem und wie es in ein QP kondensiert wird.</summary>

Am aktuellen Zustand $\mathbf x_0$ löst MPC $\min\;\mathbf x_N^\top\mathbf P\mathbf x_N+\sum_{k=0}^{N-1}(\mathbf x_k^\top\mathbf Q\mathbf x_k+\mathbf u_k^\top\mathbf R\mathbf u_k)$ unter der Dynamik $\mathbf x_{k+1}=\mathbf A\mathbf x_k+\mathbf B\mathbf u_k$ und Eingangs-/Zustandsgrenzen, wendet $\mathbf u_0$ an und recediert. Die **Kondensierung** nutzt die Linearität, um $\mathbf X=\mathbf S_x\mathbf x_0+\mathbf S_u\mathbf U$ zu schreiben (prädizierte Zustände als affine Funktion der gestapelten Eingänge $\mathbf U$), was die Kosten in ein konvexes QP $\min\tfrac12\mathbf U^\top\mathbf H\mathbf U+\mathbf x_0^\top\mathbf F^\top\mathbf U$ u. d. N. $\mathbf G\mathbf U\le\mathbf w(\mathbf x_0)$ verwandelt, mit $\mathbf H=\mathbf S_u^\top\bar{\mathbf Q}\mathbf S_u+\bar{\mathbf R}\succ0$ (eindeutiges Optimum).
</details>

<details>
<summary><b>10.</b> Wie ist MPC mit LQR verwandt, und was fügt MPC hinzu?</summary>

Lässt man die Constraints weg und den Horizont $N\to\infty$ gehen, *ist* das MPC-Problem das unendlich-Horizont-LQR-Problem, und der optimale erste Zug ist die konstante LQR-Verstärkung $\mathbf u_0=-\mathbf K\mathbf x_0$ aus der diskreten Riccati-Gleichung — also **unbeschränktes MPC = LQR**. Was MPC hinzufügt, ist die **Behandlung harter Constraints**: Es optimiert innerhalb von Eingangs- und Zustandsgrenzen, sodass MPC, wenn eine Störung den Zustand Richtung Grenze drücken würde, eine zulässige Eingangsfolge plant und innerhalb der Grenzen bleibt, während LQR (das Grenzen ignoriert) Eingänge kommandiert, die der Aktor klippen muss, was die Leistung verschlechtert und Zustandsgrenzen verletzen kann.
</details>

---

## Literatur & Quellen

**Lehrbücher — diskrete-Ereignis-Systeme & Supervisory Control**
- **Cassandras & Lafortune, *Introduction to Discrete Event Systems*** (Springer). Der maßgebliche Text zu Automaten, Sprachen, Petri-Netzen und Ramadge–Wonham-Supervisory-Control — Kapitel 2–4 decken Abschnitt 3–5 hier ab. *Vertiefend, die Standardreferenz.*
- **Wonham & Cai, *Supervisory Control of Discrete-Event Systems*** — freie Vorlesungsnotizen/Buch online. Die rigorose RW-Theorie. *Kostenlos, fortgeschritten.*
- **Murata, „Petri Nets: Properties, Analysis and Applications"**, *Proc. IEEE 1989*. Der klassische Überblick der Petri-Netz-Theorie (Inzidenzmatrix, Invarianten, Erreichbarkeit). *Kostenlos, vertiefend.*

**Lehrbücher — MPC**
- **Rawlings, Mayne & Diehl, *Model Predictive Control: Theory, Computation, and Design*** — freies PDF. Die moderne Referenz (Stabilität, Constraints, Kondensierung). *Kostenlos, vertiefend.*
- **Maciejowski, *Predictive Control with Constraints***. Sehr lesbare Einführung mit ausgearbeiteter QP-Kondensierung. *Einsteiger- bis mittelfreundlich.*
- **Borrelli, Bemporad & Morari, *Predictive Control for Linear and Hybrid Systems*** — freies PDF. Verbindet MPC, QP und hybride Systeme. *Kostenlos, fortgeschritten.*

**Schlüssel-Papers**
- **Ramadge & Wonham, „Supervisory Control of a Class of Discrete Event Processes"**, *SIAM J. Control 1987*. Der Ursprung der SCT. *Vertiefend.*
- **Mayne, Rawlings, Rao & Scokaert, „Constrained model predictive control: Stability and optimality"**, *Automatica 2000*. Die Stabilitätsreferenz für MPC. *Vertiefend.*

**Frei verfügbare Kurse**
- **Cassandras / Lafortune DES-Vorlesungen** und diverse Universitäts-DES-Kurse auf YouTube. *Kostenlos.*
- **MPC-Vorlesungen** (z. B. Borrelli in Berkeley, del Re, oder das Begleitmaterial des freien MPC-Buchs). *Kostenlos.*

**Zum Ausprobieren**
- Die **drei Projekte** bauen ein endlicher-Automat-Modell + Erreichbarkeit/Komposition (basic), eine Petri-Netz-Zelle mit Deadlock-Erkennung und Supervisor-Synthese (medium) und einen beschränkten **MPC**-Regler mit dem LQR-Vergleich (final) — alles from scratch, der beste Weg, die drei Ebenen konkret zu machen.

---

> **Nächstes Modul:** Modul 24 „Self-aware Computing" — Systeme, die sich zur Laufzeit selbst überwachen und anpassen (Selbstoptimierung, Selbstheilung), aufbauend auf den Regelungs- und Entscheidungsideen dieses Moduls und der Schätzung der Module 21–22.
