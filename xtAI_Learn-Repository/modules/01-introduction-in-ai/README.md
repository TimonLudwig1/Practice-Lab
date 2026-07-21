# Module 01 — Introduction to AI

> **Language note.** This document is bilingual. The English version comes first; the German version (*deutsche Fassung*) follows below the horizontal rule.

**What is this about?** This module is the foundation of the entire degree programme. It answers: what is artificial intelligence in the first place? How do you formalise "intelligent behaviour" so that a computer can carry it out? You will learn the classical core ideas of AI — agents, problem solving by search, games, logic, reasoning under uncertainty — and get an overview of how modern machine learning (all the way up to large language models) builds on them.

**Prerequisites:** school mathematics (sets, functions, a little probability) and Python basics (variables, loops, functions, lists/dictionaries). No prior AI knowledge required.

**To do beforehand:** nothing — this is module 1.

---

## Learning objectives

After this module you will be able to:

- explain what a **rational agent** is and describe a task using the **PEAS scheme**,
- formulate a problem as a **state space search** (states, actions, goal test, costs),
- apply the most important **search algorithms** (BFS, DFS, uniform-cost, greedy, A\*) and compare their strengths and weaknesses,
- explain what an **admissible heuristic** is and why A\* is optimal with one,
- solve two-player games using **minimax** and **alpha-beta pruning**,
- model a problem as a **constraint satisfaction problem (CSP)**,
- draw simple inferences in **propositional logic** and explain the idea of knowledge representation,
- reason under uncertainty with **Bayes' rule** and explain how a naive Bayes classifier works,
- distinguish the three main kinds of **machine learning** and roughly locate deep learning and LLMs within the AI landscape.

---

## 1. Basics

### 1.1 What is AI? Four points of view

"Artificial intelligence" has been defined in different ways since the 1950s. The standard textbook (Russell & Norvig) sorts the definitions into four quadrants:

|  | **human** | **rational** |
|--|--|--|
| **Thinking** | cognitive modelling ("thinks like a human") | logic, correct inference ("thinks correctly") |
| **Acting** | Turing test ("behaves like a human") | **rational agents ("acts as well as possible")** |

Modern AI research works almost throughout with the fourth view: **AI = building rational agents**. "Rational" here does not mean "omniscient", but rather: *the agent chooses the action that maximises its expected success — given what it perceives and knows.*

> **Intuition:** a satnav is rational if it picks the fastest route according to its maps. The fact that it does not know about an unannounced traffic jam does not make it irrational — it did the best it could with its knowledge.

**A short history in five stages:**

1. **1950–1956**: Turing poses the question "Can machines think?" (Turing test); the 1956 Dartmouth conference coins the term *artificial intelligence*.
2. **1956–1974**: early euphoria — programs solve logic puzzles and play checkers. Search and symbol manipulation dominate ("Good Old-Fashioned AI", GOFAI).
3. **1974–1980s**: the first "AI winter" (expectations disappointed, funding cut), followed by the boom of **expert systems** (rule-based domain knowledge) — and their subsequent disillusionment (second AI winter).
4. **1990s–2010**: the statistical turn — probability and machine learning instead of hand-written rules. In 1997 Deep Blue beats Kasparov at chess.
5. **From 2012**: the **deep learning** revolution (AlexNet wins ImageNet), 2016 AlphaGo, from 2017 the transformer architecture, from around 2020 large language models (GPT, Claude and others).

Note: the classical content of this module (search, logic, probability) is not "outdated" — it is the conceptual skeleton on which modern systems are also described and combined (AlphaGo, for instance, uses tree search *plus* neural networks).

### 1.2 Agents and environments

An **agent** is anything that perceives its environment through **sensors** and acts upon it through **actuators**.

```
        percepts
   Environment ────────────────────▶ Agent
   Environment ◀──────────────────── Agent
              actions
```

A task is described with the **PEAS scheme**:

| Letter | Meaning | Example: self-driving taxi |
|--|--|--|
| **P**erformance | performance measure | safe, fast, legal, comfortable |
| **E**nvironment | environment | roads, traffic, pedestrians, weather |
| **A**ctuators | actuators | steering, throttle, brake, indicators |
| **S**ensors | sensors | cameras, lidar, GPS, speedometer |

**Properties of environments** (important, because they determine which method fits):

- **fully vs. partially observable** — does the agent see the whole relevant state? (Chess: yes. Poker: no.)
- **deterministic vs. stochastic** — is the consequence of an action reliably predictable?
- **episodic vs. sequential** — does the next decision depend on earlier ones?
- **static vs. dynamic** — does the world change while the agent is deliberating?
- **discrete vs. continuous** — finitely many states/actions, or continuous ones?
- **single-agent vs. multi-agent** — are there other players or opponents?

> **Rule of thumb:** the further to the right (partially observable, stochastic, dynamic, continuous, multi-agent), the harder — and the more you need probability and learning rather than pure search.

**Agent architectures**, from simple to powerful:

1. **Simple reflex agent**: `if percept X, then action Y` (thermostat).
2. **Model-based reflex agent**: maintains an internal state about the world (vacuum robot with a map).
3. **Goal-based agent**: plans sequences of actions to reach a goal (satnav) — this leads directly to **search** (section 1.3).
4. **Utility-based agent**: rates states on a graded scale via a **utility function** (not just goal/no goal, but "how good?").
5. **Learning agent**: improves all of the above components from experience — **machine learning** (section 2.5).

### 1.3 Problem solving by search

The central idea of classical AI: many problems can be formulated as a **search in a state space**. This requires five ingredients:

1. **Initial state** — where do I start?
2. **Actions** — what can I do in a state?
3. **Transition model** — which state follows action $a$ in state $s$?
4. **Goal test** — am I done?
5. **Path costs** — what does a sequence of actions cost (steps, kilometres, time, ...)?

**Worked mini example — the 8-puzzle:** a 3x3 sliding puzzle with tiles 1–8 and one hole. State = arrangement of the tiles (there are $9!/2 = 181\,440$ reachable ones), actions = slide the hole up/down/left/right, goal = the sorted arrangement, cost = number of moves. The puzzle is thereby fully formalised as a search problem — the algorithm needs to know nothing about "puzzles".

Search builds a **search tree** starting from the initial state: nodes = states, edges = actions. The not-yet-examined boundary nodes are called the **frontier**. All search algorithms differ only in **which frontier node they expand next**.

#### Uninformed search (knows only the problem, no extra hints)

| Algorithm | Frontier strategy | complete? | optimal? | mnemonic |
|--|--|--|--|--|
| **Breadth-first search (BFS)** | queue (FIFO) | yes | yes (with equal step costs) | all neighbours first, then their neighbours |
| **Depth-first search (DFS)** | stack (LIFO) | no (infinite paths!) | no | always deeper, then backtrack |
| **Uniform-cost (Dijkstra)** | priority queue by path cost $g(n)$ | yes | yes | cheapest known path first |
| **Iterative deepening (IDS)** | DFS with growing depth limit | yes | yes (like BFS) | BFS guarantees at DFS memory cost |

- *Complete* = finds a solution if one exists. *Optimal* = finds the cheapest one.
- BFS needs **exponential memory** ($O(b^d)$ for branching factor $b$ and depth $d$) — in practice this is its death sentence on deep problems; IDS is then the trick.

#### Informed (heuristic) search

A **heuristic** $h(n)$ estimates the remaining cost from node $n$ to the goal. Route planning example: straight-line distance to the goal. 8-puzzle example: number of misplaced tiles, or better the **Manhattan distance** (sum of horizontal + vertical distances of each tile to its goal square).

- **Greedy best-first**: expand the node with the smallest $h(n)$. Fast, but neither complete nor optimal — it runs greedily towards the goal and overlooks better routes.
- **A\***: expand the node with the smallest
$$f(n) = g(n) + h(n)$$
where $g(n)$ = path cost so far, $h(n)$ = estimated remaining cost. A\* therefore combines "what has it cost" with "what will it still cost".

**The central theorem:** if $h$ is **admissible**, i.e. it never overestimates the true remaining cost ($h(n) \le h^*(n)$), then A\* is **optimal**.

> **Intuition for why this holds:** A\* always takes the node with the most optimistic total estimate. When A\* expands a goal, all other open nodes have $f$ values greater than or equal to the goal cost — and since $h$ never overestimates, no cheaper route can lead through them.

For graph search (with duplicate detection) you need the slightly stronger property of **consistency**: $h(n) \le c(n, n') + h(n')$ — along an edge the heuristic may never drop by more than the edge cost (triangle inequality). Consistent implies admissible; both the Manhattan distance and the straight-line distance are consistent.

**Comparing the quality of heuristics:** $h_2$ *dominates* $h_1$ if $h_2(n) \ge h_1(n)$ for all $n$ (both being admissible). Dominant heuristics never expand more nodes — on the 8-puzzle the Manhattan distance clearly beats "misplaced tiles". Ideally $h$ is as large as possible while still admissible.

---

## 2. Intermediate

### 2.1 Games: adversarial search

In two-player zero-sum games (chess, tic-tac-toe) an opponent actively plans *against* us. The solution: **minimax**.

**Idea:** build the game tree. Leaves get a value from the point of view of player MAX (+1 win, 0 draw, −1 loss). Then propagate upwards: MAX nodes take the **maximum** of their children (I choose my best move), MIN nodes the **minimum** (the opponent chooses the worst one for me).

$$\text{Minimax}(s) = \begin{cases} \text{Utility}(s) & s \text{ terminal} \\ \max_{a} \text{Minimax}(\text{Result}(s,a)) & \text{MAX to move} \\ \min_{a} \text{Minimax}(\text{Result}(s,a)) & \text{MIN to move} \end{cases}$$

Minimax plays **perfectly against a perfect opponent**. The problem: the game tree explodes ($b^m$ nodes; chess: $b \approx 35$, $m \approx 80$ — hopeless). Two standard ways out:

1. **Alpha-beta pruning**: cut off subtrees that provably can no longer change the result. $\alpha$ = the best option already guaranteed for MAX along the path, $\beta$ = the best for MIN. As soon as a value $\le \alpha$ appears at a MIN node (or $\ge \beta$ at a MAX node), the remaining children can be ignored — the parent would never choose this branch. With good move ordering the effort drops from $O(b^m)$ to $O(b^{m/2})$ — **twice the search depth for the same price**, and the result is *exactly* the same as with minimax.
2. **Depth limit + evaluation function**: instead of computing down to the leaves, stop at depth $d$ and estimate the position with an **evaluation function** (e.g. material balance in chess). From here on the play is no longer perfect, but it is practical.

Modern game programs (AlphaGo/AlphaZero) replace the hand-built evaluation function with a **learned neural network** and the full expansion with **Monte Carlo tree search (MCTS)** — the basic idea "tree search + position evaluation" remains.

### 2.2 Constraint satisfaction problems (CSPs)

Many problems are not a path search but an **assignment search**: find values for variables such that all constraints are satisfied.

- **Variables** $X_1, \dots, X_n$, each with a **domain** (set of values) $D_i$
- **Constraints**: permitted value combinations

**Examples:** sudoku (81 variables, domain 1–9, row/column/box constraints), map colouring (neighbouring countries get different colours), timetabling (no lecturer booked twice).

**Solution method: backtracking search** — assign variables one after another; if an assignment leads to a contradiction, take it back and try the next value. On its own this is brute force; it becomes clever through three standard improvements:

1. **MRV (minimum remaining values)**: pick as the next variable the one with the *fewest* remaining possible values ("fail first" — spot dead ends early).
2. **Forward checking**: after each assignment, delete incompatible values from the domains of neighbouring variables; if a domain becomes empty, backtrack immediately.
3. **Constraint propagation (AC-3, arc consistency)**: propagate restrictions through the whole network before search even begins. For easy sudokus, propagation alone is often enough without any search — exactly what a human does when "crossing off candidates".

> **Why CSPs get their own chapter:** the constraints give the search *structure* that generic state space search would ignore. The combination "search + propagation" is a recurring AI pattern.

### 2.3 Knowledge representation and logic

A **knowledge-based agent** stores knowledge as sentences in a formal language (**knowledge base, KB**) and derives new conclusions from it (**inference**).

**Propositional logic** — the simplest logic:

- Atoms: $P, Q, R$ (true/false), connectives: $\neg, \land, \lor, \Rightarrow, \Leftrightarrow$
- Central is the notion of **entailment**: $KB \models \alpha$ means: *in every world in which KB is true, $\alpha$ is also true.*
- Checkable e.g. by **truth table** (going through all models — correct, but exponential) or by **resolution** / **forward chaining** (applying rules).

**Worked mini example:** KB = { "if it rains, the road is wet" ($R \Rightarrow W$), "it is raining" ($R$) }. By **modus ponens**, $W$ follows. Conversely, however, $W \not\Rightarrow R$ — from "road is wet" it does not follow that it rained (perhaps it was the street cleaning truck). This mistake (confusing abduction with deduction) is also common in everyday life.

**First-order logic (FOL)** extends this with objects, relations and quantifiers:
$$\forall x\, (\text{Student}(x) \Rightarrow \text{Studies}(x)) \qquad \exists x\, \text{Passes}(x)$$
This lets you describe worlds compactly that would need infinitely many atoms in propositional logic. The price: inference in FOL is only **semi-decidable** (if something follows, you will find it; if not, the search may never terminate).

**Placing this in context:** purely logical AI historically failed on two things: the drudgery of formalising everyday knowledge by hand (**knowledge acquisition bottleneck**), and its inability to deal with *uncertainty*. The former motivates machine learning, the latter the next section. Logic lives on today in databases, verification and logic programming (see the modules *Deductive Databases*, *Logic Programming*).

### 2.4 Reasoning under uncertainty

The real world is stochastic and partially observable. The tool for this is **probability theory**.

Core notions: $P(A)$ (probability), $P(A \mid B)$ (conditional probability: probability of $A$ *given* that $B$ holds), product rule $P(A \land B) = P(A\mid B)\,P(B)$. From this follows the most important formula of the module:

$$P(H \mid E) = \frac{P(E \mid H)\; P(H)}{P(E)} \qquad \text{(Bayes' rule)}$$

Read it like this: **posterior** (belief in hypothesis $H$ after observing $E$) = **likelihood** (how well does $H$ explain the observation?) x **prior** (belief beforehand), normalised by $P(E)$.

**Worked example (the classic that almost everyone gets wrong):** a test detects a disease with 99 % sensitivity ($P(+\mid D) = 0.99$), false-positive rate 5 % ($P(+\mid \neg D) = 0.05$), and the disease affects 1 % of the population ($P(D) = 0.01$). How likely is the disease given a positive test?

$$P(D \mid +) = \frac{0.99 \cdot 0.01}{0.99 \cdot 0.01 + 0.05 \cdot 0.99} = \frac{0.0099}{0.0099 + 0.0495} \approx 0.167$$

Only about **17 %**! The low prior (a rare disease) pushes the posterior down — the many healthy people produce more false positives than the few sick people produce true positives. Seeing through this "base rate neglect" is half the value of Bayes' rule.

**Naive Bayes — Bayes' rule as a classifier:** to sort an object with features $x_1, \dots, x_n$ (e.g. the words of an email) into a class $c$ (spam/ham), one "naively" assumes the features are **conditionally independent given the class**:

$$P(c \mid x_1, \dots, x_n) \;\propto\; P(c) \prod_{i=1}^{n} P(x_i \mid c)$$

The assumption is almost always false (words depend on one another!) — but the classifier nonetheless works surprisingly well, trains in seconds, and was the backbone of spam filters for years. You will build one yourself in the final project.

**Bayesian networks** (outlook): directed graphs that compactly encode dependencies between random variables — instead of one giant table over all variable combinations, only local conditional distributions per node. They are the link between logical AI ("structure") and statistics ("uncertainty"); this is deepened in *Machine Learning 1* and *Theory of AI*.

### 2.5 Machine learning — the overview

Instead of programming behaviour, you let the agent **learn it from data**. Three basic kinds:

| Kind | Given | What is learned | Typical examples |
|--|--|--|--|
| **Supervised learning** | inputs *with* correct outputs (labels) | mapping input → output | spam filter, image recognition, price prediction |
| **Unsupervised learning** | only inputs, no labels | structure in the data | clustering customers, dimensionality reduction |
| **Reinforcement learning** | reward signal after actions | behavioural strategy (policy) | games, robotics, control |

Within supervised learning one distinguishes **classification** (discrete output: spam/ham) and **regression** (continuous output: house price).

Two notions you should learn correctly from the very beginning:

- **Generalisation**: what counts is performance on *new* data, not on the training data. That is why data is always split into a **training set and a test set**.
- **Overfitting**: an overly flexible model memorises the training data (including noise) and fails on new data. Symptom: small training error, large test error.

Much more on this in *Machine Learning 1* — here the map suffices.

---

## 3. Advanced topics

### 3.1 Local search and optimisation

Sometimes the *path* is of no interest, only the *goal state* (e.g. a valid timetable assignment, a good circuit board layout). Then you can dispense with the search tree and "wander" directly in the state space:

- **Hill climbing**: always move to the best neighbouring state. Fast and memory-free ($O(1)$), but gets stuck in **local maxima**, on plateaus and ridges.
- **Simulated annealing**: accept worsening moves with probability $e^{\Delta E / T}$, where the "temperature" $T$ slowly decreases. Initially a lot of randomness (escaping local maxima), at the end almost pure hill climbing. With sufficiently slow cooling it provably finds the global optimum — in practice it is used as a robust compromise.
- **Genetic algorithms**: a *population* of solutions; the best ones are selected, combined by **crossover** and varied by **mutation**. Useful when solutions can be meaningfully combined; often less efficient than problem-specific methods, though.
- **Gradient descent**: in *continuous* spaces you follow the negative gradient $\theta \leftarrow \theta - \eta \nabla f(\theta)$. This is the same "downhill" idea as hill climbing — and it is **the** algorithm with which neural networks are trained. Local search is therefore not a niche topic but the core of deep learning.

### 3.2 Classical planning (briefly)

**Planning** = search with *structured* states: states are sets of logical facts, actions have **preconditions** and **effects** (STRIPS/PDDL formalism):

```
Action: PickUp(x)
  Precondition: gripper free, x is clear
  Effect:       gripper holds x, not(gripper free)
```

The gain over raw state space search: from the structure of the actions one can **derive heuristics automatically** (e.g. "ignore all negative effects" → relaxation → admissible estimate). Planning is the link between search and logic, and the basis of the robotics modules.

### 3.3 From neural networks to LLMs — the modern AI landscape

An **artificial neuron** computes $y = \sigma(\sum_i w_i x_i + b)$ — a weighted sum plus a non-linear activation $\sigma$. Stacking many of these in layers (**deep learning**) lets the network learn hierarchical features: edges → shapes → objects. Training is done by gradient descent with **backpropagation** (efficient gradient computation via the chain rule).

Milestones and what they show conceptually:

- **AlexNet (2012)**: deep learning + GPUs + large amounts of data beat hand-built image features — *learn representations instead of constructing them*.
- **AlphaGo (2016)**: learned evaluation/move networks + Monte Carlo tree search — *learning and search combined*; classical AI and deep learning are not opposites.
- **Transformer (2017)** and **LLMs (from around 2020)**: models that learn to predict the next token on enormous amounts of text develop broad linguistic and factual capabilities. Instruction tuning and RLHF (reinforcement learning from human feedback) turn them into assistants.

**Placing this in the context of this module:** in the vocabulary of agents, an LLM is a learned model that can be made the core of an agent (perception: text/images; actions: text, tool calls). The classical questions — what is the performance measure? how does the agent handle uncertainty? does it plan, or does it merely react? — remain exactly the same. That is why the foundation laid in this module still pays off in the age of LLMs.

### 3.4 Limits, ethics, responsibility

Today this is a mandatory part of any introduction:

- **Bias and fairness**: models learn distortions from their training data (e.g. discriminatory patterns in historical hiring data). "The data are objective" is a fallacy.
- **Explainability (XAI)**: deep models are hard to interpret — problematic for credit, medical or judicial decisions.
- **Robustness**: adversarial examples (minimally altered inputs flip the prediction), distribution shift (the model meets data that look different from the training data).
- **Alignment**: how do you ensure that an optimising agent does what we *mean*, not what we *measure*? (Classic example: an agent maximising a reward measure often finds loopholes — "reward hacking".)
- **Regulation**: with the **AI Act** (in force since 2024, applied in stages) the EU has created a risk-based legal framework — from prohibited practices through high-risk requirements to transparency obligations for generative models.

---

## 4. Summary / cheat sheet

**Agents**
- Rational agent: maximises the expected performance measure given percepts + knowledge
- PEAS: performance, environment, actuators, sensors
- Environment axes: observable, deterministic, episodic, static, discrete, single-agent

**Search**
- Problem = (initial state, actions, transitions, goal test, costs)
- BFS: FIFO, optimal at unit costs, memory $O(b^d)$; DFS: LIFO, low memory, not optimal; UCS: by $g(n)$, optimal; IDS: BFS guarantees, DFS memory
- Greedy: only $h(n)$; **A\***: $f(n) = g(n) + h(n)$
- $h$ admissible ($h \le h^*$) implies A\* optimal; consistent: $h(n) \le c(n,n') + h(n')$
- A better (dominant) heuristic implies fewer expanded nodes

**Games**
- Minimax: MAX maximises, MIN minimises; perfect against a perfect opponent
- Alpha-beta: same result, up to $O(b^{m/2})$; move ordering is decisive
- In practice: depth limit + evaluation function (or a learned network + MCTS)

**CSP**
- Variables + domains + constraints; backtracking + MRV + forward checking + AC-3

**Logic**
- $KB \models \alpha$: $\alpha$ holds in all models of the KB
- Modus ponens: from $P \Rightarrow Q$ and $P$ follows $Q$ (the converse does not hold!)
- FOL: objects, relations, $\forall$, $\exists$ — more powerful, inference only semi-decidable

**Uncertainty**
- Bayes: $P(H\mid E) = P(E\mid H)P(H)/P(E)$ — posterior proportional to likelihood x prior
- Do not forget base rates (the 17 % example!)
- Naive Bayes: $P(c\mid x) \propto P(c)\prod_i P(x_i\mid c)$ — "naive" = conditional independence

**ML map**
- supervised (labels), unsupervised (structure), reinforcement (reward)
- Generalisation beats training performance; overfitting = memorised
- Deep learning = layered neurons + gradient descent + backpropagation

---

## 5. Self-test

Try to answer on your own first, then unfold.

<details><summary><b>1. Why does modern AI define intelligence via "rational action" rather than via "human thinking"?</b></summary>

Rational action is an *objective, measurable* criterion: maximise the expected performance measure given percepts and knowledge. Human thinking, by contrast, is hard to specify, partly faulty (cognitive biases), and as a design specification neither necessary nor sufficient — just as aeroplanes do not fly by flapping feathers. In addition, rationality can be analysed mathematically (optimality, guarantees).
</details>

<details><summary><b>2. Formulate "vacuum robot cleans a flat" as a PEAS description.</b></summary>

**P**: area cleaned per unit time, battery consumption, no damage/falls (stairs!). **E**: flat with rooms, furniture, carpets, people/pets (dynamic, partially observable). **A**: wheels/motors, suction unit, brushes. **S**: bump sensors, cliff sensors, possibly lidar/camera, battery level.
</details>

<details><summary><b>3. Why is DFS not complete, and with which trick do you get DFS memory usage and BFS guarantees at the same time?</b></summary>

DFS can run into infinitely deep (or, with cycles, endless) paths and never reach the goal even though it lies at shallow depth. The trick is **iterative deepening (IDS)**: DFS with depth limit 0, 1, 2, ... The shallow levels are expanded several times, but because the lowest level dominates exponentially, this costs only a constant factor.
</details>

<details><summary><b>4. A heuristic overestimates the remaining cost at exactly one node. What can go wrong in A*?</b></summary>

A\* can become **suboptimal**: if the overestimated node lies on the (true) optimal path, it gets too high an $f$ value and may be deferred behind a worse goal — A\* then returns the more expensive path. A\* nevertheless remains complete (on finite graphs); only the optimality guarantee depends on admissibility.
</details>

<details><summary><b>5. Why does alpha-beta pruning give exactly the same result as minimax even though it skips subtrees?</b></summary>

It only skips subtrees that *provably* cannot influence the final result: if a value $\le \alpha$ has already been found at a MIN node, we know MAX will never choose this node (MAX already has $\alpha$ secured elsewhere) — the exact value of the node is then irrelevant. Nothing is "estimated"; what is unnecessary simply is not computed.
</details>

<details><summary><b>6. Model sudoku as a CSP and explain what AC-3 achieves there.</b></summary>

81 variables (cells), domains {1,...,9} (given cells: singleton), constraints: all values in each row, column and 3x3 box pairwise different. AC-3 makes the arcs consistent: if a 5 is fixed in a cell, the 5 is removed from the domains of all row/column/box neighbours; removals trigger further checks. This corresponds to the human "crossing off candidates" and solves easy sudokus with no search at all.
</details>

<details><summary><b>7. From "if it rains, the road is wet" and "the road is wet" someone concludes "it is raining". What is this fallacy called?</b></summary>

**Affirming the consequent**: from $R \Rightarrow W$ and $W$, logically *nothing* follows about $R$ — the road may be wet for other reasons. Valid would be modus ponens ($R \Rightarrow W$, $R$ therefore $W$) or modus tollens ($R \Rightarrow W$, $\neg W$ therefore $\neg R$). As a *probabilistic* inference (abduction), "rain has become more likely" can be perfectly reasonable — for that you need Bayes' rule rather than logic.
</details>

<details><summary><b>8. Test: 99 % sensitivity, 5 % false-positive rate, disease prevalence 0.1 % (instead of 1 %). Posterior given a positive test?</b></summary>

$P(D\mid+) = \frac{0.99 \cdot 0.001}{0.99 \cdot 0.001 + 0.05 \cdot 0.999} = \frac{0.00099}{0.00099 + 0.04995} \approx 0.019$ — barely **2 %**. The rarer the disease, the more strongly the prior dominates: almost all positive tests are false positives.
</details>

<details><summary><b>9. Why does naive Bayes often work well even though its independence assumption is almost always violated?</b></summary>

For *classification*, only the **ranking** of the class posteriors has to be right, not their exact value. The independence assumption distorts the probabilities (often making them overconfident) but rarely flips the ordering. On top of that: few parameters imply little overfitting, especially with small data sets and many features (text!).
</details>

<details><summary><b>10. Categorise: is a chess program with alpha-beta "learning AI"? Is an LLM an "agent"?</b></summary>

Alpha-beta chess is AI (rational action through search) but *not* learning — all behaviour sits in the algorithm plus the evaluation function. An LLM on its own is first of all a learned *model* (text in, text out). It becomes an **agent** when embedded in a perception-action loop: it receives observations (user input, tool results), chooses actions (replies, tool calls) and pursues a goal. Agent is a *role*, not a model type.
</details>

---

## 6. Literature and sources

**Textbooks**

- **Russell & Norvig — "Artificial Intelligence: A Modern Approach" (AIMA), 4th ed.** — *the* standard work, covers this module almost one to one. Chapters 1–2 (introduction, agents), 3 (search), 5 (games), 6 (CSPs), 7–9 (logic), 12–13 (uncertainty, Bayes). *(advanced, but very readable; a German translation exists)*
- **Poole & Mackworth — "Artificial Intelligence: Foundations of Computational Agents", 3rd ed.** — completely **free** online: https://artint.info *(beginner-friendly)*

**Online courses (free)**

- **UC Berkeley CS188 — Intro to AI**: lecture videos, slides and the famous Pac-Man projects freely available: https://inst.eecs.berkeley.edu/~cs188/ *(beginner-friendly, covers exactly search/games/CSP/Bayes — the best companion to this module)*
- **Harvard CS50's Introduction to AI with Python** (edX/YouTube, free) *(very beginner-friendly, practically oriented)*
- **MIT 6.034 Artificial Intelligence** (OpenCourseWare, Patrick Winston) *(classic, excellent lectures)*

**Interactive visualisations and blog posts (free)**

- *Red Blob Games — Introduction to A\**: https://www.redblobgames.com/pathfinding/a-star/introduction.html — the best interactive A\* explanation on the web *(beginner-friendly, required reading before project 1)*
- *Setosa — Conditional probability visualized*: https://setosa.io/ev/conditional-probability/ *(beginner-friendly)*
- 3Blue1Brown: *Bayes theorem* (YouTube) — geometric intuition for Bayes' rule *(beginner-friendly)*

**Historical / advanced**

- Turing (1950): *Computing Machinery and Intelligence* — the original on the Turing test, very readable. *(free online, advanced)*
- Silver et al. (2016): *Mastering the game of Go with deep neural networks and tree search* (Nature) — AlphaGo: search + learning combined. *(advanced)*

---

**Next step:** off to the projects → `projects/01-basic/` (A\* pathfinding), then `projects/02-medium/` (tic-tac-toe with minimax), then `projects/03-final/` (spam filter with naive Bayes on real data).

---
---

# Modul 01 — Introduction in AI (deutsche Fassung)

**Worum geht es?** Dieses Modul ist das Fundament des gesamten Studiengangs. Es beantwortet: Was ist Künstliche Intelligenz überhaupt? Wie formalisiert man „intelligentes Verhalten" so, dass ein Computer es ausführen kann? Du lernst die klassischen Kernideen der KI — Agenten, Problemlösen durch Suche, Spiele, Logik, Schließen unter Unsicherheit — und bekommst einen Überblick, wie das moderne maschinelle Lernen (bis hin zu großen Sprachmodellen) darauf aufbaut.

**Vorkenntnisse:** Schulmathematik (Mengen, Funktionen, ein wenig Wahrscheinlichkeit) und Python-Grundlagen (Variablen, Schleifen, Funktionen, Listen/Dictionaries). Keine KI-Vorkenntnisse nötig.

**Vorher zu machen:** Nichts — das hier ist Modul 1.

---

## Lernziele

Nach diesem Modul kannst du:

- erklären, was ein **rationaler Agent** ist und eine Aufgabe mit dem **PEAS-Schema** beschreiben,
- ein Problem als **Zustandsraum-Suche** formulieren (Zustände, Aktionen, Zieltest, Kosten),
- die wichtigsten **Suchalgorithmen** (BFS, DFS, Uniform-Cost, Greedy, A\*) anwenden und ihre Stärken/Schwächen vergleichen,
- erklären, was eine **zulässige Heuristik** ist und warum A\* damit optimal ist,
- Zwei-Personen-Spiele mit **Minimax** und **Alpha-Beta-Pruning** lösen,
- ein Problem als **Constraint Satisfaction Problem (CSP)** modellieren,
- einfache Schlüsse in **Aussagenlogik** ziehen und die Idee der Wissensrepräsentation erklären,
- mit der **Bayes-Regel** unter Unsicherheit schließen und erklären, wie ein Naive-Bayes-Klassifikator funktioniert,
- die drei Hauptarten des **maschinellen Lernens** unterscheiden und grob einordnen, wo Deep Learning und LLMs in die KI-Landschaft gehören.

---

## 1. Grundlagen (Basics)

### 1.1 Was ist KI? Vier Sichtweisen

„Künstliche Intelligenz" wird seit den 1950ern unterschiedlich definiert. Das Standardlehrbuch (Russell & Norvig) sortiert die Definitionen in vier Quadranten:

|  | **menschlich** | **rational** |
|--|--|--|
| **Denken** | kognitive Modellierung („denkt wie ein Mensch") | Logik, korrektes Schließen („denkt richtig") |
| **Handeln** | Turing-Test („verhält sich wie ein Mensch") | **rationale Agenten („handelt bestmöglich")** |

Die moderne KI-Forschung arbeitet fast durchgängig mit der vierten Sichtweise: **KI = das Bauen rationaler Agenten**. „Rational" heißt dabei nicht „allwissend", sondern: *Der Agent wählt die Aktion, die seinen erwarteten Erfolg maximiert — gegeben das, was er wahrnimmt und weiß.*

> **Intuition:** Ein Navi ist rational, wenn es die (nach seinen Karten) schnellste Route wählt. Dass es einen unangekündigten Stau nicht kennt, macht es nicht irrational — es hat mit seinem Wissen das Beste getan.

**Kurze Geschichte in fünf Etappen:**

1. **1950–1956**: Turing stellt die Frage „Can machines think?" (Turing-Test); die Dartmouth-Konferenz 1956 prägt den Begriff *Artificial Intelligence*.
2. **1956–1974**: Frühe Euphorie — Programme lösen Logikrätsel und spielen Dame. Suche und Symbolverarbeitung dominieren („Good Old-Fashioned AI", GOFAI).
3. **1974–1980er**: Erster „AI Winter" (Erwartungen enttäuscht, Förderung gestrichen), danach Boom der **Expertensysteme** (regelbasiertes Fachwissen) — und deren Ernüchterung (zweiter AI Winter).
4. **1990er–2010**: Statistische Wende: Wahrscheinlichkeit und maschinelles Lernen statt handgeschriebener Regeln. 1997 schlägt Deep Blue Kasparow im Schach.
5. **ab 2012**: **Deep Learning**-Revolution (AlexNet gewinnt ImageNet), 2016 AlphaGo, ab 2017 die Transformer-Architektur, ab ~2020 große Sprachmodelle (GPT, Claude & Co.).

Merke: Die klassischen Inhalte dieses Moduls (Suche, Logik, Wahrscheinlichkeit) sind nicht „veraltet" — sie sind das begriffliche Skelett, auf dem auch moderne Systeme beschrieben und kombiniert werden (z. B. nutzt AlphaGo Baumsuche *plus* neuronale Netze).

### 1.2 Agenten und Umgebungen

Ein **Agent** ist alles, was seine Umgebung über **Sensoren** wahrnimmt und über **Aktuatoren** auf sie einwirkt.

```
        Wahrnehmungen (percepts)
   Umgebung ────────────────────▶ Agent
   Umgebung ◀──────────────────── Agent
              Aktionen (actions)
```

Eine Aufgabe beschreibt man mit dem **PEAS-Schema**:

| Buchstabe | Bedeutung | Beispiel: selbstfahrendes Taxi |
|--|--|--|
| **P**erformance | Erfolgsmaß | sicher, schnell, legal, komfortabel |
| **E**nvironment | Umgebung | Straßen, Verkehr, Fußgänger, Wetter |
| **A**ctuators | Aktuatoren | Lenkung, Gas, Bremse, Blinker |
| **S**ensors | Sensoren | Kameras, Lidar, GPS, Tacho |

**Eigenschaften von Umgebungen** (wichtig, weil sie bestimmen, welche Methode passt):

- **vollständig vs. teilweise beobachtbar** — sieht der Agent den ganzen relevanten Zustand? (Schach: ja. Poker: nein.)
- **deterministisch vs. stochastisch** — ist die Folge einer Aktion sicher vorhersagbar?
- **episodisch vs. sequenziell** — hängt die nächste Entscheidung von früheren ab?
- **statisch vs. dynamisch** — verändert sich die Welt, während der Agent nachdenkt?
- **diskret vs. stetig** — endlich viele Zustände/Aktionen oder kontinuierlich?
- **Einzelagent vs. Multiagent** — gibt es Mitspieler/Gegner?

> **Faustregel:** Je weiter rechts (teilweise beobachtbar, stochastisch, dynamisch, stetig, multiagent), desto schwieriger — und desto eher braucht man Wahrscheinlichkeit und Lernen statt reiner Suche.

**Agentenarchitekturen**, von einfach nach mächtig:

1. **Einfacher Reflexagent**: `wenn Wahrnehmung X, dann Aktion Y` (Thermostat).
2. **Modellbasierter Reflexagent**: hält einen internen Zustand über die Welt (Staubsaugerroboter mit Karte).
3. **Zielbasierter Agent**: plant Aktionsfolgen, um ein Ziel zu erreichen (Navi) → führt direkt zur **Suche** (Abschnitt 1.3).
4. **Nutzenbasierter Agent**: bewertet Zustände graduell über eine **Utility-Funktion** (nicht nur Ziel/kein Ziel, sondern „wie gut?").
5. **Lernender Agent**: verbessert alle obigen Komponenten aus Erfahrung → **maschinelles Lernen** (Abschnitt 2.5).

### 1.3 Problemlösen durch Suche

Die zentrale Idee der klassischen KI: Viele Probleme lassen sich als **Suche in einem Zustandsraum** formulieren. Dazu braucht man fünf Zutaten:

1. **Anfangszustand** — wo starte ich?
2. **Aktionen** — was kann ich in einem Zustand tun?
3. **Übergangsmodell** — welcher Zustand folgt auf Aktion $a$ in Zustand $s$?
4. **Zieltest** — bin ich fertig?
5. **Pfadkosten** — was kostet eine Aktionsfolge (Schritte, Kilometer, Zeit …)?

**Durchgerechnetes Mini-Beispiel — das 8-Puzzle:** Ein 3×3-Schiebepuzzle mit Steinen 1–8 und einem Loch. Zustand = Anordnung der Steine (es gibt $9!/2 = 181\,440$ erreichbare), Aktionen = Loch nach oben/unten/links/rechts schieben, Ziel = sortierte Anordnung, Kosten = Anzahl Züge. Damit ist das Puzzle vollständig als Suchproblem formalisiert — der Algorithmus muss nichts über „Puzzles" wissen.

Die Suche baut vom Anfangszustand aus einen **Suchbaum**: Knoten = Zustände, Kanten = Aktionen. Die noch nicht untersuchten Randknoten heißen **Frontier** (Grenze). Alle Suchalgorithmen unterscheiden sich nur darin, **welchen Frontier-Knoten sie als Nächstes expandieren**.

#### Uninformierte Suche (kennt nur das Problem, keine Zusatzhinweise)

| Algorithmus | Frontier-Strategie | vollständig? | optimal? | Merksatz |
|--|--|--|--|--|
| **Breitensuche (BFS)** | Warteschlange (FIFO) | ja | ja (bei gleichen Schrittkosten) | erst alle Nachbarn, dann deren Nachbarn |
| **Tiefensuche (DFS)** | Stapel (LIFO) | nein (Endlospfade!) | nein | immer weiter in die Tiefe, dann Backtracking |
| **Uniform-Cost (Dijkstra)** | Prioritätswarteschlange nach Pfadkosten $g(n)$ | ja | ja | billigsten bekannten Pfad zuerst |
| **Iterative Vertiefung (IDS)** | DFS mit wachsendem Tiefenlimit | ja | ja (wie BFS) | BFS-Garantien mit DFS-Speicherbedarf |

- *Vollständig* = findet eine Lösung, wenn eine existiert. *Optimal* = findet die billigste.
- BFS braucht **exponentiell viel Speicher** ($O(b^d)$ bei Verzweigungsfaktor $b$ und Tiefe $d$) — das ist in der Praxis sein Todesurteil bei tiefen Problemen; IDS ist dann der Trick.

#### Informierte (heuristische) Suche

Eine **Heuristik** $h(n)$ schätzt die Restkosten vom Knoten $n$ zum Ziel. Beispiel Routenplanung: Luftlinie zum Ziel. Beispiel 8-Puzzle: Anzahl falsch liegender Steine, oder besser die **Manhattan-Distanz** (Summe der horizontalen+vertikalen Abstände jedes Steins zu seinem Zielfeld).

- **Greedy Best-First**: expandiere den Knoten mit kleinstem $h(n)$. Schnell, aber weder vollständig noch optimal — rennt gierig Richtung Ziel und übersieht bessere Wege.
- **A\***: expandiere den Knoten mit kleinstem
$$f(n) = g(n) + h(n)$$
wobei $g(n)$ = bisherige Pfadkosten, $h(n)$ = geschätzte Restkosten. A\* kombiniert also „was hat es gekostet" mit „was wird es noch kosten".

**Der zentrale Satz:** Ist $h$ **zulässig** (admissible), d. h. überschätzt die echten Restkosten nie ($h(n) \le h^*(n)$), dann ist A\* **optimal**.

> **Intuition, warum das stimmt:** A\* nimmt sich immer den Knoten mit dem optimistischsten Gesamtschätzwert vor. Wenn A\* ein Ziel expandiert, haben alle anderen offenen Knoten $f$-Werte ≥ den Zielkosten — und da $h$ nie überschätzt, kann über sie kein billigerer Weg mehr führen.

Für Graphensuche (mit Duplikat-Erkennung) braucht man die etwas stärkere **Konsistenz**: $h(n) \le c(n, n') + h(n')$ — die Heuristik darf entlang einer Kante nie stärker fallen als die Kantenkosten (Dreiecksungleichung). Konsistent ⇒ zulässig; die Manhattan-Distanz und die Luftlinie sind beides.

**Qualitätsvergleich von Heuristiken:** $h_2$ *dominiert* $h_1$, wenn $h_2(n) \ge h_1(n)$ für alle $n$ (bei Zulässigkeit beider). Dominante Heuristiken expandieren nie mehr Knoten — beim 8-Puzzle schlägt Manhattan-Distanz die „falsch liegenden Steine" deutlich. Ideal ist $h$ so groß wie möglich, aber noch zulässig.

---

## 2. Aufbau (Intermediate)

### 2.1 Spiele: Adversariale Suche

Bei Zwei-Personen-Nullsummenspielen (Schach, Tic-Tac-Toe) plant ein Gegner aktiv *gegen* uns. Lösung: **Minimax**.

**Idee:** Baue den Spielbaum auf. Blätter bekommen einen Wert aus Sicht von Spieler MAX (+1 Gewinn, 0 Remis, −1 Verlust). Dann propagiere nach oben: MAX-Knoten nehmen das **Maximum** ihrer Kinder (ich wähle meinen besten Zug), MIN-Knoten das **Minimum** (der Gegner wählt den für mich schlimmsten).

$$\text{Minimax}(s) = \begin{cases} \text{Utility}(s) & s \text{ terminal} \\ \max_{a} \text{Minimax}(\text{Result}(s,a)) & \text{MAX am Zug} \\ \min_{a} \text{Minimax}(\text{Result}(s,a)) & \text{MIN am Zug} \end{cases}$$

Minimax spielt **perfekt gegen einen perfekten Gegner**. Problem: Der Spielbaum explodiert ($b^m$ Knoten; Schach: $b \approx 35$, $m \approx 80$ → hoffnungslos). Zwei Standard-Auswege:

1. **Alpha-Beta-Pruning**: schneide Teilbäume ab, die das Ergebnis beweisbar nicht mehr ändern können. $\alpha$ = beste bereits garantierte Option für MAX auf dem Pfad, $\beta$ = beste für MIN. Sobald an einem MIN-Knoten ein Wert ≤ $\alpha$ auftaucht (oder an einem MAX-Knoten ≥ $\beta$), kann man den Rest der Kinder ignorieren — der Elternknoten würde diesen Zweig nie wählen. Bei guter Zugsortierung sinkt der Aufwand von $O(b^m)$ auf $O(b^{m/2})$ — **doppelte Suchtiefe zum gleichen Preis**, und das Ergebnis ist *exakt* dasselbe wie bei Minimax.
2. **Tiefenlimit + Bewertungsfunktion**: statt bis zu den Blättern zu rechnen, brich bei Tiefe $d$ ab und schätze die Stellung mit einer **Evaluationsfunktion** (z. B. Materialbilanz im Schach). Ab hier ist das Spiel nicht mehr perfekt, aber praktikabel.

Moderne Spielprogramme (AlphaGo/AlphaZero) ersetzen die handgebaute Bewertungsfunktion durch ein **gelerntes neuronales Netz** und die vollständige Expansion durch **Monte-Carlo Tree Search (MCTS)** — die Grundidee „Baumsuche + Stellungsbewertung" bleibt.

### 2.2 Constraint Satisfaction Problems (CSPs)

Viele Probleme sind keine Pfadsuche, sondern eine **Belegungssuche**: Finde Werte für Variablen, sodass alle Nebenbedingungen erfüllt sind.

- **Variablen** $X_1, \dots, X_n$, jede mit **Domäne** (Wertemenge) $D_i$
- **Constraints**: erlaubte Wertkombinationen

**Beispiele:** Sudoku (81 Variablen, Domäne 1–9, Zeilen/Spalten/Box-Constraints), Kartenfärbung (Nachbarländer verschieden färben), Stundenplanung (kein Dozent doppelt belegt).

**Lösungsverfahren: Backtracking-Suche** — belege Variablen der Reihe nach; führt eine Belegung zum Widerspruch, nimm sie zurück und probiere den nächsten Wert. Das allein ist rohe Gewalt; klug wird es durch drei Standard-Verbesserungen:

1. **MRV (Minimum Remaining Values)**: wähle als nächstes die Variable mit den *wenigsten* noch möglichen Werten („fail first" — Sackgassen früh erkennen).
2. **Forward Checking**: nach jeder Belegung streiche unvereinbare Werte aus den Domänen der Nachbarvariablen; wird eine Domäne leer → sofort backtracken.
3. **Constraint Propagation (AC-3, Kantenkonsistenz)**: propagiere Einschränkungen durchs ganze Netz, bevor überhaupt gesucht wird. Bei leichten Sudokus reicht Propagation oft ganz ohne Suche — genau das macht ein Mensch, der „Kandidaten streicht".

> **Warum CSPs ein eigenes Kapitel sind:** Die Constraints geben der Suche *Struktur*, die generische Zustandsraumsuche ignorieren würde. Die Kombination „Suche + Propagation" ist ein wiederkehrendes KI-Muster.

### 2.3 Wissensrepräsentation und Logik

Ein **wissensbasierter Agent** speichert Wissen als Sätze in einer formalen Sprache (**Knowledge Base, KB**) und leitet daraus neue Schlüsse ab (**Inferenz**).

**Aussagenlogik (propositional logic)** — die einfachste Logik:

- Atome: $P, Q, R$ (wahr/falsch), Junktoren: $\neg, \land, \lor, \Rightarrow, \Leftrightarrow$
- Zentral ist der Begriff der **Folgerung (entailment)**: $KB \models \alpha$ heißt: *In jeder Welt, in der KB wahr ist, ist auch $\alpha$ wahr.*
- Prüfbar z. B. per **Wahrheitstabelle** (alle Modelle durchgehen — korrekt, aber exponentiell) oder per **Resolution** / **Forward Chaining** (Regeln anwenden).

**Durchgerechnetes Mini-Beispiel:** KB = { „Wenn es regnet, ist die Straße nass" ($R \Rightarrow N$), „Es regnet" ($R$) }. Mit **Modus Ponens** folgt $N$. Umgekehrt gilt aber $N \not\Rightarrow R$ — aus „Straße nass" folgt nicht „Regen" (vielleicht war es der Sprengwagen). Dieser Fehler (Abduktion mit Deduktion verwechseln) ist auch im Alltag häufig.

**Prädikatenlogik (first-order logic, FOL)** erweitert das um Objekte, Relationen und Quantoren:
$$\forall x\, (\text{Student}(x) \Rightarrow \text{Lernt}(x)) \qquad \exists x\, \text{Besteht}(x)$$
Damit kann man Welten kompakt beschreiben, die in Aussagenlogik unendlich viele Atome bräuchten. Preis: Inferenz in FOL ist nur noch **semi-entscheidbar** (wenn etwas folgt, findet man es; wenn nicht, terminiert die Suche evtl. nie).

**Einordnung:** Reine Logik-KI scheiterte historisch an zwei Dingen: der Mühsal, Alltagswissen von Hand zu formalisieren (**Knowledge Acquisition Bottleneck**), und ihrer Unfähigkeit, mit *Unsicherheit* umzugehen. Ersteres motiviert maschinelles Lernen, letzteres den nächsten Abschnitt. Logik lebt heute u. a. in Datenbanken, Verifikation und logischer Programmierung weiter (→ Module *Deduktive Datenbanken*, *Logische Programmierung*).

### 2.4 Schließen unter Unsicherheit

Die echte Welt ist stochastisch und teilweise beobachtbar. Das Werkzeug dafür ist die **Wahrscheinlichkeitstheorie**.

Kernbegriffe: $P(A)$ (Wahrscheinlichkeit), $P(A \mid B)$ (bedingte W'keit: Wahrscheinlichkeit von $A$, *gegeben* dass $B$ gilt), Produktregel $P(A \land B) = P(A\mid B)\,P(B)$. Daraus folgt die wichtigste Formel des Moduls:

$$P(H \mid E) = \frac{P(E \mid H)\; P(H)}{P(E)} \qquad \text{(Bayes-Regel)}$$

Lies sie so: **Posterior** (Glaube an Hypothese $H$ nach Beobachtung $E$) = **Likelihood** (wie gut erklärt $H$ die Beobachtung?) × **Prior** (Glaube vorher), normiert durch $P(E)$.

**Durchgerechnetes Beispiel (der Klassiker, den fast jeder falsch schätzt):** Ein Test erkennt eine Krankheit zu 99 % ($P(+\mid K) = 0{,}99$), Falsch-Positiv-Rate 5 % ($P(+\mid \neg K) = 0{,}05$), die Krankheit betrifft 1 % der Bevölkerung ($P(K) = 0{,}01$). Wie wahrscheinlich ist Krankheit bei positivem Test?

$$P(K \mid +) = \frac{0{,}99 \cdot 0{,}01}{0{,}99 \cdot 0{,}01 + 0{,}05 \cdot 0{,}99} = \frac{0{,}0099}{0{,}0099 + 0{,}0495} \approx 0{,}167$$

Nur **~17 %**! Der niedrige Prior (seltene Krankheit) drückt den Posterior — die vielen Gesunden produzieren mehr Falsch-Positive als die wenigen Kranken echte Positive. Diese „Base-Rate-Vernachlässigung" zu durchschauen ist der halbe Wert der Bayes-Regel.

**Naive Bayes — die Bayes-Regel als Klassifikator:** Um ein Objekt mit Merkmalen $x_1, \dots, x_n$ (z. B. die Wörter einer E-Mail) in eine Klasse $c$ (Spam/Ham) einzuordnen, nimmt man „naiv" an, die Merkmale seien **gegeben die Klasse unabhängig**:

$$P(c \mid x_1, \dots, x_n) \;\propto\; P(c) \prod_{i=1}^{n} P(x_i \mid c)$$

Die Annahme ist fast immer falsch (Wörter hängen voneinander ab!) — aber der Klassifikator funktioniert trotzdem erstaunlich gut, ist in Sekunden trainiert und war jahrelang das Rückgrat von Spamfiltern. Du baust im Final-Projekt selbst einen.

**Bayes-Netze** (Ausblick): gerichtete Graphen, die Abhängigkeiten zwischen Zufallsvariablen kompakt kodieren — statt einer Riesentabelle über alle Variablenkombinationen nur lokale bedingte Verteilungen pro Knoten. Sie sind das Bindeglied zwischen Logik-KI („Struktur") und Statistik („Unsicherheit"); Vertiefung folgt in *Machine Learning 1* und *Theorie der KI*.

### 2.5 Maschinelles Lernen — der Überblick

Statt Verhalten zu programmieren, lässt man den Agenten es **aus Daten lernen**. Drei Grundarten:

| Art | Gegeben | Gelernt wird | Typische Beispiele |
|--|--|--|--|
| **Supervised Learning** | Eingaben *mit* richtigen Ausgaben (Labels) | Abbildung Eingabe → Ausgabe | Spamfilter, Bilderkennung, Preisvorhersage |
| **Unsupervised Learning** | nur Eingaben, keine Labels | Struktur in den Daten | Clustering von Kunden, Dimensionsreduktion |
| **Reinforcement Learning** | Belohnungssignal nach Aktionen | Verhaltensstrategie (Policy) | Spiele, Robotik, Regelung |

Beim Supervised Learning unterscheidet man **Klassifikation** (diskrete Ausgabe: Spam/Ham) und **Regression** (stetige Ausgabe: Hauspreis).

Zwei Begriffe, die du von Anfang an richtig lernen solltest:

- **Generalisierung**: Es zählt die Leistung auf *neuen* Daten, nicht auf den Trainingsdaten. Deshalb teilt man Daten immer in **Trainings- und Testmenge**.
- **Overfitting**: Ein zu flexibles Modell lernt die Trainingsdaten auswendig (inklusive Rauschen) und versagt auf neuen Daten. Symptom: Trainingsfehler klein, Testfehler groß.

Mehr dazu ausführlich in *Machine Learning 1* — hier reicht die Landkarte.

---

## 3. Advanced-Themen

### 3.1 Lokale Suche und Optimierung

Manchmal interessiert nicht der *Pfad*, sondern nur der *Zielzustand* (z. B. eine gültige Stundenplan-Belegung, ein gutes Platinenlayout). Dann kann man auf den Suchbaum verzichten und direkt im Zustandsraum „wandern":

- **Hill Climbing**: gehe immer zum besten Nachbarzustand. Schnell und speicherfrei ($O(1)$), bleibt aber in **lokalen Maxima**, auf Plateaus und Graten hängen.
- **Simulated Annealing**: akzeptiere Verschlechterungen mit Wahrscheinlichkeit $e^{\Delta E / T}$, wobei die „Temperatur" $T$ langsam sinkt. Anfangs viel Zufall (Entkommen aus lokalen Maxima), am Ende fast reines Hill Climbing. Bei hinreichend langsamer Abkühlung findet es beweisbar das globale Optimum — praktisch nutzt man es als robusten Kompromiss.
- **Genetische Algorithmen**: eine *Population* von Lösungen; die besten werden selektiert, per **Crossover** kombiniert und per **Mutation** variiert. Nützlich, wenn Lösungen sich sinnvoll kombinieren lassen; oft aber weniger effizient als problemspezifische Verfahren.
- **Gradient Descent**: in *stetigen* Räumen folgt man dem negativen Gradienten $\theta \leftarrow \theta - \eta \nabla f(\theta)$. Das ist derselbe „bergab"-Gedanke wie Hill Climbing — und **der** Algorithmus, mit dem neuronale Netze trainiert werden. Lokale Suche ist also kein Nischenthema, sondern der Kern des Deep Learning.

### 3.2 Klassische Planung (kurz)

**Planung** = Suche mit *strukturierten* Zuständen: Zustände sind Mengen logischer Fakten, Aktionen haben **Vorbedingungen** und **Effekte** (STRIPS/PDDL-Formalismus):

```
Aktion: Aufheben(x)
  Vorbedingung: Greifer frei, x liegt oben
  Effekt:       Greifer hält x, ¬(Greifer frei)
```

Der Gewinn gegenüber roher Zustandsraumsuche: Aus der Aktionsstruktur lassen sich **automatisch Heuristiken** ableiten (z. B. „ignoriere alle negativen Effekte" → Relaxation → zulässige Schätzung). Planung ist das Bindeglied zwischen Suche und Logik und Grundlage der Robotik-Module.

### 3.3 Von neuronalen Netzen zu LLMs — die moderne KI-Landschaft

Ein **künstliches Neuron** berechnet $y = \sigma(\sum_i w_i x_i + b)$ — gewichtete Summe plus nichtlineare Aktivierung $\sigma$. Schichtet man viele davon (**Deep Learning**), kann das Netz hierarchische Merkmale lernen: Kanten → Formen → Objekte. Trainiert wird per Gradient Descent mit **Backpropagation** (effiziente Gradientenberechnung durch Kettenregel).

Meilensteine und was sie konzeptuell zeigen:

- **AlexNet (2012)**: Deep Learning + GPUs + große Datenmengen schlagen handgebaute Bildmerkmale → *Repräsentationen lernen statt konstruieren*.
- **AlphaGo (2016)**: gelernte Bewertungs-/Zugnetze + Monte-Carlo-Baumsuche → *Lernen und Suche kombiniert*, klassische KI und Deep Learning sind keine Gegensätze.
- **Transformer (2017)** und **LLMs (ab ~2020)**: Modelle, die auf riesigen Textmengen lernen, das nächste Token vorherzusagen, entwickeln breite sprachliche und faktische Fähigkeiten. Mit Instruction-Tuning und RLHF (Reinforcement Learning from Human Feedback) werden daraus Assistenten.

**Einordnung für dieses Modul:** Ein LLM ist im Agenten-Vokabular ein gelerntes Modell, das man zum Kern eines Agenten machen kann (Wahrnehmung: Text/Bilder; Aktionen: Text, Tool-Aufrufe). Die klassischen Fragen — Was ist das Erfolgsmaß? Wie geht der Agent mit Unsicherheit um? Plant er, oder reagiert er nur? — bleiben exakt dieselben. Deshalb lohnt sich das Fundament dieses Moduls auch im LLM-Zeitalter.

### 3.4 Grenzen, Ethik, Verantwortung

Gehört heute zwingend zur Einführung:

- **Bias & Fairness**: Modelle lernen Verzerrungen aus ihren Trainingsdaten (z. B. diskriminierende Muster in historischen Einstellungsdaten). „Die Daten sind objektiv" ist ein Trugschluss.
- **Erklärbarkeit (XAI)**: Tiefe Modelle sind schwer zu interpretieren — problematisch bei Kredit-, Medizin- oder Justizentscheidungen.
- **Robustheit**: Adversariale Beispiele (minimal veränderte Eingaben kippen die Vorhersage), Verteilungsverschiebung (Modell trifft auf Daten, die anders aussehen als das Training).
- **Alignment**: Wie stellt man sicher, dass ein optimierender Agent das tut, was wir *meinen*, nicht was wir *messen*? (Klassisches Beispiel: ein Agent, der ein Belohnungsmaß maximiert, findet oft Schlupflöcher — „reward hacking".)
- **Regulierung**: Die EU hat mit dem **AI Act** (in Kraft seit 2024, gestaffelte Anwendung) einen risikobasierten Rechtsrahmen geschaffen — von verbotenen Praktiken über Hochrisiko-Anforderungen bis zu Transparenzpflichten für generative Modelle.

---

## 4. Zusammenfassung / Cheat-Sheet

**Agenten**
- Rationaler Agent: maximiert erwartetes Erfolgsmaß gegeben Wahrnehmung + Wissen
- PEAS: Performance, Environment, Actuators, Sensors
- Umgebungsachsen: beobachtbar · deterministisch · episodisch · statisch · diskret · Einzelagent

**Suche**
- Problem = (Anfangszustand, Aktionen, Übergänge, Zieltest, Kosten)
- BFS: FIFO, optimal bei Einheitskosten, Speicher $O(b^d)$ · DFS: LIFO, speicherarm, nicht optimal · UCS: nach $g(n)$, optimal · IDS: BFS-Garantien, DFS-Speicher
- Greedy: nur $h(n)$ · **A\***: $f(n) = g(n) + h(n)$
- $h$ zulässig ($h \le h^*$) ⇒ A\* optimal; konsistent: $h(n) \le c(n,n') + h(n')$
- Bessere (dominante) Heuristik ⇒ weniger expandierte Knoten

**Spiele**
- Minimax: MAX maximiert, MIN minimiert; perfekt gegen perfekten Gegner
- Alpha-Beta: gleiches Ergebnis, bis zu $O(b^{m/2})$; Zugsortierung entscheidend
- Praxis: Tiefenlimit + Evaluationsfunktion (oder gelerntes Netz + MCTS)

**CSP**
- Variablen + Domänen + Constraints; Backtracking + MRV + Forward Checking + AC-3

**Logik**
- $KB \models \alpha$: $\alpha$ gilt in allen Modellen der KB
- Modus Ponens: aus $P \Rightarrow Q$ und $P$ folgt $Q$ (Umkehrung gilt nicht!)
- FOL: Objekte, Relationen, $\forall$, $\exists$ — mächtiger, Inferenz nur semi-entscheidbar

**Unsicherheit**
- Bayes: $P(H\mid E) = P(E\mid H)P(H)/P(E)$ — Posterior ∝ Likelihood × Prior
- Basisraten nicht vergessen (17-%-Beispiel!)
- Naive Bayes: $P(c\mid x) \propto P(c)\prod_i P(x_i\mid c)$ — „naiv" = bedingte Unabhängigkeit

**ML-Landkarte**
- supervised (Labels) · unsupervised (Struktur) · reinforcement (Belohnung)
- Generalisierung > Trainingsleistung; Overfitting = auswendig gelernt
- Deep Learning = geschichtete Neuronen + Gradient Descent + Backpropagation

---

## 5. Selbsttest

Versuche erst selbst zu antworten, dann aufklappen.

<details><summary><b>1. Warum definiert die moderne KI Intelligenz über „rationales Handeln" statt über „menschliches Denken"?</b></summary>

Rationales Handeln ist ein *objektives, messbares* Kriterium: Maximiere das erwartete Erfolgsmaß gegeben Wahrnehmung und Wissen. Menschliches Denken ist dagegen schwer zu spezifizieren, teils fehlerhaft (kognitive Verzerrungen) und als Bauvorgabe weder nötig noch hinreichend — so wie Flugzeuge nicht mit Federschlag fliegen. Außerdem lässt sich Rationalität mathematisch analysieren (Optimalität, Garantien).
</details>

<details><summary><b>2. Formuliere „Staubsaugerroboter reinigt eine Wohnung" als PEAS-Beschreibung.</b></summary>

**P**: gereinigte Fläche pro Zeit, Akkuverbrauch, keine Schäden/Abstürze (Treppen!). **E**: Wohnung mit Räumen, Möbeln, Teppichen, Menschen/Haustieren (dynamisch, teilweise beobachtbar). **A**: Räder/Motoren, Saugeinheit, Bürsten. **S**: Stoßsensoren, Abgrundsensoren, ggf. Lidar/Kamera, Ladezustand.
</details>

<details><summary><b>3. Warum ist DFS nicht vollständig, und mit welchem Trick bekommt man DFS-Speicherbedarf und BFS-Garantien zugleich?</b></summary>

DFS kann in unendlich tiefe (oder bei Zyklen: endlose) Pfade laufen und das Ziel nie erreichen, obwohl es in geringer Tiefe liegt. Der Trick ist **Iterative Deepening (IDS)**: DFS mit Tiefenlimit 0, 1, 2, … Die flachen Ebenen werden zwar mehrfach expandiert, aber weil die unterste Ebene exponentiell dominiert, kostet das nur einen konstanten Faktor.
</details>

<details><summary><b>4. Eine Heuristik überschätzt die Restkosten an genau einem Knoten. Was kann bei A* schiefgehen?</b></summary>

A\* kann **suboptimal** werden: Liegt der überschätzte Knoten auf dem (echten) optimalen Pfad, bekommt er einen zu hohen $f$-Wert und wird evtl. hinter einem schlechteren Ziel zurückgestellt — A\* gibt dann den teureren Pfad aus. Vollständig bleibt A\* (bei endlichen Graphen) trotzdem; nur die Optimalitätsgarantie hängt an der Zulässigkeit.
</details>

<details><summary><b>5. Warum liefert Alpha-Beta-Pruning exakt dasselbe Ergebnis wie Minimax, obwohl es Teilbäume überspringt?</b></summary>

Es überspringt nur Teilbäume, die das Endergebnis *beweisbar* nicht beeinflussen können: Wenn an einem MIN-Knoten schon ein Wert ≤ $\alpha$ gefunden ist, weiß man, dass MAX diesen Knoten nie wählen wird (MAX hat anderswo bereits $\alpha$ sicher) — der genaue Wert des Knotens ist dann irrelevant. Es wird also nichts „geschätzt", sondern nur Unnötiges nicht ausgerechnet.
</details>

<details><summary><b>6. Modelliere Sudoku als CSP und erkläre, was AC-3 dort leistet.</b></summary>

81 Variablen (Felder), Domänen {1,…,9} (vorgegebene Felder: einelementig), Constraints: alle Werte in jeder Zeile, Spalte und 3×3-Box paarweise verschieden. AC-3 macht die Kanten konsistent: Steht in einem Feld eine 5 fest, wird die 5 aus den Domänen aller Zeilen-/Spalten-/Box-Nachbarn gestrichen; Streichungen stoßen weitere Prüfungen an. Das entspricht dem menschlichen „Kandidaten streichen" und löst leichte Sudokus ganz ohne Suche.
</details>

<details><summary><b>7. Aus „Wenn es regnet, ist die Straße nass" und „Die Straße ist nass" schließt jemand „Es regnet". Wie heißt der Fehler?</b></summary>

**Affirming the consequent** (Bejahung des Konsequens): Aus $R \Rightarrow N$ und $N$ folgt logisch *nichts* über $R$ — die Straße kann aus anderen Gründen nass sein. Gültig wären Modus Ponens ($R \Rightarrow N$, $R$ ⊢ $N$) oder Modus Tollens ($R \Rightarrow N$, $\neg N$ ⊢ $\neg R$). Als *probabilistischer* Schluss (Abduktion) kann „Regen ist wahrscheinlicher geworden" aber durchaus vernünftig sein — dafür braucht man die Bayes-Regel statt Logik.
</details>

<details><summary><b>8. Test: 99 % Sensitivität, 5 % Falsch-Positiv-Rate, Krankheit bei 0,1 % (statt 1 %). Posterior bei positivem Test?</b></summary>

$P(K\mid+) = \frac{0{,}99 \cdot 0{,}001}{0{,}99 \cdot 0{,}001 + 0{,}05 \cdot 0{,}999} = \frac{0{,}00099}{0{,}00099 + 0{,}04995} \approx 0{,}019$ — knapp **2 %**. Je seltener die Krankheit, desto stärker dominiert der Prior: Fast alle positiven Tests sind Falsch-Positive.
</details>

<details><summary><b>9. Warum funktioniert Naive Bayes oft gut, obwohl seine Unabhängigkeitsannahme fast immer verletzt ist?</b></summary>

Für die *Klassifikation* muss nur die **Rangfolge** der Klassen-Posteriors stimmen, nicht ihr exakter Wert. Die Unabhängigkeitsannahme verzerrt die Wahrscheinlichkeiten (macht sie oft übertrieben sicher), kippt aber selten die Reihenfolge. Dazu kommt: Wenige Parameter ⇒ wenig Overfitting, gerade bei kleinen Datenmengen und vielen Merkmalen (Text!).
</details>

<details><summary><b>10. Ordne ein: Ist ein Schachprogramm mit Alpha-Beta „lernende KI"? Ist ein LLM ein „Agent"?</b></summary>

Alpha-Beta-Schach ist KI (rationales Handeln durch Suche), aber *kein* Lernen — alles Verhalten steckt in Algorithmus + Bewertungsfunktion. Ein LLM allein ist erst mal ein gelerntes *Modell* (Text rein, Text raus). Zum **Agenten** wird es, wenn man es in eine Wahrnehmungs-Handlungs-Schleife einbettet: Es bekommt Beobachtungen (Nutzereingaben, Tool-Ergebnisse), wählt Aktionen (Antworten, Tool-Aufrufe) und verfolgt ein Ziel. Agent ist eine *Rolle*, kein Modelltyp.
</details>

---

## 6. Literatur & Quellen

**Lehrbücher**

- **Russell & Norvig — „Artificial Intelligence: A Modern Approach" (AIMA), 4. Aufl.** — *das* Standardwerk, deckt dieses Modul fast 1:1 ab. Kapitel 1–2 (Einführung, Agenten), 3 (Suche), 5 (Spiele), 6 (CSPs), 7–9 (Logik), 12–13 (Unsicherheit, Bayes). *(vertiefend, aber gut lesbar; deutsche Übersetzung existiert)*
- **Poole & Mackworth — „Artificial Intelligence: Foundations of Computational Agents", 3. Aufl.** — komplett **kostenlos** online: https://artint.info *(einsteigerfreundlich)*

**Onlinekurse (kostenlos)**

- **UC Berkeley CS188 — Intro to AI**: Vorlesungsvideos, Folien und die berühmten Pac-Man-Projekte frei verfügbar: https://inst.eecs.berkeley.edu/~cs188/ *(einsteigerfreundlich, deckt exakt Suche/Spiele/CSP/Bayes ab — beste Ergänzung zu diesem Modul)*
- **Harvard CS50's Introduction to AI with Python** (edX/YouTube, kostenlos) *(sehr einsteigerfreundlich, praktisch orientiert)*
- **MIT 6.034 Artificial Intelligence** (OpenCourseWare, Patrick Winston) *(klassisch, hervorragende Vorlesungen)*

**Interaktive Visualisierungen & Blogposts (kostenlos)**

- *Red Blob Games — Introduction to A\**: https://www.redblobgames.com/pathfinding/a-star/introduction.html — die beste interaktive A\*-Erklärung im Netz *(einsteigerfreundlich, Pflichtlektüre vor Projekt 1)*
- *Setosa — Conditional probability visualized*: https://setosa.io/ev/conditional-probability/ *(einsteigerfreundlich)*
- 3Blue1Brown: *Bayes theorem* (YouTube) — geometrische Intuition für die Bayes-Regel *(einsteigerfreundlich)*

**Historisches / Vertiefendes**

- Turing (1950): *Computing Machinery and Intelligence* — das Original zum Turing-Test, gut lesbar. *(kostenlos online, vertiefend)*
- Silver et al. (2016): *Mastering the game of Go with deep neural networks and tree search* (Nature) — AlphaGo: Suche + Lernen kombiniert. *(vertiefend)*

---

**Nächster Schritt:** Ab in die Projekte → `projects/01-basic/` (A\*-Wegsuche), dann `projects/02-medium/` (Tic-Tac-Toe mit Minimax), dann `projects/03-final/` (Spamfilter mit Naive Bayes auf echten Daten).
