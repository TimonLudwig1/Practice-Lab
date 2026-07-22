# Module 13 — Reinforcement Learning and Computational Decision-Making

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** An **agent** is placed in an **environment**, chooses **actions**,
> receives **rewards** for them and sees **states** — and is supposed to learn, through *trial
> and error*, to act so that the **long-term** reward is maximal. Nobody tells it the correct
> action (no supervisor with labels); it only gets an *evaluative signal*, often **delayed**.
> That is **reinforcement learning (RL)**. In this module we build the classical **tabular**
> RL theory from the ground up: from the **Markov decision process** via the **Bellman
> equations** to **Monte Carlo**, **temporal-difference learning**, **SARSA** and
> **Q-learning**, the **explore-exploit dilemma** and an outlook on **function
> approximation** (the bridge to deep RL in module 14). Everything runs with pure `numpy` on
> the CPU — tabular RL is computationally cheap.

**Helpful prior knowledge:** probability theory (expectation, conditional probability), some
linear algebra, the basic idea of dynamic programming, NumPy.

**Modules you should have done first:**
- **Module 07 (Theory of AI 2)** — there you already built **MDPs**, **value iteration** and
  **policy iteration** on the 4×3 gridworld by hand. That is the case where the model
  (transitions $P$ and rewards $R$) is **known**. RL begins exactly where this knowledge is
  **missing** — the agent does not have to know $P$ and $R$ but learns from *experience*. We
  tie in directly and reference the results there.
- **Module 04/05 (Machine Learning 1/2)** — terms like *learning rate*, *sample estimator*,
  *bias/variance*, *function approximation* return here.

---

## Learning objectives

After this module you can …

- frame the **RL problem** formally as a **Markov decision process**
  $(\mathcal S,\mathcal A,P,R,\gamma)$ and distinguish it from *supervised/unsupervised
  learning*;
- derive and interpret **return**, **discount**, **state-** and **action-value functions**
  ($V^\pi$, $Q^\pi$) as well as the **Bellman expectation** and **Bellman optimality
  equations**;
- explain precisely the difference between **model-based** (planning: value/policy iteration)
  and **model-free** (learning: MC, TD);
- contrast **Monte-Carlo prediction/control**, **TD(0)**, **n-step TD** and **TD(λ)** with
  eligibility traces (bias/variance, bootstrapping);
- implement **SARSA** (on-policy) and **Q-learning** (off-policy), justify their difference on
  the **cliff-walking** example and place **expected SARSA**;
- understand the **explore-exploit dilemma** and compare **ε-greedy**, **optimistic
  initialization**, **UCB** and **Boltzmann/softmax** — first on the **multi-armed bandit**;
- name the **convergence conditions** (Robbins-Monro, GLIE);
- explain why **tabular** RL fails on large state spaces and how **function approximation**
  (linear, semi-gradient) and the **"deadly triad"** change the picture — as a transition to
  **deep RL (module 14)**.

---

## 1 · Basics — the RL problem and the MDP

### 1.1 The agent ↔ environment loop

RL formalizes **sequential decision-making**. At discrete time steps $t=0,1,2,\dots$ the
following loop runs:

```
        action A_t
   ┌──────────────────────►┌───────────────┐
   │                        │  Environment  │
┌──┴────┐                   │               │
│ Agent │                   └───────┬───────┘
└──▲────┘   state S_{t+1}           │
   │        reward R_{t+1}          │
   └────────────────────────────────┘
```

The agent observes the **state** $S_t\in\mathcal S$, chooses an **action** $A_t\in\mathcal A$,
and the environment responds with a **reward** $R_{t+1}\in\mathbb R$ and a **successor state**
$S_{t+1}$. This produces a **trajectory**
$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots$$

The decisive difference from **supervised learning**: there is **no label** "correct action".
The feedback is only **evaluative** (how good was it?), not **instructive** (what would have
been right?), and often **delayed** — a bad reward now can be the consequence of an action 20
steps ago (the **credit assignment problem**). In addition, the agent influences, through its
own acting, **which data** it sees next (not i.i.d.!) — it has to **explore** itself.

### 1.2 The Markov decision process (MDP)

A (finite) **MDP** is a tuple $(\mathcal S,\mathcal A,P,R,\gamma)$:

- $\mathcal S$ — the finite **state set**;
- $\mathcal A$ — the finite **action set** (possibly state-dependent $\mathcal A(s)$);
- $P(s'\mid s,a)=\Pr[S_{t+1}=s'\mid S_t=s,A_t=a]$ — the **transition dynamics**;
- $R(s,a)$ resp. $R(s,a,s')$ — the expected **reward**;
- $\gamma\in[0,1]$ — the **discount factor**.

The **Markov property** is the central assumption: the future depends only on the **current**
state, not on the entire past —
$$\Pr[S_{t+1}\mid S_t,A_t] = \Pr[S_{t+1}\mid S_0,A_0,\dots,S_t,A_t].$$
"The state summarizes everything relevant." If that is violated, you have a **POMDP**
(partially observable) — more on that at the end.

> **Full dynamics.** Most compactly one writes everything in one function
> $p(s',r\mid s,a)=\Pr[S_{t+1}=s', R_{t+1}=r \mid S_t=s, A_t=a]$, from which $P$ and $R$ follow
> by marginalization: $P(s'\mid s,a)=\sum_r p(s',r\mid s,a)$ and
> $R(s,a)=\sum_{s',r} r\,p(s',r\mid s,a)$.

### 1.3 Policy, return, discount

A **policy** $\pi$ is the agent's strategy — a mapping from states to (distributions over)
actions:
$$\pi(a\mid s)=\Pr[A_t=a\mid S_t=s]\quad(\text{stochastic}),\qquad a=\pi(s)\ (\text{deterministic}).$$

The goal is **not** the next reward but the **return** — the cumulative (discounted) future
reward from $t$ on:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}.$$

Why **discount** (the factor $\gamma^k$)?
- **Mathematically:** for $\gamma<1$ and bounded rewards the infinite sum converges
  ($|G_t|\le R_{\max}/(1-\gamma)$).
- **Modeling:** an immediate reward is worth more than a distant one (like an interest rate);
  $\gamma$ near 0 = "myopic/greedy", $\gamma$ near 1 = "far-sighted".
- For **episodic** tasks (with a terminal state, e.g. a game ends) $\gamma=1$ can be sensible;
  for **continuing** tasks one needs $\gamma<1$.

The practical recursion (the basis for everything that follows):
$$G_t = R_{t+1} + \gamma\,G_{t+1}.$$

### 1.4 Value functions

The **state-value function** $V^\pi(s)$ is the **expected return** when one starts in $s$ and
then follows $\pi$:
$$V^\pi(s) = \mathbb E_\pi\!\left[G_t \mid S_t=s\right].$$

The **action-value function** $Q^\pi(s,a)$ is the expected return when one takes **$a$ first**
in $s$ and then follows $\pi$:
$$Q^\pi(s,a) = \mathbb E_\pi\!\left[G_t \mid S_t=s, A_t=a\right].$$

Relationship: $V^\pi(s)=\sum_a \pi(a\mid s)\,Q^\pi(s,a)$. $Q$ is more practical for **control**,
because without a model one can read off the best action directly: $\arg\max_a Q(s,a)$.

### 1.5 The Bellman equations

Substituting the return recursion $G_t = R_{t+1}+\gamma G_{t+1}$ into the definition yields the
**Bellman expectation equation** — a *linear* system of equations that determines $V^\pi$
uniquely:
$$\boxed{\;V^\pi(s) = \sum_a \pi(a\mid s)\sum_{s',r} p(s',r\mid s,a)\big[r + \gamma V^\pi(s')\big]\;}$$
and analogously
$$Q^\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\Big[r + \gamma \sum_{a'}\pi(a'\mid s')\,Q^\pi(s',a')\Big].$$

Intuition: "the value of a state = immediate reward + discounted value of the successor state,
averaged over policy and dynamics." It is a **consistency condition**: the value *now* has to
match the value *afterwards*.

**Optimality.** There exists (for a finite MDP) an **optimal policy** $\pi_*$ that maximizes
$V^\pi(s)$ for *all* $s$ simultaneously. Its value functions $V_*=V^{\pi_*}$, $Q_*=Q^{\pi_*}$
satisfy the **Bellman optimality equations** (now *nonlinear* due to the $\max$):
$$\boxed{\;V_*(s) = \max_a \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma V_*(s')\big]\;}$$
$$\boxed{\;Q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma \max_{a'} Q_*(s',a')\big]\;}$$

From $Q_*$ one reads off the **optimal policy** *greedily*: $\pi_*(s)=\arg\max_a Q_*(s,a)$. That
is the core: **if you know $Q_*$, you have optimal behavior** — entirely without a model.

---

## 2 · Building up — from planning to learning

We organize all methods along **two axes**:

| | **Model known** ($p$ given) | **Model unknown** (only experience) |
|---|---|---|
| **Prediction** (evaluate a given $\pi$) | policy evaluation (DP) | **Monte Carlo**, **TD(0)** |
| **Control** (find the best $\pi$) | **value/policy iteration** (DP) | **MC control**, **SARSA**, **Q-learning** |

The left column ("**planning**") is module 07. The right column ("**learning**") is the core of
this module.

### 2.1 Recap: dynamic programming (model known)

When $p$ and $R$ are known, one turns the Bellman equations into **update rules**:

- **Policy evaluation:** iterate $V_{k+1}(s)\leftarrow\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V_k(s')]$
  until convergence (the fixed point = $V^\pi$).
- **Policy iteration:** alternate (1) evaluate, (2) *improve greedily*
  $\pi'(s)=\arg\max_a Q^\pi(s,a)$ — ends in finitely many steps.
- **Value iteration:** apply the Bellman **optimality** update directly,
  $V_{k+1}(s)\leftarrow\max_a\sum_{s',r}p(s',r|s,a)[r+\gamma V_k(s')]$.

> **Module 07 bridge.** That is exactly what you built in **module 07, project 03** on the 4×3
> gridworld: value iteration converged in 34, policy iteration in 5 iterations to the *same*
> optimal policy, with the AIMA utilities. The **Bellman operator** is a **contraction** with
> factor $\gamma$ (Banach fixed-point theorem) → guaranteed convergence. **The catch:** one
> needs $p$ and $R$ **explicitly**. In the real world one rarely knows them. RL solves that.

### 2.2 Monte-Carlo methods (learning from complete episodes)

**Idea:** if you don't know $p$, you simply *average* **observed returns**. $V^\pi(s)$ is an
expectation — so estimate it by the sample mean of the returns that actually occurred after
visits to $s$.

For each episode compute $G_t$ backwards; for each visited state $s$ (**first-visit**: only at
the first occurrence per episode; **every-visit**: at every one):
$$N(s)\leftarrow N(s)+1,\qquad V(s)\leftarrow V(s) + \tfrac1{N(s)}\big(G_t - V(s)\big).$$
That is the **incremental mean**. The form $\text{new}\leftarrow\text{old}+\alpha(\text{target}-\text{old})$
with the **error** $(\text{target}-\text{old})$ is the **basic pattern of every RL update**;
with a fixed learning rate $\alpha$ instead of $1/N$ one "forgets" old experience exponentially
(good for non-stationary environments).

**Properties:** MC is **unbiased** (the return is a genuine sample of $G_t$), but **high
variance** (all the randomness of an episode is in it). MC needs **complete, terminating**
episodes (no bootstrapping) and learns **only at the end of the episode**.

### 2.3 Temporal-difference learning (TD) — the breakthrough

**TD** combines the best of DP and MC: it learns **from experience** (like MC), but
**bootstraps** (like DP) — it updates an estimate with *another* estimate, without waiting for
the end of the episode. The **TD(0)** update for prediction:
$$\boxed{\;V(S_t) \leftarrow V(S_t) + \alpha\underbrace{\big[\,\overbrace{R_{t+1}+\gamma V(S_{t+1})}^{\text{TD target}} - V(S_t)\,\big]}_{\delta_t\ =\ \text{TD error}}\;}$$

The **TD error** $\delta_t = R_{t+1}+\gamma V(S_{t+1}) - V(S_t)$ measures the surprise ("was the
successor state better or worse than expected?"). TD can learn **online**, after *every step*,
even in **non-terminating** tasks.

**Bias/variance comparison:**

| | MC | TD(0) |
|---|---|---|
| target | the true return $G_t$ | $R_{t+1}+\gamma V(S_{t+1})$ (estimated) |
| bias | **unbiased** | **biased** (bootstrapping) |
| variance | **high** | **low** |
| needs the end of the episode? | yes | no |
| exploits Markov? | no | yes (uses the structure) |

TD is usually **faster** and **more data-efficient** in practice — the price is the bias, which
vanishes with a better estimate.

**In between: n-step & TD(λ).** There is a whole spectrum. The **n-step return**
$$G_{t:t+n}=R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^n V(S_{t+n})$$
interpolates between TD ($n=1$) and MC ($n=\infty$). **TD(λ)** averages **all** n-step returns
geometrically with weights $(1-\lambda)\lambda^{n-1}$ into the **λ-return** $G_t^\lambda$.
Efficiently one implements this *not* by looking ahead but by **eligibility traces** $e(s)$
that distribute "responsibility" backwards (section 3.2).

### 2.4 Model-free control: SARSA vs. Q-learning

For **control** (finding the best $\pi$) we learn $Q(s,a)$ instead of $V(s)$, because without a
model one needs $Q$ to be able to act greedily. Both follow the TD pattern, but differ in the
**target**.

**SARSA (on-policy).** The name comes from the tuple $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$. One
chooses $A_{t+1}$ **with the current (exploring) policy** and bootstraps with *that* value:
$$\boxed{\;Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma\,Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\big]\;}$$
SARSA learns the value of the policy **it actually executes** (including exploration).

**Q-learning (off-policy).** One bootstraps with the **greedy** (best) action, independent of
what was actually chosen:
$$\boxed{\;Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma\,\max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)\big]\;}$$
Q-learning directly approximates $Q_*$ — it learns the **optimal** policy while following a
different (exploring) **behavior policy**. That is the core of *off-policy*: the policy one
*learns* ≠ the policy one *executes*.

**Expected SARSA.** Replaces the $Q(S_{t+1},A_{t+1})$ by the **expectation** over the policy —
reducing the variance from the random choice of $A_{t+1}$:
$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\Big[R_{t+1}+\gamma\sum_{a'}\pi(a'\mid S_{t+1})Q(S_{t+1},a') - Q(S_t,A_t)\Big].$$
For a greedy $\pi$, expected SARSA **becomes** Q-learning — it is the generalizing bracket.

> **The cliff-walking aha (project 02).** On a gridworld with a "cliff" (falling off = −100),
> **Q-learning** finds the *optimal, risky* route right along the edge — but learns it while
> occasionally (through ε-exploration) falling off, i.e. with a **worse online return**.
> **SARSA** learns a *safer* route with distance to the cliff, because it *prices in* the
> exploration costs in its values, and achieves **more reward online**. Conclusion: "optimal in
> the limit" (Q-learning) ≠ "good while still exploring" (SARSA).

### 2.5 Generalized policy iteration & ε-greedy control

All control methods are instances of **generalized policy iteration (GPI)**: a dance of (a)
**evaluation** (bringing values closer to the policy) and (b) **improvement** (making the policy
greedy w.r.t. the values). They drive each other to the common fixed point $\pi_*,Q_*$.

For control to work, the agent has to **explore** — if it always acted greedily, it would never
see alternatives. The simplest solution is an **ε-greedy** policy:
$$\pi(a\mid s)=\begin{cases}1-\varepsilon+\varepsilon/|\mathcal A| & a=\arg\max_{a'}Q(s,a')\\[2pt]\varepsilon/|\mathcal A| & \text{otherwise.}\end{cases}$$
With probability $1-\varepsilon$ the best known action, with $\varepsilon$ a random one. For
**convergence to the optimal policy** one demands **GLIE** (*greedy in the limit with infinite
exploration*): every $(s,a)$ is visited infinitely often **and** $\varepsilon\to0$ (e.g.
$\varepsilon_k=1/k$). Together with the **Robbins-Monro** conditions on the learning rate,
$$\sum_k \alpha_k = \infty,\qquad \sum_k \alpha_k^2 < \infty,$$
tabular Q-learning converges **with probability 1** to $Q_*$ (Watkins & Dayan 1992). In
practice one often takes a constant, small $\alpha$ and a slowly decaying $\varepsilon$ — theory
and practice deliberately diverge here.

---

## 3 · Advanced topics

### 3.1 The explore-exploit dilemma & multi-armed bandits

The **bandit** is an MDP with *one* state: $k$ "arms" (actions), each delivering a reward from an
unknown distribution with mean $q_*(a)$. It isolates the **explore-exploit dilemma** in its
purest form: **exploit** (pull the best arm so far) vs. **explore** (test an uncertain arm that
*might* be better). Too much exploit → you get stuck in a local optimum; too much explore → you
waste reward. One measures the quality via the **regret**
$\rho_T = T\,q_*(a^*) - \sum_{t=1}^{T}\mathbb E[q_*(A_t)]$ (the reward missed relative to the best
arm).

Strategies (project 01 compares them):
- **ε-greedy** — simple, but explores *uniformly* and *forever* (linear regret for a fixed ε).
- **Optimistic initialization** — set $Q_0(a)$ deliberately too high; every action not yet tried
  looks attractive → *automatic* initial exploration, greedy afterwards. A trick, not a panacea
  (works only early, not for non-stationary).
- **UCB (upper confidence bound)** — choose $A_t=\arg\max_a\big[Q(a)+c\sqrt{\ln t / N(a)}\big]$:
  "optimism in the face of uncertainty". The bonus term is large for rarely pulled arms and
  shrinks with knowledge. UCB1 achieves **logarithmic** regret $O(\ln T)$ — provably
  near-optimal.
- **Boltzmann/softmax** — choose $a$ with probability $\propto e^{Q(a)/\tau}$; the **temperature**
  $\tau$ steers from greedy ($\tau\to0$) to uniform ($\tau\to\infty$).
- **Thompson sampling** (Bayesian) — maintain a posterior distribution over each $q_*(a)$, draw a
  sample per arm, play the winner. Often empirically the best; only mentioned here.

### 3.2 Eligibility traces — TD(λ) efficiently

An **eligibility trace** $e_t(s)$ (resp. $e_t(s,a)$) is a short-term memory that marks which
states were visited *recently and frequently* and are therefore **"responsible"** for a current
TD error. **Accumulating trace:**
$$e_t(s)=\gamma\lambda\,e_{t-1}(s)+\mathbb 1[S_t=s],\qquad V(s)\leftarrow V(s)+\alpha\,\delta_t\,e_t(s)\ \ \forall s.$$
The single TD error $\delta_t$ is distributed over *all* recently visited states, graded by
$(\gamma\lambda)^{k}$. For $\lambda=0$ this gives TD(0), for $\lambda=1$ (approximately) MC. This
solves the **credit-assignment** problem elegantly and often speeds up learning. The control
variants are called **SARSA(λ)** and **Watkins's Q(λ)**.

### 3.3 Function approximation — why tables are not enough

So far $Q$ was a **table** $|\mathcal S|\times|\mathcal A|$. That fails as soon as the state
space is large/continuous (chess $\sim10^{47}$, Go $\sim10^{170}$, images as states). Solution:
**parametrize** $\hat V(s;\mathbf w)\approx V^\pi(s)$ with few weights $\mathbf w$ — linear
$\hat V(s;\mathbf w)=\mathbf w^\top\mathbf x(s)$ with a **feature vector** $\mathbf x(s)$, or
nonlinear (a neural network → deep RL). One then **generalizes** over similar states.

The learning goal becomes a regression problem; the **semi-gradient TD(0)** update reads
$$\mathbf w \leftarrow \mathbf w + \alpha\big[R_{t+1}+\gamma\hat V(S_{t+1};\mathbf w) - \hat V(S_t;\mathbf w)\big]\,\nabla_{\mathbf w}\hat V(S_t;\mathbf w).$$
"Semi"-gradient, because one **ignores** the gradient of the bootstrapped target (it itself
depends on $\mathbf w$) — one treats the TD target as fixed.

> **The "deadly triad".** If you combine **(1) function approximation + (2) bootstrapping +
> (3) off-policy training**, RL can **diverge** (values run off to infinity). Each component
> alone is fine, but all three together are dangerous. Deep Q-networks (module 14) have exactly
> this triad and therefore need tricks (**experience replay**, **target network**) to tame it.
> That is the conceptual bridge to module 14.

### 3.4 Policy-gradient methods (outlook)

Instead of learning values and deriving a policy greedily from them, one can parametrize the
**policy directly**, $\pi_\theta(a\mid s)$, and optimize $\theta$ by gradient ascent on the
expected return $J(\theta)=\mathbb E_{\pi_\theta}[G_0]$. The **policy gradient theorem** gives
$$\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\!\big[\nabla_\theta\log\pi_\theta(A_t\mid S_t)\,Q^{\pi_\theta}(S_t,A_t)\big],$$
from which the **REINFORCE** algorithm follows (a Monte-Carlo estimate of $Q$ by $G_t$, often
with a **baseline** $b(s)$ for variance reduction). Advantages: natural **stochastic** policies,
**continuous** action spaces. **Actor-critic** combines both (an *actor* $\pi_\theta$, a *critic*
$\hat V$). That is the basis of modern methods (A2C/A3C, PPO, DDPG, SAC) — the topic of
**module 14 (Deep RL for Optimal Control)**.

### 3.5 When Markov is violated: POMDPs

If the agent does not see the full state but only an **observation** $O_t$, you have a
**partially observable MDP (POMDP)**. Optimal acting then requires a **belief state** (a
posterior over the true state) — often intractable exactly. Practical RL answers: build the state
from a **window** of past observations or let a **recurrent** network (RNN/LSTM, module 09)
summarize the history.

---

## 4 · Summary / cheat sheet

**Terms.** MDP $(\mathcal S,\mathcal A,P,R,\gamma)$ · policy $\pi(a|s)$ · return
$G_t=\sum_k\gamma^k R_{t+k+1}$ · $V^\pi,Q^\pi$ · optimal $V_*,Q_*$, $\pi_*(s)=\arg\max_a Q_*(s,a)$.

**Bellman.**
- expectation: $V^\pi(s)=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V^\pi(s')]$
- optimality: $Q_*(s,a)=\sum_{s',r}p(s',r|s,a)[r+\gamma\max_{a'}Q_*(s',a')]$

**Update rules (all: $\text{old}\leftarrow\text{old}+\alpha(\text{target}-\text{old})$).**

| Method | Target for the update |
|---|---|
| MC | $G_t$ (the true return) |
| TD(0) prediction | $R_{t+1}+\gamma V(S_{t+1})$ |
| **SARSA** (on-policy) | $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$ |
| **Expected SARSA** | $R_{t+1}+\gamma\sum_{a'}\pi(a'|S_{t+1})Q(S_{t+1},a')$ |
| **Q-learning** (off-policy) | $R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')$ |

**Exploration.** ε-greedy · optimistic init · UCB $Q(a)+c\sqrt{\ln t/N(a)}$ · softmax($\tau$).

**Convergence.** GLIE (∞ exploration + $\varepsilon\to0$) + Robbins-Monro ($\sum\alpha=\infty,\sum\alpha^2<\infty$).

**Axes.** model-based (planning, DP) ↔ model-free (learning) · on-policy ↔ off-policy ·
bootstrapping (TD/DP) ↔ sampling full returns (MC) · table ↔ function approximation.

**Deadly triad.** FA + bootstrapping + off-policy → possible divergence (→ deep RL needs replay/target-net).

---

## 5 · Self-test

<details>
<summary><b>1.</b> How does RL differ fundamentally from supervised learning?</summary>

No supervisor/label with a "correct action" — only an **evaluative**, often **delayed** reward
signal. The data are **not i.i.d.**: the agent influences, through its own acting, which states
it sees next, and has to actively **explore**. The goal is the **long-term** cumulative reward
(the return), not the immediate one.
</details>

<details>
<summary><b>2.</b> Why does one discount the return, and what does γ do?</summary>

So that the (for continuing tasks infinite) sum **converges** ($|G|\le R_{\max}/(1-\gamma)$) and
to **weight nearer rewards higher**. $\gamma\to0$ = myopic/greedy, $\gamma\to1$ = far-sighted.
For episodic tasks $\gamma=1$ is often fine.
</details>

<details>
<summary><b>3.</b> What does the Bellman optimality equation state, and why is it "harder" than the expectation equation?</summary>

$V_*(s)=\max_a\sum_{s',r}p(s',r|s,a)[r+\gamma V_*(s')]$ — the optimal value is the **best**
attainable expected return. Due to the $\max$ it is **nonlinear** (no longer a linear system), so
one solves it iteratively (value iteration) rather than directly.
</details>

<details>
<summary><b>4.</b> Name the core difference between Monte Carlo and TD(0) in bias/variance.</summary>

MC uses the **true return** → **unbiased**, but **high variance**, needs the end of the episode.
TD(0) uses a **bootstrapped target** $R+\gamma V(S')$ → **biased** (uses its own estimate), but
**low variance**, learns **online** after every step.
</details>

<details>
<summary><b>5.</b> SARSA vs. Q-learning: what is on- resp. off-policy, and what does that mean in practice?</summary>

**SARSA (on-policy)** bootstraps with the **actually chosen** next action $A_{t+1}\sim\pi$ →
learns the value of the executed (exploring) policy. **Q-learning (off-policy)** bootstraps with
$\max_{a'}Q(S_{t+1},a')$ → directly learns $Q_*$ (the *optimal* policy), although it follows an
exploring behavior policy. In practice: on cliff walking SARSA chooses the **safe** route,
Q-learning the **optimal-but-risky** one.
</details>

<details>
<summary><b>6.</b> Explain the explore-exploit dilemma and how UCB addresses it.</summary>

**Exploit** = use the best known action; **explore** = test uncertain actions that *might* be
better. UCB chooses $\arg\max_a[Q(a)+c\sqrt{\ln t/N(a)}]$ — the bonus is large for **rarely**
pulled arms and shrinks with experience ("optimism under uncertainty"), achieving **logarithmic**
regret.
</details>

<details>
<summary><b>7.</b> What is GLIE, and why is it needed?</summary>

*Greedy in the limit with infinite exploration*: (a) every $(s,a)$ is visited **infinitely
often** **and** (b) the policy becomes **greedy** in the limit ($\varepsilon\to0$). Needed so
that ε-greedy control (e.g. SARSA) converges to the **optimal** policy — otherwise residual
exploration remains and one is only ε-optimal.
</details>

<details>
<summary><b>8.</b> What is the "deadly triad" and why is it relevant for deep RL?</summary>

The combination **function approximation + bootstrapping + off-policy** can lead to
**divergence**. Deep Q-networks (module 14) have all three → one needs **experience replay** and
a **target network** to stabilize.
</details>

<details>
<summary><b>9.</b> What are eligibility traces for?</summary>

They solve the **credit-assignment** problem and realize **TD(λ)** efficiently: a trace $e(s)$
remembers recently visited states, so that a current TD error $\delta_t$ is distributed
retroactively over **all** responsible states (graded by $(\gamma\lambda)^k$) — interpolating
between TD(0) and MC.
</details>

<details>
<summary><b>10.</b> Why is tabular Q-learning unsuitable for chess/Go, and what is the alternative?</summary>

The table would have $|\mathcal S|\times|\mathcal A|$ entries — impossible to store *or* visit at
$10^{47}$/$10^{170}$ states. Alternative: **function approximation** $\hat Q(s,a;\mathbf w)$ with
features/a neural network that **generalizes** over similar states (→ deep RL).
</details>

---

## 6 · Literature & sources

**The standard work (free!):**
- **Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed., 2018)** — *the*
  reference, didactically outstanding, **free as a PDF**
  (incompleteideas.net/book/the-book.html). Beginner-friendly *and* complete. Ch. 2 (bandits),
  3 (MDP), 4 (DP), 5 (MC), 6 (TD), 7 (n-step), 12 (eligibility traces), 13 (policy gradient)
  cover exactly this module. **The best single source.**

**Lectures (free, online):**
- **David Silver — *RL Course* (DeepMind/UCL, 10 lectures, YouTube)** — the classic, roughly
  follows Sutton & Barto. (beginner → in-depth)
- **Stanford CS234 *Reinforcement Learning* (Emma Brunskill)** — lecture videos + slides free.
  (in-depth)
- **UC Berkeley CS285 *Deep RL* (Sergey Levine)** — for the jump to module 14. (in-depth)

**Interactive/blog (beginner-friendly):**
- **Gymnasium documentation** (gymnasium.farama.org) — the standard API for RL environments (the
  successor of OpenAI Gym). We build *without* Gym in the module (didactically), but for practice
  you should know the API.
- **Andrej Karpathy — *Deep Reinforcement Learning: Pong from Pixels*** (karpathy.github.io) — a
  legendary blog post on policy gradient. (beginner → in-depth)
- **Lilian Weng — *A (Long) Peek into Reinforcement Learning*** (lilianweng.github.io) — a
  compact, precise overview. (in-depth)
- **Spinning Up in Deep RL** (OpenAI, spinningup.openai.com) — for module 14, clean
  implementations + theory. (in-depth, free)

**Classical papers (in-depth):**
- Watkins & Dayan (1992), *Q-learning* — the convergence proof.
- Sutton (1988), *Learning to Predict by the Methods of Temporal Differences* — the TD origin.
- Auer, Cesa-Bianchi & Fischer (2002), *Finite-time Analysis of the Multiarmed Bandit* — UCB1.

---

## Next module

**Module 14 — Deep Reinforcement Learning for Optimal Control** replaces the Q-**table** with a
**neural network** (DQN, PyTorch from modules 05/09), tames the *deadly triad* with replay &
target-network and moves on to **policy-gradient/actor-critic** (REINFORCE → PPO) — for
**continuous** control (optimal control). This module provides the complete conceptual foundation
for that.
# Modul 13 — Reinforcement Learning and Computational Decision-Making (deutsche Fassung)

> **Worum geht es?** Ein **Agent** steht in einer **Umgebung**, wählt **Aktionen**, erhält
> dafür **Belohnungen** und sieht **Zustände** — und soll durch *Ausprobieren* lernen, so zu
> handeln, dass die **langfristige** Belohnung maximal wird. Niemand sagt ihm die richtige
> Aktion (kein Supervisor mit Labels); er bekommt nur ein *Bewertungssignal*, oft
> **verzögert**. Das ist **Reinforcement Learning (RL)**. In diesem Modul bauen wir die
> klassische, **tabellarische** RL-Theorie von Grund auf: vom **Markov-Entscheidungsprozess**
> über die **Bellman-Gleichungen** zu **Monte-Carlo**, **Temporal-Difference-Lernen**,
> **SARSA** und **Q-Learning**, dem **Explore-Exploit-Dilemma** und einem Ausblick auf
> **Funktionsapproximation** (die Brücke zu Deep RL in Modul 14). Alles läuft mit reinem
> `numpy` auf der CPU — tabellarisches RL ist rechnerisch billig.

**Hilfreiche Vorkenntnisse:** Wahrscheinlichkeitsrechnung (Erwartungswert, bedingte
Wahrscheinlichkeit), etwas lineare Algebra, Grundidee der dynamischen Programmierung, NumPy.

**Diese Module solltest du vorher gemacht haben:**
- **Modul 07 (Theorie der KI 2)** — dort hast du **MDPs**, **Value Iteration** und **Policy
  Iteration** auf der 4×3-Gridworld schon von Hand gebaut. Das ist der Fall, in dem das Modell
  (Übergänge $P$ und Belohnungen $R$) **bekannt** ist. RL beginnt genau da, wo dieses Wissen
  **fehlt** — der Agent muss $P$ und $R$ nicht kennen, sondern lernt aus *Erfahrung*. Wir
  knüpfen direkt an und referenzieren die dortigen Ergebnisse.
- **Modul 04/05 (Machine Learning 1/2)** — Begriffe wie *Lernrate*, *Stichprobenschätzer*,
  *Bias/Varianz*, *Funktionsapproximation* kehren hier wieder.

---

## Lernziele

Nach diesem Modul kannst du …

- das **RL-Problem** formal als **Markov-Entscheidungsprozess** $(\mathcal S,\mathcal A,P,R,\gamma)$
  fassen und von *supervised/unsupervised learning* abgrenzen;
- **Return**, **Discount**, **Zustands-** und **Aktionswertfunktionen** ($V^\pi$, $Q^\pi$) sowie
  die **Bellman-Erwartungs-** und **Bellman-Optimalitätsgleichungen** herleiten und deuten;
- den Unterschied zwischen **modellbasiert** (Planung: Value/Policy Iteration) und
  **modellfrei** (Lernen: MC, TD) präzise erklären;
- **Monte-Carlo-Prädiktion/-Kontrolle**, **TD(0)**, **n-step TD** und **TD(λ)** mit
  Eligibility Traces gegenüberstellen (Bias/Varianz, Bootstrapping);
- **SARSA** (on-policy) und **Q-Learning** (off-policy) implementieren, ihren Unterschied am
  **Cliff-Walking**-Beispiel begründen und **Expected SARSA** einordnen;
- das **Explore-Exploit-Dilemma** verstehen und **ε-greedy**, **optimistische Initialisierung**,
  **UCB** und **Boltzmann/Softmax** vergleichen — zuerst am **Multi-armed Bandit**;
- **Konvergenzbedingungen** (Robbins-Monro, GLIE) benennen;
- erklären, warum **tabellarisches** RL bei großen Zustandsräumen scheitert und wie
  **Funktionsapproximation** (linear, semi-gradient) und die **„deadly triad"** das Bild
  verändern — als Übergang zu **Deep RL (Modul 14)**.

---

## 1 · Grundlagen — Das RL-Problem und der MDP

### 1.1 Der Regelkreis Agent ↔ Umgebung

RL formalisiert **sequenzielle Entscheidungsfindung**. Zu diskreten Zeitschritten
$t=0,1,2,\dots$ läuft folgende Schleife:

```
        Aktion A_t
   ┌──────────────────────►┌───────────────┐
   │                        │   Umgebung    │
┌──┴────┐                   │  (Environment)│
│ Agent │                   └───────┬───────┘
└──▲────┘   Zustand S_{t+1}         │
   │        Belohnung R_{t+1}       │
   └────────────────────────────────┘
```

Der Agent beobachtet den **Zustand** $S_t\in\mathcal S$, wählt eine **Aktion**
$A_t\in\mathcal A$, und die Umgebung antwortet mit einer **Belohnung** $R_{t+1}\in\mathbb R$
und einem **Folgezustand** $S_{t+1}$. Das erzeugt eine **Trajektorie**
$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots$$

Der entscheidende Unterschied zu **überwachtem Lernen**: es gibt **kein Label** „richtige
Aktion". Das Feedback ist nur **evaluativ** (wie gut war es?), nicht **instruktiv** (was wäre
richtig gewesen?), und oft **verzögert** — eine schlechte Belohnung jetzt kann Folge einer
Aktion vor 20 Schritten sein (**credit assignment problem**). Zusätzlich beeinflusst der Agent
durch sein Handeln, **welche Daten** er als Nächstes sieht (nicht i.i.d.!) — er muss selbst
**explorieren**.

### 1.2 Der Markov-Entscheidungsprozess (MDP)

Ein (endlicher) **MDP** ist ein Tupel $(\mathcal S,\mathcal A,P,R,\gamma)$:

- $\mathcal S$ — endliche **Zustandsmenge**;
- $\mathcal A$ — endliche **Aktionsmenge** (ggf. zustandsabhängig $\mathcal A(s)$);
- $P(s'\mid s,a)=\Pr[S_{t+1}=s'\mid S_t=s,A_t=a]$ — **Übergangsdynamik**;
- $R(s,a)$ bzw. $R(s,a,s')$ — erwartete **Belohnung**;
- $\gamma\in[0,1]$ — **Discount-Faktor**.

Die **Markov-Eigenschaft** ist die zentrale Annahme: die Zukunft hängt nur vom **aktuellen**
Zustand ab, nicht von der ganzen Vergangenheit —
$$\Pr[S_{t+1}\mid S_t,A_t] = \Pr[S_{t+1}\mid S_0,A_0,\dots,S_t,A_t].$$
„Der Zustand fasst alles Relevante zusammen." Ist das verletzt, hat man ein **POMDP**
(partiell beobachtbar) — dazu am Ende mehr.

> **Vollständige Dynamik.** Am kompaktesten schreibt man alles in eine Funktion
> $p(s',r\mid s,a)=\Pr[S_{t+1}=s', R_{t+1}=r \mid S_t=s, A_t=a]$, aus der sich
> $P$ und $R$ durch Marginalisierung ergeben:
> $P(s'\mid s,a)=\sum_r p(s',r\mid s,a)$ und $R(s,a)=\sum_{s',r} r\,p(s',r\mid s,a)$.

### 1.3 Policy, Return, Discount

Eine **Policy** $\pi$ ist die Strategie des Agenten — eine Abbildung von Zuständen auf
(Verteilungen über) Aktionen:
$$\pi(a\mid s)=\Pr[A_t=a\mid S_t=s]\quad(\text{stochastisch}),\qquad a=\pi(s)\ (\text{deterministisch}).$$

Ziel ist **nicht** die nächste Belohnung, sondern der **Return** — die kumulierte
(diskontierte) zukünftige Belohnung ab $t$:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}.$$

Warum **diskontieren** (Faktor $\gamma^k$)?
- **Mathematisch:** bei $\gamma<1$ und beschränkten Belohnungen konvergiert die unendliche
  Summe ($|G_t|\le R_{\max}/(1-\gamma)$).
- **Modellierung:** unmittelbare Belohnung ist mehr wert als ferne (wie ein Zinssatz);
  $\gamma$ nahe 0 = „myopisch/gierig", $\gamma$ nahe 1 = „weitsichtig".
- Bei **episodischen** Aufgaben (mit Terminalzustand, z. B. ein Spiel endet) kann
  $\gamma=1$ sinnvoll sein; bei **kontinuierlichen** Aufgaben braucht man $\gamma<1$.

Praktische Rekursion (Basis für alles Folgende):
$$G_t = R_{t+1} + \gamma\,G_{t+1}.$$

### 1.4 Wertfunktionen

Die **Zustandswertfunktion** $V^\pi(s)$ ist der **erwartete Return**, wenn man in $s$ startet
und danach $\pi$ folgt:
$$V^\pi(s) = \mathbb E_\pi\!\left[G_t \mid S_t=s\right].$$

Die **Aktionswertfunktion** $Q^\pi(s,a)$ ist der erwartete Return, wenn man in $s$ **zuerst
$a$** nimmt und danach $\pi$ folgt:
$$Q^\pi(s,a) = \mathbb E_\pi\!\left[G_t \mid S_t=s, A_t=a\right].$$

Zusammenhang: $V^\pi(s)=\sum_a \pi(a\mid s)\,Q^\pi(s,a)$. $Q$ ist praktischer für **Kontrolle**,
weil man ohne Modell direkt die beste Aktion ablesen kann: $\arg\max_a Q(s,a)$.

### 1.5 Die Bellman-Gleichungen

Setzt man die Return-Rekursion $G_t = R_{t+1}+\gamma G_{t+1}$ in die Definition ein, erhält man
die **Bellman-Erwartungsgleichung** — ein *lineares* Gleichungssystem, das $V^\pi$ eindeutig
festlegt:
$$\boxed{\;V^\pi(s) = \sum_a \pi(a\mid s)\sum_{s',r} p(s',r\mid s,a)\big[r + \gamma V^\pi(s')\big]\;}$$
und analog
$$Q^\pi(s,a) = \sum_{s',r} p(s',r\mid s,a)\Big[r + \gamma \sum_{a'}\pi(a'\mid s')\,Q^\pi(s',a')\Big].$$

Intuition: „Der Wert eines Zustands = sofortige Belohnung + diskontierter Wert des
Folgezustands, gemittelt über Policy und Dynamik." Es ist eine **Konsistenzbedingung**: der
Wert *jetzt* muss zum Wert *danach* passen.

**Optimalität.** Es existiert (bei endlichem MDP) eine **optimale Policy** $\pi_*$, die $V^\pi(s)$
für *alle* $s$ gleichzeitig maximiert. Ihre Wertfunktionen $V_*=V^{\pi_*}$, $Q_*=Q^{\pi_*}$
erfüllen die **Bellman-Optimalitätsgleichungen** (jetzt *nichtlinear* durch das $\max$):
$$\boxed{\;V_*(s) = \max_a \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma V_*(s')\big]\;}$$
$$\boxed{\;Q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r+\gamma \max_{a'} Q_*(s',a')\big]\;}$$

Aus $Q_*$ liest man die **optimale Policy** *greedy* ab: $\pi_*(s)=\arg\max_a Q_*(s,a)$. Das ist
der Kern: **kennt man $Q_*$, hat man optimales Verhalten** — ganz ohne Modell.

---

## 2 · Aufbau — Von Planung zu Lernen

Wir ordnen alle Verfahren nach **zwei Achsen**:

| | **Modell bekannt** ($p$ gegeben) | **Modell unbekannt** (nur Erfahrung) |
|---|---|---|
| **Prädiktion** (evaluiere gegebenes $\pi$) | Policy Evaluation (DP) | **Monte Carlo**, **TD(0)** |
| **Kontrolle** (finde bestes $\pi$) | **Value/Policy Iteration** (DP) | **MC-Control**, **SARSA**, **Q-Learning** |

Die linke Spalte („**Planung**") ist Modul 07. Die rechte Spalte („**Lernen**") ist der Kern
dieses Moduls.

### 2.1 Rückblick: Dynamische Programmierung (Modell bekannt)

Wenn $p$ und $R$ bekannt sind, macht man aus den Bellman-Gleichungen **Update-Regeln**:

- **Policy Evaluation:** iteriere $V_{k+1}(s)\leftarrow\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V_k(s')]$
  bis zur Konvergenz (Fixpunkt = $V^\pi$).
- **Policy Iteration:** abwechselnd (1) evaluieren, (2) *greedy verbessern*
  $\pi'(s)=\arg\max_a Q^\pi(s,a)$ — endet in endlich vielen Schritten.
- **Value Iteration:** wende direkt das Bellman-**Optimalitäts**-Update an,
  $V_{k+1}(s)\leftarrow\max_a\sum_{s',r}p(s',r|s,a)[r+\gamma V_k(s')]$.

> **Modul-07-Brücke.** Genau das hast du in **Modul 07, Projekt 03** auf der 4×3-Gridworld
> gebaut: Value Iteration konvergierte in 34, Policy Iteration in 5 Iterationen zur *gleichen*
> optimalen Policy, mit den AIMA-Utilities. Der **Bellman-Operator** ist eine **Kontraktion**
> mit Faktor $\gamma$ (Banach-Fixpunktsatz) → garantierte Konvergenz. **Der Haken:** man
> braucht $p$ und $R$ **explizit**. In der echten Welt kennt man sie selten. RL löst das.

### 2.2 Monte-Carlo-Methoden (lernen aus vollständigen Episoden)

**Idee:** Wenn man $p$ nicht kennt, *mittelt* man einfach **beobachtete Returns**. $V^\pi(s)$ ist
ein Erwartungswert — also schätze ihn durch den Stichprobenmittelwert der Returns, die nach
Besuchen von $s$ tatsächlich eintraten.

Für jede Episode berechne $G_t$ rückwärts; für jeden besuchten Zustand $s$ (**first-visit**:
nur beim ersten Auftreten pro Episode; **every-visit**: bei jedem):
$$N(s)\leftarrow N(s)+1,\qquad V(s)\leftarrow V(s) + \tfrac1{N(s)}\big(G_t - V(s)\big).$$
Das ist der **inkrementelle Mittelwert**. Die Form $\text{neu}\leftarrow\text{alt}+\alpha(\text{Ziel}-\text{alt})$
mit **Fehler** $(\text{Ziel}-\text{alt})$ ist das **Grundmuster jedes RL-Updates**; mit fester
Lernrate $\alpha$ statt $1/N$ „vergisst" man alte Erfahrung exponentiell (gut für
nichtstationäre Umgebungen).

**Eigenschaften:** MC ist **unverzerrt** (der Return ist eine echte Stichprobe von $G_t$), aber
**hochvariant** (der ganze Zufall einer Episode steckt drin). MC braucht **vollständige,
terminierende** Episoden (kein Bootstrapping) und lernt **erst am Episodenende**.

### 2.3 Temporal-Difference-Lernen (TD) — der Durchbruch

**TD** kombiniert das Beste aus DP und MC: es lernt **aus Erfahrung** (wie MC), aber
**bootstrappt** (wie DP) — es aktualisiert eine Schätzung mit einer *anderen* Schätzung, ohne
das Episodenende abzuwarten. Das **TD(0)**-Update für Prädiktion:
$$\boxed{\;V(S_t) \leftarrow V(S_t) + \alpha\underbrace{\big[\,\overbrace{R_{t+1}+\gamma V(S_{t+1})}^{\text{TD-Ziel}} - V(S_t)\,\big]}_{\delta_t\ =\ \text{TD-Fehler}}\;}$$

Der **TD-Fehler** $\delta_t = R_{t+1}+\gamma V(S_{t+1}) - V(S_t)$ misst die Überraschung
(„war der Folgezustand besser oder schlechter als erwartet?"). TD kann **online**, nach *jedem
Schritt* lernen, auch in **nicht-terminierenden** Aufgaben.

**Bias/Varianz-Vergleich:**

| | MC | TD(0) |
|---|---|---|
| Ziel | echter Return $G_t$ | $R_{t+1}+\gamma V(S_{t+1})$ (geschätzt) |
| Bias | **unverzerrt** | **verzerrt** (bootstrapping) |
| Varianz | **hoch** | **niedrig** |
| braucht Episodenende? | ja | nein |
| Markov ausgenutzt? | nein | ja (nutzt Struktur) |

TD ist in der Praxis meist **schneller** und **dateneffizienter** — der Preis ist der Bias, der
mit besserer Schätzung verschwindet.

**Dazwischen: n-step & TD(λ).** Es gibt ein ganzes Spektrum. Der **n-step-Return**
$$G_{t:t+n}=R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^n V(S_{t+n})$$
interpoliert zwischen TD ($n=1$) und MC ($n=\infty$). **TD(λ)** mittelt geometrisch **alle**
n-step-Returns mit Gewichten $(1-\lambda)\lambda^{n-1}$ zum **λ-Return** $G_t^\lambda$. Effizient
implementiert man das *nicht* durch Vorausschauen, sondern durch **Eligibility Traces**
$e(s)$, die rückwärts „Verantwortlichkeit" verteilen (Abschnitt 3.2).

### 2.4 Modellfreie Kontrolle: SARSA vs. Q-Learning

Für **Kontrolle** (bestes $\pi$ finden) lernen wir $Q(s,a)$ statt $V(s)$, denn ohne Modell
braucht man $Q$, um greedy handeln zu können. Beide folgen dem TD-Muster, unterscheiden sich
aber im **Ziel**.

**SARSA (on-policy).** Der Name kommt vom Tupel $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$. Man wählt
$A_{t+1}$ **mit der aktuellen (explorierenden) Policy** und bootstrappt mit *diesem* Wert:
$$\boxed{\;Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma\,Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\big]\;}$$
SARSA lernt den Wert der Policy, **die es tatsächlich ausführt** (inkl. Exploration).

**Q-Learning (off-policy).** Man bootstrappt mit der **greedy** (besten) Aktion, unabhängig
davon, was tatsächlich gewählt wurde:
$$\boxed{\;Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma\,\max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)\big]\;}$$
Q-Learning approximiert direkt $Q_*$ — es lernt die **optimale** Policy, während es einer
anderen (explorierenden) **Verhaltens-Policy** folgt. Das ist der Kern von *off-policy*: die
Policy, die man *lernt*, ≠ die Policy, die man *ausführt*.

**Expected SARSA.** Ersetzt das $Q(S_{t+1},A_{t+1})$ durch den **Erwartungswert** über die
Policy — reduziert die Varianz aus der zufälligen Wahl von $A_{t+1}$:
$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\Big[R_{t+1}+\gamma\sum_{a'}\pi(a'\mid S_{t+1})Q(S_{t+1},a') - Q(S_t,A_t)\Big].$$
Bei greedy $\pi$ wird Expected SARSA **zu** Q-Learning — es ist die verallgemeinernde Klammer.

> **Das Cliff-Walking-Aha (Projekt 02).** Auf einer Gridworld mit einer „Klippe" (Absturz =
> −100) findet **Q-Learning** die *optimale, riskante* Route direkt an der Kante entlang —
> lernt sie aber, während es gelegentlich (durch ε-Exploration) abstürzt, also mit **schlechterem
> Online-Ertrag**. **SARSA** lernt eine *sicherere* Route mit Abstand zur Klippe, weil es die
> Explorationskosten in seinen Werten *mit einpreist*, und erzielt **online mehr Belohnung**.
> Fazit: „optimal im Grenzwert" (Q-Learning) ≠ „gut, während man noch exploriert" (SARSA).

### 2.5 Generalized Policy Iteration & ε-greedy-Kontrolle

Alle Kontrollverfahren sind Instanzen von **Generalized Policy Iteration (GPI)**: ein Tanz aus
(a) **Evaluation** (Werte an die Policy annähern) und (b) **Improvement** (Policy greedy bzgl.
Werten machen). Sie treiben sich gegenseitig zum gemeinsamen Fixpunkt $\pi_*,Q_*$.

Damit Kontrolle funktioniert, muss der Agent **explorieren** — würde er immer nur greedy
handeln, sähe er nie Alternativen. Die einfachste Lösung ist eine **ε-greedy**-Policy:
$$\pi(a\mid s)=\begin{cases}1-\varepsilon+\varepsilon/|\mathcal A| & a=\arg\max_{a'}Q(s,a')\\[2pt]\varepsilon/|\mathcal A| & \text{sonst.}\end{cases}$$
Mit Wahrscheinlichkeit $1-\varepsilon$ die beste bekannte Aktion, mit $\varepsilon$ eine
zufällige. Für **Konvergenz zur Optimal-Policy** verlangt man **GLIE** (*Greedy in the Limit
with Infinite Exploration*): jeder $(s,a)$ wird unendlich oft besucht **und** $\varepsilon\to0$
(z. B. $\varepsilon_k=1/k$). Zusammen mit den **Robbins-Monro**-Bedingungen an die Lernrate,
$$\sum_k \alpha_k = \infty,\qquad \sum_k \alpha_k^2 < \infty,$$
konvergiert tabellarisches Q-Learning **mit Wahrscheinlichkeit 1** gegen $Q_*$ (Watkins &
Dayan 1992). In der Praxis nimmt man oft ein konstantes, kleines $\alpha$ und ein langsam
fallendes $\varepsilon$ — Theorie und Praxis weichen hier bewusst ab.

---

## 3 · Advanced-Themen

### 3.1 Das Explore-Exploit-Dilemma & Multi-armed Bandits

Der **Bandit** ist ein MDP mit *einem* Zustand: $k$ „Arme" (Aktionen), jeder liefert Belohnung
aus einer unbekannten Verteilung mit Mittelwert $q_*(a)$. Er isoliert das **Explore-Exploit-
Dilemma** in Reinform: **exploitieren** (den bisher besten Arm ziehen) vs. **explorieren**
(einen unsicheren Arm testen, der *vielleicht* besser ist). Zu viel Exploit → man bleibt in
einem lokalen Optimum stecken; zu viel Explore → man verschenkt Belohnung. Man misst die Güte
über den **Regret** $\rho_T = T\,q_*(a^*) - \sum_{t=1}^{T}\mathbb E[q_*(A_t)]$ (entgangene
Belohnung gegenüber dem besten Arm).

Strategien (Projekt 01 vergleicht sie):
- **ε-greedy** — simpel, aber exploriert *uniform* und *ewig* (linearer Regret bei festem ε).
- **Optimistische Initialisierung** — setze $Q_0(a)$ absichtlich zu hoch; jede noch nicht
  probierte Aktion wirkt attraktiv → *automatische* Anfangsexploration, danach greedy. Trick,
  kein Allheilmittel (wirkt nur früh, nicht bei nichtstationär).
- **UCB (Upper Confidence Bound)** — wähle $A_t=\arg\max_a\big[Q(a)+c\sqrt{\ln t / N(a)}\big]$:
  „Optimismus angesichts von Unsicherheit". Der Bonusterm ist groß für selten gezogene Arme und
  schrumpft mit Wissen. UCB1 erreicht **logarithmischen** Regret $O(\ln T)$ — beweisbar
  fast-optimal.
- **Boltzmann/Softmax** — wähle $a$ mit Wahrscheinlichkeit $\propto e^{Q(a)/\tau}$; die
  **Temperatur** $\tau$ steuert von gierig ($\tau\to0$) zu uniform ($\tau\to\infty$).
- **Thompson Sampling** (bayesianisch) — halte eine Posterior-Verteilung über jedes $q_*(a)$,
  ziehe eine Stichprobe je Arm, spiele den Sieger. Oft empirisch das Beste; hier nur erwähnt.

### 3.2 Eligibility Traces — TD(λ) effizient

Ein **Eligibility Trace** $e_t(s)$ (bzw. $e_t(s,a)$) ist ein Kurzzeitgedächtnis, das markiert,
welche Zustände *kürzlich und häufig* besucht wurden und daher für einen aktuellen TD-Fehler
**„verantwortlich"** sind. **Accumulating trace:**
$$e_t(s)=\gamma\lambda\,e_{t-1}(s)+\mathbb 1[S_t=s],\qquad V(s)\leftarrow V(s)+\alpha\,\delta_t\,e_t(s)\ \ \forall s.$$
Der eine TD-Fehler $\delta_t$ wird auf *alle* kürzlich besuchten Zustände verteilt, abgestuft
nach $ (\gamma\lambda)^{k}$. Für $\lambda=0$ ergibt sich TD(0), für $\lambda=1$ (näherungsweise)
MC. Das löst das **credit-assignment**-Problem elegant und beschleunigt oft das Lernen. Die
Kontroll-Varianten heißen **SARSA(λ)** und **Watkins's Q(λ)**.

### 3.3 Funktionsapproximation — warum Tabellen nicht reichen

Bisher war $Q$ eine **Tabelle** $|\mathcal S|\times|\mathcal A|$. Das scheitert, sobald der
Zustandsraum groß/kontinuierlich ist (Schach $\sim10^{47}$, Go $\sim10^{170}$, Bilder als
Zustand). Lösung: **parametrisiere** $\hat V(s;\mathbf w)\approx V^\pi(s)$ mit wenigen
Gewichten $\mathbf w$ — linear $\hat V(s;\mathbf w)=\mathbf w^\top\mathbf x(s)$ mit
**Feature-Vektor** $\mathbf x(s)$, oder nichtlinear (neuronales Netz → Deep RL). Man
**verallgemeinert** dann über ähnliche Zustände.

Das Lernziel wird ein Regressionsproblem; das **semi-gradient TD(0)**-Update lautet
$$\mathbf w \leftarrow \mathbf w + \alpha\big[R_{t+1}+\gamma\hat V(S_{t+1};\mathbf w) - \hat V(S_t;\mathbf w)\big]\,\nabla_{\mathbf w}\hat V(S_t;\mathbf w).$$
„Semi"-Gradient, weil man den Gradienten des bootstrapped Ziels **ignoriert** (es hängt selbst
von $\mathbf w$ ab) — man behandelt das TD-Ziel als fix.

> **Die „deadly triad".** Kombiniert man **(1) Funktionsapproximation + (2) Bootstrapping +
> (3) off-policy-Training**, kann RL **divergieren** (Werte laufen ins Unendliche). Jede
> Komponente einzeln ist ok, aber alle drei zusammen sind gefährlich. Deep Q-Networks (Modul 14)
> haben genau diese Triade und brauchen deshalb Tricks (**Experience Replay**, **Target
> Network**), um sie zu zähmen. Das ist die konzeptuelle Brücke zu Modul 14.

### 3.4 Policy-Gradient-Methoden (Ausblick)

Statt Werte zu lernen und daraus greedy eine Policy abzuleiten, kann man die **Policy direkt**
parametrisieren, $\pi_\theta(a\mid s)$, und $\theta$ per Gradientenaufstieg auf die erwartete
Rendite $J(\theta)=\mathbb E_{\pi_\theta}[G_0]$ optimieren. Das **Policy-Gradient-Theorem** gibt
$$\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\!\big[\nabla_\theta\log\pi_\theta(A_t\mid S_t)\,Q^{\pi_\theta}(S_t,A_t)\big],$$
woraus der **REINFORCE**-Algorithmus folgt (Monte-Carlo-Schätzung von $Q$ durch $G_t$, oft mit
einer **Baseline** $b(s)$ zur Varianzreduktion). Vorteile: natürliche **stochastische** Policies,
**kontinuierliche** Aktionsräume. **Actor-Critic** verbindet beides (ein *Actor* $\pi_\theta$, ein
*Critic* $\hat V$). Das ist die Grundlage moderner Verfahren (A2C/A3C, PPO, DDPG, SAC) — Thema
von **Modul 14 (Deep RL for Optimal Control)**.

### 3.5 Wenn Markov verletzt ist: POMDPs

Sieht der Agent nicht den vollen Zustand, sondern nur eine **Beobachtung** $O_t$, hat man einen
**Partially Observable MDP (POMDP)**. Optimales Handeln erfordert dann einen **Belief State**
(Posterior über den wahren Zustand) — exakt oft unlösbar. Praktische RL-Antworten: den Zustand
aus einem **Fenster** vergangener Beobachtungen bauen oder ein **rekurrentes** Netz (RNN/LSTM,
Modul 09) die Historie zusammenfassen lassen.

---

## 4 · Zusammenfassung / Cheat-Sheet

**Begriffe.** MDP $(\mathcal S,\mathcal A,P,R,\gamma)$ · Policy $\pi(a|s)$ · Return
$G_t=\sum_k\gamma^k R_{t+k+1}$ · $V^\pi,Q^\pi$ · Optimal $V_*,Q_*$, $\pi_*(s)=\arg\max_a Q_*(s,a)$.

**Bellman.**
- Erwartung: $V^\pi(s)=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V^\pi(s')]$
- Optimalität: $Q_*(s,a)=\sum_{s',r}p(s',r|s,a)[r+\gamma\max_{a'}Q_*(s',a')]$

**Update-Regeln (alle: $\text{alt}\leftarrow\text{alt}+\alpha(\text{Ziel}-\text{alt})$).**

| Methode | Ziel für Update |
|---|---|
| MC | $G_t$ (echter Return) |
| TD(0) Prädiktion | $R_{t+1}+\gamma V(S_{t+1})$ |
| **SARSA** (on-policy) | $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$ |
| **Expected SARSA** | $R_{t+1}+\gamma\sum_{a'}\pi(a'|S_{t+1})Q(S_{t+1},a')$ |
| **Q-Learning** (off-policy) | $R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')$ |

**Exploration.** ε-greedy · optimistische Init · UCB $Q(a)+c\sqrt{\ln t/N(a)}$ · Softmax($\tau$).

**Konvergenz.** GLIE (∞ Exploration + $\varepsilon\to0$) + Robbins-Monro ($\sum\alpha=\infty,\sum\alpha^2<\infty$).

**Achsen.** modellbasiert (Planung, DP) ↔ modellfrei (Lernen) · on-policy ↔ off-policy ·
Bootstrapping (TD/DP) ↔ Sampling volle Returns (MC) · Tabelle ↔ Funktionsapproximation.

**Deadly Triad.** FA + Bootstrapping + off-policy → mögliche Divergenz (→ Deep RL braucht Replay/Target-Net).

---

## 5 · Selbsttest

<details>
<summary><b>1.</b> Worin unterscheidet sich RL grundlegend von überwachtem Lernen?</summary>

Kein Supervisor/Label mit „richtiger Aktion" — nur ein **evaluatives**, oft **verzögertes**
Belohnungssignal. Die Daten sind **nicht i.i.d.**: der Agent beeinflusst durch sein Handeln,
welche Zustände er als Nächstes sieht, und muss aktiv **explorieren**. Ziel ist die **langfristige**
kumulierte Belohnung (Return), nicht die momentane.
</details>

<details>
<summary><b>2.</b> Warum diskontiert man den Return, und was bewirkt γ?</summary>

Damit die (bei kontinuierlichen Aufgaben unendliche) Summe **konvergiert** ($|G|\le R_{\max}/(1-\gamma)$)
und um **nähere Belohnungen höher zu gewichten**. $\gamma\to0$ = myopisch/gierig, $\gamma\to1$ =
weitsichtig. Bei episodischen Aufgaben ist $\gamma=1$ oft ok.
</details>

<details>
<summary><b>3.</b> Was besagt die Bellman-Optimalitätsgleichung, und warum ist sie „schwerer" als die Erwartungsgleichung?</summary>

$V_*(s)=\max_a\sum_{s',r}p(s',r|s,a)[r+\gamma V_*(s')]$ — der optimale Wert ist der **beste**
erreichbare erwartete Return. Durch das $\max$ ist sie **nichtlinear** (kein lineares
Gleichungssystem mehr), daher löst man sie iterativ (Value Iteration) statt direkt.
</details>

<details>
<summary><b>4.</b> Nenne den Kernunterschied zwischen Monte Carlo und TD(0) in Bias/Varianz.</summary>

MC nutzt den **echten Return** → **unverzerrt**, aber **hohe Varianz**, braucht Episodenende.
TD(0) nutzt ein **bootstrapped Ziel** $R+\gamma V(S')$ → **verzerrt** (nutzt eigene Schätzung),
aber **niedrige Varianz**, lernt **online** nach jedem Schritt.
</details>

<details>
<summary><b>5.</b> SARSA vs. Q-Learning: Was ist on- bzw. off-policy, und was heißt das praktisch?</summary>

**SARSA (on-policy)** bootstrappt mit der **tatsächlich gewählten** nächsten Aktion
$A_{t+1}\sim\pi$ → lernt den Wert der ausgeführten (explorierenden) Policy.
**Q-Learning (off-policy)** bootstrappt mit $\max_{a'}Q(S_{t+1},a')$ → lernt direkt $Q_*$
(die *optimale* Policy), obwohl es einer explorierenden Verhaltens-Policy folgt. Praktisch: bei
Cliff Walking wählt SARSA die **sichere**, Q-Learning die **optimale-aber-riskante** Route.
</details>

<details>
<summary><b>6.</b> Erkläre das Explore-Exploit-Dilemma und wie UCB es adressiert.</summary>

**Exploit** = beste bekannte Aktion nutzen; **Explore** = unsichere Aktionen testen, die
*vielleicht* besser sind. UCB wählt $\arg\max_a[Q(a)+c\sqrt{\ln t/N(a)}]$ — der Bonus ist groß
für **selten** gezogene Arme und schrumpft mit Erfahrung („Optimismus bei Unsicherheit"),
erreicht **logarithmischen** Regret.
</details>

<details>
<summary><b>7.</b> Was ist GLIE, und warum braucht man es?</summary>

*Greedy in the Limit with Infinite Exploration*: (a) jeder $(s,a)$ wird **unendlich oft**
besucht **und** (b) die Policy wird im Grenzwert **greedy** ($\varepsilon\to0$). Nötig, damit
ε-greedy-Kontrolle (z. B. SARSA) gegen die **optimale** Policy konvergiert — sonst bleibt
Rest-Exploration und man ist nur ε-optimal.
</details>

<details>
<summary><b>8.</b> Was ist die „deadly triad" und warum ist sie relevant für Deep RL?</summary>

Die Kombination **Funktionsapproximation + Bootstrapping + off-policy** kann zu **Divergenz**
führen. Deep Q-Networks (Modul 14) haben alle drei → man braucht **Experience Replay** und ein
**Target Network**, um zu stabilisieren.
</details>

<details>
<summary><b>9.</b> Wozu dienen Eligibility Traces?</summary>

Sie lösen das **credit-assignment**-Problem und realisieren **TD(λ)** effizient: ein Trace
$e(s)$ merkt sich kürzlich besuchte Zustände, sodass ein aktueller TD-Fehler $\delta_t$
rückwirkend auf **alle** verantwortlichen Zustände (abgestuft nach $(\gamma\lambda)^k$) verteilt
wird — interpoliert zwischen TD(0) und MC.
</details>

<details>
<summary><b>10.</b> Warum ist tabellarisches Q-Learning für Schach/Go ungeeignet, und was ist die Alternative?</summary>

Die Tabelle hätte $|\mathcal S|\times|\mathcal A|$ Einträge — bei $10^{47}$/$10^{170}$ Zuständen
unmöglich zu speichern *oder* zu besuchen. Alternative: **Funktionsapproximation** $\hat Q(s,a;\mathbf w)$
mit Features/neuronalem Netz, die über ähnliche Zustände **verallgemeinert** (→ Deep RL).
</details>

---

## 6 · Literatur & Quellen

**Das Standardwerk (kostenlos!):**
- 📗 **Sutton & Barto — *Reinforcement Learning: An Introduction* (2. Aufl., 2018)** —
  *die* Referenz, didaktisch hervorragend, **frei als PDF** (incompleteideas.net/book/the-book.html).
  Einsteigerfreundlich *und* vollständig. Kap. 2 (Bandits), 3 (MDP), 4 (DP), 5 (MC), 6 (TD),
  7 (n-step), 12 (Eligibility Traces), 13 (Policy Gradient) decken genau dieses Modul ab.
  **Beste Einzelquelle.**

**Vorlesungen (frei, online):**
- 🎥 **David Silver — *RL Course* (DeepMind/UCL, 10 Vorlesungen, YouTube)** — der Klassiker,
  folgt grob Sutton & Barto. *Einsteiger→vertiefend.*
- 🎥 **Stanford CS234 *Reinforcement Learning* (Emma Brunskill)** — Vorlesungsvideos + Folien frei.
  *Vertiefend.*
- 🎥 **UC Berkeley CS285 *Deep RL* (Sergey Levine)** — für den Sprung zu Modul 14. *Vertiefend.*

**Interaktiv/Blog (einsteigerfreundlich):**
- 🌐 **Gymnasium-Dokumentation** (gymnasium.farama.org) — Standard-API für RL-Umgebungen
  (Nachfolger von OpenAI Gym). Wir bauen im Modul zwar *ohne* Gym (didaktisch), aber für die
  Praxis solltest du die API kennen.
- 🌐 **Andrej Karpathy — *Deep Reinforcement Learning: Pong from Pixels*** (karpathy.github.io) —
  legendärer Blogpost zu Policy Gradient. *Einsteiger→vertiefend.*
- 🌐 **Lilian Weng — *A (Long) Peek into Reinforcement Learning*** (lilianweng.github.io) —
  kompakte, präzise Übersicht. *Vertiefend.*
- 🌐 **Spinning Up in Deep RL** (OpenAI, spinningup.openai.com) — für Modul 14, saubere
  Implementierungen + Theorie. *Vertiefend, kostenlos.*

**Klassische Paper (vertiefend):**
- Watkins & Dayan (1992), *Q-learning* — Konvergenzbeweis.
- Sutton (1988), *Learning to Predict by the Methods of Temporal Differences* — TD-Ursprung.
- Auer, Cesa-Bianchi & Fischer (2002), *Finite-time Analysis of the Multiarmed Bandit* — UCB1.

---

## Nächstes Modul

**Modul 14 — Deep Reinforcement Learning for Optimal Control** ersetzt die Q-**Tabelle** durch
ein **neuronales Netz** (DQN, PyTorch aus Modul 05/09), zähmt die *deadly triad* mit Replay &
Target-Network und geht zu **Policy-Gradient/Actor-Critic** (REINFORCE → PPO) über — für
**kontinuierliche** Steuerung (Optimal Control). Dieses Modul liefert dafür das komplette
konzeptuelle Fundament.
