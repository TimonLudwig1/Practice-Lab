# Module 14 — Deep Reinforcement Learning for Optimal Control

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** In module 13 the value function was a **table** $Q(s,a)$ — that
> works only for small, discrete state spaces. As soon as the state becomes
> high-dimensional or **continuous** (a camera image, the joint angles of a robot, the
> attitude of a drone), no table can be stored any more. **Deep RL** replaces the table with
> a **neural network** that *generalizes* over similar states. This module introduces the
> core families: **value-based** (DQN and extensions), **policy-based** (REINFORCE,
> actor-critic, PPO) and **continuous control** (DDPG/TD3/SAC) — and builds the bridge to
> **classical optimal control** (LQR, Bellman/HJB, Pontryagin), from which the module name
> comes.

**Helpful prior knowledge:** module 13 (MDP, Bellman, Q-learning, the policy-gradient idea,
the "deadly triad"), module 05/09 (neural networks & PyTorch, backprop, optimizers), the
basics of calculus/linear algebra (gradient, eigenvalues, quadratic forms).

**Modules you should have done first:**
- **Module 13 (RL)** — *mandatory*. We build directly on the Bellman equations, Q-learning,
  SARSA, ε-greedy, the policy-gradient theorem and the **deadly triad**. Every algorithm here
  is the "deep" version of a tabular method there.
- **Module 05 (ML 2)** — MLPs, backpropagation, SGD/Adam, regularization, PyTorch practice.

> **Hardware note.** Real deep RL (Atari from pixels, MuJoCo robots, large PPO) needs GPU
> hours to days and is **not** sensibly trainable on a laptop. Therefore: we explain the
> **expensive** methods fully **theoretically** in formal notation, and the projects
> deliberately use **small** tasks (a self-built CartPole, linear systems), **small networks**
> and **few episodes**, so that everything runs in minutes on CPU/MPS. The understanding is
> identical — only the scale is reduced.

---

## Learning objectives

After this module you can …

- explain **why** and **how** a neural network replaces the Q table, and name the **deadly
  triad** (function approximation + bootstrapping + off-policy) as the central source of
  instability;
- derive **DQN** completely — including **experience replay** and the **target network** — and
  justify how both tricks tame the triad; place **double DQN**, **dueling** and **prioritized
  replay**;
- explain the contrast **value-based vs. policy-based** and write down the **policy-gradient
  theorem**, **REINFORCE**, the **baseline/advantage** and **actor-critic** (A2C, GAE)
  formally;
- understand **PPO** and its **clipped surrogate objective** (why "proximal");
- place **continuous control** (deterministic policy gradient → **DDPG/TD3**, entropy → **SAC**);
- establish the connection to **classical optimal control**: **Bellman ↔ the HJB equation**,
  **LQR/Riccati** as an exactly solvable special case, **Pontryagin's maximum principle**,
  **MPC**;
- build and stabilize a small deep-RL system yourself in **PyTorch**.

---

## 1 · Basics — from the table to the network

### 1.1 Why function approximation (recap & deepening)

Tabular Q-learning stores a value per $(s,a)$. That fails twice: **memory** (Go has
$\sim10^{170}$ states) and **experience** (one can never visit every state). Continuous states
($s\in\mathbb R^n$) even have *uncountably* many entries. Solution: approximate
$$\hat q(s,a;\mathbf w)\approx q_*(s,a),\qquad \hat v(s;\mathbf w)\approx v_*(s),$$
with a parametrized function approximator (weights $\mathbf w$). A **neural network** is the
most expressive choice and learns the **features** itself (instead of hand-designing them).
The gain is **generalization**: an update in one state improves the estimate in *all similar*
states. That is why deep RL works in enormous spaces at all — and at the same time the source
of all instability.

### 1.2 The learning goal as regression — and why it is tricky

In module 13 the **semi-gradient** TD update was
$$\mathbf w \leftarrow \mathbf w + \alpha\big[\underbrace{R+\gamma \hat q(S',A';\mathbf w)}_{\text{target (bootstrapped)}} - \hat q(S,A;\mathbf w)\big]\nabla_{\mathbf w}\hat q(S,A;\mathbf w).$$
"Semi", because one pretends the target is **fixed**, although it itself depends on
$\mathbf w$. Harmless with a tabular representation; with neural approximation three problems
arise that together form the **deadly triad**:

1. **Function approximation** — an update "leaks" onto other states (can shift wrong ones).
2. **Bootstrapping** — the target contains one's own (erroneous) estimate → errors can
   amplify themselves.
3. **off-policy** — one learns from data of a different distribution than the target policy.

All three together can cause **divergence** (values run off to $\pm\infty$). Two further
violations of the usual supervised-learning assumptions come on top: the data are **strongly
correlated** (consecutive transitions resemble each other) and the **target distribution
moves** (the policy changes during learning → a *non-stationary target*). The deep-RL
algorithms are at their core **tricks to stay stable nonetheless.**

---

## 2 · Value-based deep RL: DQN

### 2.1 Deep Q-network (DQN)

**DQN** (Mnih et al. 2013/2015, the "Atari paper") is Q-learning with a neural network
$Q(s,a;\theta)$ (usually: input $s$, one output per action). One minimizes the expected
squared **TD error**:
$$L(\theta)=\mathbb E_{(s,a,r,s')\sim \mathcal D}\Big[\big(\underbrace{r+\gamma\max_{a'}Q(s',a';\theta^-)}_{\text{target }y} - Q(s,a;\theta)\big)^2\Big].$$

Two ingredients turn this into a *stably trainable* method:

**(a) Experience replay.** Store transitions $(s,a,r,s')$ in a **buffer** $\mathcal D$ (e.g.
the last $10^6$) and train on **random minibatches** from it. This (i) **decorrelates** the
data (breaks the temporal dependence) and (ii) uses each experience **multiple times** (data
efficiency). — Addresses the triad component "correlated data".

**(b) Target network.** The target $y$ uses a **frozen** network $\theta^-$ that is only pulled
towards $\theta$ every $C$ steps (or via a Polyak average
$\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-$). Without it one would "shoot at a moving
target" — the target $y$ would depend immediately on every weight update and could
oscillate/diverge. The frozen target makes learning resemble a stable **regression problem**
again. — Addresses the "non-stationary target".

**Training loop (pseudocode):**
```
initialize Q(θ), target Q(θ⁻)=θ, an empty replay buffer D
for each episode:
    s = env.reset()
    repeat:
        a = ε-greedy(Q(s,·;θ))                 # exploration as in module 13
        s', r, done = env.step(a);  D.push(s,a,r,s',done)
        minibatch B ~ D
        y = r + γ·(1-done)·max_a' Q(s',a';θ⁻)  # target with the target network
        θ ← θ - lr·∇θ  mean_B (Q(s,a;θ) - y)²   # one SGD/Adam step
        every C steps:  θ⁻ ← θ                  # pull the target network along
        s = s'
    reduce ε
```
Project 01 builds exactly this on a self-built CartPole.

### 2.2 DQN extensions (short but complete)

- **Double DQN** — the $\max$ operator in the target systematically **overestimates** (the max
  over noisy estimators ⇒ a positive bias). Solution: **decouple selection and evaluation** —
  choose the action with the *online* network, evaluate it with the *target* network:
  $$y = r + \gamma\,Q\big(s',\,\arg\max_{a'}Q(s',a';\theta);\,\theta^-\big).$$
- **Dueling DQN** — decompose $Q(s,a)=V(s)+A(s,a)$ into a **state value** and an **advantage**
  (with $A$ centered on its mean). Useful when the state value dominates and the action choice
  barely matters.
- **Prioritized experience replay** — draw transitions with a large TD error **more often**
  (there is the most to learn there), corrected via importance-sampling weights.
- **Rainbow** combines these and others (n-step, distributional RL, noisy nets).

**Limit of DQN:** it needs a discrete $\arg\max_a$ → **no continuous** actions (for those see
section 4). And: purely value-based, it learns a deterministic greedy policy.

---

## 3 · Policy-based deep RL

### 3.1 Why learn the policy directly?

Value-based methods learn $Q$ and derive the policy *indirectly* (greedily). **Policy-gradient**
methods parametrize the policy **directly**, $\pi_\theta(a\mid s)$, and optimize it by gradient
ascent. Advantages: (i) **continuous** action spaces naturally (a Gaussian policy), (ii)
**stochastic** optimal policies possible (important under partial observability/in games), (iii)
smooth improvement instead of a jumpy $\arg\max$. Disadvantage: higher **variance**, often less
data-efficient.

### 3.2 The policy-gradient theorem & REINFORCE

The goal is the expected return $J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G_0]$. The
**policy-gradient theorem** yields (remarkably: **without** a derivative of the environment
dynamics):
$$\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\!\Big[\sum_{t} \nabla_\theta\log\pi_\theta(A_t\mid S_t)\,\Psi_t\Big],$$
where $\Psi_t$ is a **credit signal**. Different choices of $\Psi_t$ give different algorithms:

| $\Psi_t$ | Method |
|---|---|
| $G_t$ (the full return) | **REINFORCE** (Monte Carlo) |
| $G_t - b(S_t)$ (baseline) | REINFORCE **with a baseline** (variance reduction) |
| $Q^\pi(S_t,A_t)$ | actor-critic (Q form) |
| $A^\pi(S_t,A_t)=Q^\pi-V^\pi$ | **advantage** actor-critic (A2C) |
| $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$ | TD actor-critic |

**REINFORCE** update (after a whole episode): $\theta\leftarrow\theta+\alpha\sum_t \nabla_\theta\log\pi_\theta(A_t|S_t)\,G_t$.
Intuition: **increase** the probability of actions that led to a high return. The $\log$
derivative is called the **score function** (the "REINFORCE trick").

**Baseline.** One may subtract an **arbitrary state-dependent** function $b(S_t)$ from $\Psi_t$
without changing the expectation (and thus the unbiasedness) — because
$\mathbb E_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a|s)]=0$. A good baseline (typically
$b=\hat V(s)$) lowers the **variance** drastically. $G_t-\hat V(S_t)$ estimates the
**advantage**: "was this action better than the average in this state?".

### 3.3 Actor-critic & GAE

**Actor-critic** combines both worlds: an **actor** $\pi_\theta$ (chooses actions) and a
**critic** $\hat V_\phi$ (evaluates states, bootstraps à la TD). The critic provides the
baseline/the advantage, the actor makes the policy-gradient step. **A2C** is the synchronous,
**A3C** the asynchronous (parallel workers) variant.

The **advantage** can be estimated over different numbers of steps — a bias/variance trade-off
like n-step in module 13. **Generalized advantage estimation (GAE)** averages them
geometrically with a parameter $\lambda$:
$$\hat A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l},\qquad \delta_t=R_{t+1}+\gamma \hat V(S_{t+1})-\hat V(S_t).$$
$\lambda=0$ → pure TD (low variance, more bias), $\lambda=1$ → the Monte-Carlo advantage.

### 3.4 PPO — proximal policy optimization

Naive policy gradient is **sensitive to the step size**: too large an update can "destroy" the
policy (it then only visits bad states from which it hardly recovers). **TRPO** solved this with
a hard KL constraint; **PPO** (Schulman et al. 2017) is the simpler, today most-used variant.
With the **probability ratio**
$r_t(\theta)=\dfrac{\pi_\theta(A_t|S_t)}{\pi_{\theta_{\text{old}}}(A_t|S_t)}$ PPO maximizes the
**clipped** objective
$$L^{\text{CLIP}}(\theta)=\mathbb E_t\Big[\min\big(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\big)\Big].$$
The **clipping** removes the incentive to drive $r_t$ far beyond $1\pm\epsilon$ — the policy
stays in the **vicinity** (*proximal*) of the old one, which stabilizes the update and allows
several epochs per dataset (more data-efficient than vanilla PG). PPO is the de-facto standard
for many continuous and discrete tasks (robotics, RLHF for LLMs).

---

## 4 · Continuous control & optimal control

### 4.1 Deterministic policy gradient: DDPG, TD3

For **continuous** actions ($a\in\mathbb R^m$), $\max_a Q(s,a)$ is no longer trivial. **DDPG**
(deep deterministic policy gradient) learns a **deterministic** actor policy $\mu_\theta(s)$ and
a critic $Q_\phi(s,a)$; the actor is moved towards a higher Q value:
$\nabla_\theta J\approx\mathbb E[\nabla_a Q_\phi(s,a)|_{a=\mu_\theta(s)}\nabla_\theta\mu_\theta(s)]$
(the chain rule — "move the action uphill in the Q landscape"). It is at its core **DQN for
continuous actions** (with replay & target networks). Exploration via additive noise.

**TD3** (twin delayed DDPG) fixes DDPG's overestimation with three tricks: **two** critics (take
the minimum → less overestimation), **delayed** actor updates, and **target-policy smoothing**
(noise in the target).

### 4.2 Maximum-entropy RL: SAC

**Soft actor-critic (SAC)** maximizes the return **plus** the entropy of the policy:
$$J(\pi)=\sum_t\mathbb E\big[R_{t+1}+\alpha\,\mathcal H(\pi(\cdot\mid S_t))\big].$$
The **entropy bonus** $\alpha\mathcal H$ rewards "as random as possible, as long as the task is
solved" → better exploration, more robust policies, more stable training. SAC is off-policy,
sample-efficient and one of the strongest algorithms for continuous control.

### 4.3 The bridge to classical optimal control

The module name is **"for optimal control"** — RL is the **data-/model-free** sister of
classical **optimal control**. The central connections:

- **Bellman ↔ HJB.** The Bellman optimality equation is the **discrete-time** form of the
  **Hamilton-Jacobi-Bellman (HJB)** equation of continuous optimal control. Both say: "the
  optimal value now = the immediate utility + the optimal value afterwards". RL solves them by
  **learning from samples** when the model is unknown.
- **LQR — the exactly solvable case.** For a **linear** system $x_{t+1}=Ax_t+Bu_t$ with
  **quadratic** costs $\sum_t (x_t^\top Q x_t + u_t^\top R u_t)$, the optimal control is a
  **linear feedback** $u_t=-Kx_t$, where $K$ follows from the **algebraic Riccati equation**.
  This is "value iteration with a closed-form solution" — and serves (in project 03) as the
  **exact reference** that a learned RL controller has to measure up to (exactly as value
  iteration was the reference in module 13).
- **Pontryagin's maximum principle** — the second major approach of optimal control (necessary
  conditions via the *adjoint state / costate*), complementary to the HJB/dynamic-programming
  approach.
- **MPC (model predictive control)** — if a (learned or known) model is available: optimize at
  each step over a **finite horizon**, execute only the first step, repeat. Closely related to
  **model-based RL**.

### 4.4 Model-based RL (outlook)

Instead of only learning values/a policy, one can learn a **model** of the dynamics
$\hat p(s'|s,a)$ and *plan* in it (Dyna, PILCO, world models, MuZero). Advantage: **data
efficiency** (planning in the model is cheap). Disadvantage: model errors propagate. Hybrid
methods (Dyna: combine real + model-generated experience) are an active field.

---

## 5 · Practical pitfalls (deep RL is notoriously fragile)

- **Reproducibility** — results fluctuate strongly over random seeds; always average over
  several seeds.
- **Reward shaping** — badly chosen rewards lead to *reward hacking* (the agent optimizes the
  wrong thing).
- **Hyperparameters** — learning rates, network size, replay size, target-update frequency,
  $\gamma$ are sensitive; small changes tip the training over.
- **Exploration** — in environments with sparse reward, ε-greedy/Gaussian noise is often not
  enough (→ intrinsic motivation, curiosity).
- **Sim-to-real** — policies learned in simulation transfer poorly to real hardware (the
  *reality gap*); a countermeasure: domain randomization.
- **Debugging** — first verify on a *tiny*, solvable task (exactly our approach), watch learning
  curves & Q values (do they diverge?).

---

## 6 · Summary / cheat sheet

**Map.**
```
                         Deep RL
        ┌───────────────────┼─────────────────────┐
   value-based         policy-based          continuous control
   DQN                 REINFORCE             DDPG  (det. PG)
   +Double/Dueling     +baseline             TD3   (2 critics)
   +prioritized replay actor-critic (A2C)    SAC   (max-entropy)
   (discrete actions)  PPO (clipped)         └── optimal control: LQR/HJB/MPC
```

**DQN target:** $y=r+\gamma\max_{a'}Q(s',a';\theta^-)$ · loss $=(y-Q(s,a;\theta))^2$ · **replay**
+ **target network** = taming the deadly triad.

**Double-DQN target:** $y=r+\gamma Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$.

**Policy gradient:** $\nabla_\theta J=\mathbb E[\sum_t\nabla_\theta\log\pi_\theta(A_t|S_t)\,\Psi_t]$,
$\Psi_t\in\{G_t,\ G_t-b(s),\ A^\pi(s,a),\ \delta_t\}$.

**Advantage:** $A(s,a)=Q(s,a)-V(s)$ · **GAE**: $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$.

**PPO:** maximize $\mathbb E[\min(r_t\hat A_t,\ \text{clip}(r_t,1{-}\epsilon,1{+}\epsilon)\hat A_t)]$,
$r_t=\pi_\theta/\pi_{\theta_{old}}$.

**Optimal control:** Bellman ↔ **HJB**; linear+quadratic ⇒ **LQR** $u=-Kx$ (Riccati) = the exact
reference; **MPC** = plan in the model over a horizon.

---

## 7 · Self-test

<details>
<summary><b>1.</b> Why does deep RL replace the Q table with a network, and what does one gain/risk?</summary>

Because tables are neither storable nor visitable for large/continuous state spaces. A network
**generalizes** over similar states (one update helps many). Risk: the **deadly triad** arises
(function approximation + bootstrapping + off-policy) → possible divergence, plus correlated data
and a moving target.
</details>

<details>
<summary><b>2.</b> What do experience replay and the target network serve in DQN — each exactly one problem?</summary>

**Experience replay**: breaks the **temporal correlation** of the data (and uses experience
multiple times → data efficiency). **Target network**: freezes the bootstrapping **target** →
prevents "shooting at a moving target" (the non-stationary target) and makes learning resemble a
stable regression problem.
</details>

<details>
<summary><b>3.</b> What is the overestimation bias in DQN and how does double DQN fix it?</summary>

$\max_{a'}Q(s',a')$ takes the maximum over *noisy* estimators → systematically **too high** (a
positive bias). **Double DQN** decouples **selection** (the online network $\theta$) and
**evaluation** (the target network $\theta^-$): $y=r+\gamma Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$.
</details>

<details>
<summary><b>4.</b> State the policy-gradient theorem and the use of a baseline.</summary>

$\nabla_\theta J=\mathbb E[\sum_t\nabla_\theta\log\pi_\theta(A_t|S_t)\,\Psi_t]$. A state-dependent
**baseline** $b(s)$ (typically $\hat V(s)$) may be subtracted from $\Psi_t$ **without** changing
the expectation (unbiased), but reduces the **variance** strongly; $G_t-\hat V(s)$ estimates the
**advantage**.
</details>

<details>
<summary><b>5.</b> What distinguishes actor-critic from REINFORCE?</summary>

REINFORCE uses the **full Monte-Carlo return** $G_t$ (unbiased, high variance, needs the end of
the episode). **Actor-critic** additionally has a **critic** $\hat V_\phi$ that **bootstraps**
(TD) and serves as the baseline/advantage → lower variance, online updates. Actor = the policy,
critic = the value estimator.
</details>

<details>
<summary><b>6.</b> Why "proximal"? What does the clipping in PPO do?</summary>

Too large policy updates can destroy the policy. PPO keeps the new policy **close** (*proximal*)
to the old one by **clipping** the ratio $r_t=\pi_\theta/\pi_{\theta_{old}}$ to
$[1-\epsilon,1+\epsilon]$ — this removes the incentive to drive $r_t$ far, and allows several
update epochs per data batch (stable + data-efficient).
</details>

<details>
<summary><b>7.</b> Why can DQN not handle continuous actions, and what does DDPG do about it?</summary>

DQN needs $\arg\max_a Q(s,a)$ — for continuous $a$ not solvable in closed form. **DDPG** learns a
**deterministic actor** $\mu_\theta(s)$ and moves it **uphill** in the critic $Q_\phi$ via the
chain rule: $\nabla_\theta J\approx\mathbb E[\nabla_aQ_\phi\,\nabla_\theta\mu_\theta]$. It is "DQN
for continuous actions".
</details>

<details>
<summary><b>8.</b> What is the connection between the Bellman equation and HJB, and what is LQR?</summary>

The Bellman optimality equation is the **discrete-time** form of the **Hamilton-Jacobi-Bellman**
equation of continuous optimal control. **LQR** is the special case of *linear dynamics +
quadratic costs*: the optimal control is the linear feedback $u=-Kx$ with $K$ from the **Riccati
equation** — an exactly solvable reference.
</details>

<details>
<summary><b>9.</b> What does the entropy objective in SAC do?</summary>

SAC maximizes the return **+** the policy **entropy** ($\alpha\mathcal H(\pi)$). The bonus rewards
"as random as possible, as long as the task is solved" → better **exploration**, more robust
policies, more stable training.
</details>

<details>
<summary><b>10.</b> Name three reasons why deep RL is notoriously unstable/hard to reproduce.</summary>

Any three: the deadly triad (divergence), high **seed variance**, sensitive **hyperparameters**,
**correlated/non-stationary** data, sparse reward/exploration, **reward hacking**, the sim-to-real
gap. Therefore: first verify on tiny tasks, average over seeds.
</details>

---

## 8 · Literature & sources

**Books & courses (free):**
- **Sutton & Barto — *Reinforcement Learning: An Introduction* (2018)**, ch. 9–13 (function
  approximation, policy gradient). Free as a PDF. *Foundation.*
- **OpenAI Spinning Up in Deep RL** (spinningup.openai.com) — *the* practical introduction: clean
  derivations (VPG→TRPO→PPO→DDPG→TD3→SAC) **plus** reference-able code. Free. *Beginner →
  in-depth, highly recommended.*
- **UC Berkeley CS285 *Deep Reinforcement Learning* (Sergey Levine)** — videos + slides free.
  *In-depth, comprehensive.*
- **DeepMind × UCL RL Lecture Series** — the successor of Silver's course. *Beginner → in-depth.*

**Key papers (in-depth):**
- Mnih et al. (2015), *Human-level control through deep RL* (**DQN/Nature**).
- van Hasselt et al. (2016), *Deep RL with Double Q-learning*.
- Wang et al. (2016), *Dueling Network Architectures*.
- Schulman et al. (2015), *High-Dimensional Continuous Control Using GAE*.
- Schulman et al. (2017), *Proximal Policy Optimization* (**PPO**).
- Lillicrap et al. (2016), *Continuous control with deep RL* (**DDPG**); Fujimoto et al. (2018),
  **TD3**; Haarnoja et al. (2018), **SAC**.

**Optimal control (the bridge):**
- **Bertsekas — *Dynamic Programming and Optimal Control*** / *Reinforcement Learning and Optimal
  Control* (2019) — connects both worlds formally. *In-depth.*
- **Steven Brunton — *Control Bootcamp* (YouTube)** — LQR, Riccati, HJB illustratively. *Beginner.*

**Practice/tooling:**
- **Gymnasium** (gymnasium.farama.org) — the standard environment API (CartPole, Pendulum,
  MuJoCo). We build *without* Gym in the module (didactically), but you should know the API.
- **Stable-Baselines3** — maintained, tested implementations (DQN/PPO/SAC/TD3) for practice (not
  for learning the internals).

---

## Next module

With this the RL block (modules 13–14) is complete. **Module 15 — Machine Learning for Networks
1** follows. The foundation built in this module (value- vs. policy-based, actor-critic, optimal
control) is the basis for RL applications in robotics (modules 21/22), advanced automation (23)
and everywhere that **sequential decisions under uncertainty** are made.
# Modul 14 — Deep Reinforcement Learning for Optimal Control (deutsche Fassung)

> **Worum geht es?** In Modul 13 war die Wertfunktion eine **Tabelle** $Q(s,a)$ — das
> funktioniert nur bei kleinen, diskreten Zustandsräumen. Sobald der Zustand hochdimensional
> oder **kontinuierlich** wird (Kamerabild, Gelenkwinkel eines Roboters, Lage einer Drohne),
> ist keine Tabelle mehr speicherbar. **Deep RL** ersetzt die Tabelle durch ein **neuronales
> Netz**, das über ähnliche Zustände *verallgemeinert*. Dieses Modul führt die Kernfamilien
> ein: **wertbasiert** (DQN und Erweiterungen), **policy-basiert** (REINFORCE, Actor-Critic,
> PPO) und **kontinuierliche Steuerung** (DDPG/TD3/SAC) — und schlägt die Brücke zur
> **klassischen Optimalsteuerung** (LQR, Bellman/HJB, Pontryagin), von der der Modulname kommt.

**Hilfreiche Vorkenntnisse:** Modul 13 (MDP, Bellman, Q-Learning, Policy Gradient-Idee, die
„deadly triad"), Modul 05/09 (neuronale Netze & PyTorch, Backprop, Optimierer), Grundlagen
Analysis/lineare Algebra (Gradient, Eigenwerte, quadratische Formen).

**Diese Module solltest du vorher gemacht haben:**
- **Modul 13 (RL)** — *zwingend*. Wir bauen direkt auf Bellman-Gleichungen, Q-Learning, SARSA,
  ε-greedy, dem Policy-Gradient-Theorem und der **deadly triad** auf. Jeder Algorithmus hier ist
  die „tiefe" Version eines dortigen tabellarischen Verfahrens.
- **Modul 05 (ML 2)** — MLPs, Backpropagation, SGD/Adam, Regularisierung, PyTorch-Praxis.

> **⚠️ Hardware-Hinweis.** Echtes Deep RL (Atari aus Pixeln, MuJoCo-Roboter, großes PPO) braucht
> GPU-Stunden bis -Tage und ist auf einem Laptop **nicht** sinnvoll trainierbar. Deshalb: die
> **teuren** Verfahren erklären wir vollständig **theoretisch** in formaler Notation, und die
> Projekte nutzen bewusst **kleine** Aufgaben (selbstgebautes CartPole, lineare Systeme),
> **kleine Netze** und **wenige Episoden**, sodass alles in Minuten auf CPU/MPS läuft. Das
> Verständnis ist identisch — nur der Maßstab ist reduziert.

---

## Lernziele

Nach diesem Modul kannst du …

- erklären, **warum** und **wie** ein neuronales Netz die Q-Tabelle ersetzt, und die
  **deadly triad** (Funktionsapproximation + Bootstrapping + off-policy) als zentrale
  Instabilitätsquelle benennen;
- **DQN** vollständig herleiten — inkl. **Experience Replay** und **Target Network** — und
  begründen, wie beide Tricks die Triade zähmen; **Double DQN**, **Dueling** und
  **Prioritized Replay** einordnen;
- den Gegensatz **wertbasiert vs. policy-basiert** erklären und das **Policy-Gradient-Theorem**,
  **REINFORCE**, die **Baseline/Advantage** und **Actor-Critic** (A2C, GAE) formal aufschreiben;
- **PPO** und sein **clipped surrogate objective** verstehen (warum „proximal");
- **kontinuierliche Steuerung** (deterministischer Policy-Gradient → **DDPG/TD3**, Entropie →
  **SAC**) einordnen;
- die Verbindung zur **klassischen Optimalsteuerung** herstellen: **Bellman ↔ HJB-Gleichung**,
  **LQR/Riccati** als exakt lösbaren Spezialfall, **Pontryagins Maximumprinzip**, **MPC**;
- ein kleines Deep-RL-System in **PyTorch** selbst bauen und stabilisieren.

---

## 1 · Grundlagen — Von der Tabelle zum Netz

### 1.1 Warum Funktionsapproximation (Rückblick & Vertiefung)

Tabellarisches Q-Learning speichert einen Wert pro $(s,a)$. Das scheitert doppelt: **Speicher**
(Go hat $\sim10^{170}$ Zustände) und **Erfahrung** (man kann nie jeden Zustand besuchen).
Kontinuierliche Zustände ($s\in\mathbb R^n$) haben sogar *überabzählbar* viele Einträge. Lösung:
approximiere
$$\hat q(s,a;\mathbf w)\approx q_*(s,a),\qquad \hat v(s;\mathbf w)\approx v_*(s),$$
mit einem parametrisierten Funktionsapproximator (Gewichte $\mathbf w$). Ein **neuronales Netz**
ist die ausdrucksstärkste Wahl und lernt die **Features** selbst (statt sie von Hand zu bauen).
Der Gewinn ist **Generalisierung**: ein Update in einem Zustand verbessert die Schätzung in
*allen ähnlichen* Zuständen. Das ist der Grund, warum Deep RL überhaupt in riesigen Räumen
funktioniert — und zugleich die Quelle aller Instabilität.

### 1.2 Das Lernziel als Regression — und warum es tückisch ist

In Modul 13 war das **semi-gradient**-TD-Update
$$\mathbf w \leftarrow \mathbf w + \alpha\big[\underbrace{R+\gamma \hat q(S',A';\mathbf w)}_{\text{Ziel (bootstrapped)}} - \hat q(S,A;\mathbf w)\big]\nabla_{\mathbf w}\hat q(S,A;\mathbf w).$$
„Semi", weil man so tut, als sei das Ziel **fix**, obwohl es selbst von $\mathbf w$ abhängt. Bei
tabellarischer Darstellung harmlos; bei neuronaler Approximation entstehen drei Probleme, die
zusammen die **deadly triad** bilden:

1. **Funktionsapproximation** — ein Update „leckt" auf andere Zustände (kann falsche
   verschieben).
2. **Bootstrapping** — das Ziel enthält die eigene (fehlerhafte) Schätzung → Fehler können sich
   selbst verstärken.
3. **off-policy** — man lernt über Daten einer anderen Verteilung als die Zielpolitik.

Alle drei zusammen können **Divergenz** verursachen (Werte laufen nach $\pm\infty$). Zwei
weitere Verletzungen der üblichen Supervised-Learning-Annahmen kommen hinzu: die Daten sind
**stark korreliert** (aufeinanderfolgende Transitionen ähneln sich) und die **Zielverteilung
bewegt sich** (die Policy ändert sich beim Lernen → *non-stationary target*). Die Deep-RL-
Algorithmen sind im Kern **Tricks, um trotzdem stabil zu bleiben.**

---

## 2 · Wertbasiertes Deep RL: DQN

### 2.1 Deep Q-Network (DQN)

**DQN** (Mnih et al. 2013/2015, das „Atari-Paper") ist Q-Learning mit einem neuronalen Netz
$Q(s,a;\theta)$ (meist: Eingang $s$, ein Ausgang je Aktion). Man minimiert den erwarteten
quadratischen **TD-Fehler**:
$$L(\theta)=\mathbb E_{(s,a,r,s')\sim \mathcal D}\Big[\big(\underbrace{r+\gamma\max_{a'}Q(s',a';\theta^-)}_{\text{Ziel }y} - Q(s,a;\theta)\big)^2\Big].$$

Zwei Zutaten machen daraus ein *stabil trainierbares* Verfahren:

**(a) Experience Replay.** Speichere Transitionen $(s,a,r,s')$ in einem **Puffer** $\mathcal D$
(z. B. letzte $10^6$) und trainiere auf **zufälligen Minibatches** daraus. Das (i) **entkorreliert**
die Daten (bricht die zeitliche Abhängigkeit) und (ii) nutzt jede Erfahrung **mehrfach**
(Dateneffizienz). — Adressiert Triaden-Komponente „korrelierte Daten".

**(b) Target Network.** Das Ziel $y$ verwendet ein **eingefrorenes** Netz $\theta^-$, das nur
alle $C$ Schritte (oder per Polyak-Mittel $\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-$) auf
$\theta$ nachgezogen wird. Ohne das würde man „auf ein sich bewegendes Ziel schießen" — das Ziel
$y$ hinge sofort von jedem Gewichtsupdate ab und könnte oszillieren/divergieren. Das eingefrorene
Ziel macht das Lernen wieder einem stabilen **Regressionsproblem** ähnlich. — Adressiert
„non-stationary target".

**Trainingsschleife (Pseudocode):**
```
initialisiere Q(θ), Target Q(θ⁻)=θ, leeren Replay-Puffer D
für jede Episode:
    s = env.reset()
    wiederhole:
        a = ε-greedy(Q(s,·;θ))                 # Exploration wie in Modul 13
        s', r, done = env.step(a);  D.push(s,a,r,s',done)
        Minibatch B ~ D
        y = r + γ·(1-done)·max_a' Q(s',a';θ⁻)  # Ziel mit Target-Netz
        θ ← θ - lr·∇θ  mean_B (Q(s,a;θ) - y)²   # ein SGD/Adam-Schritt
        alle C Schritte:  θ⁻ ← θ                # Target-Netz nachziehen
        s = s'
    reduziere ε
```
Projekt 01 baut genau das auf einem selbstgebauten CartPole.

### 2.2 DQN-Erweiterungen (kurz, aber vollständig)

- **Double DQN** — der $\max$-Operator im Ziel **überschätzt** systematisch (max über
  verrauschte Schätzer ⇒ positiver Bias). Lösung: **entkopple Auswahl und Bewertung** —
  wähle die Aktion mit dem *Online*-Netz, bewerte sie mit dem *Target*-Netz:
  $$y = r + \gamma\,Q\big(s',\,\arg\max_{a'}Q(s',a';\theta);\,\theta^-\big).$$
- **Dueling DQN** — zerlege $Q(s,a)=V(s)+A(s,a)$ in **Zustandswert** und **Vorteil**
  (mit $A$ um seinen Mittelwert zentriert). Nützlich, wenn der Zustandswert dominiert und die
  Aktionswahl kaum zählt.
- **Prioritized Experience Replay** — ziehe Transitionen mit großem TD-Fehler **häufiger**
  (dort ist am meisten zu lernen), korrigiert per Importance-Sampling-Gewichten.
- **Rainbow** kombiniert diese und weitere (n-step, verteilungsbasiertes RL, Noisy Nets).

**Grenze von DQN:** braucht ein diskretes $\arg\max_a$ → **keine kontinuierlichen** Aktionen
(dafür Abschnitt 4). Und: rein wertbasiert, lernt eine deterministische greedy-Policy.

---

## 3 · Policy-basiertes Deep RL

### 3.1 Warum die Policy direkt lernen?

Wertbasierte Methoden lernen $Q$ und leiten die Policy *indirekt* (greedy) ab. **Policy-
Gradient**-Methoden parametrisieren die Policy **direkt**, $\pi_\theta(a\mid s)$, und optimieren
sie per Gradientenaufstieg. Vorteile: (i) **kontinuierliche** Aktionsräume natürlich (Gaußsche
Policy), (ii) **stochastische** Optimalpolicies möglich (wichtig bei partieller
Beobachtbarkeit/Spielen), (iii) glatte Verbesserung statt sprunghaftem $\arg\max$. Nachteil:
höhere **Varianz**, oft weniger dateneffizient.

### 3.2 Das Policy-Gradient-Theorem & REINFORCE

Ziel ist die erwartete Rendite $J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G_0]$. Das
**Policy-Gradient-Theorem** liefert (bemerkenswert: **ohne** Ableitung der Umgebungsdynamik):
$$\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\!\Big[\sum_{t} \nabla_\theta\log\pi_\theta(A_t\mid S_t)\,\Psi_t\Big],$$
wobei $\Psi_t$ ein **Kredit-Signal** ist. Verschiedene Wahlen von $\Psi_t$ ergeben verschiedene
Algorithmen:

| $\Psi_t$ | Verfahren |
|---|---|
| $G_t$ (voller Return) | **REINFORCE** (Monte Carlo) |
| $G_t - b(S_t)$ (Baseline) | REINFORCE **mit Baseline** (Varianzreduktion) |
| $Q^\pi(S_t,A_t)$ | Actor-Critic (Q-Form) |
| $A^\pi(S_t,A_t)=Q^\pi-V^\pi$ | **Advantage** Actor-Critic (A2C) |
| $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$ | TD-Actor-Critic |

**REINFORCE**-Update (nach ganzer Episode): $\theta\leftarrow\theta+\alpha\sum_t \nabla_\theta\log\pi_\theta(A_t|S_t)\,G_t$.
Intuition: **erhöhe** die Wahrscheinlichkeit von Aktionen, die zu hohem Return führten. Die
$\log$-Ableitung heißt **score function** (der „REINFORCE-Trick").

**Baseline.** Man darf von $\Psi_t$ eine **beliebige zustandsabhängige** Funktion $b(S_t)$
abziehen, ohne den Erwartungswert (und damit die Unverzerrtheit) zu ändert — denn
$\mathbb E_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a|s)]=0$. Eine gute Baseline (typisch
$b=\hat V(s)$) senkt die **Varianz** drastisch. $G_t-\hat V(S_t)$ schätzt den **Advantage**:
„war diese Aktion besser als der Durchschnitt in diesem Zustand?".

### 3.3 Actor-Critic & GAE

**Actor-Critic** kombiniert beide Welten: ein **Actor** $\pi_\theta$ (wählt Aktionen) und ein
**Critic** $\hat V_\phi$ (bewertet Zustände, bootstrappt à la TD). Der Critic liefert die
Baseline/den Advantage, der Actor macht den Policy-Gradient-Schritt. **A2C** ist die synchrone,
**A3C** die asynchrone (parallele Worker) Variante.

Der **Advantage** kann über verschieden viele Schritte geschätzt werden — ein Bias/Varianz-
Trade-off wie bei n-step in Modul 13. **Generalized Advantage Estimation (GAE)** mittelt sie
geometrisch mit einem Parameter $\lambda$:
$$\hat A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l},\qquad \delta_t=R_{t+1}+\gamma \hat V(S_{t+1})-\hat V(S_t).$$
$\lambda=0$ → reines TD (niedrige Varianz, mehr Bias), $\lambda=1$ → Monte-Carlo-Advantage.

### 3.4 PPO — Proximal Policy Optimization

Naives Policy-Gradient ist **empfindlich gegenüber der Schrittweite**: ein zu großer Update
kann die Policy „zerstören" (sie besucht dann nur noch schlechte Zustände, aus denen sie sich
kaum erholt). **TRPO** löste das mit einer harten KL-Nebenbedingung; **PPO** (Schulman et al.
2017) ist die einfachere, heute meistgenutzte Variante. Mit dem
**Wahrscheinlichkeitsverhältnis** $r_t(\theta)=\dfrac{\pi_\theta(A_t|S_t)}{\pi_{\theta_{\text{old}}}(A_t|S_t)}$
maximiert PPO das **geklippte** Ziel
$$L^{\text{CLIP}}(\theta)=\mathbb E_t\Big[\min\big(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\big)\Big].$$
Das **Clipping** entfernt den Anreiz, $r_t$ weit über $1\pm\epsilon$ zu treiben — die Policy
bleibt in der **Nähe** (*proximal*) der alten, was den Update stabilisiert und mehrere
Epochen pro Datensatz erlaubt (dateneffizienter als Vanilla-PG). PPO ist der De-facto-Standard
für viele kontinuierliche und diskrete Aufgaben (Robotik, RLHF für LLMs).

---

## 4 · Kontinuierliche Steuerung & Optimal Control

### 4.1 Deterministischer Policy-Gradient: DDPG, TD3

Bei **kontinuierlichen** Aktionen ($a\in\mathbb R^m$) ist $\max_a Q(s,a)$ nicht mehr trivial.
**DDPG** (Deep Deterministic Policy Gradient) lernt eine **deterministische** Actor-Policy
$\mu_\theta(s)$ und einen Critic $Q_\phi(s,a)$; der Actor wird in Richtung höheren Q-Werts
bewegt: $\nabla_\theta J\approx\mathbb E[\nabla_a Q_\phi(s,a)|_{a=\mu_\theta(s)}\nabla_\theta\mu_\theta(s)]$
(Kettenregel — „bewege die Aktion bergauf im Q-Gebirge"). Es ist im Kern **DQN für
kontinuierliche Aktionen** (mit Replay & Target-Netzen). Exploration per additivem Rauschen.

**TD3** (Twin Delayed DDPG) fixt DDPGs Überschätzung mit drei Tricks: **zwei** Critics (nimm das
Minimum → weniger Überschätzung), **verzögerte** Actor-Updates, und **Target-Policy-Smoothing**
(Rauschen im Ziel).

### 4.2 Maximum-Entropy-RL: SAC

**Soft Actor-Critic (SAC)** maximiert Rendite **plus Entropie** der Policy:
$$J(\pi)=\sum_t\mathbb E\big[R_{t+1}+\alpha\,\mathcal H(\pi(\cdot\mid S_t))\big].$$
Der **Entropie-Bonus** $\alpha\mathcal H$ belohnt „so zufällig wie möglich, solange die Aufgabe
gelöst wird" → bessere Exploration, robustere Policies, stabileres Training. SAC ist off-policy,
sample-effizient und einer der stärksten Algorithmen für kontinuierliche Steuerung.

### 4.3 Die Brücke zur klassischen Optimalsteuerung

Der Modulname ist **„for Optimal Control"** — RL ist die **daten-/modellfreie** Schwester der
klassischen **optimalen Regelung**. Die zentralen Verbindungen:

- **Bellman ↔ HJB.** Die Bellman-Optimalitätsgleichung ist die **zeitdiskrete** Form der
  **Hamilton-Jacobi-Bellman (HJB)**-Gleichung der kontinuierlichen Optimalsteuerung. Beide sagen:
  „optimaler Wert jetzt = sofortiger Nutzen + optimaler Wert danach". RL löst sie durch **Lernen
  aus Stichproben**, wenn das Modell unbekannt ist.
- **LQR — der exakt lösbare Fall.** Für ein **lineares** System $x_{t+1}=Ax_t+Bu_t$ mit
  **quadratischen** Kosten $\sum_t (x_t^\top Q x_t + u_t^\top R u_t)$ ist die optimale Steuerung
  eine **lineare Rückführung** $u_t=-Kx_t$, wobei $K$ aus der **algebraischen Riccati-Gleichung**
  folgt. Das ist die „Value Iteration mit geschlossener Lösung" — und dient (in Projekt 03) als
  **exakte Referenz**, an der sich ein gelernter RL-Regler messen muss (genau wie Value Iteration
  in Modul 13 die Referenz war).
- **Pontryagins Maximumprinzip** — der zweite große Zugang der Optimalsteuerung (notwendige
  Bedingungen über den *adjungierten Zustand/das Kostenfunktional*), komplementär zum
  HJB/Dynamic-Programming-Zugang.
- **MPC (Model Predictive Control)** — wenn ein (gelerntes oder bekanntes) Modell vorliegt:
  optimiere in jedem Schritt über einen **endlichen Horizont**, führe nur den ersten Schritt aus,
  wiederhole. Enge Verwandtschaft zu **modellbasiertem RL**.

### 4.4 Modellbasiertes RL (Ausblick)

Statt nur Werte/Policy zu lernen, kann man ein **Modell** der Dynamik $\hat p(s'|s,a)$ lernen und
darin *planen* (Dyna, PILCO, world models, MuZero). Vorteil: **Dateneffizienz** (Planung im
Modell ist billig). Nachteil: Modellfehler pflanzen sich fort. Hybridverfahren (Dyna:
kombiniere echte + modell-generierte Erfahrung) sind ein aktives Feld.

---

## 5 · Praxis-Stolpersteine (Deep RL ist berüchtigt fragil)

- **Reproduzierbarkeit** — Ergebnisse schwanken stark über Random-Seeds; immer über mehrere
  Seeds mitteln.
- **Reward Shaping** — schlecht gewählte Belohnungen führen zu *reward hacking* (Agent
  optimiert das Falsche).
- **Hyperparameter** — Lernraten, Netzgröße, Replay-Größe, Target-Update-Frequenz, $\gamma$ sind
  sensibel; kleine Änderungen kippen das Training.
- **Exploration** — in Umgebungen mit spärlicher Belohnung reicht ε-greedy/Gauß-Rauschen oft
  nicht (→ intrinsische Motivation, Curiosity).
- **Sim-to-Real** — in Simulation gelernte Policies übertragen sich schlecht auf echte Hardware
  (*reality gap*); Gegenmittel: Domain Randomization.
- **Debugging** — erst auf einer *winzigen*, lösbaren Aufgabe verifizieren (genau unser Ansatz),
  Lernkurven & Q-Werte beobachten (divergieren sie?).

---

## 6 · Zusammenfassung / Cheat-Sheet

**Landkarte.**
```
                         Deep RL
        ┌───────────────────┼─────────────────────┐
   wertbasiert         policy-basiert        kontinuierl. Control
   DQN                 REINFORCE             DDPG  (det. PG)
   +Double/Dueling     +Baseline             TD3   (2 Critics)
   +Prioritized Replay Actor-Critic (A2C)    SAC   (max-entropy)
   (diskrete Aktionen) PPO (clipped)         └── Optimal Control: LQR/HJB/MPC
```

**DQN-Ziel:** $y=r+\gamma\max_{a'}Q(s',a';\theta^-)$ · Loss $=(y-Q(s,a;\theta))^2$ · **Replay**
+ **Target-Netz** = Zähmung der deadly triad.

**Double-DQN-Ziel:** $y=r+\gamma Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$.

**Policy-Gradient:** $\nabla_\theta J=\mathbb E[\sum_t\nabla_\theta\log\pi_\theta(A_t|S_t)\,\Psi_t]$,
$\Psi_t\in\{G_t,\ G_t-b(s),\ A^\pi(s,a),\ \delta_t\}$.

**Advantage:** $A(s,a)=Q(s,a)-V(s)$ · **GAE**: $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$.

**PPO:** maximiere $\mathbb E[\min(r_t\hat A_t,\ \text{clip}(r_t,1{-}\epsilon,1{+}\epsilon)\hat A_t)]$,
$r_t=\pi_\theta/\pi_{\theta_{old}}$.

**Optimal Control:** Bellman ↔ **HJB**; linear+quadratisch ⇒ **LQR** $u=-Kx$ (Riccati) = exakte
Referenz; **MPC** = planen im Modell über Horizont.

---

## 7 · Selbsttest

<details>
<summary><b>1.</b> Warum ersetzt Deep RL die Q-Tabelle durch ein Netz, und was gewinnt/riskiert man?</summary>

Weil Tabellen bei großen/kontinuierlichen Zustandsräumen weder speicherbar noch besuchbar sind.
Ein Netz **generalisiert** über ähnliche Zustände (ein Update hilft vielen). Risiko: es entsteht
die **deadly triad** (Funktionsapproximation + Bootstrapping + off-policy) → mögliche Divergenz,
plus korrelierte Daten und bewegtes Ziel.
</details>

<details>
<summary><b>2.</b> Wofür dienen Experience Replay und Target Network in DQN — je genau ein Problem?</summary>

**Experience Replay**: bricht die **zeitliche Korrelation** der Daten (und nutzt Erfahrung
mehrfach → Dateneffizienz). **Target Network**: friert das Bootstrapping-**Ziel** ein →
verhindert das „Schießen auf ein bewegtes Ziel" (non-stationary target) und macht das Lernen
einem stabilen Regressionsproblem ähnlich.
</details>

<details>
<summary><b>3.</b> Was ist der Überschätzungs-Bias in DQN und wie behebt ihn Double DQN?</summary>

$\max_{a'}Q(s',a')$ nimmt das Maximum über *verrauschte* Schätzer → systematisch **zu hoch**
(positiver Bias). **Double DQN** entkoppelt **Auswahl** (Online-Netz $\theta$) und **Bewertung**
(Target-Netz $\theta^-$): $y=r+\gamma Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$.
</details>

<details>
<summary><b>4.</b> Formuliere das Policy-Gradient-Theorem und den Nutzen einer Baseline.</summary>

$\nabla_\theta J=\mathbb E[\sum_t\nabla_\theta\log\pi_\theta(A_t|S_t)\,\Psi_t]$. Eine
zustandsabhängige **Baseline** $b(s)$ (typisch $\hat V(s)$) darf von $\Psi_t$ abgezogen werden,
**ohne** den Erwartungswert zu ändern (unverzerrt), reduziert aber die **Varianz** stark;
$G_t-\hat V(s)$ schätzt den **Advantage**.
</details>

<details>
<summary><b>5.</b> Was unterscheidet Actor-Critic von REINFORCE?</summary>

REINFORCE nutzt den **vollen Monte-Carlo-Return** $G_t$ (unverzerrt, hohe Varianz, braucht
Episodenende). **Actor-Critic** hat zusätzlich einen **Critic** $\hat V_\phi$, der **bootstrappt**
(TD) und als Baseline/Advantage dient → niedrigere Varianz, Online-Updates. Actor = Policy,
Critic = Wertschätzer.
</details>

<details>
<summary><b>6.</b> Warum „proximal"? Was bewirkt das Clipping in PPO?</summary>

Zu große Policy-Updates können die Policy zerstören. PPO hält die neue Policy **nahe**
(*proximal*) an der alten, indem es das Verhältnis $r_t=\pi_\theta/\pi_{\theta_{old}}$ auf
$[1-\epsilon,1+\epsilon]$ **klippt** — das entfernt den Anreiz, $r_t$ weit zu treiben, und erlaubt
mehrere Update-Epochen pro Datenbatch (stabil + dateneffizient).
</details>

<details>
<summary><b>7.</b> Warum kann DQN keine kontinuierlichen Aktionen, und was tut DDPG dagegen?</summary>

DQN braucht $\arg\max_a Q(s,a)$ — bei kontinuierlichem $a$ nicht in geschlossener Form lösbar.
**DDPG** lernt einen **deterministischen Actor** $\mu_\theta(s)$ und bewegt ihn per Kettenregel
**bergauf** im Critic $Q_\phi$: $\nabla_\theta J\approx\mathbb E[\nabla_aQ_\phi\,\nabla_\theta\mu_\theta]$.
Es ist „DQN für kontinuierliche Aktionen".
</details>

<details>
<summary><b>8.</b> Was ist der Zusammenhang zwischen Bellman-Gleichung und HJB, und was ist LQR?</summary>

Die Bellman-Optimalitätsgleichung ist die **zeitdiskrete** Form der **Hamilton-Jacobi-Bellman**-
Gleichung der kontinuierlichen Optimalsteuerung. **LQR** ist der Spezialfall *linearer Dynamik +
quadratischer Kosten*: die optimale Steuerung ist die lineare Rückführung $u=-Kx$ mit $K$ aus der
**Riccati-Gleichung** — eine exakt lösbare Referenz.
</details>

<details>
<summary><b>9.</b> Was macht das Entropie-Ziel in SAC?</summary>

SAC maximiert Rendite **+** Policy-**Entropie** ($\alpha\mathcal H(\pi)$). Der Bonus belohnt
„so zufällig wie möglich, solange die Aufgabe gelöst wird" → bessere **Exploration**, robustere
Policies, stabileres Training.
</details>

<details>
<summary><b>10.</b> Nenne drei Gründe, warum Deep RL notorisch instabil/schwer reproduzierbar ist.</summary>

Beliebige drei: deadly triad (Divergenz), hohe **Seed-Varianz**, sensible **Hyperparameter**,
**korrelierte/nicht-stationäre** Daten, spärliche Belohnung/Exploration, **reward hacking**,
Sim-to-Real-Gap. Deshalb: erst auf winzigen Aufgaben verifizieren, über Seeds mitteln.
</details>

---

## 8 · Literatur & Quellen

**Bücher & Kurse (kostenlos):**
- 📗 **Sutton & Barto — *Reinforcement Learning: An Introduction* (2018)**, Kap. 9–13
  (Funktionsapproximation, Policy Gradient). Frei als PDF. *Fundament.*
- 🌐 **OpenAI Spinning Up in Deep RL** (spinningup.openai.com) — **die** praktische Einführung:
  saubere Herleitungen (VPG→TRPO→PPO→DDPG→TD3→SAC) **plus** referenzierbarer Code. Frei.
  *Einsteiger→vertiefend, sehr empfohlen.*
- 🎥 **UC Berkeley CS285 *Deep Reinforcement Learning* (Sergey Levine)** — Videos + Folien frei.
  *Vertiefend, umfassend.*
- 🎥 **DeepMind × UCL RL Lecture Series** — Nachfolger von Silvers Kurs. *Einsteiger→vertiefend.*

**Schlüssel-Paper (vertiefend):**
- Mnih et al. (2015), *Human-level control through deep RL* (**DQN/Nature**).
- van Hasselt et al. (2016), *Deep RL with Double Q-learning*.
- Wang et al. (2016), *Dueling Network Architectures*.
- Schulman et al. (2015), *High-Dimensional Continuous Control Using GAE*.
- Schulman et al. (2017), *Proximal Policy Optimization* (**PPO**).
- Lillicrap et al. (2016), *Continuous control with deep RL* (**DDPG**); Fujimoto et al. (2018),
  **TD3**; Haarnoja et al. (2018), **SAC**.

**Optimal Control (Brücke):**
- 📘 **Bertsekas — *Dynamic Programming and Optimal Control*** / *Reinforcement Learning and
  Optimal Control* (2019) — verbindet beide Welten formal. *Vertiefend.*
- 🌐 **Steven Brunton — *Control Bootcamp* (YouTube)** — LQR, Riccati, HJB anschaulich. *Einsteiger.*

**Praxis/Tooling:**
- 🌐 **Gymnasium** (gymnasium.farama.org) — Standard-Umgebungs-API (CartPole, Pendulum, MuJoCo).
  Wir bauen im Modul *ohne* Gym (didaktisch), aber die API solltest du kennen.
- 🌐 **Stable-Baselines3** — gepflegte, getestete Implementierungen (DQN/PPO/SAC/TD3) für die
  Praxis (nicht zum Lernen der Interna).

---

## Nächstes Modul

Damit ist der RL-Block (Module 13–14) abgeschlossen. Es folgt **Modul 15 — Machine Learning for
Networks 1**. Das in diesem Modul gebaute Fundament (wert- vs. policy-basiert, Actor-Critic,
Optimalsteuerung) ist Grundlage für RL-Anwendungen in Robotik (Module 21/22), Advanced
Automation (23) und überall dort, wo **sequenzielle Entscheidungen unter Unsicherheit** getroffen
werden.
