# Module 24 — Self-aware Computing

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** Module 23 automated a *plant* — a machine outside the computer. This module turns the same idea **inward**: a software system that observes **itself**, builds a **model of itself**, predicts what will happen if the load or the configuration changes, and **reconfigures itself** to keep meeting its goals — without a human in the loop. A web service that scales its own capacity when traffic surges, notices that a replica has gone sick and replaces it, and re-learns its own performance model when the code changes: that is a **self-aware computing system**. The mathematics is **queueing theory** (the self-model), **feedback control** (the adaptation loop), and **online learning** (keeping the model true).
>
> **Prior knowledge**: probability (exponential distributions, expectation), basic statistics, feedback control basics. From this repo the following build in: **module 23** (the MAPE-K loop is a control loop, and model-based scaling *is* the MPC idea — predict, optimise, act, repeat), **module 14** (LQR/optimal control), **module 15/16** (anomaly detection and the base-rate fallacy — indispensable for self-healing), **module 13** (RL, the learning route to adaptation policies), **module 02/03** (statistics/estimation). Module 23 is the most useful preceding module.

> **Note on the scope.** No official module description is available, so I scoped the content myself — but here the choice is unusually well anchored: **Samuel Kounev**, who holds the chair for Software Engineering at the University of Würzburg, is one of the field's defining figures and the lead editor of the standard reference *Self-Aware Computing Systems* (Springer 2017). I follow that canon: Kounev's three properties (**self-reflective, self-predictive, self-adaptive**), IBM's **MAPE-K** loop, **models at run-time**, and the quantitative core of **queueing theory** and **elasticity**. The module is deliberately built as one arc — **model → control loop → learning system** — because that is exactly what distinguishes a *self-aware* system from a merely *adaptive* one: it does not just react, it **reasons with a model of itself**. Everything is from scratch in `numpy`/`scipy` (a discrete-event simulator, queueing formulas, controllers) — no cloud account, no Kubernetes, CPU seconds.

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

- define a **self-aware computing system** (Kounev): it **learns models** of itself and its environment *on an ongoing basis*, **reasons** with those models, and **acts** according to higher-level goals — and distinguish the three properties **self-reflective / self-predictive / self-adaptive**.
- describe the **MAPE-K** loop (Monitor–Analyze–Plan–Execute over a shared **Knowledge** base) and the **self-\*** properties (self-configuration, self-healing, self-optimisation, self-protection).
- apply the **operational laws** — **Little's Law** $N=X\cdot R$, the **Utilisation Law** $U=X\cdot S$, the **Forced Flow Law** and the **Service Demand Law** — to reason about a running system from measurements alone.
- derive and use the **M/M/1** results ($\rho$, $N$, $R$, the response-time distribution and its quantiles), explain **why response time explodes as $\rho\to1$**, and extend to **M/M/c**; find the **bottleneck** and state the **asymptotic bounds**.
- design and compare **auto-scaling** (elasticity) policies — **threshold/reactive**, **control-theoretic**, and **model-based predictive** — and evaluate them honestly with **SLO violations, cost, and oscillation (flapping)**; explain **hysteresis** and **cooldown**.
- explain **models at run-time** and why a self-aware system must **re-estimate** its model online (drift), and how **anomaly detection** feeds **self-healing** — including why the **base-rate fallacy** (module 15) makes naive alerting useless.

---

## Basics

### 1. From automation to self-awareness

Classical automation (module 23) controls an external plant with a model an *engineer* wrote. A modern software system, though, is its own plant: it runs on shared, changing infrastructure, its load varies by orders of magnitude between night and peak, its own code changes weekly, and no engineer is watching at 3 a.m. **Self-aware computing** asks the system to take over that engineer's job.

Kounev's definition (2017) is the reference and each clause matters:

> A **self-aware computing system** *learns models* capturing knowledge about itself and its environment **on an ongoing basis**, and *reasons* using these models, enabling it to *act* based on its knowledge and reasoning, in accordance with **higher-level goals**.

Three properties follow, and they are the ladder this module climbs:

1. **Self-reflective** — it *knows itself*: what components exist, what the current load, utilisation and response time are. (Monitoring, model structure.)
2. **Self-predictive** — it can *predict the consequences* of a change **before** making it: "if load doubles, will I still meet the SLO? how many replicas would I need?" This is what a **model** buys you, and it is what separates self-awareness from blind reaction.
3. **Self-adaptive** — it *acts*: reconfigures, scales, restarts, reroutes, to keep meeting its goals.

A thermostat is adaptive but not self-aware (no model of itself, no prediction). A system that computes "at 800 requests/s with 3 replicas my predicted 95th-percentile latency is 240 ms, which breaks my 200 ms SLO, so I need a 4th replica **now**" is self-aware.

### 2. The MAPE-K loop and the self-\* properties

The organising architecture comes from IBM's **autonomic computing** manifesto (2003): an autonomic manager wrapped around a managed element, running the **MAPE-K** loop:

```
            +--------------------------------------------------+
            |                 Autonomic manager                |
            |   Monitor -> Analyze -> Plan -> Execute          |
            |        \        |         |        /             |
            |         +---- Knowledge (models, history) ---+   |
            +--------------------------------------------------+
                 ^                                    |
             sensors                               effectors
                 |                                    v
            +--------------------------------------------------+
            |        Managed element (the software system)      |
            +--------------------------------------------------+
```

- **Monitor** — collect metrics through sensors (arrival rate, utilisation, response time, error rate).
- **Analyze** — decide whether the goals are (about to be) violated; this is where the **model** and the **prediction** live.
- **Plan** — choose the adaptation (scale out by 2, restart replica 7, shed load).
- **Execute** — apply it through effectors.
- **Knowledge** — the shared, *persistent* part: the system model, the SLOs, the measurement history. This K is precisely what makes the loop *self-aware* rather than a reflex.

The classic goals are the **self-\*** (or **self-CHOP**) properties: **self-configuration** (install/wire itself), **self-healing** (detect, diagnose and repair failures), **self-optimisation** (tune itself for performance/cost), **self-protection** (defend against attacks and cascading failures).

> **The link to module 23.** MAPE-K *is* a feedback control loop with a model in it — Monitor = measurement, Analyze+Plan = the optimiser, Execute = actuation, Knowledge = the model. When the plan is computed by predicting a horizon with a model and applying only the first action, you have literally rebuilt **MPC** (module 23 P03), now with the *computer itself* as the plant. That is the cleanest way to understand model-based auto-scaling.

### 3. Models at run-time

A design-time model (a UML diagram, a capacity spreadsheet) is dead the moment the system changes. A **model at run-time** (*models@run.time*) is a live, causally connected abstraction: it is **kept in sync with the running system** by monitoring, and **changing it changes the system**. Kounev's Descartes Modelling Language is one research instance; a simple, honest instance — and the one we build — is a **queueing model whose parameters are continuously re-estimated from measurements**.

That is the crux: a model is only useful while it is *true*. Code deployments, noisy neighbours, and cache warm-up all shift the real service rate. A self-aware system therefore **re-estimates its own parameters online** and, when the model stops matching reality, notices that too. The **final project** builds exactly this, including the drift.

---

## Intermediate

### 4. The operational laws: reasoning about a system from measurements

Before any stochastic assumption, a set of **exact, assumption-free** identities holds for any system observed over a finite interval. They are the working vocabulary of performance self-awareness. Over an observation window, define arrivals/completions $C$, busy time $B$, window length $T$:

- **Throughput** $X = C/T$, **utilisation** $U = B/T$, **mean service time** $S = B/C$.
- **Utilisation Law**: $\;\boxed{U = X\cdot S}\;$ — utilisation is throughput times service time. (Directly: $B/T = (C/T)(B/C)$.)
- **Little's Law**: $\;\boxed{N = X\cdot R}\;$ — the mean number in the system equals throughput times mean response time. It holds for *any* stable system, no distributional assumption whatsoever. It is the single most useful formula in performance engineering: measure any two, get the third.
- **Forced Flow Law**: $X_k = V_k\,X$ — if each request visits resource $k$ on average $V_k$ times, that resource's throughput is a fixed multiple of the system's.
- **Service Demand Law**: $\;\boxed{D_k = V_k S_k = U_k / X}\;$ — the total demand a request places on resource $k$ is its utilisation divided by system throughput. This lets you extract per-resource demands **from black-box measurements**, which is how a system estimates its own model.

**The bottleneck** is the resource with the largest demand $D_{\max}=\max_k D_k$. It caps everything: $X \le 1/D_{\max}$. With $D=\sum_k D_k$ (total demand) and think time $Z$, the **asymptotic bounds** are

$$X(N) \le \min\Big(\frac{1}{D_{\max}},\;\frac{N}{D+Z}\Big),\qquad R(N)\ \ge\ \max\big(D,\;N D_{\max}-Z\big).$$

Two regimes: at low load the system is *latency-bound* ($R\approx D$), at high load *bottleneck-bound* (throughput saturates at $1/D_{\max}$ and response time grows linearly with $N$). Knowing where you are is self-awareness at its most practical.

### 5. The M/M/1 queue: why response time explodes

Add the standard stochastic assumptions — Poisson arrivals at rate $\lambda$, exponential service at rate $\mu$, one server, infinite FIFO queue — and you get the **M/M/1** queue, the single most instructive model in computing. Define the **utilisation** (traffic intensity)

$$\rho = \frac{\lambda}{\mu}\qquad(\text{stable iff } \rho<1).$$

The queue is a birth–death chain with birth rate $\lambda$ and death rate $\mu$; balance gives the stationary distribution $P(N=n) = (1-\rho)\rho^{n}$ — geometric. From it:

$$\boxed{\;N = \frac{\rho}{1-\rho},\qquad R = \frac{1}{\mu-\lambda} = \frac{S}{1-\rho},\qquad W = R - S = \frac{\rho S}{1-\rho}\;}$$

(with $S=1/\mu$; $N=XR$ is Little's Law again, as it must be). Moreover the **response time is exponentially distributed** with rate $\mu-\lambda$, so its quantiles are available in closed form:

$$R_p = \frac{\ln\!\big(1/(1-p)\big)}{\mu-\lambda}\quad\Longrightarrow\quad R_{95} = \frac{\ln 20}{\mu-\lambda}\approx 3\,R,\qquad R_{99}\approx 4.6\,R.$$

This is why SLOs are written on **percentiles**, not means: the tail is *three to five times* the mean.

**The explosion.** The factor $1/(1-\rho)$ is the whole story of capacity planning:

| $\rho$ | 0.5 | 0.8 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|
| $R/S$ | 2 | 5 | 10 | 20 | 100 |

Going from 50 % to 90 % utilisation costs **5× the latency**; 90 % to 99 % costs another **10×**. Response time is **non-linear and unbounded** in utilisation — which is precisely why "the CPU is only at 85 %, we're fine" is a dangerous sentence, and why an auto-scaler must act on *predicted latency*, not on utilisation alone. The **basic project** verifies every one of these formulas against a simulator.

**M/M/c.** With $c$ parallel servers (replicas!) and $\rho=\lambda/(c\mu)$, the probability that an arrival must queue is the **Erlang-C** formula

$$P_Q \;=\; \frac{\dfrac{(c\rho)^c}{c!\,(1-\rho)}}{\displaystyle\sum_{k=0}^{c-1}\frac{(c\rho)^k}{k!} + \frac{(c\rho)^c}{c!\,(1-\rho)}},\qquad W = \frac{P_Q}{c\mu-\lambda},\qquad R = W + \frac1\mu.$$

The practical consequence is **economies of scale**: at equal utilisation, many servers behind *one* queue give far lower waiting time than the same servers with separate queues — one reason load balancers and shared thread pools exist. M/M/c is the model an auto-scaler uses to answer "how many replicas do I need for this load?"

---

## Advanced topics

### 6. Elasticity and auto-scaling

**Elasticity** is the degree to which a system **autonomously adapts its capacity to the current demand**, as closely and as promptly as possible (Herbst, Kounev & Reussner). Note the three qualities hidden in that definition — *accuracy*, *timeliness*, and *stability* — and that they trade off. Three policy families:

- **Reactive / threshold-based** (what most cloud auto-scalers ship with): if $U > 0.7$ for $k$ intervals, add a replica; if $U<0.3$, remove one. Simple and robust, but it is **purely reactive**: it acts only *after* the SLO is already at risk, and during the provisioning delay (VM boot, container start, cache warm-up — seconds to minutes) the system stays overloaded.
- **Control-theoretic**: treat capacity as the control input and the SLO as the setpoint; use a **PI/PID** controller (module 21) or an LQR/MPC formulation (modules 14/23). Principled, tunable, with stability guarantees — but it needs a model or at least identified gains.
- **Model-based predictive**: **forecast** the load (seasonal patterns, module 16) and use the **queueing model** to compute the capacity that will meet the SLO *at that future load* — then provision **ahead of** the surge. This is MPC with the M/M/c model as the plant model, and it is the only family that can beat the provisioning delay.

**Two mechanisms every real scaler needs.** Both fight **oscillation (flapping)** — adding and removing capacity repeatedly, which costs money and, because each change has a warm-up cost, actively harms performance:
- **Hysteresis**: use *different* thresholds for scaling out and in (e.g. out at 0.7, in at 0.3). The gap is a dead zone in which nothing happens.
- **Cooldown** (damping): after any action, refuse to act again for a fixed period, so the system's response can settle before it is measured.

**Evaluating elasticity honestly.** Never report only "it scaled". The community's metrics (Herbst et al.) are:
- **Under-provisioning** accuracy/timeshare: how much, and for what fraction of time, capacity was *below* demand — this is where **SLO violations** live.
- **Over-provisioning** accuracy/timeshare: how much capacity was wasted — this is **cost**.
- **Instability / jitter**: the number of adaptations (or sign changes) — this is **flapping**.

A scaler that never violates the SLO by running 10× over-provisioned is not good, and neither is a cheap one that misses the SLO 20 % of the time. The **medium project** measures exactly this triple for all three policy families.

### 7. Self-healing and the anomaly-detection trap

**Self-healing** = detect a failure, diagnose it, repair it (restart, replace, reroute), ideally before users notice. Detection usually means **anomaly detection** on metric streams (latency, error rate, saturation) — and here the module must issue a warning that modules 15/16 already earned.

Failures are **rare**. Let a detector have a true-positive rate $\mathrm{TPR}$ and false-positive rate $\mathrm{FPR}$, and let failures occur with base rate $\pi$. The probability that an alert is real is

$$P(\text{failure}\mid\text{alert}) = \frac{\mathrm{TPR}\cdot\pi}{\mathrm{TPR}\cdot\pi + \mathrm{FPR}\cdot(1-\pi)}.$$

With an excellent detector ($\mathrm{TPR}=0.99$, $\mathrm{FPR}=0.01$) and a realistic base rate ($\pi=10^{-3}$), this is $\approx 9\%$ — **nine out of ten alerts are false**, and an automated healer wired to them would restart healthy replicas nine times out of ten, *causing* the outage it was meant to prevent. This is the **base-rate fallacy** (module 15), and in self-healing it is not academic: it is the difference between a system that stabilises itself and one that thrashes. Remedies: raise the evidence threshold (accept lower TPR for a much lower FPR), **require persistence** (anomalous for $k$ consecutive intervals), **corroborate** across independent signals, and make the repair **cheap and reversible** so that a false positive is survivable.

### 8. Learning and keeping the model true

The "on an ongoing basis" in Kounev's definition is the hard part. Three threads:

- **Online parameter estimation.** The model's parameters (service demands $D_k$, service rate $\mu$) are **estimated from measurements** — via the Service Demand Law ($D_k=U_k/X$) or by regression of response time against load — and re-estimated continuously, e.g. with an exponentially weighted moving average or a Kalman filter (module 21) so the estimate tracks change while damping noise.
- **Drift and model invalidation.** After a deployment the real $\mu$ may drop by 30 %. A self-aware system must **notice** that its predictions have become biased (monitor the *prediction error*, not just the metric) and re-learn — model **validity** is itself something to be aware of.
- **Learning the policy.** Rather than deriving the adaptation rule, learn it: **reinforcement learning** (modules 13/14) can learn a scaling policy from experience, with the well-known costs — sample inefficiency and unsafe exploration in production — which is why the practical state of the art is usually a **model-based** controller with learned *parameters*, not a learned policy. The **final project** takes exactly that route.

---

## Summary / cheat sheet

**Self-aware system (Kounev)**: learns models of itself/environment **continuously**, **reasons** with them, **acts** toward higher-level goals. Properties: **self-reflective** (knows itself) · **self-predictive** (predicts consequences) · **self-adaptive** (acts).

**MAPE-K**: Monitor → Analyze → Plan → Execute, over shared **Knowledge** (models, history, SLOs). Self-\* goals: self-**C**onfiguration, self-**H**ealing, self-**O**ptimisation, self-**P**rotection. MAPE-K with model-based planning ≡ **MPC** (module 23) applied to the computer itself.

**Operational laws** (exact, no assumptions): $U = X S$ · **Little: $N = X R$** · $X_k=V_k X$ · **Service demand: $D_k = U_k/X$**.
**Bottleneck** $D_{\max}$: $X\le\min(1/D_{\max},\,N/(D+Z))$, $R\ge\max(D,\,N D_{\max}-Z)$.

**M/M/1**: $\rho=\lambda/\mu$ (stable iff $<1$); $P(N=n)=(1-\rho)\rho^n$; $N=\rho/(1-\rho)$; $R=S/(1-\rho)=1/(\mu-\lambda)$; response time $\sim\mathrm{Exp}(\mu-\lambda)$ ⇒ $R_p=\ln(1/(1-p))/(\mu-\lambda)$, $R_{95}\approx3R$, $R_{99}\approx4.6R$.
**The $1/(1-\rho)$ explosion**: $\rho=0.5\to R/S=2$; $0.9\to10$; $0.99\to100$. Never reason on utilisation alone.
**M/M/c**: Erlang-C $P_Q$, $W=P_Q/(c\mu-\lambda)$ — pooling beats separate queues (economies of scale).

**Elasticity** = autonomously match capacity to demand, accurately and promptly. Policies: **reactive/threshold** (acts too late), **control-theoretic** (PI/PID, LQR), **model-based predictive** (forecast + queueing model ⇒ beats the provisioning delay). Always add **hysteresis** + **cooldown** against **flapping**.
**Evaluate with the triple**: under-provisioning (**SLO violations**) · over-provisioning (**cost**) · instability (**adaptations/flapping**).

**Self-healing**: detect → diagnose → repair. **Base-rate fallacy** (module 15): $P(\text{fail}\mid\text{alert})=\frac{\mathrm{TPR}\,\pi}{\mathrm{TPR}\,\pi+\mathrm{FPR}(1-\pi)}$ — TPR .99/FPR .01 at $\pi=10^{-3}$ gives **9 %**. Require persistence, corroboration, cheap reversible repairs.

**Models at run-time**: keep the model **causally connected** and **re-estimate online** ($D_k=U_k/X$, EWMA/Kalman); monitor the **prediction error** to detect drift and model invalidation.

---

## Self-test

<details>
<summary><b>1.</b> Give Kounev's definition of a self-aware computing system and name the three properties.</summary>

A self-aware computing system **learns models** capturing knowledge about itself and its environment **on an ongoing basis**, **reasons** using these models, and **acts** on that knowledge and reasoning in accordance with **higher-level goals**. The three properties are **self-reflective** (it knows its own structure and state), **self-predictive** (it can predict the consequences of a change *before* making it), and **self-adaptive** (it acts to keep meeting its goals). The middle one — prediction via a model — is what separates self-awareness from mere reactive adaptation.
</details>

<details>
<summary><b>2.</b> Describe the MAPE-K loop and say what the K contributes.</summary>

**Monitor** (collect metrics via sensors) → **Analyze** (are goals violated or about to be?) → **Plan** (choose an adaptation) → **Execute** (apply it via effectors), all operating over a shared **Knowledge** base. The **K** holds the system model, the SLOs and the measurement history: it is what lets Analyze *predict* and Plan *reason*, rather than merely reflex-react. Without K the loop is a thermostat; with a model in K, model-based planning makes it exactly the MPC pattern of module 23 applied to the software system itself.
</details>

<details>
<summary><b>3.</b> State Little's Law and say what makes it remarkable. Use it: 200 req/s, mean response time 50 ms — how many requests are in the system?</summary>

**$N = X\cdot R$**: the mean number of requests in the system equals throughput times mean response time. It is remarkable because it is **exact for any stable system** — no assumption about arrival or service distributions, scheduling discipline or number of servers. Measure any two quantities and you get the third.

With $X=200\,\mathrm{s^{-1}}$ and $R=0.05\,\mathrm s$: $N = 200\cdot0.05 = \mathbf{10}$ requests resident on average.
</details>

<details>
<summary><b>4.</b> What is the Service Demand Law and why is it central to self-awareness?</summary>

$D_k = V_k S_k = U_k / X$ — the total service demand a request places on resource $k$ equals that resource's utilisation divided by system throughput. It is central because it lets a system **estimate its own model parameters from black-box measurements**: utilisation and throughput are routinely monitored, so the system can compute its per-resource demands, find its bottleneck ($D_{\max}$) and parameterise a queueing model **at run-time**, without any instrumentation of internals.
</details>

<details>
<summary><b>5.</b> Derive/state the M/M/1 mean response time and explain the explosion as $\rho\to1$.</summary>

With $\rho=\lambda/\mu<1$, the stationary queue length is geometric, $P(N=n)=(1-\rho)\rho^n$, giving $N=\rho/(1-\rho)$; Little's Law ($N=\lambda R$) then yields $R = \dfrac{1}{\mu-\lambda} = \dfrac{S}{1-\rho}$.

The factor $1/(1-\rho)$ **explodes** as utilisation approaches 1: $\rho=0.5\Rightarrow R=2S$, $0.9\Rightarrow10S$, $0.99\Rightarrow100S$. Intuitively, at high utilisation the server is almost never idle when work arrives, so almost every request waits behind a queue that takes ever longer to drain. Practically: response time is **non-linear and unbounded** in utilisation, so a capacity decision must be made on predicted *latency*, not on a utilisation number.
</details>

<details>
<summary><b>6.</b> Why are SLOs written on percentiles? Quantify for M/M/1.</summary>

Because the **tail** is far worse than the mean. In M/M/1 the response time is **exponentially distributed** with rate $\mu-\lambda$, so $R_p = \ln\!\big(1/(1-p)\big)/(\mu-\lambda)$, i.e. $R_p = \ln(1/(1-p))\cdot R$. Hence $R_{95}=\ln 20\cdot R\approx 3R$ and $R_{99}=\ln100\cdot R\approx 4.6R$. A system meeting a 100 ms *mean* can easily be violating a 200 ms *95th-percentile* SLO — which is why real SLOs are stated as percentiles, and why an auto-scaler should target one.
</details>

<details>
<summary><b>7.</b> Compare reactive, control-theoretic and predictive auto-scaling. What does the provisioning delay imply?</summary>

**Reactive/threshold** scaling acts only *after* a metric crosses a bound — simple and robust, but always late. **Control-theoretic** scaling (PI/PID, LQR) treats the SLO as a setpoint and capacity as the input, giving principled tuning and stability, but needs identified gains/model. **Model-based predictive** scaling **forecasts** the load and uses a **queueing model** to compute the capacity that will satisfy the SLO at that future load, provisioning *ahead* of the surge (MPC applied to the system itself).

The **provisioning delay** (VM boot, container start, warm-up) is decisive: during it, a reactive scaler leaves the system overloaded, so its SLO violations are essentially unavoidable. Only a predictive policy — which starts capacity *before* the demand arrives — can hide that delay.
</details>

<details>
<summary><b>8.</b> What is flapping and what two mechanisms prevent it?</summary>

**Flapping** (oscillation/instability) is repeatedly adding and removing capacity around a threshold. It costs money, and because every change carries a warm-up/migration cost, it also *degrades* performance — the adaptation makes things worse. The two standard mechanisms: **hysteresis** — different thresholds for scaling out and in (e.g. out at $U>0.7$, in at $U<0.3$), creating a dead zone; and **cooldown/damping** — after any adaptation, refuse to adapt again for a fixed interval so the effect can settle before being measured.
</details>

<details>
<summary><b>9.</b> How should elasticity be evaluated, and why is one number not enough?</summary>

With a **triple**, because the qualities trade off: **under-provisioning** (how much and how long capacity was below demand — i.e. **SLO violations**), **over-provisioning** (wasted capacity — i.e. **cost**), and **instability/jitter** (number of adaptations — i.e. **flapping**). One number is not enough because any single metric is trivially gamed: a policy that provisions 10× always meets the SLO at absurd cost, and the cheapest policy violates it constantly. Only the triple shows the actual operating point.
</details>

<details>
<summary><b>10.</b> Why is naive anomaly-triggered self-healing dangerous? Compute an example.</summary>

Because failures are **rare**, so even an excellent detector produces mostly false alarms — the **base-rate fallacy**. With $P(\text{failure}\mid\text{alert}) = \frac{\mathrm{TPR}\,\pi}{\mathrm{TPR}\,\pi+\mathrm{FPR}(1-\pi)}$, a detector with $\mathrm{TPR}=0.99$, $\mathrm{FPR}=0.01$ at base rate $\pi=10^{-3}$ gives $\frac{0.99\cdot0.001}{0.99\cdot0.001+0.01\cdot0.999}\approx\mathbf{9\%}$: **91 % of alerts are false**. An automated healer wired directly to it would restart healthy replicas nine times out of ten and could cause the very outage it exists to prevent. Remedies: much lower FPR (accepting lower TPR), **persistence** over $k$ intervals, **corroboration** across independent signals, and **cheap, reversible** repair actions.
</details>

---

## Literature & sources

**Textbooks & the field's reference**
- **Kounev, Kephart, Milenkoski & Zhu (eds.), *Self-Aware Computing Systems*** (Springer 2017). **The** reference for this module — definitions, the three properties, models at run-time, architectures. Samuel Kounev holds the Software Engineering chair at the **University of Würzburg**. *In-depth, the canonical source.*
- **Lazowska, Zahorjan, Graham & Sevcik, *Quantitative System Performance*** — **free online**. The classic on operational laws, bottleneck analysis and queueing-network models (sections 4–5 here). *Free, beginner-friendly, still unmatched.*
- **Menascé, Almeida & Dowdy, *Performance by Design*** — capacity planning with queueing models, very practical. *Intermediate.*
- **Harchol-Balter, *Performance Modeling and Design of Computer Systems*** (Cambridge). Excellent, rigorous and readable on M/M/1, M/M/c and beyond. *In-depth, highly recommended.*
- **Hellerstein, Diao, Parekh & Tilbury, *Feedback Control of Computing Systems*** (Wiley). Control theory applied to software systems — the theoretical backing for section 6. *In-depth.*

**Key papers**
- **Kephart & Chess, "The Vision of Autonomic Computing"**, *IEEE Computer 2003*. The **MAPE-K** origin. *Free, beginner-friendly, a classic.*
- **Herbst, Kounev & Reussner, "Elasticity in Cloud Computing: What It Is, and What It Is Not"**, *ICAC 2013*. The definition and the **elasticity metrics** of section 6. *Free, essential.*
- **Kounev et al., "The Notion of Self-aware Computing"** (chapter 1 of the book above) — the definition used here. *In-depth.*
- **Lorido-Botrán, Miguel-Alonso & Lozano, "A Review of Auto-scaling Techniques for Elastic Applications in Cloud Environments"**, *JGC 2014*. The survey of the policy families. *Free, survey.*
- **Axelsson, "The Base-Rate Fallacy and the Difficulty of Intrusion Detection"**, *TISSEC 2000* — section 7's warning, already met in module 15. *Free.*

**Freely available courses / materials**
- **Descartes Research** (descartes.tools, Kounev's group) — the Descartes Modelling Language and elasticity benchmarks (BUNGEE). *Free.*
- Lectures on **performance modelling / queueing theory** and **autonomic & self-adaptive systems** (e.g. SEAMS community tutorials). *Free.*

**For hands-on practice**
- The **three projects** build the self-model (queueing theory verified against a simulator), the adaptation loop (three auto-scaling policies measured on the elasticity triple) and a complete self-aware system (MAPE-K with a model at run-time, online re-estimation under drift, and self-healing) — all from scratch, the honest way to see why prediction beats reaction.

---

> **Next module:** Module 25 "Interaktive Computergraphik" — the rendering pipeline, transformations and shading; it returns to the 3D geometry line of modules 19–20 from the graphics side.

---
---

# Modul 24 — Self-aware Computing (deutsche Fassung)

> **Worum geht es?** Modul 23 automatisierte eine *Anlage* — eine Maschine außerhalb des Computers. Dieses Modul dreht dieselbe Idee **nach innen**: ein Softwaresystem, das **sich selbst** beobachtet, ein **Modell von sich selbst** baut, vorhersagt, was passiert, wenn sich Last oder Konfiguration ändern, und sich **selbst rekonfiguriert**, um seine Ziele weiter zu erfüllen — ohne Menschen im Regelkreis. Ein Webdienst, der seine Kapazität bei einem Lastanstieg selbst hochfährt, bemerkt, dass eine Replik krank geworden ist, und sie ersetzt, und der sein eigenes Performance-Modell neu lernt, wenn sich der Code ändert: das ist ein **self-aware computing system**. Die Mathematik dahinter ist **Warteschlangentheorie** (das Selbstmodell), **Regelungstechnik** (die Adaptionsschleife) und **Online-Lernen** (das Modell wahr halten).
>
> **Vorkenntnisse**: Wahrscheinlichkeitsrechnung (Exponentialverteilung, Erwartungswert), Grundstatistik, Regelungsgrundlagen. Aus diesem Repo bauen ein: **Modul 23** (die MAPE-K-Schleife *ist* ein Regelkreis, und modellbasiertes Skalieren *ist* die MPC-Idee — prädizieren, optimieren, handeln, wiederholen), **Modul 14** (LQR/Optimalregelung), **Modul 15/16** (Anomalieerkennung und die Base-Rate-Fallacy — unverzichtbar für Self-Healing), **Modul 13** (RL, der lernende Weg zu Adaptionsstrategien), **Modul 02/03** (Statistik/Schätzung). Modul 23 ist das nützlichste Vormodul.

> **Hinweis zum Zuschnitt.** Es liegt keine offizielle Modulbeschreibung vor, ich habe den Inhalt also selbst zugeschnitten — hier ist die Wahl aber ungewöhnlich gut verankert: **Samuel Kounev**, Inhaber des Lehrstuhls für Software Engineering an der **Universität Würzburg**, ist eine der prägenden Figuren des Feldes und Hauptherausgeber des Standardwerks *Self-Aware Computing Systems* (Springer 2017). Ich folge diesem Kanon: Kounevs drei Eigenschaften (**self-reflective, self-predictive, self-adaptive**), IBMs **MAPE-K**-Schleife, **Modelle zur Laufzeit** und der quantitative Kern aus **Warteschlangentheorie** und **Elastizität**. Das Modul ist bewusst als ein Bogen gebaut — **Modell → Regelkreis → lernendes System** —, denn genau das unterscheidet ein *self-aware* System von einem bloß *adaptiven*: Es reagiert nicht nur, es **schließt mit einem Modell von sich selbst**. Alles from scratch in `numpy`/`scipy` (ein Discrete-Event-Simulator, Warteschlangenformeln, Regler) — kein Cloud-Account, kein Kubernetes, CPU-Sekunden.

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

- ein **self-aware computing system** definieren können (Kounev): Es **lernt Modelle** von sich und seiner Umgebung *fortlaufend*, **schließt** mit diesen Modellen und **handelt** gemäß übergeordneten Zielen — und die drei Eigenschaften **self-reflective / self-predictive / self-adaptive** unterscheiden.
- die **MAPE-K**-Schleife beschreiben (Monitor–Analyze–Plan–Execute über einer gemeinsamen **Knowledge**-Basis) sowie die **self-\***-Eigenschaften (Selbstkonfiguration, Selbstheilung, Selbstoptimierung, Selbstschutz).
- die **operational laws** anwenden — **Little's Law** $N=X\cdot R$, das **Auslastungsgesetz** $U=X\cdot S$, das **Forced-Flow-Gesetz** und das **Service-Demand-Gesetz** —, um allein aus Messungen über ein laufendes System zu schließen.
- die **M/M/1**-Resultate herleiten und nutzen ($\rho$, $N$, $R$, die Antwortzeitverteilung und ihre Quantile), erklären, **warum die Antwortzeit bei $\rho\to1$ explodiert**, und auf **M/M/c** erweitern; den **Engpass** finden und die **asymptotischen Schranken** angeben.
- **Auto-Scaling**-Strategien (Elastizität) entwerfen und vergleichen — **schwellenbasiert/reaktiv**, **regelungstheoretisch** und **modellbasiert-prädiktiv** — und sie ehrlich mit **SLO-Verletzungen, Kosten und Oszillation (Flapping)** bewerten; **Hysterese** und **Cooldown** erklären.
- **Modelle zur Laufzeit** erklären und warum ein self-aware System sein Modell **online nachschätzen** muss (Drift), und wie **Anomalieerkennung** die **Selbstheilung** speist — inklusive der Frage, warum die **Base-Rate-Fallacy** (Modul 15) naives Alarmieren unbrauchbar macht.

---

## Grundlagen (Basics)

### 1. Von der Automatisierung zur Selbstwahrnehmung

Klassische Automatisierung (Modul 23) regelt eine externe Anlage mit einem Modell, das ein *Ingenieur* geschrieben hat. Ein modernes Softwaresystem ist aber seine eigene Anlage: Es läuft auf geteilter, wechselnder Infrastruktur, seine Last schwankt zwischen Nacht und Spitze um Größenordnungen, sein eigener Code ändert sich wöchentlich, und um 3 Uhr nachts schaut niemand zu. **Self-aware Computing** verlangt vom System, die Aufgabe dieses Ingenieurs zu übernehmen.

Kounevs Definition (2017) ist die Referenz, und jeder Teilsatz zählt:

> Ein **self-aware computing system** *lernt Modelle*, die Wissen über sich selbst und seine Umgebung erfassen, **fortlaufend**, und *schließt* mit diesen Modellen, was es befähigt, auf Basis seines Wissens und Schließens zu *handeln*, im Einklang mit **übergeordneten Zielen**.

Daraus folgen drei Eigenschaften — die Leiter, die dieses Modul erklimmt:

1. **Self-reflective** — es *kennt sich selbst*: welche Komponenten existieren, wie Last, Auslastung und Antwortzeit gerade sind. (Monitoring, Modellstruktur.)
2. **Self-predictive** — es kann die *Konsequenzen einer Änderung vorhersagen*, **bevor** es sie vornimmt: „Wenn die Last sich verdoppelt, halte ich dann noch das SLO? Wie viele Repliken bräuchte ich?" Das ist es, was ein **Modell** einbringt, und es trennt Selbstwahrnehmung von blinder Reaktion.
3. **Self-adaptive** — es *handelt*: rekonfiguriert, skaliert, startet neu, routet um, um seine Ziele weiter zu erfüllen.

Ein Thermostat ist adaptiv, aber nicht self-aware (kein Modell von sich, keine Prädiktion). Ein System, das rechnet „bei 800 Requests/s mit 3 Repliken ist meine prädizierte 95%-Latenz 240 ms, das bricht mein 200-ms-SLO, also brauche ich **jetzt** eine 4. Replik", ist self-aware.

### 2. Die MAPE-K-Schleife und die self-\*-Eigenschaften

Die ordnende Architektur stammt aus IBMs **Autonomic-Computing**-Manifest (2003): ein Autonomic Manager, der ein Managed Element umschließt und die **MAPE-K**-Schleife fährt:

```
            +--------------------------------------------------+
            |                 Autonomic Manager                |
            |   Monitor -> Analyze -> Plan -> Execute          |
            |        \        |         |        /             |
            |         +---- Knowledge (Modelle, Historie) --+  |
            +--------------------------------------------------+
                 ^                                    |
             Sensoren                             Effektoren
                 |                                    v
            +--------------------------------------------------+
            |      Managed Element (das Softwaresystem)         |
            +--------------------------------------------------+
```

- **Monitor** — Metriken über Sensoren erfassen (Ankunftsrate, Auslastung, Antwortzeit, Fehlerrate).
- **Analyze** — entscheiden, ob die Ziele (bald) verletzt werden; hier leben **Modell** und **Prädiktion**.
- **Plan** — die Adaption wählen (um 2 hochskalieren, Replik 7 neu starten, Last abwerfen).
- **Execute** — sie über Effektoren anwenden.
- **Knowledge** — der gemeinsame, *persistente* Teil: das Systemmodell, die SLOs, die Messhistorie. Genau dieses K macht die Schleife *self-aware* statt zu einem Reflex.

Die klassischen Ziele sind die **self-\***- (oder **self-CHOP**-) Eigenschaften: **Selbstkonfiguration** (sich selbst installieren/verdrahten), **Selbstheilung** (Fehler erkennen, diagnostizieren, reparieren), **Selbstoptimierung** (sich auf Performance/Kosten trimmen), **Selbstschutz** (Angriffe und Kaskadenfehler abwehren).

> **Der Bezug zu Modul 23.** MAPE-K *ist* ein Regelkreis mit einem Modell darin — Monitor = Messung, Analyze+Plan = der Optimierer, Execute = Stellen, Knowledge = das Modell. Wird der Plan berechnet, indem man mit einem Modell einen Horizont prädiziert und nur die erste Aktion anwendet, hat man buchstäblich **MPC** (Modul 23 P03) nachgebaut — nun mit dem *Computer selbst* als Anlage. Das ist der klarste Weg, modellbasiertes Auto-Scaling zu verstehen.

### 3. Modelle zur Laufzeit

Ein Entwurfszeit-Modell (ein UML-Diagramm, eine Kapazitäts-Tabelle) ist in dem Moment tot, in dem sich das System ändert. Ein **Modell zur Laufzeit** (*models@run.time*) ist eine lebende, kausal verbundene Abstraktion: Es wird durch Monitoring **mit dem laufenden System synchron gehalten**, und es zu ändern **ändert das System**. Kounevs Descartes Modelling Language ist eine Forschungsinstanz davon; eine einfache, ehrliche Instanz — und die, die wir bauen — ist ein **Warteschlangenmodell, dessen Parameter fortlaufend aus Messungen nachgeschätzt werden**.

Das ist der Kern: Ein Modell nützt nur, solange es *wahr* ist. Deployments, laute Nachbarn und Cache-Aufwärmen verschieben alle die reale Servicerate. Ein self-aware System **schätzt seine Parameter daher online nach** und bemerkt es auch, wenn das Modell nicht mehr zur Realität passt. Das **Final-Projekt** baut genau das, inklusive der Drift.

---

## Aufbau (Intermediate)

### 4. Die operational laws: aus Messungen über ein System schließen

Vor jeder stochastischen Annahme gilt eine Menge **exakter, annahmefreier** Identitäten für jedes über ein endliches Intervall beobachtete System. Sie sind das Arbeitsvokabular der Performance-Selbstwahrnehmung. Über ein Beobachtungsfenster definiere Ankünfte/Fertigstellungen $C$, Busy-Zeit $B$, Fensterlänge $T$:

- **Durchsatz** $X = C/T$, **Auslastung** $U = B/T$, **mittlere Servicezeit** $S = B/C$.
- **Auslastungsgesetz**: $\;\boxed{U = X\cdot S}\;$ — Auslastung ist Durchsatz mal Servicezeit. (Direkt: $B/T = (C/T)(B/C)$.)
- **Little's Law**: $\;\boxed{N = X\cdot R}\;$ — die mittlere Anzahl im System ist Durchsatz mal mittlere Antwortzeit. Es gilt für *jedes* stabile System, ohne jede Verteilungsannahme. Es ist die nützlichste Formel der Performance-Technik: Miss zwei Größen, erhalte die dritte.
- **Forced-Flow-Gesetz**: $X_k = V_k\,X$ — besucht jeder Request Ressource $k$ im Mittel $V_k$-mal, ist deren Durchsatz ein festes Vielfaches des Systemdurchsatzes.
- **Service-Demand-Gesetz**: $\;\boxed{D_k = V_k S_k = U_k / X}\;$ — der Gesamtbedarf, den ein Request an Ressource $k$ stellt, ist deren Auslastung geteilt durch den Systemdurchsatz. Damit lassen sich Ressourcenbedarfe **aus Black-Box-Messungen** gewinnen — so schätzt ein System sein eigenes Modell.

**Der Engpass** ist die Ressource mit dem größten Bedarf $D_{\max}=\max_k D_k$. Er deckelt alles: $X \le 1/D_{\max}$. Mit $D=\sum_k D_k$ (Gesamtbedarf) und Denkzeit $Z$ lauten die **asymptotischen Schranken**

$$X(N) \le \min\Big(\frac{1}{D_{\max}},\;\frac{N}{D+Z}\Big),\qquad R(N)\ \ge\ \max\big(D,\;N D_{\max}-Z\big).$$

Zwei Regime: bei geringer Last ist das System *latenzbegrenzt* ($R\approx D$), bei hoher Last *engpassbegrenzt* (der Durchsatz sättigt bei $1/D_{\max}$, die Antwortzeit wächst linear in $N$). Zu wissen, wo man steht, ist Selbstwahrnehmung in ihrer praktischsten Form.

### 5. Die M/M/1-Warteschlange: warum die Antwortzeit explodiert

Nimmt man die Standardannahmen hinzu — Poisson-Ankünfte mit Rate $\lambda$, exponentielle Bedienung mit Rate $\mu$, ein Server, unendliche FIFO-Schlange —, erhält man die **M/M/1**-Warteschlange, das lehrreichste Modell der Informatik. Definiere die **Auslastung** (Verkehrsintensität)

$$\rho = \frac{\lambda}{\mu}\qquad(\text{stabil genau dann wenn } \rho<1).$$

Die Schlange ist eine Geburts-Todes-Kette mit Geburtsrate $\lambda$ und Todesrate $\mu$; die Bilanz liefert die stationäre Verteilung $P(N=n) = (1-\rho)\rho^{n}$ — geometrisch. Daraus:

$$\boxed{\;N = \frac{\rho}{1-\rho},\qquad R = \frac{1}{\mu-\lambda} = \frac{S}{1-\rho},\qquad W = R - S = \frac{\rho S}{1-\rho}\;}$$

(mit $S=1/\mu$; $N=XR$ ist wieder Little's Law, wie es sein muss). Zudem ist die **Antwortzeit exponentialverteilt** mit Rate $\mu-\lambda$, ihre Quantile liegen also geschlossen vor:

$$R_p = \frac{\ln\!\big(1/(1-p)\big)}{\mu-\lambda}\quad\Longrightarrow\quad R_{95} = \frac{\ln 20}{\mu-\lambda}\approx 3\,R,\qquad R_{99}\approx 4{,}6\,R.$$

Deshalb werden SLOs auf **Perzentile** geschrieben, nicht auf Mittelwerte: Der Schwanz ist das *Drei- bis Fünffache* des Mittelwerts.

**Die Explosion.** Der Faktor $1/(1-\rho)$ ist die ganze Geschichte der Kapazitätsplanung:

| $\rho$ | 0,5 | 0,8 | 0,9 | 0,95 | 0,99 |
|---|---|---|---|---|---|
| $R/S$ | 2 | 5 | 10 | 20 | 100 |

Von 50 % auf 90 % Auslastung kostet die **5-fache Latenz**; von 90 % auf 99 % nochmals das **10-fache**. Die Antwortzeit ist **nichtlinear und unbeschränkt** in der Auslastung — genau deshalb ist „die CPU ist erst bei 85 %, alles gut" ein gefährlicher Satz, und genau deshalb muss ein Auto-Scaler auf *prädizierte Latenz* handeln, nicht auf Auslastung allein. Das **Basic-Projekt** verifiziert jede dieser Formeln gegen einen Simulator.

**M/M/c.** Mit $c$ parallelen Servern (Repliken!) und $\rho=\lambda/(c\mu)$ ist die Wahrscheinlichkeit, dass ein Ankömmling warten muss, die **Erlang-C**-Formel

$$P_Q \;=\; \frac{\dfrac{(c\rho)^c}{c!\,(1-\rho)}}{\displaystyle\sum_{k=0}^{c-1}\frac{(c\rho)^k}{k!} + \frac{(c\rho)^c}{c!\,(1-\rho)}},\qquad W = \frac{P_Q}{c\mu-\lambda},\qquad R = W + \frac1\mu.$$

Die praktische Folge sind **Skaleneffekte**: Bei gleicher Auslastung liefern viele Server hinter *einer* Schlange weit geringere Wartezeit als dieselben Server mit getrennten Schlangen — ein Grund, warum es Load Balancer und geteilte Thread-Pools gibt. M/M/c ist das Modell, mit dem ein Auto-Scaler „wie viele Repliken brauche ich für diese Last?" beantwortet.

---

## Advanced-Themen

### 6. Elastizität und Auto-Scaling

**Elastizität** ist der Grad, in dem ein System seine Kapazität **autonom an die aktuelle Nachfrage anpasst**, so genau und so zeitnah wie möglich (Herbst, Kounev & Reussner). Man beachte die drei in dieser Definition versteckten Qualitäten — *Genauigkeit*, *Zeitnähe* und *Stabilität* — und dass sie sich gegenseitig bedingen. Drei Strategiefamilien:

- **Reaktiv / schwellenbasiert** (womit die meisten Cloud-Auto-Scaler ausgeliefert werden): Ist $U > 0{,}7$ für $k$ Intervalle, füge eine Replik hinzu; ist $U<0{,}3$, nimm eine weg. Einfach und robust, aber **rein reaktiv**: Es handelt erst, *nachdem* das SLO bereits gefährdet ist, und während der Bereitstellungsverzögerung (VM-Boot, Container-Start, Cache-Aufwärmen — Sekunden bis Minuten) bleibt das System überlastet.
- **Regelungstheoretisch**: Kapazität als Stellgröße, das SLO als Sollwert; ein **PI/PID**-Regler (Modul 21) oder eine LQR/MPC-Formulierung (Module 14/23). Prinzipiengeleitet, einstellbar, mit Stabilitätsgarantien — braucht aber ein Modell oder zumindest identifizierte Verstärkungen.
- **Modellbasiert-prädiktiv**: Die Last **prognostizieren** (saisonale Muster, Modul 16) und mit dem **Warteschlangenmodell** die Kapazität berechnen, die das SLO *bei dieser künftigen Last* hält — und dann **vor** dem Anstieg bereitstellen. Das ist MPC mit dem M/M/c-Modell als Anlagenmodell, und es ist die einzige Familie, die die Bereitstellungsverzögerung schlagen kann.

**Zwei Mechanismen, die jeder reale Scaler braucht.** Beide bekämpfen **Oszillation (Flapping)** — wiederholtes Hinzufügen und Wegnehmen von Kapazität, das Geld kostet und, weil jede Änderung Aufwärmkosten hat, die Performance aktiv *verschlechtert*:
- **Hysterese**: *unterschiedliche* Schwellen für Hoch- und Runterskalieren (z. B. hoch bei 0,7, runter bei 0,3). Der Abstand ist eine Totzone, in der nichts passiert.
- **Cooldown** (Dämpfung): Nach jeder Aktion für eine feste Zeit keine weitere zulassen, damit sich die Systemantwort setzen kann, bevor gemessen wird.

**Elastizität ehrlich bewerten.** Nie nur „es hat skaliert" berichten. Die Metriken der Community (Herbst et al.) sind:
- **Under-Provisioning**-Genauigkeit/-Zeitanteil: wie stark und für welchen Zeitanteil die Kapazität *unter* der Nachfrage lag — hier leben die **SLO-Verletzungen**.
- **Over-Provisioning**-Genauigkeit/-Zeitanteil: wie viel Kapazität verschwendet wurde — das sind die **Kosten**.
- **Instabilität / Jitter**: die Zahl der Adaptionen (oder Vorzeichenwechsel) — das ist das **Flapping**.

Ein Scaler, der das SLO nie verletzt, weil er 10-fach überprovisioniert, ist nicht gut — und ein billiger, der das SLO zu 20 % der Zeit reißt, auch nicht. Das **Medium-Projekt** misst genau dieses Tripel für alle drei Familien.

### 7. Selbstheilung und die Anomalieerkennungs-Falle

**Selbstheilung** = einen Fehler erkennen, diagnostizieren, reparieren (Neustart, Ersatz, Umrouten), idealerweise bevor Nutzer es merken. Erkennung heißt meist **Anomalieerkennung** auf Metrikströmen (Latenz, Fehlerrate, Sättigung) — und hier muss das Modul eine Warnung aussprechen, die sich die Module 15/16 bereits verdient haben.

Fehler sind **selten**. Habe ein Detektor die Richtig-Positiv-Rate $\mathrm{TPR}$ und die Falsch-Positiv-Rate $\mathrm{FPR}$, und träten Fehler mit Basisrate $\pi$ auf. Die Wahrscheinlichkeit, dass ein Alarm echt ist, beträgt

$$P(\text{Fehler}\mid\text{Alarm}) = \frac{\mathrm{TPR}\cdot\pi}{\mathrm{TPR}\cdot\pi + \mathrm{FPR}\cdot(1-\pi)}.$$

Mit einem exzellenten Detektor ($\mathrm{TPR}=0{,}99$, $\mathrm{FPR}=0{,}01$) und realistischer Basisrate ($\pi=10^{-3}$) sind das $\approx 9\,\%$ — **neun von zehn Alarmen sind falsch**, und ein daran verdrahteter automatischer Heiler würde in neun von zehn Fällen gesunde Repliken neu starten und damit den Ausfall *verursachen*, den er verhindern sollte. Das ist die **Base-Rate-Fallacy** (Modul 15), und in der Selbstheilung ist sie nicht akademisch: Sie entscheidet zwischen einem System, das sich stabilisiert, und einem, das sich aufschaukelt. Gegenmittel: Evidenzschwelle anheben (niedrigere TPR für viel niedrigere FPR akzeptieren), **Persistenz fordern** (anomal für $k$ aufeinanderfolgende Intervalle), über unabhängige Signale **korroborieren**, und die Reparatur **billig und reversibel** machen, damit ein Fehlalarm überlebbar ist.

### 8. Lernen und das Modell wahr halten

Das „fortlaufend" in Kounevs Definition ist der schwierige Teil. Drei Stränge:

- **Online-Parameterschätzung.** Die Modellparameter (Ressourcenbedarfe $D_k$, Servicerate $\mu$) werden **aus Messungen geschätzt** — über das Service-Demand-Gesetz ($D_k=U_k/X$) oder per Regression der Antwortzeit gegen die Last — und fortlaufend nachgeschätzt, z. B. mit einem exponentiell gewichteten gleitenden Mittel oder einem Kalman-Filter (Modul 21), damit die Schätzung Änderungen folgt und Rauschen dämpft.
- **Drift und Modell-Invalidierung.** Nach einem Deployment kann das reale $\mu$ um 30 % fallen. Ein self-aware System muss **bemerken**, dass seine Prädiktionen verzerrt geworden sind (den *Prädiktionsfehler* überwachen, nicht nur die Metrik), und neu lernen — die **Gültigkeit** des Modells ist selbst etwas, dessen man sich bewusst sein muss.
- **Die Strategie lernen.** Statt die Adaptionsregel herzuleiten, sie lernen: **Reinforcement Learning** (Module 13/14) kann eine Skalierungsstrategie aus Erfahrung lernen, mit den bekannten Kosten — Sample-Ineffizienz und unsichere Exploration in Produktion —, weshalb der praktische Stand der Technik meist ein **modellbasierter** Regler mit gelernten *Parametern* ist, keine gelernte Policy. Das **Final-Projekt** geht genau diesen Weg.

---

## Zusammenfassung / Cheat-Sheet

**Self-aware System (Kounev)**: lernt Modelle von sich/Umgebung **fortlaufend**, **schließt** mit ihnen, **handelt** auf übergeordnete Ziele hin. Eigenschaften: **self-reflective** (kennt sich) · **self-predictive** (sagt Folgen vorher) · **self-adaptive** (handelt).

**MAPE-K**: Monitor → Analyze → Plan → Execute, über gemeinsamer **Knowledge** (Modelle, Historie, SLOs). Self-\*-Ziele: Selbst-**K**onfiguration, Selbst-**H**eilung, Selbst-**O**ptimierung, Selbst-**S**chutz. MAPE-K mit modellbasierter Planung ≡ **MPC** (Modul 23), angewandt auf den Computer selbst.

**Operational laws** (exakt, annahmefrei): $U = X S$ · **Little: $N = X R$** · $X_k=V_k X$ · **Service Demand: $D_k = U_k/X$**.
**Engpass** $D_{\max}$: $X\le\min(1/D_{\max},\,N/(D+Z))$, $R\ge\max(D,\,N D_{\max}-Z)$.

**M/M/1**: $\rho=\lambda/\mu$ (stabil gdw. $<1$); $P(N=n)=(1-\rho)\rho^n$; $N=\rho/(1-\rho)$; $R=S/(1-\rho)=1/(\mu-\lambda)$; Antwortzeit $\sim\mathrm{Exp}(\mu-\lambda)$ ⇒ $R_p=\ln(1/(1-p))/(\mu-\lambda)$, $R_{95}\approx3R$, $R_{99}\approx4{,}6R$.
**Die $1/(1-\rho)$-Explosion**: $\rho=0{,}5\to R/S=2$; $0{,}9\to10$; $0{,}99\to100$. Nie auf Auslastung allein schließen.
**M/M/c**: Erlang-C $P_Q$, $W=P_Q/(c\mu-\lambda)$ — Pooling schlägt getrennte Schlangen (Skaleneffekte).

**Elastizität** = Kapazität autonom an die Nachfrage anpassen, genau und zeitnah. Strategien: **reaktiv/schwellenbasiert** (handelt zu spät), **regelungstheoretisch** (PI/PID, LQR), **modellbasiert-prädiktiv** (Prognose + Warteschlangenmodell ⇒ schlägt die Bereitstellungsverzögerung). Immer **Hysterese** + **Cooldown** gegen **Flapping**.
**Mit dem Tripel bewerten**: Under-Provisioning (**SLO-Verletzungen**) · Over-Provisioning (**Kosten**) · Instabilität (**Adaptionen/Flapping**).

**Selbstheilung**: erkennen → diagnostizieren → reparieren. **Base-Rate-Fallacy** (Modul 15): $P(\text{Fehler}\mid\text{Alarm})=\frac{\mathrm{TPR}\,\pi}{\mathrm{TPR}\,\pi+\mathrm{FPR}(1-\pi)}$ — TPR 0,99/FPR 0,01 bei $\pi=10^{-3}$ ergibt **9 %**. Persistenz, Korroboration, billige reversible Reparaturen fordern.

**Modelle zur Laufzeit**: Modell **kausal verbunden** halten und **online nachschätzen** ($D_k=U_k/X$, EWMA/Kalman); den **Prädiktionsfehler** überwachen, um Drift und Modell-Invalidierung zu erkennen.

---

## Selbsttest

<details>
<summary><b>1.</b> Gib Kounevs Definition eines self-aware computing system an und nenne die drei Eigenschaften.</summary>

Ein self-aware computing system **lernt Modelle**, die Wissen über sich selbst und seine Umgebung erfassen, **fortlaufend**, **schließt** mit diesen Modellen und **handelt** auf Basis dieses Wissens und Schließens im Einklang mit **übergeordneten Zielen**. Die drei Eigenschaften sind **self-reflective** (es kennt eigene Struktur und Zustand), **self-predictive** (es kann die Folgen einer Änderung vorhersagen, *bevor* es sie vornimmt) und **self-adaptive** (es handelt, um seine Ziele weiter zu erfüllen). Die mittlere — Prädiktion per Modell — trennt Selbstwahrnehmung von bloß reaktiver Adaption.
</details>

<details>
<summary><b>2.</b> Beschreibe die MAPE-K-Schleife und sage, was das K beiträgt.</summary>

**Monitor** (Metriken über Sensoren erfassen) → **Analyze** (werden Ziele verletzt oder bald verletzt?) → **Plan** (eine Adaption wählen) → **Execute** (sie über Effektoren anwenden), alles über einer gemeinsamen **Knowledge**-Basis. Das **K** hält das Systemmodell, die SLOs und die Messhistorie: Es ist das, was Analyze *prädizieren* und Plan *schließen* lässt, statt bloß reflexhaft zu reagieren. Ohne K ist die Schleife ein Thermostat; mit einem Modell im K macht modellbasierte Planung sie exakt zum MPC-Muster aus Modul 23, angewandt auf das Softwaresystem selbst.
</details>

<details>
<summary><b>3.</b> Formuliere Little's Law und sage, was es bemerkenswert macht. Wende es an: 200 Req/s, mittlere Antwortzeit 50 ms — wie viele Requests sind im System?</summary>

**$N = X\cdot R$**: Die mittlere Zahl der Requests im System ist Durchsatz mal mittlere Antwortzeit. Bemerkenswert ist es, weil es **exakt für jedes stabile System** gilt — ohne jede Annahme über Ankunfts- oder Bedienverteilungen, Scheduling-Disziplin oder Serverzahl. Miss zwei beliebige Größen und erhalte die dritte.

Mit $X=200\,\mathrm{s^{-1}}$ und $R=0{,}05\,\mathrm s$: $N = 200\cdot0{,}05 = \mathbf{10}$ Requests im Mittel im System.
</details>

<details>
<summary><b>4.</b> Was ist das Service-Demand-Gesetz und warum ist es zentral für Selbstwahrnehmung?</summary>

$D_k = V_k S_k = U_k / X$ — der gesamte Servicebedarf, den ein Request an Ressource $k$ stellt, ist deren Auslastung geteilt durch den Systemdurchsatz. Zentral ist es, weil ein System damit **seine eigenen Modellparameter aus Black-Box-Messungen schätzen** kann: Auslastung und Durchsatz werden ohnehin überwacht, das System kann also seine Ressourcenbedarfe berechnen, seinen Engpass finden ($D_{\max}$) und ein Warteschlangenmodell **zur Laufzeit** parametrieren, ganz ohne Instrumentierung der Interna.
</details>

<details>
<summary><b>5.</b> Leite/nenne die mittlere M/M/1-Antwortzeit und erkläre die Explosion für $\rho\to1$.</summary>

Mit $\rho=\lambda/\mu<1$ ist die stationäre Warteschlangenlänge geometrisch, $P(N=n)=(1-\rho)\rho^n$, also $N=\rho/(1-\rho)$; Little's Law ($N=\lambda R$) liefert dann $R = \dfrac{1}{\mu-\lambda} = \dfrac{S}{1-\rho}$.

Der Faktor $1/(1-\rho)$ **explodiert**, wenn die Auslastung gegen 1 geht: $\rho=0{,}5\Rightarrow R=2S$, $0{,}9\Rightarrow10S$, $0{,}99\Rightarrow100S$. Anschaulich: Bei hoher Auslastung ist der Server fast nie frei, wenn Arbeit ankommt, fast jeder Request wartet also hinter einer Schlange, die immer länger zum Abbauen braucht. Praktisch: Die Antwortzeit ist **nichtlinear und unbeschränkt** in der Auslastung, eine Kapazitätsentscheidung muss also auf prädizierter *Latenz* getroffen werden, nicht auf einer Auslastungszahl.
</details>

<details>
<summary><b>6.</b> Warum werden SLOs auf Perzentile geschrieben? Quantifiziere für M/M/1.</summary>

Weil der **Schwanz** weit schlechter ist als der Mittelwert. In M/M/1 ist die Antwortzeit **exponentialverteilt** mit Rate $\mu-\lambda$, also $R_p = \ln\!\big(1/(1-p)\big)/(\mu-\lambda)$, d. h. $R_p = \ln(1/(1-p))\cdot R$. Damit ist $R_{95}=\ln 20\cdot R\approx 3R$ und $R_{99}=\ln100\cdot R\approx 4{,}6R$. Ein System, das einen *Mittelwert* von 100 ms einhält, kann ein 200-ms-*95-%-SLO* leicht verletzen — deshalb werden reale SLOs als Perzentile formuliert, und deshalb sollte ein Auto-Scaler auf eines davon zielen.
</details>

<details>
<summary><b>7.</b> Vergleiche reaktives, regelungstheoretisches und prädiktives Auto-Scaling. Was folgt aus der Bereitstellungsverzögerung?</summary>

**Reaktives/schwellenbasiertes** Skalieren handelt erst, *nachdem* eine Metrik eine Grenze überschreitet — einfach und robust, aber immer zu spät. **Regelungstheoretisches** Skalieren (PI/PID, LQR) nimmt das SLO als Sollwert und die Kapazität als Stellgröße, mit prinzipiengeleiteter Einstellung und Stabilität, braucht aber identifizierte Verstärkungen/ein Modell. **Modellbasiert-prädiktives** Skalieren **prognostiziert** die Last und berechnet mit dem **Warteschlangenmodell** die Kapazität, die das SLO bei dieser künftigen Last erfüllt, und stellt *vor* dem Anstieg bereit (MPC auf das System selbst angewandt).

Die **Bereitstellungsverzögerung** (VM-Boot, Container-Start, Aufwärmen) ist entscheidend: Während ihrer bleibt das System bei einem reaktiven Scaler überlastet, seine SLO-Verletzungen sind also praktisch unvermeidbar. Nur eine prädiktive Strategie — die Kapazität *vor* dem Eintreffen der Nachfrage startet — kann diese Verzögerung verbergen.
</details>

<details>
<summary><b>8.</b> Was ist Flapping und welche zwei Mechanismen verhindern es?</summary>

**Flapping** (Oszillation/Instabilität) ist das wiederholte Hinzufügen und Wegnehmen von Kapazität um eine Schwelle herum. Es kostet Geld, und weil jede Änderung Aufwärm-/Migrationskosten trägt, *verschlechtert* es zusätzlich die Performance — die Adaption macht es schlimmer. Die zwei Standardmechanismen: **Hysterese** — unterschiedliche Schwellen für Hoch- und Runterskalieren (z. B. hoch bei $U>0{,}7$, runter bei $U<0{,}3$), was eine Totzone schafft; und **Cooldown/Dämpfung** — nach jeder Adaption für ein festes Intervall keine weitere zulassen, damit sich die Wirkung setzen kann, bevor gemessen wird.
</details>

<details>
<summary><b>9.</b> Wie sollte Elastizität bewertet werden, und warum genügt eine Zahl nicht?</summary>

Mit einem **Tripel**, weil sich die Qualitäten gegenseitig bedingen: **Under-Provisioning** (wie stark und wie lange die Kapazität unter der Nachfrage lag — also **SLO-Verletzungen**), **Over-Provisioning** (verschwendete Kapazität — also **Kosten**) und **Instabilität/Jitter** (Zahl der Adaptionen — also **Flapping**). Eine Zahl genügt nicht, weil jede Einzelmetrik trivial zu manipulieren ist: Eine Strategie, die stets 10-fach provisioniert, hält das SLO zu absurden Kosten, und die billigste Strategie verletzt es ständig. Erst das Tripel zeigt den tatsächlichen Betriebspunkt.
</details>

<details>
<summary><b>10.</b> Warum ist naive anomaliegetriebene Selbstheilung gefährlich? Rechne ein Beispiel.</summary>

Weil Fehler **selten** sind, produziert selbst ein exzellenter Detektor überwiegend Fehlalarme — die **Base-Rate-Fallacy**. Mit $P(\text{Fehler}\mid\text{Alarm}) = \frac{\mathrm{TPR}\,\pi}{\mathrm{TPR}\,\pi+\mathrm{FPR}(1-\pi)}$ ergibt ein Detektor mit $\mathrm{TPR}=0{,}99$, $\mathrm{FPR}=0{,}01$ bei Basisrate $\pi=10^{-3}$ gerade $\frac{0{,}99\cdot0{,}001}{0{,}99\cdot0{,}001+0{,}01\cdot0{,}999}\approx\mathbf{9\,\%}$: **91 % der Alarme sind falsch**. Ein direkt daran verdrahteter automatischer Heiler würde in neun von zehn Fällen gesunde Repliken neu starten und könnte genau den Ausfall verursachen, den zu verhindern er existiert. Gegenmittel: deutlich niedrigere FPR (bei Inkaufnahme geringerer TPR), **Persistenz** über $k$ Intervalle, **Korroboration** über unabhängige Signale und **billige, reversible** Reparaturaktionen.
</details>

---

## Literatur & Quellen

**Lehrbücher & die Referenz des Feldes**
- **Kounev, Kephart, Milenkoski & Zhu (Hrsg.), *Self-Aware Computing Systems*** (Springer 2017). **Die** Referenz für dieses Modul — Definitionen, die drei Eigenschaften, Modelle zur Laufzeit, Architekturen. Samuel Kounev hat den Lehrstuhl für Software Engineering an der **Universität Würzburg**. *Vertiefend, die kanonische Quelle.*
- **Lazowska, Zahorjan, Graham & Sevcik, *Quantitative System Performance*** — **frei online**. Der Klassiker zu operational laws, Engpassanalyse und Warteschlangennetz-Modellen (Abschnitt 4–5 hier). *Kostenlos, einsteigerfreundlich, bis heute unübertroffen.*
- **Menascé, Almeida & Dowdy, *Performance by Design*** — Kapazitätsplanung mit Warteschlangenmodellen, sehr praxisnah. *Mittel.*
- **Harchol-Balter, *Performance Modeling and Design of Computer Systems*** (Cambridge). Exzellent, rigoros und lesbar zu M/M/1, M/M/c und darüber hinaus. *Vertiefend, sehr empfohlen.*
- **Hellerstein, Diao, Parekh & Tilbury, *Feedback Control of Computing Systems*** (Wiley). Regelungstechnik auf Softwaresysteme angewandt — die theoretische Grundlage für Abschnitt 6. *Vertiefend.*

**Schlüssel-Papers**
- **Kephart & Chess, „The Vision of Autonomic Computing"**, *IEEE Computer 2003*. Der **MAPE-K**-Ursprung. *Kostenlos, einsteigerfreundlich, ein Klassiker.*
- **Herbst, Kounev & Reussner, „Elasticity in Cloud Computing: What It Is, and What It Is Not"**, *ICAC 2013*. Die Definition und die **Elastizitätsmetriken** aus Abschnitt 6. *Kostenlos, essenziell.*
- **Kounev et al., „The Notion of Self-aware Computing"** (Kapitel 1 des obigen Buchs) — die hier genutzte Definition. *Vertiefend.*
- **Lorido-Botrán, Miguel-Alonso & Lozano, „A Review of Auto-scaling Techniques for Elastic Applications in Cloud Environments"**, *JGC 2014*. Der Überblick über die Strategiefamilien. *Kostenlos, Survey.*
- **Axelsson, „The Base-Rate Fallacy and the Difficulty of Intrusion Detection"**, *TISSEC 2000* — die Warnung aus Abschnitt 7, bereits in Modul 15 begegnet. *Kostenlos.*

**Frei verfügbare Kurse / Materialien**
- **Descartes Research** (descartes.tools, Kounevs Gruppe) — die Descartes Modelling Language und Elastizitäts-Benchmarks (BUNGEE). *Kostenlos.*
- Vorlesungen zu **Performance-Modellierung / Warteschlangentheorie** und **autonomen & selbstadaptiven Systemen** (z. B. Tutorials der SEAMS-Community). *Kostenlos.*

**Zum Ausprobieren**
- Die **drei Projekte** bauen das Selbstmodell (Warteschlangentheorie gegen einen Simulator verifiziert), die Adaptionsschleife (drei Auto-Scaling-Strategien am Elastizitäts-Tripel gemessen) und ein vollständiges self-aware System (MAPE-K mit Modell zur Laufzeit, Online-Nachschätzung unter Drift und Selbstheilung) — alles from scratch, der ehrliche Weg zu sehen, warum Prädiktion die Reaktion schlägt.

---

> **Nächstes Modul:** Modul 25 „Interaktive Computergraphik" — die Rendering-Pipeline, Transformationen und Shading; es kehrt von der Graphik-Seite her zur 3D-Geometrie-Linie der Module 19–20 zurück.
