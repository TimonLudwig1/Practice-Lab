# Project 03 (final) — A decision-theoretic MDP agent

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 07 — Theory of AI 2** · Format: **a Python project, built from scratch by you**

> **The final project of the module.** There is **no given code** — you design
> and implement everything yourself. The project consolidates part 3 (utility
> theory, MDPs, the Bellman equations, value/policy iteration) and builds the
> bridge to reinforcement learning (module 13). Level: a genuine master's
> examination piece.

## Why this format and this topic?

An MDP agent is the direct, executable implementation of the Bellman optimality
equation and of the MEU principle. If you write **value iteration** and **policy
iteration** yourself and *observe* their convergence, you understand the core of
sequential decision making — and you see exactly the building blocks that RL is
built on (only that there $P$ and $R$ are unknown). A real, modularized code base
is right here; the procedure itself is too structured for a throwaway notebook.

## Goal

Build an agent that computes the optimal policy of a stochastic MDP — by **two**
routes, whose agreement you check:

- **Value iteration** — iterate the Bellman optimality operator to convergence;
- **Policy iteration** — alternate policy evaluation and policy improvement.

The environment is the classical **stochastic 4×3 gridworld** (Russell &
Norvig): movement succeeds in the intended direction with probability 80 %, with
10 % each perpendicular to it; one wall cell; a +1 and a −1 terminal; a living
reward of $-0.04$ per step. You are to **visualize** the optimal policy (a grid
of arrows), check the **convergence** empirically and investigate **how $\gamma$
and the living reward change the optimal policy**.

## Prior knowledge

Part 3 of the script, in particular:
- the MDP definition, policy, value function, the **Bellman (optimality) equation**;
- **value iteration** and the **contraction proof** (why $B$ converges);
- **policy iteration** (policy evaluation as a linear system, policy improvement).

Python: `dict`, iteration, optionally `matplotlib` for a nicer visualization.

## What you should build — the components

1. **The MDP environment.** Represent the states (grid cells), the actions
   (N/S/E/W), the **noisy transition model** $P(s'\mid s,a)$ (0.8/0.1/0.1, against
   a wall or the border → stay put), the reward $R(s)$ (the living reward, the ±1
   terminals) and $\gamma$. Terminal states are absorbing.

2. **Value iteration.** Implement the Bellman optimality update
   $$V_{k+1}(s) \leftarrow R(s) + \gamma \max_a \sum_{s'} P(s'\mid s,a)\,V_k(s'),$$
   iterate until the **maximum Bellman residual** $\lVert V_{k+1}-V_k\rVert_\infty$
   falls below a bound, and count the iterations. Read off the **greedy policy**
   $\pi^\ast(s)=\arg\max_a\sum_{s'}P(s'\mid s,a)V^\ast(s')$.

3. **Policy iteration.** Implement **policy evaluation** (solve or iterate
   $V^\pi(s)=R(s)+\gamma\sum_{s'}P(s'\mid s,\pi(s))V^\pi(s')$) and **policy
   improvement** (greedy with respect to $V^\pi$); alternate until the policy is
   stable.

4. **Analysis and visualization.**
   - Print $V^\ast$ as a grid of numbers and $\pi^\ast$ as a **grid of arrows**.
   - Show that VI and PI deliver **the same** policy and (up to a tolerance) the
     same values, and compare the **number of iterations**.
   - **The $\gamma$ study:** compute $\pi^\ast$ for several $\gamma$ (e.g. 1.0,
     0.9, 0.5, 0.2) and describe how the policy changes.
   - **The living-reward study:** vary $R$ (e.g. $-0.04, -0.5, -2.0, 0.0$) and
     explain the qualitative behaviour (with a strongly negative $R$ the agent
     takes risky shortcuts, even towards the $-1$ terminal).

## Acceptance criteria

- [ ] Value iteration reproduces the known AIMA utilities of the 4×3 world
      (among others $V(0,0)\approx0.705$, $V(2,2)\approx0.918$, $V(3,0)\approx0.388$;
      tolerance $<0.005$);
- [ ] the optimal policy read off from them agrees with the classical solution
      (the top row → → →, below ↑ ← ← ←, with ↑ next to the wall);
- [ ] policy iteration delivers **the same** policy and values as value iteration
      (and needs **considerably fewer** outer iterations for it);
- [ ] the $\gamma$ and living-reward studies show comprehensible changes of the
      policy;
- [ ] VI reports the number of iterations until convergence (reference: 34 at
      $\gamma=1$, with a residual $<10^{-8}$).

## Self-check questions (answer them in writing)

1. **Why does value iteration converge?** Sketch the contraction argument (the
   Bellman operator is a $\gamma$-contraction in $\lVert\cdot\rVert_\infty$, the
   Banach fixed point theorem). Why does it converge in this world **even at
   $\gamma=1$** (keyword: absorbing terminals / a proper policy)?
2. **Why does policy iteration need so many fewer iterations** than value
   iteration, even though every iteration is more expensive?
3. **Why does a smaller $\gamma$ change the policy** towards shorter and riskier
   routes? Argue via the discounted return.
4. **Why does the agent take shortcuts even towards the $-1$ terminal when the
   living reward is strongly negative?** Connect that with the sign and the
   magnitude of $R$.
5. **Where exactly is the bridge to reinforcement learning (module 13)?** What
   does an RL agent *not* know that your MDP agent assumes here, and how does
   that change the approach (keyword: sampling / temporal difference instead of
   an exact $P,R$)?

## Extensions (optional, for going deeper)

- **Q-values and modified policy iteration** (policy evaluation with only a few sweeps).
- **A matplotlib heatmap** of the utilities with the policy arrows overlaid.
- **A convergence curve** ($\lVert V_{k+1}-V_k\rVert_\infty$ over $k$, on a log
  scale) — confirm the *geometric* rate $\gamma^k$.
- **Asynchronous/prioritized value iteration** or a larger gridworld of your own
  design with traps and several goals.

## Reference solution

**`solution/`** holds a complete, tested reference implementation:
- `gridworld.py` — the 4×3 MDP environment + the ASCII visualization (arrows/values),
- `mdp.py` — value iteration, the greedy policy, policy iteration (with policy evaluation),
- `demo.py` — the utilities/policy, the VI-vs-PI comparison, the $\gamma$ and living-reward studies,
- `test_mdp.py` — the acceptance test against the AIMA reference values.

Reference: VI about 34 iterations ($\gamma=1$), PI about 5 iterations, an
**identical policy**. **Look only after your own attempt.**

```bash
source ../../../../.venv/bin/activate    # only the standard library is needed
cd solution && python demo.py            # the utilities, policy, gamma/reward studies
python test_mdp.py                       # the acceptance test
```

---
---

# Projekt 03 (final) — Ein entscheidungstheoretischer MDP-Agent (deutsche Fassung)

**Modul 07 — Theorie der KI 2** · Format: **Python-Projekt, von Grund auf selbst gebaut**

> **Abschlussprojekt des Moduls.** Es gibt **keinen vorgegebenen Code** — du
> entwirfst und implementierst alles selbst. Das Projekt konsolidiert Teil 3
> (Nutzentheorie, MDPs, Bellman-Gleichungen, Value/Policy Iteration) und schlägt
> die Brücke zum Reinforcement Learning (Modul 13). Niveau: echte
> Master-Prüfungsleistung.

## Warum dieses Format & dieses Thema?

Ein MDP-Agent ist die direkte, ausführbare Umsetzung der Bellman-Optimalitäts­gleichung
und des MEU-Prinzips. Wenn du **Value Iteration** und **Policy Iteration** selbst
schreibst und ihre Konvergenz *beobachtest*, verstehst du den Kern der
sequenziellen Entscheidungsfindung — und siehst genau die Bausteine wieder, auf
denen RL aufbaut (nur dass dort $P$ und $R$ unbekannt sind). Eine echte,
modularisierte Codebasis ist hier richtig; das reine Verfahren ist zu strukturiert
für ein Wegwerf-Notebook.

## Ziel

Baue einen Agenten, der für einen **Markov-Entscheidungsprozess** $(S, A, P, R,
\gamma)$ die optimale Wertfunktion $V^\ast$ und die optimale Policy $\pi^\ast$
berechnet — auf **zwei** Wegen, deren Übereinstimmung du prüfst:

- **Value Iteration** — iteriere den Bellman-Optimalitäts-Operator bis Konvergenz;
- **Policy Iteration** — alterniere Policy Evaluation und Policy Improvement.

Als Umgebung dient die klassische **stochastische 4×3-Gridworld** (Russell &
Norvig): Bewegung gelingt zu 80 % in die gewünschte Richtung, je 10 % senkrecht
daneben; ein Wandfeld; ein +1- und ein −1-Terminal; Living Reward $-0{,}04$ pro
Schritt. Du sollst die optimale Policy **visualisieren** (Pfeilgitter), die
**Konvergenz** empirisch prüfen und untersuchen, **wie $\gamma$ und der Living
Reward die optimale Policy verändern**.

## Vorwissen

Skript Teil 3, insbesondere:
- MDP-Definition, Policy, Wertfunktion, **Bellman-(Optimalitäts-)Gleichung**;
- **Value Iteration** und der **Kontraktionsbeweis** (warum $B$ konvergiert);
- **Policy Iteration** (Policy Evaluation als lineares System, Policy Improvement).

Python: `dict`, Iteration, optional `matplotlib` für eine schönere Visualisierung.

## Was du bauen sollst — die Komponenten

1. **Die MDP-Umgebung.** Repräsentiere Zustände (Gitterzellen), Aktionen
   (N/S/E/W), das **verrauschte Übergangsmodell** $P(s'\mid s,a)$ (0.8/0.1/0.1,
   gegen Wand/Rand → stehen bleiben), die Belohnung $R(s)$ (Living Reward,
   ±1-Terminals) und $\gamma$. Terminalzustände sind absorbierend.

2. **Value Iteration.** Implementiere den Bellman-Optimalitäts-Update
   $$V_{k+1}(s) \leftarrow R(s) + \gamma \max_a \sum_{s'} P(s'\mid s,a)\,V_k(s'),$$
   iteriere bis das **maximale Bellman-Residuum** $\lVert V_{k+1}-V_k\rVert_\infty$
   unter eine Schranke fällt, und zähle die Iterationen. Lies die **greedy Policy**
   $\pi^\ast(s)=\arg\max_a\sum_{s'}P(s'\mid s,a)V^\ast(s')$ ab.

3. **Policy Iteration.** Implementiere **Policy Evaluation** (löse bzw. iteriere
   $V^\pi(s)=R(s)+\gamma\sum_{s'}P(s'\mid s,\pi(s))V^\pi(s')$) und **Policy
   Improvement** (greedy bzgl. $V^\pi$); alterniere bis die Policy stabil ist.

4. **Analyse & Visualisierung.**
   - Gib $V^\ast$ als Zahlengitter und $\pi^\ast$ als **Pfeilgitter** aus.
   - Zeige, dass VI und PI **dieselbe** Policy und (bis auf Toleranz) dieselben
     Werte liefern, und vergleiche die **Iterationszahlen**.
   - **$\gamma$-Studie:** Berechne $\pi^\ast$ für mehrere $\gamma$ (z. B. 1.0, 0.9,
     0.5, 0.2) und beschreibe, wie sich die Policy ändert.
   - **Living-Reward-Studie:** Variiere $R$ (z. B. $-0{,}04, -0{,}5, -2{,}0, 0{,}0$)
     und erkläre das qualitative Verhalten (bei stark negativem $R$ nimmt der Agent
     riskante Abkürzungen sogar Richtung $-1$).

## Akzeptanzkriterien (Abnahmetest)

- [ ] Value Iteration reproduziert die bekannten AIMA-Utilities der 4×3-Welt
      (u. a. $V(0,0)\approx0{,}705$, $V(2,2)\approx0{,}918$, $V(3,0)\approx0{,}388$;
      Toleranz $<0{,}005$);
- [ ] die daraus abgelesene optimale Policy stimmt mit der klassischen Lösung
      überein (obere Reihe → → →, unten ↑ ← ← ←, mit ↑ neben der Wand);
- [ ] Policy Iteration liefert **dieselbe** Policy und Werte wie Value Iteration
      (und braucht dafür **deutlich weniger** äußere Iterationen);
- [ ] die $\gamma$- und Living-Reward-Studien zeigen nachvollziehbare
      Policy-Änderungen;
- [ ] VI meldet die Zahl der Iterationen bis zur Konvergenz (Referenz: 34 bei
      $\gamma=1$, Residuum $<10^{-8}$).

## Selbstcheck-Fragen (schriftlich beantworten)

1. **Warum konvergiert Value Iteration?** Skizziere das Kontraktionsargument
   (Bellman-Operator ist $\gamma$-Kontraktion in $\lVert\cdot\rVert_\infty$,
   Banachscher Fixpunktsatz). Warum konvergiert es in dieser Welt **auch bei
   $\gamma=1$** (Stichwort: absorbierende Terminals / proper policy)?
2. **Warum braucht Policy Iteration so viel weniger Iterationen** als Value
   Iteration, obwohl jede Iteration teurer ist?
3. **Wieso ändert kleineres $\gamma$ die Policy** in Richtung kürzerer/riskanterer
   Wege? Argumentiere über den diskontierten Return.
4. **Warum nimmt der Agent bei stark negativem Living Reward** Abkürzungen sogar in
   Richtung des $-1$-Terminals? Verbinde das mit dem Vorzeichen und der Größe von $R$.
5. **Wo genau ist die Brücke zum Reinforcement Learning (Modul 13)?** Was kennt ein
   RL-Agent *nicht*, das dein MDP-Agent hier voraussetzt, und wie ändert das das
   Vorgehen (Stichwort: Sampling / Temporal-Difference statt exaktem $P,R$)?

## Erweiterungen (optional, für Vertiefung)

- **Q-Werte & modifizierte Policy Iteration** (Policy Evaluation nur wenige Sweeps).
- **matplotlib-Heatmap** der Utilities mit überlagerten Policy-Pfeilen.
- **Konvergenzkurve** ($\lVert V_{k+1}-V_k\rVert_\infty$ über $k$, log-Skala) —
  bestätige die *geometrische* Rate $\gamma^k$.
- **Asynchrones/priorisiertes Value Iteration** oder eine größere, selbst
  entworfene Gridworld mit Fallen und mehreren Zielen.

## Musterlösung

In **`solution/`** liegt eine vollständige, getestete Referenzimplementierung:
- `gridworld.py` — die 4×3-MDP-Umgebung + ASCII-Visualisierung (Pfeile/Werte),
- `mdp.py` — Value Iteration, greedy Policy, Policy Iteration (mit Policy Evaluation),
- `demo.py` — Utilities/Policy, VI-vs-PI-Vergleich, $\gamma$- und Living-Reward-Studie,
- `test_mdp.py` — Abnahmetest gegen die AIMA-Referenzwerte.

Referenz: VI ~34 Iterationen ($\gamma=1$), PI ~5 Iterationen, **identische Policy**.
**Erst nach eigenem Versuch ansehen.**

```bash
source ../../../../.venv/bin/activate    # nur Standardbibliothek nötig
cd solution && python demo.py            # Utilities, Policy, gamma-/Reward-Studie
python test_mdp.py                       # Abnahmetest
```
