# Project 02 (medium) — Policy gradient: REINFORCE & actor-critic

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Format: Python project** (`.py` + tests). **Why?** Three algorithm variants that share an
environment and an experiment harness and that are secured by a test suite — that is structured
code, not an exploration notebook. The separation environment ↔ agent ↔ experiment corresponds to
real RL software.

---

## Goal

Project 01 was **value-based** (DQN learns $Q$, the policy follows greedily). Here you switch to
the **policy-based** family: the policy $\pi_\theta(a\mid s)$ is parametrized **directly** and
optimized by gradient ascent. You implement and compare three variants:

1. **REINFORCE without a baseline** — $\Psi_t = G_t$ (the full return-to-go). Unbiased, but
   **high variance**.
2. **REINFORCE with a baseline** — $G_t$ normalized. Still unbiased, but **markedly** more stable.
3. **Actor-critic (A2C)** — a **critic** $\hat V_\phi$ provides the baseline, $\Psi_t = G_t-\hat V(s_t)$
   (the **advantage**).

The guiding question: **how much does variance reduction help?**

## Prior knowledge

Script **module 14, section 3** (the policy-gradient theorem, REINFORCE, baseline, actor-critic).
Project 01 (PyTorch practice, CartPole). Module 13 (return, the Monte-Carlo recursion).

## Files

| File | Role |
|---|---|
| `cartpole.py` | The environment (like project 01). **Given** — infrastructure. |
| `policy_gradient.py` | Returns, REINFORCE, actor-critic. **This is your work** (5 TODOs). |
| `run.py` | Experiment: averages over 5 seeds, prints the comparison, draws the plot. |
| `test_pg.py` | Test suite (9 tests) — returns, distribution, update logic, ability to learn. |

## Assignment

In `policy_gradient.py` **five** places are marked with `# TODO`. Given are only the two MLP
classes and the constructors — the RL core is yours:

1. **`compute_returns(rewards, gamma)`** — return-to-go, backwards in $O(T)$ (the recursion
   $G = r + \gamma G$, like Monte Carlo in module 13).
2. **`REINFORCE.select_action`** — sample from $\pi(\cdot|s)$, return `(action, log_prob)`.
3. **`REINFORCE.update`** — form the returns, optionally normalize (baseline),
   $\text{loss}=-\sum_t \log\pi(a_t|s_t)\,\Psi_t$, an optimizer step.
4. **`ActorCritic.select_action`** — additionally return $\hat V(s)$.
5. **`ActorCritic.update`** — advantage $=(G-V)$**`.detach()`**, normalize, actor loss +
   `value_coef`·critic loss.

**Two pitfalls** the tests watch for:
- **Sign:** optimizers *minimize*, the policy gradient wants to *maximize* → `loss` is the
  **negative** of the objective.
- **`.detach()` on the advantage:** the advantage is only a *weight* for the actor — **no**
  gradient may flow through it back into the critic.

## What should work in the end

`python test_pg.py` → **all 9 tests green** (~3 s). `python run.py` → a comparison over **5 seeds**
(~60 s). Expected order of magnitude (episodes until "solved", i.e. the mean of the last 20 ≥ 475):

| Method | episodes until solved | solved seeds |
|---|---|---|
| REINFORCE (no baseline) | ~443 ± 81 | **3/5** |
| REINFORCE (with baseline) | **~211 ± 43** | 5/5 |
| actor-critic (A2C) | ~274 ± 116 | 5/5 |

**The core message:** the **baseline halves** the episodes to solution — and makes learning
**reliable** in the first place (5/5 instead of 3/5 seeds). Pure REINFORCE is, due to its high
variance, not only slow but **unreliable**. That is exactly what baselines and critics are for.

> **Honest observation:** A2C is *not* faster than REINFORCE-with-baseline here. On CartPole with
> full episodes the simple normalization is already a very strong baseline, while the critic is
> itself bad at the start and introduces **bias**. A2C's advantages (bootstrapping, online
> updates, no whole episodes needed) only pay off on longer/continuous tasks — see project 03.

> **Why 5 seeds?** Deep RL fluctuates massively over random seeds (script 5). A single run is
> **worthless** as a comparison — that is itself a lesson of this project.

## Running / setup

The repo `venv` (`torch`, `numpy`, `matplotlib`). From the project folder:

```bash
/.../xtAI_Learn-Repository/.venv/bin/python test_pg.py   # 9 tests, ~3 s
/.../xtAI_Learn-Repository/.venv/bin/python run.py       # 5-seed comparison + plot, ~60 s
```
CPU, small networks. `pytest` optional (the tests have a `__main__` runner).

## Solution

Complete in [`solution/`](solution/) (identical environment/tests, a solved `policy_gradient.py`).
Try it yourself first — the tests tell you fairly precisely which building block still snags.

## Thinking further

- **Entropy bonus:** add $-\beta\,\mathcal H(\pi)$ to the actor loss (script 4.2, the SAC idea) —
  prevents premature collapse onto one action.
- **GAE** (script 3.3): replace $G_t-\hat V$ by $\hat A_t^{\text{GAE}}=\sum_l(\gamma\lambda)^l\delta_{t+l}$
  and vary $\lambda$ — the bias/variance knob.
- **Baseline without normalization:** use only $G_t-\hat V(s_t)$ *without* standardization. How
  important is the scaling really?
- **PPO** (script 3.4): add the clipped objective with several epochs per batch.

---

# Projekt 02 (medium) — Policy Gradient: REINFORCE & Actor-Critic (deutsche Fassung)

**Format: Python-Projekt** (`.py` + Tests). **Warum?** Drei Algorithmen-Varianten, die sich eine
Umgebung und ein Experiment-Harness teilen und die per Test-Suite abgesichert werden — das ist
strukturierter Code, kein Explorations-Notebook. Die Trennung Umgebung ↔ Agent ↔ Experiment
entspricht echter RL-Software.

---

## Ziel

Projekt 01 war **wertbasiert** (DQN lernt $Q$, Policy folgt greedy). Hier wechselst du zur
**policy-basierten** Familie: die Policy $\pi_\theta(a\mid s)$ wird **direkt** parametrisiert und
per Gradientenaufstieg optimiert. Du implementierst und vergleichst drei Varianten:

1. **REINFORCE ohne Baseline** — $\Psi_t = G_t$ (voller Return-to-go). Unverzerrt, aber
   **hochvariant**.
2. **REINFORCE mit Baseline** — $G_t$ normalisiert. Immer noch unverzerrt, aber **deutlich**
   stabiler.
3. **Actor-Critic (A2C)** — ein **Critic** $\hat V_\phi$ liefert die Baseline, $\Psi_t = G_t-\hat V(s_t)$
   (**Advantage**).

Die Leitfrage: **Wie stark hilft Varianzreduktion?**

## Vorwissen

Skript **Modul 14, Abschnitt 3** (Policy-Gradient-Theorem, REINFORCE, Baseline, Actor-Critic).
Projekt 01 (PyTorch-Praxis, CartPole). Modul 13 (Return, Monte-Carlo-Rekursion).

## Dateien

| Datei | Rolle |
|---|---|
| `cartpole.py` | Die Umgebung (wie Projekt 01). **Vorgegeben** — Infrastruktur. |
| `policy_gradient.py` | Returns, REINFORCE, Actor-Critic. **Hier ist deine Arbeit** (5 TODOs). |
| `run.py` | Experiment: mittelt über 5 Seeds, druckt Vergleich, zeichnet den Plot. |
| `test_pg.py` | Test-Suite (9 Tests) — Returns, Verteilung, Update-Logik, Lernfähigkeit. |

## Aufgabe

In `policy_gradient.py` sind **fünf** Stellen mit `# TODO`. Vorgegeben sind nur die zwei
MLP-Klassen und die Konstruktoren — der RL-Kern ist deiner:

1. **`compute_returns(rewards, gamma)`** — Return-to-go, rückwärts in $O(T)$ (Rekursion
   $G = r + \gamma G$, wie Monte Carlo in Modul 13).
2. **`REINFORCE.select_action`** — aus $\pi(\cdot|s)$ sampeln, `(action, log_prob)` zurückgeben.
3. **`REINFORCE.update`** — Returns bilden, optional normalisieren (Baseline),
   $\text{loss}=-\sum_t \log\pi(a_t|s_t)\,\Psi_t$, Optimierer-Schritt.
4. **`ActorCritic.select_action`** — zusätzlich $\hat V(s)$ zurückgeben.
5. **`ActorCritic.update`** — Advantage $=(G-V)$**`.detach()`**, normalisieren, Actor-Loss +
   `value_coef`·Critic-Loss.

**Zwei Fallen**, auf die die Tests achten:
- **Vorzeichen:** Optimierer *minimieren*, der Policy-Gradient will *maximieren* → `loss` ist
  das **Negative** des Ziels.
- **`.detach()` beim Advantage:** der Advantage ist für den Actor nur ein *Gewicht* — durch ihn
  darf **kein** Gradient in den Critic zurückfließen.

## Was am Ende funktionieren soll

`python test_pg.py` → **alle 9 Tests grün** (~3 s). `python run.py` → Vergleich über **5 Seeds**
(~60 s). Erwartete Größenordnung (Episoden bis „gelöst", d. h. Mittel der letzten 20 ≥ 475):

| Verfahren | Episoden bis gelöst | gelöste Seeds |
|---|---|---|
| REINFORCE (ohne Baseline) | ~443 ± 81 | **3/5** |
| REINFORCE (mit Baseline) | **~211 ± 43** | 5/5 |
| Actor-Critic (A2C) | ~274 ± 116 | 5/5 |

**Die Kernbotschaft:** Die **Baseline halbiert** die Episoden bis zur Lösung — und macht das
Lernen überhaupt erst **zuverlässig** (5/5 statt 3/5 Seeds). Reines REINFORCE ist wegen seiner
hohen Varianz nicht nur langsam, sondern **unzuverlässig**. Genau dafür gibt es Baselines und
Critics.

> **Ehrliche Beobachtung:** A2C ist hier *nicht* schneller als REINFORCE-mit-Baseline. Auf
> CartPole mit vollen Episoden ist die simple Normalisierung schon eine sehr starke Baseline,
> während der Critic anfangs selbst schlecht ist und **Bias** einbringt. A2Cs Vorteile
> (Bootstrapping, Online-Updates, keine ganzen Episoden nötig) zahlen sich erst bei längeren/
> kontinuierlichen Aufgaben aus — siehe Projekt 03.

> **Warum 5 Seeds?** Deep RL schwankt massiv über Random-Seeds (Skript 5). Ein Einzellauf ist
> **wertlos** als Vergleich — das ist selbst eine Lektion dieses Projekts.

## Ausführen / Setup

Repo-`venv` (`torch`, `numpy`, `matplotlib`). Aus dem Projektordner:

```bash
/.../xtAI_Learn-Repository/.venv/bin/python test_pg.py   # 9 Tests, ~3 s
/.../xtAI_Learn-Repository/.venv/bin/python run.py       # 5-Seed-Vergleich + Plot, ~60 s
```
CPU, kleine Netze. `pytest` optional (Tests haben `__main__`-Runner).

## Lösung

Vollständig in [`solution/`](solution/) (identische Umgebung/Tests, gelöste `policy_gradient.py`).
Erst selbst versuchen — die Tests sagen dir ziemlich genau, welcher Baustein noch hakt.

## Weiterdenken

- **Entropie-Bonus:** addiere $-\beta\,\mathcal H(\pi)$ zum Actor-Loss (Skript 4.2, SAC-Idee) —
  verhindert vorzeitiges Kollabieren auf eine Aktion.
- **GAE** (Skript 3.3): ersetze $G_t-\hat V$ durch $\hat A_t^{\text{GAE}}=\sum_l(\gamma\lambda)^l\delta_{t+l}$
  und variiere $\lambda$ — der Bias/Varianz-Regler.
- **Baseline ohne Normalisierung:** nutze nur $G_t-\hat V(s_t)$ *ohne* Standardisierung. Wie
  wichtig ist das Skalieren wirklich?
- **PPO** (Skript 3.4): ergänze das geklippte Ziel mit mehreren Epochen pro Batch.
