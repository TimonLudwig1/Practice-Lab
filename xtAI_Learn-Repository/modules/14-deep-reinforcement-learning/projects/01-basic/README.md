# Project 01 (basic) — DQN from scratch: balancing CartPole

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format: Jupyter notebook (PyTorch)** (`dqn_cartpole.ipynb`). **Why?** Deep RL lives off
*watching* — jittery learning curves, a pole that eventually stays up. A notebook connects code,
training and visualization ideally for your first own **deep Q-network**.

---

## Goal

You build a **DQN** (Mnih et al. 2015, in miniature) and teach an agent to solve the
**CartPole** task: balancing a pole on a movable cart upright through left/right pushes. We build
the physics environment **ourselves** — without `gym`.

You implement the **two conceptual core pieces**; everything else (physics, replay buffer, the
network, the training loop with a best-model checkpoint, plots) is given:

1. **ε-greedy action selection** (`select_action`) — as in module 13, only with the network
   instead of the table.
2. **The DQN target & loss** (`learn`): $y = r + \gamma(1-\text{done})\max_{a'}Q(s',a';\theta^-)$
   with the **target network** $\theta^-$, and a Huber loss against the prediction
   $Q(s,a;\theta)$.

## Prior knowledge

Script **module 14, section 2** (DQN, experience replay, target network). Module 13 (Q-learning,
ε-greedy). Module 05 (PyTorch: `nn.Module`, Adam, backprop).

## Assignment (step by step)

In the notebook the environment, the network and the replay buffer are given. In the class
`DQNAgent`:

1. **`select_action(state)`** — with probability $\varepsilon$ random, otherwise
   $\arg\max_a Q(s,a;\theta)$ (`with torch.no_grad()`).
2. **`learn()`** — fill in three lines: the prediction `q_sa` (via `gather`), the target `y`
   (with `self.qt` = the target network, in the `no_grad` block), and `loss` (`smooth_l1_loss`).

Then: run the training cell, interpret the learning curve and the **greedy evaluation**.

## What should work in the end

- A learning curve that rises from ~15 steps to several hundred (not monotonic — that is normal!).
- The **greedy evaluation** reaches on average close to **500** steps (the episode maximum) — the
  pole stays up. In our case: "solved" after ~160 episodes, greedy mean ~478/500.
- **Runtime: a few seconds up to ~30 s** on the CPU (a small network, early stop on solving).

## Running / setup

The repo `venv` (contains `torch` 2.12). Open the notebook in Jupyter:
`/.../xtAI_Learn-Repository/.venv/bin/python -m jupyter lab` (or the `.venv` kernel in VS Code).
Only `torch`, `numpy`, `matplotlib`. By default **CPU** — for such small networks faster and more
reproducible than MPS.

## Solution

Fully solved and **executed** in
[`solution/dqn_cartpole_solution.ipynb`](solution/dqn_cartpole_solution.ipynb). Try it yourself
first! It contains four extension tasks (drop the target network → instability, **double DQN**,
vary the network size/ε-decay, watch the Q values for divergence).

## What you learn here (transfer)

This is Q-learning from module 13 with a **network** instead of the table — plus the two
stability tricks (**experience replay**, **target network**) that tame the **deadly triad**. The
jittery curve and the *catastrophic forgetting* (the best-model checkpoints catch it) show
concretely why deep RL is considered fragile. Project 02 then switches to the **policy-based**
family (REINFORCE / actor-critic).

---

# Projekt 01 (basic) — DQN von Grund auf: CartPole balancieren (deutsche Fassung)

**Format: Jupyter Notebook (PyTorch)** (`dqn_cartpole.ipynb`). **Warum?** Deep RL lebt vom
*Beobachten* — zappelnde Lernkurven, ein Pol, der irgendwann stehen bleibt. Ein Notebook
verbindet Code, Training und Visualisierung ideal für den ersten eigenen **Deep-Q-Network**.

---

## Ziel

Du baust ein **DQN** (Mnih et al. 2015, im Kleinformat) und bringst einem Agenten bei, die
**CartPole**-Aufgabe zu lösen: einen Stab auf einem fahrbaren Wagen durch Links/Rechts-Schübe
aufrecht zu balancieren. Die Physik-Umgebung bauen wir **selbst** — ohne `gym`.

Du implementierst die **zwei konzeptuellen Kernstücke**; alles andere (Physik, Replay-Puffer,
Netz, Trainingsschleife mit Best-Model-Checkpoint, Plots) ist vorgegeben:

1. **ε-greedy-Aktionswahl** (`select_action`) — wie in Modul 13, nur mit dem Netz statt der Tabelle.
2. **DQN-Ziel & Verlust** (`learn`): $y = r + \gamma(1-\text{done})\max_{a'}Q(s',a';\theta^-)$ mit
   dem **Target-Netz** $\theta^-$, und Huber-Verlust gegen die Vorhersage $Q(s,a;\theta)$.

## Vorwissen

Skript **Modul 14, Abschnitt 2** (DQN, Experience Replay, Target Network). Modul 13 (Q-Learning,
ε-greedy). Modul 05 (PyTorch: `nn.Module`, Adam, Backprop).

## Aufgabe (Schritt für Schritt)

Im Notebook sind Umgebung, Netz und Replay-Puffer vorgegeben. In der Klasse `DQNAgent`:

1. **`select_action(state)`** — mit Wkt. $\varepsilon$ zufällig, sonst $\arg\max_a Q(s,a;\theta)$
   (`with torch.no_grad()`).
2. **`learn()`** — fülle drei Zeilen: Vorhersage `q_sa` (per `gather`), Ziel `y` (mit
   `self.qt` = Target-Netz, im `no_grad`-Block), und `loss` (`smooth_l1_loss`).

Danach: Trainingszelle laufen lassen, Lernkurve und **greedy-Auswertung** interpretieren.

## Was am Ende funktionieren soll

- Eine Lernkurve, die von ~15 Schritten auf mehrere Hundert steigt (nicht monoton — das ist
  normal!).
- Die **greedy-Auswertung** erreicht im Mittel nahe **500** Schritte (Episoden-Maximum) — der
  Pol bleibt stehen. Bei uns: „Gelöst" nach ~160 Episoden, greedy-Mittel ~478/500.
- **Laufzeit: wenige Sekunden bis ~30 s** auf der CPU (kleines Netz, Early-Stop bei Lösung).

## Ausführen / Setup

Repo-`venv` (enthält `torch` 2.12). Notebook in Jupyter öffnen:
`/.../xtAI_Learn-Repository/.venv/bin/python -m jupyter lab` (oder `.venv`-Kernel in VS Code).
Nur `torch`, `numpy`, `matplotlib`. Standardmäßig **CPU** — für so kleine Netze schneller und
reproduzierbarer als MPS.

## Lösung

Vollständig gelöst und **ausgeführt** in
[`solution/dqn_cartpole_solution.ipynb`](solution/dqn_cartpole_solution.ipynb). Erst selbst
probieren! Enthält vier Erweiterungs-Aufgaben (Target-Netz weglassen → Instabilität, **Double
DQN**, Netzgröße/ε-Decay variieren, Q-Werte auf Divergenz beobachten).

## Was du hier lernst (Transfer)

Das ist Q-Learning aus Modul 13 mit einem **Netz** statt der Tabelle — plus die zwei
Stabilitäts-Tricks (**Experience Replay**, **Target Network**), die die **deadly triad**
zähmen. Die zappelige Kurve und das *catastrophic forgetting* (die Best-Model-Checkpoints
fangen es ab) zeigen konkret, warum Deep RL als fragil gilt. Projekt 02 wechselt dann zur
**policy-basierten** Familie (REINFORCE / Actor-Critic).
