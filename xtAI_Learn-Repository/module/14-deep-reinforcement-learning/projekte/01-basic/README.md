# Projekt 01 (basic) — DQN von Grund auf: CartPole balancieren

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
[`loesung/dqn_cartpole_loesung.ipynb`](loesung/dqn_cartpole_loesung.ipynb). Erst selbst
probieren! Enthält vier Erweiterungs-Aufgaben (Target-Netz weglassen → Instabilität, **Double
DQN**, Netzgröße/ε-Decay variieren, Q-Werte auf Divergenz beobachten).

## Was du hier lernst (Transfer)

Das ist Q-Learning aus Modul 13 mit einem **Netz** statt der Tabelle — plus die zwei
Stabilitäts-Tricks (**Experience Replay**, **Target Network**), die die **deadly triad**
zähmen. Die zappelige Kurve und das *catastrophic forgetting* (die Best-Model-Checkpoints
fangen es ab) zeigen konkret, warum Deep RL als fragil gilt. Projekt 02 wechselt dann zur
**policy-basierten** Familie (REINFORCE / Actor-Critic).
