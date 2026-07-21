# Projekt 02 (medium) — Policy Gradient: REINFORCE & Actor-Critic

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
