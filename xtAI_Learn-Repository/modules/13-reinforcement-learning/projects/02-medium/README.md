# Project 02 (medium) — SARSA vs. Q-learning on cliff walking

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Format: Python project** (`.py` + tests). **Why?** Here it is about a real environment,
reusable agents and a test suite that *secures* the behavior — that is structured code, not an
exploration notebook. Exactly the separation environment ↔ agent ↔ training loop that every RL
codebase (and `gymnasium`) has, you build here yourself — **without** an external RL library.

---

## Goal

You implement the two central **model-free control algorithms** and bring out their difference
on the canonical example **cliff walking** (Sutton & Barto, example 6.6):

- **SARSA** — *on-policy*: learns the value of the policy it **executes** (including exploration).
- **Q-learning** — *off-policy*: learns directly the **optimal** policy $Q_*$, while following
  an exploring policy.

The **one** difference is in the bootstrapping target of the TD update — nothing more. That is
exactly what you should see and understand.

## Prior knowledge

Script **module 13, section 2.4–2.5** (SARSA, Q-learning, GPI, ε-greedy). Project 01 (ε-greedy,
the incremental update). Basic Python (classes, NumPy).

## Files

| File | Role |
|---|---|
| `cliff_walking.py` | The environment (4×12 grid with a cliff). **Given** — infrastructure. |
| `td_control.py` | The agent + training loop. **This is your work** (3 TODOs). |
| `run.py` | Experiment: averages over many runs, prints results, draws the plot. |
| `test_td.py` | Test suite (10 tests) — secures the environment *and* the agent. |

## Assignment (step by step)

In `td_control.py` **three** places are marked with `# TODO`:

1. **`TDAgent.select_action`** — ε-greedy: with probability ε random, otherwise
   $\arg\max_a Q(s,a)$ (with fair tie-breaking).
2. **`TDAgent.update`** — the TD update. Case distinction:
   - `done` → target $=r$;
   - `sarsa` → target $=r+\gamma\,Q(s',a')$;
   - `qlearning` → target $=r+\gamma\max_a Q(s',a)$;
   - then $Q(s,a)\mathrel{+}=\alpha\,(\text{target}-Q(s,a))$.
3. **`train`** — the episode loop. **Attention SARSA:** the next action $a'$ must be drawn from
   $s'$ *before* the update and actually executed in the next step (the
   $(s,a)\!\to\!(s',a')$ transition). The docstring sketches the flow.

Then: run `python run.py` and interpret the two greedy policies + the learning curve.

## What should work in the end

`python test_td.py` → **all 10 tests green**. `python run.py` reproduces the textbook figure:

```
algorithm     online return (last 100 ep.)   greedy return
sarsa                 ~ -27                          -17
qlearning             ~ -52                          -13
```

- **Q-learning** finds the **optimal, risky** route right along the cliff edge (greedy return
  **−13**) — but loses more reward during training because ε-exploration occasionally makes it
  fall off (online ~ −52).
- **SARSA** learns a **safer** route with distance to the cliff (greedy return **−17**) and in
  return achieves **online** markedly more (~ −27), because it *prices in* the exploration costs.

The greedy policies (an ASCII map) show it directly: Q-learning walks the row *right above* the
cliff, SARSA dodges upwards.

> **This is the core message of the module:** "optimal in the limit" (Q-learning) ≠ "good while
> still exploring and acting" (SARSA). *Off-policy* learns the best; *on-policy* learns the best
> **taking its own exploration behavior into account**.

## Running / setup

Repo `venv`, only `numpy` (+ `matplotlib` optional for the plot). From the project folder:

```bash
/.../xtAI_Learn-Repository/.venv/bin/python test_td.py   # tests
/.../xtAI_Learn-Repository/.venv/bin/python run.py        # comparison + plot (cliff_comparison.png)
```

Runs in **seconds** on the CPU (tabular). `pytest` is optional — the tests have their own
`__main__` runner.

## Solution

Complete in [`solution/`](solution/) (identical environment/tests, a solved `td_control.py`).
Try it yourself first! If a test snags, its message usually says exactly which case (target,
terminal case, tie-breaking, SARSA-vs-Q) is not yet right.

## Thinking further

- Set `epsilon=0` in `run.py` — both algorithms collapse to the same greedy route. Why?
  (Without exploration the online difference vanishes — SARSA *is* Q-learning then.)
- Add **expected SARSA** (target $=r+\gamma\sum_{a'}\pi(a'|s')Q(s',a')$) and show that for a
  greedy $\pi$ it becomes Q-learning (script 2.4).
- Let $\varepsilon$ decay over the episodes ($\varepsilon_k\propto1/k$, GLIE) — does SARSA then
  also converge to the optimal route?

---

# Projekt 02 (medium) — SARSA vs. Q-Learning auf Cliff Walking (deutsche Fassung)

**Format: Python-Projekt** (`.py` + Tests). **Warum?** Hier geht es um eine echte
Umgebung, wiederverwendbare Agenten und eine Test-Suite, die das Verhalten *absichert* — das
ist strukturierter Code, kein Explorations-Notebook. Genau die Trennung
Umgebung ↔ Agent ↔ Trainingsschleife, die jede RL-Codebasis (und `gymnasium`) hat, baust du
hier selbst — **ohne** externe RL-Bibliothek.

---

## Ziel

Du implementierst die beiden zentralen **modellfreien Kontroll-Algorithmen** und stellst ihren
Unterschied am kanonischen Beispiel **Cliff Walking** (Sutton & Barto, Beispiel 6.6) heraus:

- **SARSA** — *on-policy*: lernt den Wert der Policy, die es **ausführt** (inkl. Exploration).
- **Q-Learning** — *off-policy*: lernt direkt die **optimale** Policy $Q_*$, während es einer
  explorierenden Policy folgt.

Der **eine** Unterschied steckt im Bootstrapping-Ziel des TD-Updates — mehr nicht. Genau das
sollst du sehen und verstehen.

## Vorwissen

Skript **Modul 13, Abschnitt 2.4–2.5** (SARSA, Q-Learning, GPI, ε-greedy). Projekt 01 (ε-greedy,
inkrementelles Update). Grundlegendes Python (Klassen, NumPy).

## Dateien

| Datei | Rolle |
|---|---|
| `cliff_walking.py` | Die Umgebung (4×12-Gitter mit Klippe). **Vorgegeben** — Infrastruktur. |
| `td_control.py` | Der Agent + Trainingsschleife. **Hier ist deine Arbeit** (3 TODOs). |
| `run.py` | Experiment: mittelt über viele Läufe, druckt Ergebnisse, zeichnet den Plot. |
| `test_td.py` | Test-Suite (10 Tests) — sichert Umgebung *und* Agenten ab. |

## Aufgabe (Schritt für Schritt)

In `td_control.py` sind **drei** Stellen mit `# TODO` markiert:

1. **`TDAgent.select_action`** — ε-greedy: mit Wkt. ε zufällig, sonst $\arg\max_a Q(s,a)$
   (mit fairem Tie-Breaking).
2. **`TDAgent.update`** — das TD-Update. Fallunterscheidung:
   - `done` → Ziel $=r$;
   - `sarsa` → Ziel $=r+\gamma\,Q(s',a')$;
   - `qlearning` → Ziel $=r+\gamma\max_a Q(s',a)$;
   - dann $Q(s,a)\mathrel{+}=\alpha\,(\text{Ziel}-Q(s,a))$.
3. **`train`** — die Episoden-Schleife. **Achtung SARSA:** die nächste Aktion $a'$ muss *vor*
   dem Update aus $s'$ gezogen und im nächsten Schritt tatsächlich ausgeführt werden (der
   $(s,a)\!\to\!(s',a')$-Übergang). Der Docstring skizziert den Ablauf.

Danach: `python run.py` ausführen und die zwei greedy-Policies + die Lernkurve interpretieren.

## Was am Ende funktionieren soll

`python test_td.py` → **alle 10 Tests grün**. `python run.py` reproduziert die Lehrbuch-Figur:

```
Algorithmus   Online-Ertrag (letzte 100 Ep.)   greedy-Ertrag
sarsa                 ~ -27                          -17
qlearning             ~ -52                          -13
```

- **Q-Learning** findet die **optimale, riskante** Route direkt an der Klippenkante
  (greedy-Ertrag **−13**) — verliert aber während des Trainings mehr Belohnung, weil
  ε-Exploration es gelegentlich abstürzen lässt (Online ~ −52).
- **SARSA** lernt eine **sicherere** Route mit Abstand zur Klippe (greedy-Ertrag **−17**) und
  erzielt dafür **online** deutlich mehr (~ −27), weil es die Explorationskosten *einpreist*.

Die greedy-Policies (ASCII-Karte) zeigen es direkt: Q-Learning geht die Reihe *direkt über* der
Klippe entlang, SARSA weicht nach oben aus.

> **Das ist die Kernbotschaft des Moduls:** „optimal im Grenzwert" (Q-Learning) ≠ „gut, während
> man noch exploriert und handelt" (SARSA). *Off-policy* lernt das Beste; *on-policy* lernt das
> Beste **unter Berücksichtigung des eigenen Explorationsverhaltens**.

## Ausführen / Setup

Repo-`venv`, nur `numpy` (+ `matplotlib` optional für den Plot). Aus dem Projektordner:

```bash
/.../xtAI_Learn-Repository/.venv/bin/python test_td.py   # Tests
/.../xtAI_Learn-Repository/.venv/bin/python run.py        # Vergleich + Plot (cliff_comparison.png)
```

Läuft in **Sekunden** auf der CPU (tabellarisch). `pytest` ist optional — die Tests haben einen
eigenen `__main__`-Runner.

## Lösung

Vollständig in [`solution/`](solution/) (identische Umgebung/Tests, gelöste `td_control.py`).
Erst selbst versuchen! Wenn ein Test hakt, sagt seine Meldung meist genau, welcher Fall
(Ziel, Terminalfall, Tie-Breaking, SARSA-vs-Q) noch nicht stimmt.

## Weiterdenken

- Setze `epsilon=0` in `run.py` — beide Algorithmen kollabieren zur gleichen greedy-Route.
  Warum? (Ohne Exploration verschwindet der Online-Unterschied — SARSA *ist* dann Q-Learning.)
- Ergänze **Expected SARSA** (Ziel $=r+\gamma\sum_{a'}\pi(a'|s')Q(s',a')$) und zeige, dass es
  bei greedy $\pi$ zu Q-Learning wird (Skript 2.4).
- Lass $\varepsilon$ über die Episoden abfallen ($\varepsilon_k\propto1/k$, GLIE) — konvergiert
  SARSA dann auch zur optimalen Route?
