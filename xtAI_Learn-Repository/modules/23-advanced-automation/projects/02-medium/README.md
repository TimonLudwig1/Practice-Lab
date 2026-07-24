# P02 (medium) — Petri nets & supervisory control: deadlock detection and supervisor synthesis

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 23 — Advanced Automation** · Format: **Python module + test suite**

## Goal

You model a **resource-sharing manufacturing cell** as a **Petri net**, find where it **deadlocks**, and **synthesise a supervisor** that provably prevents the deadlock while keeping as much concurrency as safely possible — the core of discrete automation safety.

1. the Petri-net **state equation** $\mathbf M' = \mathbf M + \mathbf C\,\mathbf u$ and the **reachability graph** (script ch. 4),
2. **deadlock** and **blocking** detection on the reachability graph,
3. **supervisor synthesis** by the Ramadge–Wonham **forbidden-state backward fixed-point** (script ch. 5), respecting **controllable vs. uncontrollable** events.

## Why this format?

A **Python module with a test suite** — the net, the reachability graph and the supervisor are exact, testable objects (the deadlock is a specific marking; the supervised system must have zero deadlocks and still reach the goal), and the experiments vary the controllable/uncontrollable split.

## Why synthetic data?

The cell is a **model** (a Petri net), not a dataset — the classic two-jobs/two-resources circular-wait deadlock, minimal but faithful, lets every property be computed exactly and the Ramadge–Wonham subtlety be exhibited by flipping one event's controllability.

## Prior knowledge

**P01** of this module (automata, reachability), the module 23 script ch. 4–5, linear algebra (the incidence matrix), graph search.

## Task

Open `petri.py`. The net data structure, firing semantics (`enabled`, `fire`) and the example cell (`make_cell`) are given — you implement the **three core functions** (`# TODO` / `NotImplementedError`):

1. **`reachability_graph(net)`** — BFS from $\mathbf M_0$ firing every enabled transition; return the states and edges.
2. **`find_deadlocks(net, states)`** — the reachable markings with no enabled transition (excluding the goal).
3. **`synthesize_supervisor(net, states, edges, forbidden)`** — grow the bad set backward over **uncontrollable** edges to a fixed point, then disable the **controllable** edges from safe states into it.

Then:

```bash
cd modules/23-advanced-automation/projects/02-medium
/Users/.../.venv/bin/python test_petri.py   # 6 tests -> all PASS
/Users/.../.venv/bin/python run.py           # 3 experiments + plot
```

## What should come out (expected values)

**Experiment 1 — the plant.** Two jobs, two shared resources R1/R2: **9 reachable markings, 10 edges**, the goal reachable, and exactly **1 deadlock** — the marking `{A_r1:1, B_r2:1}` (A holds R1, B holds R2; `tA2` needs R2 and `tB2` needs R1, both held by the other job). The classic circular wait.

**Experiment 2 — supervisor synthesis.** The forbidden set (deadlock + blocking) has 1 state; the backward fixed-point leaves the bad set at **1** (its predecessors are reached by controllable events). The supervisor disables **2 controllable transitions**: `tB1` when A already holds R1, and `tA1` when B already holds R2 — exactly the two "second grab" moves that close the circular wait. Result: **8 of 9** states remain reachable, **0 deadlocks**, goal still reachable — **maximally permissive** (it removes only the deadlock).

**Experiment 3 — uncontrollable events (Ramadge–Wonham).** Make `tB1` uncontrollable (B grabs R2 on its own). Now the bad set grows backward to **2** (the state where A can still grab R1 becomes unsafe, because B might then *autonomously* deadlock it), and the supervisor must act earlier: only **5 of 9** states remain reachable (vs. 8/9 when everything is controllable).

| controllability | bad set | disabled | supervised states |
|---|---|---|---|
| all acquires controllable | 1 | 2 | 8 / 9 |
| `tB1` uncontrollable | 2 | 2 | 5 / 9 |

> **The lesson.** A Petri net models concurrency and shared resources compactly, and its **reachability graph** exposes the safety failure of resource sharing — **deadlock**, the circular wait. The **supervisor** is computed, not guessed: a backward fixed-point marks every state from which badness is unavoidable, and the controller disables exactly the controllable moves into that set — the **least restrictive** safe controller. The decisive subtlety is the **controllable/uncontrollable split**: an event the automation cannot prevent forces the supervisor to be conservative *earlier*, shrinking the safe behaviour. That asymmetry is the whole content of Ramadge–Wonham supervisory control.

## Solution

The complete reference is in [`solution/`](solution/). Try it yourself first!

## What comes next

**P03 (final)**: **Model Predictive Control** — the continuous layer's optimising controller. Solve a constrained optimal-control problem every step (a condensed QP), respect hard input/state limits where LQR would saturate, and verify that unconstrained MPC *equals* the LQR of module 14. No code given.

---
---

# P02 (medium) — Petri-Netze & Supervisory Control: Deadlock-Erkennung und Supervisor-Synthese (deutsche Fassung)

**Modul 23 — Advanced Automation** · Format: **Python-Modul + Testsuite**

## Ziel

Du modellierst eine **ressourcenteilende Fertigungszelle** als **Petri-Netz**, findest, wo sie **verklemmt (Deadlock)**, und **synthetisierst einen Supervisor**, der den Deadlock beweisbar verhindert und dabei so viel Nebenläufigkeit wie sicher möglich erhält — der Kern der diskreten Automatisierungssicherheit.

1. die Petri-Netz-**Zustandsgleichung** $\mathbf M' = \mathbf M + \mathbf C\,\mathbf u$ und den **Erreichbarkeitsgraphen** (Skript Kap. 4),
2. **Deadlock**- und **Blocking**-Erkennung auf dem Erreichbarkeitsgraphen,
3. **Supervisor-Synthese** per Ramadge–Wonham-**Forbidden-State-Rückwärts-Fixpunkt** (Skript Kap. 5), unter Beachtung **steuerbarer vs. nicht-steuerbarer** Ereignisse.

## Warum dieses Format?

Ein **Python-Modul mit Testsuite** — das Netz, der Erreichbarkeitsgraph und der Supervisor sind exakte, testbare Objekte (der Deadlock ist eine bestimmte Markierung; das überwachte System muss null Deadlocks haben und das Ziel noch erreichen), und die Experimente variieren die steuerbar/nicht-steuerbar-Teilung.

## Warum synthetische Daten?

Die Zelle ist ein **Modell** (ein Petri-Netz), kein Datensatz — der klassische Zwei-Jobs/zwei-Ressourcen-Zirkulär-Warte-Deadlock, minimal aber treu, lässt jede Eigenschaft exakt berechnen und die Ramadge–Wonham-Subtilität durch Umschalten der Steuerbarkeit eines Ereignisses zeigen.

## Vorwissen

**P01** dieses Moduls (Automaten, Erreichbarkeit), das Modul-23-Skript Kap. 4–5, lineare Algebra (die Inzidenzmatrix), Graphsuche.

## Aufgabenstellung

Öffne `petri.py`. Die Netz-Datenstruktur, die Feuersemantik (`enabled`, `fire`) und die Beispielzelle (`make_cell`) sind vorgegeben — du implementierst die **drei Kernfunktionen** (`# TODO` / `NotImplementedError`):

1. **`reachability_graph(net)`** — BFS von $\mathbf M_0$, jede aktivierte Transition feuern; Zustände und Kanten zurückgeben.
2. **`find_deadlocks(net, states)`** — die erreichbaren Markierungen ohne aktivierte Transition (außer dem Ziel).
3. **`synthesize_supervisor(net, states, edges, forbidden)`** — die schlechte Menge rückwärts über **nicht-steuerbare** Kanten bis zum Fixpunkt wachsen, dann die **steuerbaren** Kanten von sicheren Zuständen hinein deaktivieren.

Dann:

```bash
cd modules/23-advanced-automation/projects/02-medium
/Users/.../.venv/bin/python test_petri.py   # 6 Tests -> alle PASS
/Users/.../.venv/bin/python run.py           # 3 Experimente + Plot
```

## Was am Ende herauskommt (Erwartungswerte)

**Experiment 1 — die Anlage.** Zwei Jobs, zwei geteilte Ressourcen R1/R2: **9 erreichbare Markierungen, 10 Kanten**, das Ziel erreichbar, und genau **1 Deadlock** — die Markierung `{A_r1:1, B_r2:1}` (A hält R1, B hält R2; `tA2` braucht R2 und `tB2` braucht R1, beide vom anderen Job gehalten). Das klassische zirkuläre Warten.

**Experiment 2 — Supervisor-Synthese.** Die verbotene Menge (Deadlock + Blocking) hat 1 Zustand; der Rückwärts-Fixpunkt lässt die schlechte Menge bei **1** (ihre Vorgänger werden über steuerbare Ereignisse erreicht). Der Supervisor deaktiviert **2 steuerbare Transitionen**: `tB1`, wenn A bereits R1 hält, und `tA1`, wenn B bereits R2 hält — genau die zwei „zweiter Griff"-Züge, die das zirkuläre Warten schließen. Ergebnis: **8 von 9** Zuständen bleiben erreichbar, **0 Deadlocks**, Ziel weiter erreichbar — **maximal permissiv** (entfernt nur den Deadlock).

**Experiment 3 — nicht-steuerbare Ereignisse (Ramadge–Wonham).** Mache `tB1` nicht-steuerbar (B greift R2 von selbst). Nun wächst die schlechte Menge rückwärts auf **2** (der Zustand, in dem A noch R1 greifen kann, wird unsicher, weil B ihn dann *autonom* verklemmen könnte), und der Supervisor muss früher handeln: nur **5 von 9** Zuständen bleiben erreichbar (gegen 8/9 bei voller Steuerbarkeit).

| Steuerbarkeit | schlechte Menge | deaktiviert | überwachte Zustände |
|---|---|---|---|
| alle Griffe steuerbar | 1 | 2 | 8 / 9 |
| `tB1` nicht-steuerbar | 2 | 2 | 5 / 9 |

> **Die Lehre.** Ein Petri-Netz modelliert Nebenläufigkeit und geteilte Ressourcen kompakt, und sein **Erreichbarkeitsgraph** legt den Sicherheitsfehler des Ressourcenteilens offen — **Deadlock**, das zirkuläre Warten. Der **Supervisor** wird berechnet, nicht geraten: Ein Rückwärts-Fixpunkt markiert jeden Zustand, von dem Schlechtigkeit unvermeidbar ist, und der Regler deaktiviert genau die steuerbaren Züge in diese Menge — der **am wenigsten restriktive** sichere Regler. Die entscheidende Subtilität ist die **steuerbar/nicht-steuerbar-Teilung**: Ein Ereignis, das die Automatisierung nicht verhindern kann, zwingt den Supervisor, *früher* konservativ zu sein, und schrumpft das sichere Verhalten. Diese Asymmetrie ist der ganze Inhalt der Ramadge–Wonham-Supervisory-Control.

## Lösung

Die vollständige Referenz liegt in [`solution/`](solution/). Erst selbst versuchen!

## Weiter geht's

**P03 (final)**: **Model Predictive Control** — der optimierende Regler der kontinuierlichen Ebene. Löse bei jedem Schritt ein beschränktes Optimalregelungsproblem (ein kondensiertes QP), respektiere harte Eingangs-/Zustandsgrenzen, wo LQR sättigen würde, und verifiziere, dass unbeschränktes MPC *gleich* dem LQR aus Modul 14 ist. Keine Code-Vorgabe.
