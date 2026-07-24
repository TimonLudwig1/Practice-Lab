# P03 (final) — Model Predictive Control: constrained optimal control by a condensed QP

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 23 — Advanced Automation** · Format: **Python project (free implementation, no code given)**

> Final project. **No code is given** — you build the MPC controller yourself. The reference solution is in [`solution/`](solution/); **try it yourself first**. This README is the specification.

## What it is about

The continuous layer of automation needs a controller that respects **hard limits** — actuators saturate, speeds and temperatures must stay in band. **Model Predictive Control (MPC)** is the modern workhorse: at every step it solves a finite-horizon constrained optimal-control problem, applies the first input, and re-optimises (receding horizon). You build it from scratch for an automated positioning stage (a double integrator), **condense** the problem into a **quadratic program**, and demonstrate the two facts that define MPC: **unconstrained MPC equals the LQR of module 14**, and **MPC honours constraints that LQR simply violates**.

## Learning objective

You derive and implement the **condensation** ($\mathbf X=\mathbf S_x\mathbf x_0+\mathbf S_u\mathbf U$ → a QP in $\mathbf U$), the **receding-horizon** loop, the **terminal cost** (and why it makes MPC exact), and **input and state constraints** — and you verify each claim numerically against the LQR baseline.

## Prior knowledge

The module 23 script ch. 6, **LQR / the discrete Riccati equation (module 14)**, linear systems, quadratic programming. `scipy.linalg.solve_discrete_are` for the LQR baseline and `scipy.optimize` for the QP.

---

## Task (specification)

### 1. Plant and LQR baseline

- A **double integrator** (automated positioning stage): state $(x,y)=$ (position, velocity), input = force; discretise with $\Delta t=0.1$. Weights $\mathbf Q=\mathrm{diag}(1,0.1)$, $\mathbf R=[0.1]$.
- The **LQR** gain and cost-to-go from the discrete algebraic Riccati equation — the unconstrained optimum and the reference to beat.

### 2. Condensation into a QP

- **Prediction matrices** $\mathbf S_x,\mathbf S_u$ from powers of $\mathbf A,\mathbf B$ so that $\mathbf X=\mathbf S_x\mathbf x_0+\mathbf S_u\mathbf U$ (stacked predicted states as an affine function of the stacked inputs). **Verify** they reproduce a plain forward simulation.
- Build the QP $\min_{\mathbf U}\tfrac12\mathbf U^\top\mathbf H\mathbf U+\mathbf x_0^\top\mathbf F^\top\mathbf U$ with $\mathbf H=\mathbf S_u^\top\bar{\mathbf Q}\mathbf S_u+\bar{\mathbf R}$ (check $\mathbf H\succ0$) and a **terminal weight** $\mathbf P$ on $\mathbf x_N$.

### 3. The MPC controller

- **Unconstrained**: solve the QP in closed form $\mathbf U=-\mathbf H^{-1}\mathbf F\mathbf x_0$ and apply $\mathbf u_0$.
- **Constrained**: add input bounds $|\mathbf u_k|\le u_{\max}$ and/or a state (speed) bound $|v_k|\le v_{\max}$ and solve the QP with a solver (`scipy.optimize`). Apply $\mathbf u_0$, step, repeat — the receding horizon.
- Two **terminal-cost** modes: the **LQR** cost-to-go (Riccati $\mathbf P$) and the plain stage cost ($\mathbf P=\mathbf Q$).

### 4. Evaluation (three experiments)

- **A — MPC = LQR**: with the LQR terminal cost, unconstrained MPC equals the LQR action for *any* horizon; with $\mathbf P=\mathbf Q$, it converges to LQR as $N$ grows.
- **B — input constraint**: LQR commands an input far over the actuator limit and must be naively clipped; MPC plans within the limit — compare the closed-loop cost.
- **C — state constraint**: a speed limit LQR **violates** but MPC **respects**.

Plots go to `results/` (gitignored), the test suite is a `__main__` runner.

---

## What should come out (reference orders of magnitude)

**Experiment A — MPC = LQR.** LQR first input at $x_0=(2,0)$: $u=-5.5247$. With the **LQR terminal cost**, MPC gives $-5.5247$ at $N=5$ (diff **~1e-14**) — exact for any horizon. With **$\mathbf P=\mathbf Q$**, MPC converges as the horizon grows: $N=1\to-0.099$, $N=5\to-2.17$, $N=20\to-5.44$, $N=60\to-5.5247$ (error $\to$ ~2e-6).

**Experiment B — input constraint** $|u|\le0.5$. LQR's unclipped command is $|u|=5.52$ — **11× over the limit** — so it must be clipped (suboptimal); MPC plans within. Both keep the applied input at $0.5$, but MPC achieves a **lower closed-loop cost** (~**65.6** vs ~**66.4**) and reaches the target.

**Experiment C — state constraint** $|v|\le0.6$. LQR peak speed **1.558** → **violates** the limit; MPC peak speed **0.600** → **respects** it (to solver tolerance), and both still reach the target. This is MPC's decisive advantage.

> **The lesson.** MPC is the automation controller that plans within reality's limits. The **condensation** turns a constrained optimal-control problem into a single convex **QP** solved every step, and re-solving with fresh measurements is what makes it **feedback**. Two facts frame it: **unconstrained MPC *is* LQR** (module 14) — with the Riccati terminal cost it matches to machine precision, so MPC is a strict generalisation, not a different idea; and the **terminal cost** is not cosmetic — it encodes the infinite tail, which is why a short-horizon MPC without it is myopic. What MPC adds over LQR is the one thing that matters on a real plant: it **honours hard constraints** that an unconstrained controller blows straight through.

## Setup & running

```bash
cd modules/23-advanced-automation/projects/03-final
# write your own implementation, then:
/Users/.../.venv/bin/python test_mpc.py   # test suite
/Users/.../.venv/bin/python run.py         # 3 experiments + plots
```

Only `numpy`, `scipy` (`solve_discrete_are` + `optimize`), `matplotlib`. Runtime ~1 s.

## Solution

The complete reference is in [`solution/`](solution/): `mpc.py` (plant, LQR, prediction matrices, condensation, the `MPC` controller, closed-loop simulation), `run.py` (3 experiments + plots), `test_mpc.py` (7 tests).

## Looking back & ahead

This closes module 23 — the automation stack from **discrete logic** (P01 automata) through **resources and safety** (P02 Petri nets + supervisory control) to **constrained optimising control** (P03 MPC). Next, **module 24 "Self-aware Computing"** turns the lens inward: systems that monitor and adapt *themselves* at runtime, built on exactly this module's feedback-control and decision machinery and the estimation of modules 21–22.

---
---

# P03 (final) — Model Predictive Control: beschränkte Optimalregelung per kondensiertem QP (deutsche Fassung)

**Modul 23 — Advanced Automation** · Format: **Python-Projekt (freie Umsetzung, keine Code-Vorgabe)**

> Abschlussprojekt. **Kein vorgegebener Code** — du baust den MPC-Regler selbst. Die Referenzlösung liegt in [`solution/`](solution/); **erst selbst versuchen**. Diese README ist die Spezifikation.

## Worum es geht

Die kontinuierliche Ebene der Automatisierung braucht einen Regler, der **harte Grenzen** respektiert — Aktoren sättigen, Geschwindigkeiten und Temperaturen müssen im Band bleiben. **Model Predictive Control (MPC)** ist das moderne Arbeitspferd: Bei jedem Schritt löst es ein beschränktes Optimalregelungsproblem über endlichem Horizont, wendet den ersten Eingang an und optimiert neu (Receding Horizon). Du baust es from scratch für eine automatisierte Positionierachse (einen doppelten Integrator), **kondensierst** das Problem in ein **quadratisches Programm** und zeigst die zwei Tatsachen, die MPC definieren: **unbeschränktes MPC gleicht dem LQR aus Modul 14**, und **MPC hält Constraints ein, die LQR schlicht verletzt**.

## Lernziel

Du leitest die **Kondensierung** her und implementierst sie ($\mathbf X=\mathbf S_x\mathbf x_0+\mathbf S_u\mathbf U$ → ein QP in $\mathbf U$), die **Receding-Horizon**-Schleife, die **Endkosten** (und warum sie MPC exakt machen) und **Eingangs- und Zustands-Constraints** — und verifizierst jede Aussage numerisch gegen die LQR-Baseline.

## Vorwissen

Das Modul-23-Skript Kap. 6, **LQR / die diskrete Riccati-Gleichung (Modul 14)**, lineare Systeme, quadratische Programmierung. `scipy.linalg.solve_discrete_are` für die LQR-Baseline und `scipy.optimize` für das QP.

---

## Aufgabenstellung (Spezifikation)

### 1. Anlage und LQR-Baseline

- Ein **doppelter Integrator** (automatisierte Positionierachse): Zustand $(x,y)=$ (Position, Geschwindigkeit), Eingang = Kraft; mit $\Delta t=0.1$ diskretisieren. Gewichte $\mathbf Q=\mathrm{diag}(1,0.1)$, $\mathbf R=[0.1]$.
- Die **LQR**-Verstärkung und Cost-to-go aus der diskreten algebraischen Riccati-Gleichung — das unbeschränkte Optimum und die zu schlagende Referenz.

### 2. Kondensierung in ein QP

- **Prädiktionsmatrizen** $\mathbf S_x,\mathbf S_u$ aus Potenzen von $\mathbf A,\mathbf B$, sodass $\mathbf X=\mathbf S_x\mathbf x_0+\mathbf S_u\mathbf U$ (gestapelte prädizierte Zustände als affine Funktion der gestapelten Eingänge). **Verifiziere**, dass sie eine schlichte Vorwärtssimulation reproduzieren.
- Baue das QP $\min_{\mathbf U}\tfrac12\mathbf U^\top\mathbf H\mathbf U+\mathbf x_0^\top\mathbf F^\top\mathbf U$ mit $\mathbf H=\mathbf S_u^\top\bar{\mathbf Q}\mathbf S_u+\bar{\mathbf R}$ (prüfe $\mathbf H\succ0$) und einem **Endgewicht** $\mathbf P$ auf $\mathbf x_N$.

### 3. Der MPC-Regler

- **Unbeschränkt**: das QP in geschlossener Form $\mathbf U=-\mathbf H^{-1}\mathbf F\mathbf x_0$ lösen und $\mathbf u_0$ anwenden.
- **Beschränkt**: Eingangsgrenzen $|\mathbf u_k|\le u_{\max}$ und/oder eine Zustands- (Geschwindigkeits-) Grenze $|v_k|\le v_{\max}$ hinzufügen und das QP mit einem Solver (`scipy.optimize`) lösen. $\mathbf u_0$ anwenden, Schritt, wiederholen — der Receding Horizon.
- Zwei **Endkosten**-Modi: die **LQR**-Cost-to-go (Riccati $\mathbf P$) und die reine Stufenkosten ($\mathbf P=\mathbf Q$).

### 4. Evaluation (drei Experimente)

- **A — MPC = LQR**: mit den LQR-Endkosten gleicht unbeschränktes MPC der LQR-Aktion für *jeden* Horizont; mit $\mathbf P=\mathbf Q$ konvergiert es mit wachsendem $N$ gegen LQR.
- **B — Eingangs-Constraint**: LQR kommandiert einen Eingang weit über der Aktorgrenze und muss naiv geklippt werden; MPC plant innerhalb der Grenze — die geschlossene-Kreis-Kosten vergleichen.
- **C — Zustands-Constraint**: eine Geschwindigkeitsgrenze, die LQR **verletzt**, MPC aber **respektiert**.

Plots nach `results/` (gitignored), die Testsuite als `__main__`-Runner.

---

## Was am Ende herauskommen soll (Referenz-Größenordnungen)

**Experiment A — MPC = LQR.** LQR-erster-Eingang bei $x_0=(2,0)$: $u=-5.5247$. Mit den **LQR-Endkosten** gibt MPC $-5.5247$ bei $N=5$ (Diff **~1e-14**) — exakt für jeden Horizont. Mit **$\mathbf P=\mathbf Q$** konvergiert MPC mit wachsendem Horizont: $N=1\to-0.099$, $N=5\to-2.17$, $N=20\to-5.44$, $N=60\to-5.5247$ (Fehler $\to$ ~2e-6).

**Experiment B — Eingangs-Constraint** $|u|\le0.5$. LQRs ungeklippter Befehl ist $|u|=5.52$ — **11× über der Grenze** — muss also geklippt werden (suboptimal); MPC plant innerhalb. Beide halten den angewandten Eingang bei $0.5$, aber MPC erreicht **niedrigere geschlossene-Kreis-Kosten** (~**65.6** vs ~**66.4**) und erreicht das Ziel.

**Experiment C — Zustands-Constraint** $|v|\le0.6$. LQR-Spitzengeschwindigkeit **1.558** → **verletzt** die Grenze; MPC-Spitzengeschwindigkeit **0.600** → **respektiert** sie (bis auf Solver-Toleranz), und beide erreichen das Ziel. Das ist MPCs entscheidender Vorteil.

> **Die Lehre.** MPC ist der Automatisierungsregler, der innerhalb der Grenzen der Realität plant. Die **Kondensierung** verwandelt ein beschränktes Optimalregelungsproblem in ein einziges konvexes **QP**, das jeden Schritt gelöst wird, und das Neu-Lösen mit frischen Messungen ist es, was es zu **Feedback** macht. Zwei Tatsachen rahmen es ein: **unbeschränktes MPC *ist* LQR** (Modul 14) — mit den Riccati-Endkosten stimmt es bis auf Maschinengenauigkeit überein, MPC ist also eine strikte Verallgemeinerung, keine andere Idee; und die **Endkosten** sind nicht kosmetisch — sie kodieren den unendlichen Schwanz, weshalb ein kurz-horizontiges MPC ohne sie myopisch ist. Was MPC über LQR hinaus hinzufügt, ist das Eine, das an einer realen Anlage zählt: Es **hält harte Constraints ein**, durch die ein unbeschränkter Regler geradewegs hindurchfährt.

## Setup & Ausführen

```bash
cd modules/23-advanced-automation/projects/03-final
# eigene Umsetzung schreiben, dann:
/Users/.../.venv/bin/python test_mpc.py   # Testsuite
/Users/.../.venv/bin/python run.py         # 3 Experimente + Plots
```

Nur `numpy`, `scipy` (`solve_discrete_are` + `optimize`), `matplotlib`. Laufzeit ~1 s.

## Lösung

Die vollständige Referenz liegt in [`solution/`](solution/): `mpc.py` (Anlage, LQR, Prädiktionsmatrizen, Kondensierung, der `MPC`-Regler, geschlossene-Kreis-Simulation), `run.py` (3 Experimente + Plots), `test_mpc.py` (7 Tests).

## Rückblick & Ausblick

Damit schließt Modul 23 — der Automatisierungs-Stack von der **diskreten Logik** (P01 Automaten) über **Ressourcen und Sicherheit** (P02 Petri-Netze + Supervisory Control) zur **beschränkten optimierenden Regelung** (P03 MPC). Als Nächstes richtet **Modul 24 „Self-aware Computing"** den Blick nach innen: Systeme, die sich zur Laufzeit *selbst* überwachen und anpassen, gebaut auf genau der Regelungs- und Entscheidungsmaschinerie dieses Moduls und der Schätzung der Module 21–22.
