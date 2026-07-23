# P01 (basic) — manipulator dynamics: the mass matrix and a forward-dynamics simulator

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 22 — Robotics 2** · Format: **Jupyter notebook**

## Goal

Robotics 1 answered *where* the hand is. This project answers *what makes it move* — the **dynamics**. You build the three terms of the manipulator equation $\mathbf M(\mathbf q)\ddot{\mathbf q}+\mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q}+\mathbf g(\mathbf q)=\boldsymbol\tau$ for the planar 2-link arm, turn them into a **forward-dynamics simulator** (a physics engine in a few lines) and validate everything with a physical law: **energy is conserved** when no torque and no friction act.

1. the **mass matrix** $\mathbf M(\mathbf q)$ (script ch. 4),
2. the **Coriolis** term $\mathbf C(\mathbf q,\dot{\mathbf q})$ and the **gravity** term $\mathbf g(\mathbf q)$,
3. the **forward dynamics** $\ddot{\mathbf q}=\mathbf M^{-1}(\boldsymbol\tau-\mathbf C\dot{\mathbf q}-\mathbf g)$ and its RK4 integration,
4. the structural **properties** and **energy conservation** as correctness tests.

## Why this format?

A **notebook**, because dynamics wants the equations, the numbers and the plots (joint trajectory, energy over time) side by side.

## Why synthetic data?

This is about the **laws of motion**, not a dataset. A self-defined arm (known masses and lengths) lets every quantity be checked against a physical law — energy conservation, the positive-definiteness of $\mathbf M$, the skew-symmetry of $\dot{\mathbf M}-2\mathbf C$ — which is exactly what the notebook does throughout.

## Prior knowledge

The module 22 script ch. 1–5, the 2-link kinematics of module 21, linear algebra.

## Task (step by step)

Open `dynamics.ipynb`. Much is given; at the `# TODO` spots you build the cores:

- **Part A** — `mass_matrix(q)`: the $2\times2$ inertia matrix (symmetric, depends only on $q_2$).
- **Part B** — `coriolis(q, dq)` and `gravity(q)`: the velocity-dependent Coriolis matrix and the gravity torque.
- **Part C** (given) — the property checks: $\mathbf M$ positive definite, $\dot{\mathbf M}-2\mathbf C$ skew-symmetric.
- **Part D** — `forward_dynamics(q, dq, tau)`: solve for $\ddot{\mathbf q}$ (use `np.linalg.solve`); the RK4 integrator is given.
- **Part E** (given) — energy conservation: simulate the passive arm with RK4 vs. explicit Euler.

## What should come out (expected values)

- **Part A/B**: `mass_matrix` is symmetric and unchanged when only $q_1$ varies; `coriolis` has a zero in the bottom-right; `gravity` at a raised pose has two positive entries.
- **Part C**: $\mathbf M$ positive definite on all 200 random samples; `max |N + N^T|` $\approx$ **9.9e-10** (skew-symmetry of $\dot{\mathbf M}-2\mathbf C$, limited only by the finite-difference $\dot{\mathbf M}$).
- **Part D**: feeding exactly $\boldsymbol\tau=\mathbf g(\mathbf q)$ to the passive arm keeps it still (drift ~0).
- **Part E**: **RK4 energy drift ~8e-9** over 6 s (conserved) vs. **explicit Euler +10.1%** (energy injected from nowhere). The flat RK4 curve validates all three dynamics terms.

> **The lesson.** The manipulator equation has three terms and a wrong sign in any of them is easy to make and hard to spot — so you check against **physics**, not against a reference number: $\mathbf M$ must be positive definite, $\dot{\mathbf M}-2\mathbf C$ must be skew-symmetric (passivity), and a passive arm must conserve energy. These are the tests a real robotics engineer runs.

## Setup

```bash
cd modules/22-robotics-2/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # dynamics.ipynb
```

Only `numpy` + `matplotlib`. Runtime a few seconds.

## Solution

The complete, executed solution is in [`solution/dynamics_solution.ipynb`](solution/dynamics_solution.ipynb) — **try it yourself first!**

## What comes next

- **P02 (medium)**: **computed-torque control** — use exactly these $\mathbf M,\mathbf C,\mathbf g$ to cancel the dynamics and make the arm track a trajectory, and compare with PD + gravity compensation.
- **P03 (final)**: **pose-graph SLAM** — estimate the trajectory *and* the map at once.

---
---

# P01 (basic) — Manipulator-Dynamik: die Massenmatrix und ein Vorwärtsdynamik-Simulator (deutsche Fassung)

**Modul 22 — Robotics 2** · Format: **Jupyter-Notebook**

## Ziel

Robotics 1 beantwortete, *wo* die Hand ist. Dieses Projekt beantwortet, *was sie bewegt* — die **Dynamik**. Du baust die drei Terme der Manipulatorgleichung $\mathbf M(\mathbf q)\ddot{\mathbf q}+\mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q}+\mathbf g(\mathbf q)=\boldsymbol\tau$ für den planaren 2-Gelenk-Arm, machst daraus einen **Vorwärtsdynamik-Simulator** (eine Physik-Engine in wenigen Zeilen) und validierst alles mit einem physikalischen Gesetz: **Energie ist erhalten**, wenn kein Moment und keine Reibung wirken.

1. die **Massenmatrix** $\mathbf M(\mathbf q)$ (Skript Kap. 4),
2. den **Coriolis**-Term $\mathbf C(\mathbf q,\dot{\mathbf q})$ und den **Gravitations**-Term $\mathbf g(\mathbf q)$,
3. die **Vorwärtsdynamik** $\ddot{\mathbf q}=\mathbf M^{-1}(\boldsymbol\tau-\mathbf C\dot{\mathbf q}-\mathbf g)$ und ihre RK4-Integration,
4. die strukturellen **Eigenschaften** und **Energieerhaltung** als Korrektheits-Tests.

## Warum dieses Format?

Ein **Notebook**, weil die Dynamik die Gleichungen, die Zahlen und die Plots (Gelenktrajektorie, Energie über der Zeit) nebeneinander will.

## Warum synthetische Daten?

Es geht um die **Bewegungsgesetze**, nicht um einen Datensatz. Ein selbst definierter Arm (bekannte Massen und Längen) erlaubt, jede Größe gegen ein physikalisches Gesetz zu prüfen — Energieerhaltung, positive Definitheit von $\mathbf M$, Schiefsymmetrie von $\dot{\mathbf M}-2\mathbf C$ — genau das macht das Notebook durchgehend.

## Vorwissen

Das Modul-22-Skript Kap. 1–5, die 2-Gelenk-Kinematik aus Modul 21, lineare Algebra.

## Aufgabenstellung (Schritt für Schritt)

Öffne `dynamics.ipynb`. Vieles ist vorgegeben; an den `# TODO`-Stellen baust du die Kerne:

- **Teil A** — `mass_matrix(q)`: die $2\times2$-Trägheitsmatrix (symmetrisch, hängt nur von $q_2$ ab).
- **Teil B** — `coriolis(q, dq)` und `gravity(q)`: die geschwindigkeitsabhängige Coriolis-Matrix und das Gravitationsmoment.
- **Teil C** (vorgegeben) — die Eigenschafts-Checks: $\mathbf M$ positiv definit, $\dot{\mathbf M}-2\mathbf C$ schiefsymmetrisch.
- **Teil D** — `forward_dynamics(q, dq, tau)`: nach $\ddot{\mathbf q}$ auflösen (nutze `np.linalg.solve`); der RK4-Integrator ist vorgegeben.
- **Teil E** (vorgegeben) — Energieerhaltung: den passiven Arm mit RK4 vs. explizitem Euler simulieren.

## Was am Ende herauskommt (Erwartungswerte)

- **Teil A/B**: `mass_matrix` ist symmetrisch und unverändert, wenn nur $q_1$ variiert; `coriolis` hat eine Null unten rechts; `gravity` hat bei angehobener Pose zwei positive Einträge.
- **Teil C**: $\mathbf M$ auf allen 200 Zufallsstichproben positiv definit; `max |N + N^T|` $\approx$ **9.9e-10** (Schiefsymmetrie von $\dot{\mathbf M}-2\mathbf C$, nur durch die Finite-Differenzen-$\dot{\mathbf M}$ begrenzt).
- **Teil D**: exakt $\boldsymbol\tau=\mathbf g(\mathbf q)$ in den passiven Arm zu speisen hält ihn still (Drift ~0).
- **Teil E**: **RK4-Energiedrift ~8e-9** über 6 s (erhalten) vs. **expliziter Euler +10.1%** (Energie aus dem Nichts). Die flache RK4-Kurve validiert alle drei Dynamikterme.

> **Die Lehre.** Die Manipulatorgleichung hat drei Terme, und ein Vorzeichenfehler in einem ist leicht gemacht und schwer zu finden — also prüft man gegen die **Physik**, nicht gegen eine Referenzzahl: $\mathbf M$ muss positiv definit sein, $\dot{\mathbf M}-2\mathbf C$ schiefsymmetrisch (Passivität), und ein passiver Arm muss Energie erhalten. Das sind die Tests, die eine echte Robotik-Ingenieurin fährt.

## Setup

```bash
cd modules/22-robotics-2/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # dynamics.ipynb
```

Nur `numpy` + `matplotlib`. Laufzeit wenige Sekunden.

## Lösung

Die vollständige, ausgeführte Lösung liegt in [`solution/dynamics_solution.ipynb`](solution/dynamics_solution.ipynb) — **erst selbst probieren!**

## Weiter geht's

- **P02 (medium)**: **Computed-Torque-Regelung** — nutze genau diese $\mathbf M,\mathbf C,\mathbf g$, um die Dynamik wegzuheben und den Arm eine Trajektorie verfolgen zu lassen, und vergleiche mit PD + Gravitationskompensation.
- **P03 (final)**: **Pose-Graph-SLAM** — Trajektorie *und* Karte gleichzeitig schätzen.
