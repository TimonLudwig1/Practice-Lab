# P02 (medium) — inverse kinematics: analytic, numerical and the singularity trap

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 21 — Robotics 1** · Format: **Python module + test suite**

## Goal

You invert the kinematics of P01 — from the pose to the joint angles — and get to know the central numerical trap of robotics:

1. **Analytic IK** for the 2-joint arm (law of cosines) — with **both** solutions (elbow up/down), a correct reachability check and `atan2`.
2. **Numerical IK** via the Jacobian in three variants: **transpose**, **pseudoinverse** and **damped least squares**.
3. The **singularity trap**: why the pseudoinverse **explodes** near $\det\mathbf J = 0$ (joint jumps $\sim 1/\det\mathbf J$) and how DLS tames it — plus the $\lambda$ trade-off.
4. **Redundancy**: with a 3-joint arm there is a **null space** — joint motions that do not move the end effector.

## Why this format?

A **Python module with a test suite**: the IK variants are clearly testable functions (the analytic solution must hit the target *exactly*, the Jacobian against the numerical derivative, null-space drift ≈ 0), and the experiments vary parameters systematically ($\lambda$, proximity to a singularity).

## Why synthetic data?

This is about **properties of algorithms**, not about measured data. A self-defined arm lets you **back-compute** every solution through the forward kinematics and check it exactly — and lets you set the proximity to a singularity *deliberately* (which would not be possible with real robot data).

## Prior knowledge

**P01** of this module (FK, Jacobian, singularities), **ch. 6** of the [script](../../README.md), pseudoinverse/least squares.

## Task

Open `ik.py`. FK, the joint positions and the Jacobian are given (from P01) — you implement the **three cores** (`# TODO` / `NotImplementedError`):

1. **`analytic_ik_2link(target, lengths)`** — reachability check, law of cosines, **both** signs of $q_2$, $q_1$ via `atan2`, remove the duplicate at the boundary.
2. **The three update rules in `numeric_ik`** — `transpose` (with the optimal step size), `pinv`, `dls`.
3. **`nullspace_step(q, lengths, z)`** — the projection $(\mathbf I - \mathbf J^{+}\mathbf J)\,\mathbf z$.

Then:

```bash
cd modules/21-robotics-1/projects/02-medium
/Users/.../.venv/bin/python test_ik.py   # 8 tests -> all PASS
/Users/.../.venv/bin/python run.py        # 4 experiments + plots
```

## What should come out (expected values)

**Experiment 1 — analytic IK.** The target $(1,1)$ has **two** solutions: $q=(0°, +90°)$ and $q=(90°, -90°)$ — both hit exactly. The target $(0,2)$ (stretched, the workspace boundary) has **one**; $(2.5, 0)$ is **unreachable**.

**Experiment 2 — comparison of methods** (200 random targets):

| Method | success | iterations (median) | max &#124;Δq&#124; |
|---|---|---|---|
| transpose | 1.00 | 26 | 12 |
| pinv | 0.87 | **7** | **1194** |
| dls | 0.89 | 9 | 16 |

The transpose is the most **robust** but slow (gradient descent). The pseudoinverse is the fastest — but pays for it with enormous individual steps.

**Experiment 3 — the singularity trap.** If you start ever closer to the stretched arm ($q_2\to0$, i.e. $\det\mathbf J\to0$), the pseudoinverse steps scale **exactly like $1/\det\mathbf J$**:

| $q_2^{\text{start}}$ | $\det\mathbf J$ | pinv max&#124;Δq&#124; | DLS max&#124;Δq&#124; |
|---|---|---|---|
| 0.5 | 0.479 | 4.4 | 3.7 |
| 0.1 | 0.0998 | 26.5 | 4.4 |
| 0.01 | 0.0100 | 274.7 | 2.7 |
| 0.001 | 0.00100 | 2756 | 4.9 |
| 0.0001 | 0.000100 | **27573** | **4.7** |

Every order of magnitude closer to the singularity = **ten times larger** joint jumps for pinv; DLS stays at ~3–5. The $\lambda$ sweep shows the trade-off: a small $\lambda$ is accurate but jumpy, a large $\lambda$ is smooth but slow/less accurate.

**Experiment 4 — redundancy.** With the 3-joint arm $\mathbf J$ is a $2\times3$ matrix with **null space dimension 1**. 200 random null-space steps move the end effector by at most $1.7\cdot10^{-5}$ — the joints move, the hand stands still.

> **The lesson.** The pseudoinverse is mathematically "optimal" (the smallest joint change) and practically **dangerous**: it knows no bound on $\|\Delta\mathbf q\|$. The damping in DLS is not a cosmetic flaw but the condition for a real robot not to lash out into the singularity — you **deliberately trade accuracy for bounded joint velocities**.

## Solution

The complete reference is in [`solution/`](solution/). Try it yourself first!

## What comes next

**P03 (final)**: the full **sense-plan-act cycle** of a mobile robot — RRT planning, particle filter localisation and PID path following. No code given.

---
---

# P02 (medium) — Inverse Kinematik: analytisch, numerisch und die Singularitätsfalle (deutsche Fassung)

**Modul 21 — Robotics 1** · Format: **Python-Modul + Testsuite**

## Ziel

Du drehst die Kinematik aus P01 um — von der Pose zu den Gelenkwinkeln — und lernst dabei die zentrale numerische Falle der Robotik kennen:

1. **Analytische IK** für den 2-Gelenk-Arm (Kosinussatz) — mit **beiden** Lösungen (Ellbogen oben/unten), korrekter Erreichbarkeitsprüfung und `atan2`.
2. **Numerische IK** über die Jacobi-Matrix in drei Varianten: **Transponierte**, **Pseudoinverse** und **Damped Least Squares**.
3. Die **Singularitätsfalle**: Warum die Pseudoinverse nahe $\det\mathbf J = 0$ **explodiert** (Gelenksprünge $\sim 1/\det\mathbf J$) und DLS sie zähmt — plus der $\lambda$-Trade-off.
4. **Redundanz**: Beim 3-Gelenk-Arm gibt es einen **Nullraum** — Gelenkbewegungen, die den Endeffektor nicht bewegen.

## Warum dieses Format?

Ein **Python-Modul mit Testsuite**: Die IK-Varianten sind klar testbare Funktionen (analytische Lösung muss das Ziel *exakt* treffen, Jacobi gegen numerische Ableitung, Nullraum-Drift ≈ 0), und die Experimente variieren systematisch Parameter ($\lambda$, Singularitätsnähe).

## Warum synthetische Daten?

Es geht um **Algorithmen-Eigenschaften**, nicht um Messdaten. Ein selbst definierter Arm erlaubt, jede Lösung per Vorwärtskinematik **zurückzurechnen** und exakt zu prüfen — und die Singularitätsnähe *gezielt* einzustellen (was mit echten Roboterdaten nicht ginge).

## Vorwissen

**P01** dieses Moduls (FK, Jacobi, Singularitäten), **Kap. 6** des [Skripts](../../README.md), Pseudoinverse/Least Squares.

## Aufgabenstellung

Öffne `ik.py`. FK, Gelenkpositionen und Jacobi sind vorgegeben (aus P01) — du implementierst die **drei Kerne** (`# TODO` / `NotImplementedError`):

1. **`analytic_ik_2link(target, lengths)`** — Erreichbarkeitsprüfung, Kosinussatz, **beide** Vorzeichen von $q_2$, $q_1$ per `atan2`, Duplikat an der Grenze entfernen.
2. **Die drei Update-Regeln in `numeric_ik`** — `transpose` (mit optimaler Schrittweite), `pinv`, `dls`.
3. **`nullspace_step(q, lengths, z)`** — die Projektion $(\mathbf I - \mathbf J^{+}\mathbf J)\,\mathbf z$.

Dann:

```bash
cd modules/21-robotics-1/projects/02-medium
/Users/.../.venv/bin/python test_ik.py   # 8 Tests -> alle PASS
/Users/.../.venv/bin/python run.py        # 4 Experimente + Plots
```

## Was am Ende herauskommt (Erwartungswerte)

**Experiment 1 — analytische IK.** Ziel $(1,1)$ hat **zwei** Lösungen: $q=(0°, +90°)$ und $q=(90°, -90°)$ — beide treffen exakt. Ziel $(0,2)$ (gestreckt, Arbeitsraumgrenze) hat **eine**; $(2.5, 0)$ ist **unerreichbar**.

**Experiment 2 — Methodenvergleich** (200 Zufallsziele):

| Methode | Erfolg | Iterationen (Median) | max &#124;Δq&#124; |
|---|---|---|---|
| transpose | 1.00 | 26 | 12 |
| pinv | 0.87 | **7** | **1194** |
| dls | 0.89 | 9 | 16 |

Die Transponierte ist am **robustesten**, aber langsam (Gradientenabstieg). Pseudoinverse ist am schnellsten — bezahlt das aber mit gewaltigen Einzelschritten.

**Experiment 3 — die Singularitätsfalle.** Startet man immer näher am gestreckten Arm ($q_2\to0$, also $\det\mathbf J\to0$), skalieren die Pseudoinverse-Schritte **exakt wie $1/\det\mathbf J$**:

| $q_2^{\text{start}}$ | $\det\mathbf J$ | pinv max&#124;Δq&#124; | DLS max&#124;Δq&#124; |
|---|---|---|---|
| 0.5 | 0.479 | 4.4 | 3.7 |
| 0.1 | 0.0998 | 26.5 | 4.4 |
| 0.01 | 0.0100 | 274.7 | 2.7 |
| 0.001 | 0.00100 | 2756 | 4.9 |
| 0.0001 | 0.000100 | **27573** | **4.7** |

Jede Zehnerpotenz näher an der Singularität = **zehnfach größere** Gelenksprünge bei pinv; DLS bleibt bei ~3–5. Der $\lambda$-Sweep zeigt den Trade-off: kleines $\lambda$ genau aber sprunghaft, großes $\lambda$ sanft aber langsam/ungenauer.

**Experiment 4 — Redundanz.** Beim 3-Gelenk-Arm ist $\mathbf J$ eine $2\times3$-Matrix mit **Nullraum-Dimension 1**. 200 zufällige Nullraum-Schritte bewegen den Endeffektor um maximal $1.7\cdot10^{-5}$ — die Gelenke bewegen sich, die Hand steht still.

> **Die Lehre.** Die Pseudoinverse ist mathematisch „optimal" (kleinste Gelenkänderung) und praktisch **gefährlich**: Sie kennt keine Grenze für $\|\Delta\mathbf q\|$. Die Dämpfung in DLS ist kein Schönheitsfehler, sondern die Bedingung dafür, dass ein realer Roboter nicht in die Singularität hineinschlägt — man **tauscht bewusst Genauigkeit gegen beschränkte Gelenkgeschwindigkeiten**.

## Lösung

Vollständige Referenz in [`solution/`](solution/). Erst selbst versuchen!

## Weiter geht's

**P03 (final)**: der volle **Sense-Plan-Act-Zyklus** eines mobilen Roboters — RRT-Planung, Partikelfilter-Lokalisierung und PID-Pfadverfolgung. Keine Code-Vorgabe.
