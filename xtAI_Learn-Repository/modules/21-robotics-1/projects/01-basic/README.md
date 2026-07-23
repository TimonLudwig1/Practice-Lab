# P01 (basic) — forward kinematics, workspace and the Jacobian matrix

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 21 — Robotics 1** · Format: **Jupyter notebook**

## Goal

You build the kinematic foundation of every robot arm:

1. The general **DH transformation matrix** (script ch. 4).
2. The **kinematic chain** as a matrix product — verified against the closed-form formula derived by hand.
3. The **workspace** by sampling (an annulus, with a hole for unequal link lengths).
4. The **Jacobian matrix** analytically, cross-checked by a numerical derivative — and its **singularities** $\det\mathbf J = l_1l_2\sin q_2$.

## Why this format?

A **notebook**, because kinematics thrives on the interplay of formula, number and drawing: you want to see the workspace, draw the arm poses and view the manipulability over the configuration space as a heatmap.

## Why synthetic data?

This is about **laws of geometry**, not about a dataset. A self-defined arm (known link lengths) lets you check every computation against an analytically known target value — which is exactly what the notebook does throughout (chain vs. closed-form formula, analytic vs. numerical Jacobian, measured vs. theoretical reach).

## Prior knowledge

Homogeneous $4\times4$ transformations (**module 19**), partial derivatives, **ch. 3–5** of the [module 21 script](../../README.md).

## Task (step by step)

Open `kinematics.ipynb`. Much is given; at the `# TODO` spots you build the cores:

- **Part A** — `dh_matrix(theta, d, a, alpha)`: the $4\times4$ DH matrix.
- **Part B** — `fk_joints(q, lengths)`: multiply the chain out and collect the joint positions along the way. Verification against $x=l_1\cos q_1+l_2\cos(q_1{+}q_2)$ etc.
- **Part C** (given) — sample and plot the workspace.
- **Part D** — `jacobian_analytic(q, lengths)`: fill in the $2\times2$ Jacobian; automatic comparison with the numerical derivative, then the determinant and the singularities.
- **Part E** (given) — draw the manipulability over the C-space + the arm poses.

## What should come out (expected values)

- **Part B**: `equal=True` for all test configurations; at $q=(0,0)$ the end effector sits at $(2,0)$.
- **Part C**: for $l_1=l_2$ a full disc (radius 2); for $l_1=1.2, l_2=0.5$ an **annulus**: measured max reach **1.700** ($=l_1+l_2$), min **0.700** ($=|l_1-l_2|$).
- **Part D**: `matching: True`; $\det\mathbf J = 0.891207$ **exactly** equal to $l_1l_2\sin q_2$; at $q_2=0$ and $q_2=\pi$ the $\det\mathbf J$ falls to ~$10^{-17}$ → **singular**.
- **Part E**: the manipulability heatmap shows **vertical stripes** — $w$ depends only on $q_2$, consistent with $\det\mathbf J=l_1l_2\sin q_2$.

> **The physical reading of the singularity:** in the stretched state the end effector can **no longer move radially outwards** — one direction of motion is lost. Nearby you need ever larger joint velocities for the same hand motion. That is exactly what blows up the naive pseudoinverse in P02.

## Setup

```bash
cd modules/21-robotics-1/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # kinematics.ipynb
```

Only `numpy` + `matplotlib`. Runtime a few seconds.

## Solution

The complete, executed solution is in [`solution/kinematics_solution.ipynb`](solution/kinematics_solution.ipynb) — **try it yourself first!**

## What comes next

- **P02 (medium)**: **inverse kinematics** — analytically (both elbow solutions) and numerically via the Jacobian, including **damped least squares** against the singularities of part D.
- **P03 (final)**: the complete **sense-plan-act navigation** (RRT + particle filter + PID).

---
---

# P01 (basic) — Vorwärtskinematik, Arbeitsraum und die Jacobi-Matrix (deutsche Fassung)

**Modul 21 — Robotics 1** · Format: **Jupyter-Notebook**

## Ziel

Du baust die kinematische Grundlage jedes Roboterarms:

1. Die allgemeine **DH-Transformationsmatrix** (Skript Kap. 4).
2. Die **kinematische Kette** als Matrixprodukt — verifiziert gegen die von Hand hergeleitete geschlossene Formel.
3. Den **Arbeitsraum** durch Sampling (Kreisring, mit Loch bei ungleichen Gliedlängen).
4. Die **Jacobi-Matrix** analytisch, gegengeprüft per numerischer Ableitung — und ihre **Singularitäten** $\det\mathbf J = l_1l_2\sin q_2$.

## Warum dieses Format?

Ein **Notebook**, weil Kinematik vom Zusammenspiel aus Formel, Zahl und Zeichnung lebt: Man will den Arbeitsraum sehen, die Armposen zeichnen und die Manipulierbarkeit über dem Konfigurationsraum als Heatmap betrachten.

## Warum synthetische Daten?

Es geht um **Geometrie-Gesetze**, nicht um einen Datensatz. Ein selbst definierter Arm (bekannte Gliedlängen) erlaubt, jede Rechnung gegen einen analytisch bekannten Sollwert zu prüfen — genau das macht das Notebook durchgehend (Kette vs. geschlossene Formel, analytische vs. numerische Jacobi, gemessene vs. theoretische Reichweite).

## Vorwissen

Homogene $4\times4$-Transformationen (**Modul 19**), partielle Ableitungen, **Kap. 3–5** des [Modul-21-Skripts](../../README.md).

## Aufgabenstellung (Schritt für Schritt)

Öffne `kinematics.ipynb`. Vieles ist vorgegeben; an den `# TODO`-Stellen baust du die Kerne:

- **Teil A** — `dh_matrix(theta, d, a, alpha)`: die $4\times4$-DH-Matrix.
- **Teil B** — `fk_joints(q, lengths)`: die Kette aufmultiplizieren und dabei die Gelenkpositionen sammeln. Verifikation gegen $x=l_1\cos q_1+l_2\cos(q_1{+}q_2)$ etc.
- **Teil C** (vorgegeben) — Arbeitsraum sampeln und plotten.
- **Teil D** — `jacobian_analytic(q, lengths)`: die $2\times2$-Jacobi eintragen; automatischer Vergleich mit der numerischen Ableitung, dann Determinante und Singularitäten.
- **Teil E** (vorgegeben) — Manipulierbarkeit über dem C-Space + Armposen zeichnen.

## Was am Ende herauskommt (Erwartungswerte)

- **Teil B**: `gleich=True` für alle Testkonfigurationen; bei $q=(0,0)$ steht der Endeffektor bei $(2,0)$.
- **Teil C**: bei $l_1=l_2$ eine volle Scheibe (Radius 2); bei $l_1=1.2, l_2=0.5$ ein **Ring**: gemessene max. Reichweite **1.700** ($=l_1+l_2$), min. **0.700** ($=|l_1-l_2|$).
- **Teil D**: `uebereinstimmend: True`; $\det\mathbf J = 0.891207$ **exakt** gleich $l_1l_2\sin q_2$; bei $q_2=0$ und $q_2=\pi$ fällt $\det\mathbf J$ auf ~$10^{-17}$ → **singulär**.
- **Teil E**: Die Manipulierbarkeits-Heatmap zeigt **senkrechte Streifen** — $w$ hängt nur von $q_2$ ab, konsistent mit $\det\mathbf J=l_1l_2\sin q_2$.

> **Die physikalische Lesart der Singularität:** Im gestreckten Zustand kann sich der Endeffektor **nicht weiter radial nach außen** bewegen — eine Bewegungsrichtung geht verloren. In der Nähe braucht man immer größere Gelenkgeschwindigkeiten für dieselbe Handbewegung. Genau das sprengt in P02 die naive Pseudoinverse.

## Setup

```bash
cd modules/21-robotics-1/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # kinematics.ipynb
```

Nur `numpy` + `matplotlib`. Laufzeit wenige Sekunden.

## Lösung

Vollständige, ausgeführte Lösung in [`solution/kinematics_solution.ipynb`](solution/kinematics_solution.ipynb) — **erst selbst probieren!**

## Weiter geht's

- **P02 (medium)**: **Inverse Kinematik** — analytisch (beide Ellbogen-Lösungen) und numerisch über die Jacobi-Matrix, inklusive **Damped Least Squares** gegen die Singularitäten aus Teil D.
- **P03 (final)**: die vollständige **Sense-Plan-Act-Navigation** (RRT + Partikelfilter + PID).
