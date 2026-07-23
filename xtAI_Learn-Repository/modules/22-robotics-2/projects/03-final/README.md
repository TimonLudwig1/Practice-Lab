# P03 (final) — pose-graph SLAM: Gauss-Newton on a sparse pose graph, with loop closure

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 22 — Robotics 2** · Format: **Python project (free implementation, no code given)**

> Final project. **No code is given** — you build the SLAM back-end yourself. The reference solution is in [`solution/`](solution/); **try it yourself first**. This README is the specification.

## What it is about

Module 21 localised a robot on a **known** map. Here you solve the full **SLAM** problem — estimate the robot's whole **trajectory *and* the map at once** — in its modern **graph-based** form (script ch. 10). You build a **pose graph**, turn SLAM into one sparse non-linear least-squares problem, solve it by **Gauss-Newton**, and show the central phenomenon of SLAM: a single **loop closure** collapses the unbounded odometry drift (module 21 P03) into a globally consistent map.

## Learning objective

You implement the SE(2) machinery (pose composition, the edge **error function** and its **Jacobians**), assemble the **sparse information matrix** $\mathbf H=\sum\mathbf J^\top\boldsymbol\Omega\mathbf J$, anchor the gauge freedom, iterate Gauss-Newton, and evaluate against ground truth — including the **robustness** failure mode that makes real SLAM hard.

## Prior knowledge

The module 22 script, especially **ch. 8–11**. Homogeneous transformations / SE(2) (module 19), least squares and Gauss-Newton, odometry and its drift (module 21 P03), ICP as the loop-closure front-end (module 20). `scipy.sparse` for the sparse solve.

---

## Task (specification)

### 1. SE(2) utilities

- `wrap(a)` — angle to $(-\pi,\pi]$; `v2t(x)` — pose $(x,y,\theta)\mapsto$ $3\times3$ transform; `t2v(T)` — the inverse.

### 2. Edge error and Jacobians

- **Error** of an edge $(i,j)$ with measurement $\mathbf z$: $\mathbf e_{ij}=\mathrm{t2v}(\mathbf Z^{-1}\mathbf X_i^{-1}\mathbf X_j)$, angle wrapped — zero when the poses match the measurement.
- **Jacobians** $\mathbf A=\partial\mathbf e/\partial\mathbf x_i$, $\mathbf B=\partial\mathbf e/\partial\mathbf x_j$ (each $3\times3$). Derive them analytically (Grisetti et al.) **and verify them against numerical differentiation** — a wrong Jacobian silently ruins convergence.

### 3. The Gauss-Newton optimiser

- For each edge accumulate the sparse $\mathbf H=\sum\mathbf J^\top\boldsymbol\Omega\mathbf J$ and $\mathbf b=\sum\mathbf J^\top\boldsymbol\Omega\mathbf e$ (each edge fills only the $2\times2$ block positions of poses $i,j$ — use `scipy.sparse`).
- **Anchor pose 0** (add a strong prior to its diagonal block) to remove the gauge freedom, then solve $\mathbf H\Delta\mathbf x=-\mathbf b$ (`spsolve`), update $\mathbf x\leftarrow\mathbf x+\Delta\mathbf x$ (wrap angles), iterate.
- Add an optional **Huber** robust kernel: down-weight any edge whose Mahalanobis error exceeds a threshold.

### 4. Synthetic dataset (disclosed, reproducible)

A robot drives a **closed rectangular loop**. Produce: the **ground-truth** poses; **odometry edges** (true relative pose + noise → the estimate drifts when chained); **loop-closure edges** between non-consecutive poses that pass close to each other (true relative pose + noise, higher information). Keep ground truth for the ATE.

### 5. Evaluation (three experiments)

- **A — drift collapse**: odometry-only ATE vs. optimised ATE; show that optimising the **chain alone** (no loop closures) changes nothing, and that adding loop closures collapses the drift; report the $\chi^2$ per Gauss-Newton iteration.
- **B — incremental loop closures**: add them one at a time and plot the ATE — the **first** one does almost all the work.
- **C — robustness**: inject one **false** loop closure (naive least squares is wrecked), then switch on the **Huber** kernel (the map recovers).

Plots go to `results/` (gitignored), the test suite is a `__main__` runner.

---

## What should come out (reference orders of magnitude)

**Experiment A** (125 poses, 124 odometry edges, 5 loop closures):

| | ATE |
|---|---|
| odometry only | **1.034 m** (unbounded drift) |
| optimised, no loop closures | 1.034 m (a chain has no conflicts) |
| optimised, with loop closures | **0.086 m** (~12x better) |

$\chi^2$ per Gauss-Newton iteration: **4025 → 18.4 → 17.8 → …** — the drift shows up as a huge initial error and **collapses in a single step**.

**Experiment B** (ATE vs. number of loop closures): 0 → **1.034**, 1 → **0.127**, 2 → 0.085, 3 → 0.092, 4 → 0.088, 5 → 0.086. **One loop closure** already corrects most of the drift; the rest just refine it.

**Experiment C**: correct loop closures → ATE **0.086**; + 1 **false** loop closure (naive LS) → **5.64** (worse than odometry!); + **Huber** kernel → **0.47** (the outlier is rejected).

> **The lesson.** Graph SLAM is one idea carried all the way through: **poses are nodes, relative measurements are edges, and the map is the trajectory that best satisfies every edge at once** — a sparse non-linear least-squares problem. Optimising a pure odometry chain does nothing, because a chain has no conflicting constraints; the magic is the **loop closure**, which makes the accumulated drift visible as one large edge error that Gauss-Newton redistributes over the entire loop. And the same trust that makes it powerful makes it fragile: one **false** loop closure, believed absolutely, drags the whole map to satisfy it — which is why real systems need **robust cost functions**. This is the mathematics under every modern SLAM library.

## Setup & running

```bash
cd modules/22-robotics-2/projects/03-final
# write your own implementation, then:
/Users/.../.venv/bin/python test_pose_graph.py   # test suite
/Users/.../.venv/bin/python run.py                # 3 experiments + plots
```

Only `numpy`, `scipy` (`scipy.sparse` for the sparse solve), `matplotlib`. Runtime ~7 s.

## Solution

The complete reference is in [`solution/`](solution/): `pose_graph.py` (SE(2) utilities, error + Jacobians, the sparse Gauss-Newton optimiser, the dataset generator), `run.py` (3 experiments + plots), `test_pose_graph.py` (7 tests).

## Looking back & ahead

This closes module 22 — and the robotics arc: from **dynamics** (P01) via **model-based control** (P02) to **SLAM** (P03). Point-cloud perception (module 20), transformations (module 19), estimation (module 21) and optimisation (this project) are the reusable building blocks that the broader applications block of the curriculum keeps drawing on.

---
---

# P03 (final) — Pose-Graph-SLAM: Gauß-Newton auf einem dünnbesetzten Posengraphen, mit Loop Closure (deutsche Fassung)

**Modul 22 — Robotics 2** · Format: **Python-Projekt (freie Umsetzung, keine Code-Vorgabe)**

> Abschlussprojekt. **Kein vorgegebener Code** — du baust das SLAM-Backend selbst. Die Referenzlösung liegt in [`solution/`](solution/); **erst selbst versuchen**. Diese README ist die Spezifikation.

## Worum es geht

Modul 21 lokalisierte einen Roboter auf einer **bekannten** Karte. Hier löst du das volle **SLAM**-Problem — die ganze **Trajektorie *und* die Karte gleichzeitig** schätzen — in seiner modernen **graphbasierten** Form (Skript Kap. 10). Du baust einen **Posengraphen**, verwandelst SLAM in ein dünnbesetztes nichtlineares Least-Squares-Problem, löst es per **Gauß-Newton** und zeigst das zentrale Phänomen von SLAM: Ein einziges **Loop Closure** kollabiert den unbeschränkten Odometriedrift (Modul 21 P03) in eine global konsistente Karte.

## Lernziel

Du implementierst die SE(2)-Maschinerie (Posenverkettung, die Kanten-**Fehlerfunktion** und ihre **Jacobi-Matrizen**), baust die **dünnbesetzte Informationsmatrix** $\mathbf H=\sum\mathbf J^\top\boldsymbol\Omega\mathbf J$ zusammen, verankerst die Eichfreiheit, iterierst Gauß-Newton und evaluierst gegen ground truth — inklusive der **Robustheits**-Fehlerart, die reales SLAM schwer macht.

## Vorwissen

Das Modul-22-Skript, besonders **Kap. 8–11**. Homogene Transformationen / SE(2) (Modul 19), Least Squares und Gauß-Newton, Odometrie und ihr Drift (Modul 21 P03), ICP als Loop-Closure-Frontend (Modul 20). `scipy.sparse` für den dünnbesetzten Solve.

---

## Aufgabenstellung (Spezifikation)

### 1. SE(2)-Werkzeuge

- `wrap(a)` — Winkel auf $(-\pi,\pi]$; `v2t(x)` — Pose $(x,y,\theta)\mapsto$ $3\times3$-Transformation; `t2v(T)` — die Umkehrung.

### 2. Kantenfehler und Jacobi-Matrizen

- **Fehler** einer Kante $(i,j)$ mit Messung $\mathbf z$: $\mathbf e_{ij}=\mathrm{t2v}(\mathbf Z^{-1}\mathbf X_i^{-1}\mathbf X_j)$, Winkel normiert — null, wenn die Posen zur Messung passen.
- **Jacobi-Matrizen** $\mathbf A=\partial\mathbf e/\partial\mathbf x_i$, $\mathbf B=\partial\mathbf e/\partial\mathbf x_j$ (je $3\times3$). Leite sie analytisch her (Grisetti et al.) **und verifiziere sie gegen numerische Differentiation** — eine falsche Jacobi ruiniert die Konvergenz stillschweigend.

### 3. Der Gauß-Newton-Optimierer

- Für jede Kante die dünnbesetzte $\mathbf H=\sum\mathbf J^\top\boldsymbol\Omega\mathbf J$ und $\mathbf b=\sum\mathbf J^\top\boldsymbol\Omega\mathbf e$ akkumulieren (jede Kante füllt nur die $2\times2$-Blockpositionen der Posen $i,j$ — `scipy.sparse` nutzen).
- **Pose 0 verankern** (starken Prior auf ihren Diagonalblock addieren), um die Eichfreiheit zu entfernen, dann $\mathbf H\Delta\mathbf x=-\mathbf b$ lösen (`spsolve`), $\mathbf x\leftarrow\mathbf x+\Delta\mathbf x$ aktualisieren (Winkel normieren), iterieren.
- Einen optionalen **Huber**-Robustheitskern hinzufügen: jede Kante herabgewichten, deren Mahalanobis-Fehler eine Schwelle überschreitet.

### 4. Synthetischer Datensatz (offengelegt, reproduzierbar)

Ein Roboter fährt eine **geschlossene Rechteck-Schleife**. Erzeuge: die **ground-truth**-Posen; **Odometrie-Kanten** (wahre relative Pose + Rauschen → die Schätzung driftet beim Verketten); **Loop-Closure-Kanten** zwischen nicht-aufeinanderfolgenden Posen, die nah aneinander vorbeikommen (wahre relative Pose + Rauschen, höhere Information). Ground truth für die ATE behalten.

### 5. Evaluation (drei Experimente)

- **A — Drift-Kollaps**: Odometrie-ATE vs. optimierte ATE; zeigen, dass das Optimieren der **reinen Kette** (ohne Loop Closures) nichts ändert und dass Loop Closures den Drift kollabieren lassen; $\chi^2$ pro Gauß-Newton-Iteration berichten.
- **B — inkrementelle Loop Closures**: eins nach dem anderen hinzufügen und die ATE plotten — das **erste** leistet fast die ganze Arbeit.
- **C — Robustheit**: ein **falsches** Loop Closure injizieren (naives Least Squares wird zerstört), dann den **Huber**-Kern einschalten (die Karte erholt sich).

Plots nach `results/` (gitignored), die Testsuite als `__main__`-Runner.

---

## Was am Ende herauskommen soll (Referenz-Größenordnungen)

**Experiment A** (125 Posen, 124 Odometrie-Kanten, 5 Loop Closures):

| | ATE |
|---|---|
| nur Odometrie | **1.034 m** (unbeschränkter Drift) |
| optimiert, ohne Loop Closures | 1.034 m (eine Kette hat keine Konflikte) |
| optimiert, mit Loop Closures | **0.086 m** (~12x besser) |

$\chi^2$ pro Gauß-Newton-Iteration: **4025 → 18.4 → 17.8 → …** — der Drift zeigt sich als riesiger Anfangsfehler und **kollabiert in einem einzigen Schritt**.

**Experiment B** (ATE gegen Loop-Closure-Zahl): 0 → **1.034**, 1 → **0.127**, 2 → 0.085, 3 → 0.092, 4 → 0.088, 5 → 0.086. **Ein Loop Closure** korrigiert schon den Großteil des Drifts; der Rest verfeinert nur.

**Experiment C**: korrekte Loop Closures → ATE **0.086**; + 1 **falsches** Loop Closure (naives LS) → **5.64** (schlechter als Odometrie!); + **Huber**-Kern → **0.47** (der Ausreißer wird abgelehnt).

> **Die Lehre.** Graph-SLAM ist eine Idee, konsequent durchgezogen: **Posen sind Knoten, relative Messungen sind Kanten, und die Karte ist die Trajektorie, die jede Kante zugleich am besten erfüllt** — ein dünnbesetztes nichtlineares Least-Squares-Problem. Eine reine Odometriekette zu optimieren bewirkt nichts, weil eine Kette keine widersprüchlichen Constraints hat; die Magie ist das **Loop Closure**, das den aufgelaufenen Drift als einen großen Kantenfehler sichtbar macht, den Gauß-Newton über die ganze Schleife verteilt. Und dasselbe Vertrauen, das es mächtig macht, macht es fragil: Ein **falsches** Loop Closure, absolut geglaubt, zieht die ganze Karte, um es zu erfüllen — weshalb reale Systeme **robuste Kostenfunktionen** brauchen. Das ist die Mathematik unter jeder modernen SLAM-Bibliothek.

## Setup & Ausführen

```bash
cd modules/22-robotics-2/projects/03-final
# eigene Umsetzung schreiben, dann:
/Users/.../.venv/bin/python test_pose_graph.py   # Testsuite
/Users/.../.venv/bin/python run.py                # 3 Experimente + Plots
```

Nur `numpy`, `scipy` (`scipy.sparse` für den dünnbesetzten Solve), `matplotlib`. Laufzeit ~7 s.

## Lösung

Die vollständige Referenz liegt in [`solution/`](solution/): `pose_graph.py` (SE(2)-Werkzeuge, Fehler + Jacobi-Matrizen, der dünnbesetzte Gauß-Newton-Optimierer, der Datensatz-Generator), `run.py` (3 Experimente + Plots), `test_pose_graph.py` (7 Tests).

## Rückblick & Ausblick

Damit schließt Modul 22 — und der Robotik-Bogen: von der **Dynamik** (P01) über die **modellbasierte Regelung** (P02) zum **SLAM** (P03). Punktwolken-Perzeption (Modul 20), Transformationen (Modul 19), Schätzung (Modul 21) und Optimierung (dieses Projekt) sind die wiederverwendbaren Bausteine, auf die der breitere Anwendungsblock des Curriculums weiter zurückgreift.
