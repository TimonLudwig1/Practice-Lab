# P03 (final) — sense-plan-act: RRT planning, particle filter localisation, path following

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 21 — Robotics 1** · Format: **Python project (free implementation, no code given)**

> Final project. **No code is given** — you build the complete navigation stack yourself. The reference solution is in [`solution/`](solution/); **try it yourself first**. This README is the specification.

## What it is about

You build the **complete sense-plan-act cycle** of a mobile robot (script ch. 1) — and show that it only works as a **whole**:

- **PLAN**: an **RRT** finds a collision-free path through an obstacle landscape, a **shortcut smoothing** shortens it.
- **SENSE**: the odometry **drifts** (dead reckoning); a **particle filter** corrects it with range measurements to known landmarks.
- **ACT**: a **pure pursuit controller** follows the path — on the **estimated**, not the true pose.

The core finding you demonstrate empirically: **without the SENSE everything collapses.** A robot that only believes its odometry leaves the perfectly planned path and collides.

## Learning objective

You integrate the three pillars of the module — **planning** (ch. 7–8), **state estimation** (ch. 9–11) and **control** (ch. 12) — into one system and evaluate it quantitatively.

## Prior knowledge

The module 21 [script](../../README.md), especially **ch. 7–12**. A\*/search from module 06, Bayes from module 07, sensor fusion from modules 17/18.

---

## Task (specification)

### 1. World and robot

- **World**: a 2D area (e.g. $10\times10$) with circular **obstacles**, a collision test for points *and* segments (dense sampling along the edge). A **safety margin** (the robot radius) must be configurable.
- **Robot**: a **unicycle model** with state $(x, y, \theta)$:
  $$x \mathrel{+}= v\cos\theta\,\Delta t,\quad y \mathrel{+}= v\sin\theta\,\Delta t,\quad \theta \mathrel{+}= \omega\,\Delta t$$
  The *true* motion gets **noise** on $v$ and $\omega$; the **odometry** computes with the pure commanded values (and drifts because of it).
- **Sensor**: **ranges to known landmarks**, noisy.

Why **synthetic**: only this way do you know the *true* pose and can measure the localisation error at all — with a real robot there is no ground truth.

### 2. PLAN — RRT + smoothing

Implement **RRT** (script ch. 8): sample (with a **goal bias**) → nearest tree node → step of length $\varepsilon$ → **collision test of the edge** → insert; near the goal recover the path by tracing back the parents. Then **shortcut smoothing** (repeatedly connect two path points directly when free).

> **Two pitfalls you will hit:** (1) after the smoothing the path consists of **few widely spaced corners** — the path follower then needs **densified** waypoints, otherwise its target point jumps. (2) The controller **cuts corners**; therefore plan with a **safety margin**, otherwise the robot grazes obstacles although the path was formally free.

### 3. SENSE — particle filter

Implement Monte Carlo localisation (script ch. 11):
1. **Prediction**: motion model + noise on every particle.
2. **Weighting**: $w^{[m]} \propto p(\mathbf z\mid \mathbf x^{[m]})$ — the Gaussian likelihood of the ranges. *Compute in **log** likelihoods and subtract the maximum, otherwise the weights underflow numerically.*
3. **Resampling**: **systematic**, and **only when** $N_{\text{eff}} = 1/\sum_m (w^{[m]})^2$ falls below $M/2$ (against **particle depletion**).
4. **Estimate**: the weighted mean; average the **orientation** via $\operatorname{atan2}(\sum w\sin\theta, \sum w\cos\theta)$ (not naively arithmetically!).

### 4. ACT — path following

**Pure pursuit**: find the most advanced path point within the lookahead radius, turn proportionally to the angular error towards it, drive more slowly at a large angular error. **Bound $\omega$** (saturation) — otherwise the robot turns too far per time step and oscillates.

### 5. Evaluation (three experiments)

- **A — planning**: success rate, tree size and path length over various **goal bias** values; the effect of the **smoothing**.
- **B — localisation**: the error over the drive, **particle filter vs. pure odometry**; vary the **particle count** $M$.
- **C — closed loop**: navigation with the controller on the **filter estimate** vs. on **pure odometry** — goal reached, final distance, collisions (10 runs each).

Plots go to `results/` (gitignored), the test suite is a `__main__` runner.

---

## What should come out (reference orders of magnitude)

**A — planning**: RRT always finds a solution in this world. The goal bias lowers the search effort noticeably (~200 → 120–140 nodes) and barely changes the path length. The big gain comes from the **smoothing**: ~16.3 → ~13.6 (about **17 % shorter**) — RRT is, after all, probabilistically complete but **not optimal**.

**B — localisation** (one run over ~160 steps):

| | mean error | final error |
|---|---|---|
| particle filter | **0.068 m** | **0.089 m** |
| odometry (dead reckoning) | 0.541 m | **1.370 m** |

The odometry error **grows without bound** (a random walk, above all the orientation error), the filter stays **bounded** thanks to the landmarks. Particle count: $M=10$ → 0.494 m, $M=50$ → 0.092 m, $M=200$ → 0.082 m, $M=1000$ → 0.070 m — the gain **saturates** from a few hundred particles onwards.

**C — closed loop** (10 runs):

| control on | goal reached | final distance | collisions |
|---|---|---|---|
| **particle filter** | **10/10** | 0.37 m | **0/10** |
| odometry only | 2/10 | 1.39 m | 7/10 |

> **The big lesson.** A perfect plan is worth nothing if the robot does not know where it is. Odometry is an **integrating** process — every small error stays in the state forever, the error grows without bound. Only the **correction step of the Bayes filter** (an external measurement) breaks this drift. Sense, plan and act are not three separate modules but a **loop**: the control acts on the *estimate*, and the quality of the estimate decides between success and collision.

## Setup & running

```bash
cd modules/21-robotics-1/projects/03-final
# write your own implementation, then:
/Users/.../.venv/bin/python test_navigation.py   # test suite
/Users/.../.venv/bin/python run.py                # 3 experiments + plots
```

Only `numpy` + `matplotlib`. Runtime ~10 s (pure simulation, no training).

## Solution

The complete reference is in [`solution/`](solution/): `navigation.py` (world, RRT, smoothing/densification, unicycle model, particle filter, pure pursuit), `run.py` (3 experiments + plots), `test_navigation.py` (9 tests).

## Looking back & ahead

This closes module 21: from the **forward kinematics and the Jacobian** (P01) via **inverse kinematics with DLS** (P02) to the **complete navigation stack** (P03). It continues in **module 22 "Robotics 2"** with dynamics (forces instead of geometry alone), **SLAM** (estimating map *and* pose simultaneously — the continuation of experiment B) and advanced control.

---
---

# P03 (final) — Sense-Plan-Act: RRT-Planung, Partikelfilter-Lokalisierung, Pfadverfolgung (deutsche Fassung)

**Modul 21 — Robotics 1** · Format: **Python-Projekt (freie Umsetzung, keine Code-Vorgabe)**

> Abschlussprojekt. **Kein vorgegebener Code** — du baust den kompletten Navigationsstack selbst. Referenzlösung in [`solution/`](solution/); **erst selbst versuchen**. Diese README ist die Spezifikation.

## Worum es geht

Du baust den **vollständigen Sense-Plan-Act-Zyklus** eines mobilen Roboters (Skript Kap. 1) — und zeigst, dass er nur als **Ganzes** funktioniert:

- **PLAN**: ein **RRT** findet einen kollisionsfreien Weg durch eine Hindernislandschaft, eine **Shortcut-Glättung** kürzt ihn.
- **SENSE**: Die Odometrie **driftet** (Dead Reckoning); ein **Partikelfilter** korrigiert sie mit Entfernungsmessungen zu bekannten Landmarken.
- **ACT**: Ein **Pure-Pursuit-Regler** folgt dem Pfad — und zwar auf der **geschätzten**, nicht der wahren Pose.

Der Kernbefund, den du empirisch belegst: **Ohne das SENSE bricht alles zusammen.** Ein Roboter, der nur seiner Odometrie glaubt, verlässt den perfekt geplanten Pfad und kollidiert.

## Lernziel

Du integrierst die drei Säulen des Moduls — **Planung** (Kap. 7–8), **Zustandsschätzung** (Kap. 9–11) und **Regelung** (Kap. 12) — zu einem System und evaluierst es quantitativ.

## Vorwissen

Modul-21-[Skript](../../README.md), besonders **Kap. 7–12**. A\*/Suche aus Modul 06, Bayes aus Modul 07, Sensorfusion aus Modul 17/18.

---

## Aufgabenstellung (Spezifikation)

### 1. Welt und Roboter

- **Welt**: 2D-Bereich (z. B. $10\times10$) mit kreisförmigen **Hindernissen**, Kollisionstest für Punkte *und* Strecken (dichte Abtastung entlang der Kante). Ein **Sicherheitsabstand** (Roboterradius) muss einstellbar sein.
- **Roboter**: **Einspurmodell (unicycle)** mit Zustand $(x, y, \theta)$:
  $$x \mathrel{+}= v\cos\theta\,\Delta t,\quad y \mathrel{+}= v\sin\theta\,\Delta t,\quad \theta \mathrel{+}= \omega\,\Delta t$$
  Die *wahre* Bewegung bekommt **Rauschen** auf $v$ und $\omega$; die **Odometrie** rechnet mit den reinen Sollbefehlen (und driftet dadurch).
- **Sensor**: **Entfernungen zu bekannten Landmarken**, verrauscht.

Warum **synthetisch**: Nur so kennst du die *wahre* Pose und kannst den Lokalisierungsfehler überhaupt messen — bei einem echten Roboter gibt es keine ground truth.

### 2. PLAN — RRT + Glättung

Implementiere **RRT** (Skript Kap. 8): Stichprobe (mit **Goal-Bias**) → nächster Baumknoten → Schritt der Länge $\varepsilon$ → **Kollisionstest der Kante** → einfügen; bei Zielnähe Pfad durch Rückverfolgen der Eltern. Danach **Shortcut-Glättung** (wiederholt zwei Pfadpunkte direkt verbinden, wenn frei).

> **Zwei Fallstricke, die du treffen wirst:** (1) Nach der Glättung besteht der Pfad aus **wenigen weit auseinanderliegenden Ecken** — der Pfadverfolger braucht dann **verdichtete** Stützpunkte, sonst springt sein Zielpunkt. (2) Der Regler **schneidet Kurven**; plane deshalb mit **Sicherheitsabstand**, sonst streift der Roboter Hindernisse, obwohl der Pfad formal frei war.

### 3. SENSE — Partikelfilter

Implementiere die Monte-Carlo-Lokalisierung (Skript Kap. 11):
1. **Prädiktion**: Bewegungsmodell + Rauschen auf jedes Partikel.
2. **Gewichtung**: $w^{[m]} \propto p(\mathbf z\mid \mathbf x^{[m]})$ — Gauß-Likelihood der Entfernungen. *Rechne in **log**-Likelihoods und ziehe das Maximum ab, sonst unterlaufen die Gewichte numerisch.*
3. **Resampling**: **systematisch**, und **nur wenn** $N_{\text{eff}} = 1/\sum_m (w^{[m]})^2$ unter $M/2$ fällt (gegen **Partikelverarmung**).
4. **Schätzung**: gewichteter Mittelwert; die **Orientierung** über $\operatorname{atan2}(\sum w\sin\theta, \sum w\cos\theta)$ mitteln (nicht naiv arithmetisch!).

### 4. ACT — Pfadverfolgung

**Pure Pursuit**: Suche den am weitesten fortgeschrittenen Pfadpunkt im Vorausschau-Radius, drehe proportional zum Winkelfehler dorthin, fahre bei großem Winkelfehler langsamer. **Begrenze $\omega$** (Sättigung) — sonst dreht der Roboter pro Zeitschritt zu weit und schwingt auf.

### 5. Evaluation (drei Experimente)

- **A — Planung**: Erfolgsrate, Baumgröße und Pfadlänge über verschiedene **Goal-Bias**-Werte; Wirkung der **Glättung**.
- **B — Lokalisierung**: Fehlerverlauf **Partikelfilter vs. reine Odometrie** über die Fahrt; **Partikelzahl** $M$ variieren.
- **C — Geschlossener Kreis**: Navigation mit Regelung auf der **Filter-Schätzung** vs. auf **reiner Odometrie** — Zielerreichung, Endabstand, Kollisionen (je 10 Läufe).

Plots nach `results/` (gitignored), Testsuite als `__main__`-Runner.

---

## Was am Ende herauskommen soll (Referenz-Größenordnungen)

**A — Planung**: RRT findet in dieser Welt immer eine Lösung. Der Goal-Bias senkt den Suchaufwand spürbar (~200 → 120–140 Knoten), ändert die Pfadlänge kaum. Der große Gewinn kommt vom **Glätten**: ~16.3 → ~13.6 (rund **17 % kürzer**) — RRT ist eben probabilistisch vollständig, aber **nicht optimal**.

**B — Lokalisierung** (ein Lauf über ~160 Schritte):

| | Fehler Mittel | Fehler Ende |
|---|---|---|
| Partikelfilter | **0.068 m** | **0.089 m** |
| Odometrie (dead reckoning) | 0.541 m | **1.370 m** |

Der Odometriefehler **wächst unbeschränkt** (Random Walk, v. a. der Orientierungsfehler), der Filter bleibt durch die Landmarken **beschränkt**. Partikelzahl: $M=10$ → 0.494 m, $M=50$ → 0.092 m, $M=200$ → 0.082 m, $M=1000$ → 0.070 m — der Gewinn **sättigt** ab einigen Hundert Partikeln.

**C — Geschlossener Kreis** (10 Läufe):

| Regelung auf | Ziel erreicht | Endabstand | Kollisionen |
|---|---|---|---|
| **Partikelfilter** | **10/10** | 0.37 m | **0/10** |
| nur Odometrie | 2/10 | 1.39 m | 7/10 |

> **Die große Lehre.** Ein perfekter Plan nützt nichts, wenn der Roboter nicht weiß, wo er ist. Die Odometrie ist ein **integrierender** Prozess — jeder kleine Fehler bleibt für immer im Zustand, der Fehler wächst unbeschränkt. Erst der **Korrekturschritt des Bayes-Filters** (externe Messung) bricht diesen Drift. Sense, Plan und Act sind keine drei getrennten Module, sondern ein **Kreislauf**: Die Regelung wirkt auf die *Schätzung*, und die Qualität der Schätzung entscheidet über Erfolg oder Kollision.

## Setup & Ausführen

```bash
cd modules/21-robotics-1/projects/03-final
# eigene Umsetzung schreiben, dann:
/Users/.../.venv/bin/python test_navigation.py   # Testsuite
/Users/.../.venv/bin/python run.py                # 3 Experimente + Plots
```

Nur `numpy` + `matplotlib`. Laufzeit ~10 s (reine Simulation, kein Training).

## Lösung

Vollständige Referenz in [`solution/`](solution/): `navigation.py` (Welt, RRT, Glättung/Verdichtung, Einspurmodell, Partikelfilter, Pure Pursuit), `run.py` (3 Experimente + Plots), `test_navigation.py` (9 Tests).

## Rückblick & Ausblick

Damit schließt Modul 21: von der **Vorwärtskinematik und Jacobi** (P01) über die **inverse Kinematik mit DLS** (P02) zum **vollständigen Navigationsstack** (P03). Weiter geht es in **Modul 22 „Robotics 2"** mit Dynamik (Kräfte statt nur Geometrie), **SLAM** (Karte *und* Pose gleichzeitig schätzen — die Fortsetzung von Experiment B) und fortgeschrittener Regelung.
