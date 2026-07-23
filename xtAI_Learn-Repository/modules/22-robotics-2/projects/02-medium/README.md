# P02 (medium) — model-based control: computed torque vs. PD + gravity compensation

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 22 — Robotics 2** · Format: **Python module + test suite**

## Goal

You use the dynamics from P01 to build the two workhorse controllers of manipulator control and prove *why* one of them turns a wildly non-linear, coupled arm into a set of trivial linear systems:

1. **PD + gravity compensation** — $\boldsymbol\tau = -\mathbf K_p\mathbf e - \mathbf K_d\dot{\mathbf e} + \mathbf g(\mathbf q)$ (script ch. 6).
2. **Computed torque (inverse-dynamics control)** — $\boldsymbol\tau = \mathbf M(\mathbf q)(\ddot{\mathbf q}_d - \mathbf K_d\dot{\mathbf e} - \mathbf K_p\mathbf e) + \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} + \mathbf g(\mathbf q)$ (script ch. 7).
3. Four experiments: **regulation**, **tracking**, **feedback linearisation** (the error obeys an exact linear ODE), and **model error**.

## Why this format?

A **Python module with a test suite** — the controllers are small, sharply testable functions (feed the computed torque into the true dynamics and the acceleration must equal $\ddot{\mathbf q}_d - \mathbf K_d\dot{\mathbf e} - \mathbf K_p\mathbf e$ *exactly*), and the experiments systematically vary trajectory speed and model error.

## Why synthetic data?

This is about a **control law and a stability property**, not measured data. A known model lets you verify the *exact* cancellation and set the model error deliberately (experiment 4), which no real robot allows.

## Prior knowledge

**P01** of this module (you built $\mathbf M,\mathbf C,\mathbf g$ — here they are given), the module 22 script ch. 6–7, PD control, second-order linear ODEs. The gain choice connects to **LQR (module 14)**.

## Task

Open `control.py`. The dynamics model (`ArmModel`), the reference trajectory and the closed-loop simulator are given — you implement the **two controllers** (`# TODO` / `NotImplementedError`):

1. **`pd_gravity_control(...)`** — $-\mathbf K_p\mathbf e - \mathbf K_d\dot{\mathbf e} + \mathbf g(\mathbf q)$.
2. **`computed_torque_control(...)`** — $\mathbf M(\ddot{\mathbf q}_d - \mathbf K_d\dot{\mathbf e} - \mathbf K_p\mathbf e) + \mathbf C\dot{\mathbf q} + \mathbf g$.

Then:

```bash
cd modules/22-robotics-2/projects/02-medium
/Users/.../.venv/bin/python test_control.py   # 6 tests -> all PASS
/Users/.../.venv/bin/python run.py             # 4 experiments + plots
```

The gains are $\mathbf K_p=100\,\mathbf I$, $\mathbf K_d=20\,\mathbf I$ — i.e. natural frequency $\omega_0=\sqrt{100}=10$ and **critical damping** ($\mathbf K_d=2\omega_0$).

## What should come out (expected values)

**Experiment 1 — regulation.** For a *constant* setpoint, both controllers reach it: PD+gravity final error ~**3.7e-4**, computed torque ~**4.5e-12**. PD+gravity is enough for setpoints.

**Experiment 2 — tracking** (RMS error over the second half, vs. trajectory speed $w$):

| $w$ | PD+gravity RMS | computed-torque RMS | ratio |
|---|---|---|---|
| 1.0 | 0.0411 | 6.43e-04 | 64x |
| 2.0 | 0.1735 | 1.50e-03 | 116x |
| 3.0 | 0.4188 | 2.17e-03 | 193x |
| 4.0 | 0.7379 | 2.95e-03 | 250x |

PD+gravity's error grows with speed (it ignores $\mathbf M\ddot{\mathbf q}_d$ and the Coriolis coupling); computed torque feeds those forward and stays near zero. **The gap widens with speed.**

**Experiment 3 — feedback linearisation.** Computed torque makes the error obey $\ddot{\mathbf e}+\mathbf K_d\dot{\mathbf e}+\mathbf K_p\mathbf e=\mathbf 0$, i.e. the critically-damped decay $\mathbf e(t)=\mathbf e_0(1+\omega_0 t)e^{-\omega_0 t}$. Measured deviation from this analytic curve: **~2.7e-4 in *both* configurations** (config-independent — the model truly linearises and decouples the arm). PD+gravity's decay is configuration-dependent (differs by 0.010 between poses; $\mathbf M$ is not cancelled).

**Experiment 4 — model error.** Computed torque needs the model; as the controller's mass estimate drifts from the truth, tracking degrades: RMS **0.0015 (exact) → 0.0147 (+10 %) → 0.0269 (+20 %) → 0.0539 (+50 %)**. This graceful degradation is why real systems add robust/adaptive control on top.

> **The lesson.** PD+gravity **regulates** (reaches a setpoint) but **lags** a moving target. Computed torque **cancels the dynamics through the model**, leaving an exactly linear, decoupled error system whose eigenvalues you pick — that is **feedback linearisation**, and it gives near-perfect tracking. The price is that you must *know* the model: the cancellation is only as good as $\mathbf M,\mathbf C,\mathbf g$. Choosing the gains $\mathbf K_p,\mathbf K_d$ optimally for the resulting linear system is exactly the **LQR** problem of module 14 — the two ideas compose.

## Solution

The complete reference is in [`solution/`](solution/). Try it yourself first!

## What comes next

**P03 (final)**: **pose-graph SLAM** — estimate the whole trajectory *and* the map at once by minimising a sparse non-linear least-squares problem, and watch a single loop closure collapse the odometry drift. No code given.

---
---

# P02 (medium) — modellbasierte Regelung: Computed Torque vs. PD + Gravitationskompensation (deutsche Fassung)

**Modul 22 — Robotics 2** · Format: **Python-Modul + Testsuite**

## Ziel

Du nutzt die Dynamik aus P01, um die zwei Arbeitspferd-Regler der Manipulatorregelung zu bauen und *warum* der eine einen hochgradig nichtlinearen, gekoppelten Arm in eine Menge trivialer linearer Systeme verwandelt:

1. **PD + Gravitationskompensation** — $\boldsymbol\tau = -\mathbf K_p\mathbf e - \mathbf K_d\dot{\mathbf e} + \mathbf g(\mathbf q)$ (Skript Kap. 6).
2. **Computed Torque (inverse-Dynamik-Regelung)** — $\boldsymbol\tau = \mathbf M(\mathbf q)(\ddot{\mathbf q}_d - \mathbf K_d\dot{\mathbf e} - \mathbf K_p\mathbf e) + \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} + \mathbf g(\mathbf q)$ (Skript Kap. 7).
3. Vier Experimente: **Regelung**, **Verfolgung**, **Feedback-Linearisierung** (der Fehler gehorcht einer exakten linearen DGL) und **Modellfehler**.

## Warum dieses Format?

Ein **Python-Modul mit Testsuite** — die Regler sind kleine, scharf testbare Funktionen (speist man das Computed Torque in die wahre Dynamik, muss die Beschleunigung *exakt* $\ddot{\mathbf q}_d - \mathbf K_d\dot{\mathbf e} - \mathbf K_p\mathbf e$ sein), und die Experimente variieren systematisch Trajektoriengeschwindigkeit und Modellfehler.

## Warum synthetische Daten?

Es geht um ein **Regelgesetz und eine Stabilitätseigenschaft**, nicht um Messdaten. Ein bekanntes Modell erlaubt, die *exakte* Wegkürzung zu verifizieren und den Modellfehler gezielt einzustellen (Experiment 4), was kein echter Roboter zulässt.

## Vorwissen

**P01** dieses Moduls (du hast $\mathbf M,\mathbf C,\mathbf g$ gebaut — hier vorgegeben), das Modul-22-Skript Kap. 6–7, PD-Regelung, lineare DGL zweiter Ordnung. Die Verstärkungswahl knüpft an **LQR (Modul 14)** an.

## Aufgabenstellung

Öffne `control.py`. Das Dynamikmodell (`ArmModel`), die Referenztrajektorie und der Regelkreis-Simulator sind vorgegeben — du implementierst die **zwei Regler** (`# TODO` / `NotImplementedError`):

1. **`pd_gravity_control(...)`** — $-\mathbf K_p\mathbf e - \mathbf K_d\dot{\mathbf e} + \mathbf g(\mathbf q)$.
2. **`computed_torque_control(...)`** — $\mathbf M(\ddot{\mathbf q}_d - \mathbf K_d\dot{\mathbf e} - \mathbf K_p\mathbf e) + \mathbf C\dot{\mathbf q} + \mathbf g$.

Dann:

```bash
cd modules/22-robotics-2/projects/02-medium
/Users/.../.venv/bin/python test_control.py   # 6 Tests -> alle PASS
/Users/.../.venv/bin/python run.py             # 4 Experimente + Plots
```

Die Verstärkungen sind $\mathbf K_p=100\,\mathbf I$, $\mathbf K_d=20\,\mathbf I$ — also Eigenfrequenz $\omega_0=\sqrt{100}=10$ und **kritische Dämpfung** ($\mathbf K_d=2\omega_0$).

## Was am Ende herauskommt (Erwartungswerte)

**Experiment 1 — Regelung.** Für einen *konstanten* Sollwert erreichen ihn beide Regler: PD+Gravitation Endfehler ~**3.7e-4**, Computed Torque ~**4.5e-12**. PD+Gravitation genügt für Sollwerte.

**Experiment 2 — Verfolgung** (RMS-Fehler über die zweite Hälfte, gegen die Trajektoriengeschwindigkeit $w$):

| $w$ | PD+Gravitation RMS | Computed-Torque RMS | Verhältnis |
|---|---|---|---|
| 1.0 | 0.0411 | 6.43e-04 | 64x |
| 2.0 | 0.1735 | 1.50e-03 | 116x |
| 3.0 | 0.4188 | 2.17e-03 | 193x |
| 4.0 | 0.7379 | 2.95e-03 | 250x |

PD+Gravitations-Fehler wächst mit der Geschwindigkeit (ignoriert $\mathbf M\ddot{\mathbf q}_d$ und die Coriolis-Kopplung); Computed Torque speist diese vorwärts und bleibt nahe null. **Die Lücke wächst mit der Geschwindigkeit.**

**Experiment 3 — Feedback-Linearisierung.** Computed Torque lässt den Fehler $\ddot{\mathbf e}+\mathbf K_d\dot{\mathbf e}+\mathbf K_p\mathbf e=\mathbf 0$ gehorchen, also die kritisch gedämpfte Abklingkurve $\mathbf e(t)=\mathbf e_0(1+\omega_0 t)e^{-\omega_0 t}$. Gemessene Abweichung von dieser analytischen Kurve: **~2.7e-4 in *beiden* Konfigurationen** (konfigurationsunabhängig — das Modell linearisiert und entkoppelt den Arm wirklich). PD+Gravitations-Abklingen ist konfigurationsabhängig (unterscheidet sich um 0.010 zwischen Posen; $\mathbf M$ wird nicht weggekürzt).

**Experiment 4 — Modellfehler.** Computed Torque braucht das Modell; driftet die Massenschätzung des Reglers von der Wahrheit ab, verschlechtert sich die Verfolgung: RMS **0.0015 (exakt) → 0.0147 (+10 %) → 0.0269 (+20 %) → 0.0539 (+50 %)**. Diese sanfte Degradation ist der Grund, warum reale Systeme robuste/adaptive Regelung darüber legen.

> **Die Lehre.** PD+Gravitation **regelt** (erreicht einen Sollwert), aber **läuft** einem bewegten Ziel **nach**. Computed Torque **hebt die Dynamik über das Modell weg** und lässt ein exakt lineares, entkoppeltes Fehlersystem übrig, dessen Eigenwerte du wählst — das ist **Feedback-Linearisierung**, und sie liefert nahezu perfekte Verfolgung. Der Preis: Man muss das Modell *kennen* — die Wegkürzung ist nur so gut wie $\mathbf M,\mathbf C,\mathbf g$. Die Verstärkungen $\mathbf K_p,\mathbf K_d$ für das resultierende lineare System optimal zu wählen, ist genau das **LQR**-Problem aus Modul 14 — die zwei Ideen verketten sich.

## Lösung

Die vollständige Referenz liegt in [`solution/`](solution/). Erst selbst versuchen!

## Weiter geht's

**P03 (final)**: **Pose-Graph-SLAM** — die ganze Trajektorie *und* die Karte gleichzeitig schätzen, indem man ein dünnbesetztes nichtlineares Least-Squares-Problem minimiert, und zusehen, wie ein einziges Loop Closure den Odometriedrift kollabieren lässt. Keine Code-Vorgabe.
