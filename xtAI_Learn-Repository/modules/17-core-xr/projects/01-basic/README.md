# Project 01 (basic) — head tracking: the mathematics behind presence

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format: Jupyter notebook** (`head_tracking.ipynb`). **Why?** You only understand rotations if you
**try them out**: rotate an axis, see a degree of freedom disappear, watch a drift curve run away.
Numbers and plots right next to the derivation — that is a notebook.

---

## Goal

You rebuild the mathematics that decides whether a headset produces **presence** or nausea — the
**orientation of the head**:

1. **rotations do not commute** — the source of countless XR bugs,
2. **gimbal lock** — how Euler angles lose a degree of freedom,
3. **quaternions** — why XR uses them, including the $q \equiv -q$ trap,
4. **SLERP vs. LERP** — rotating uniformly,
5. **IMU sensor fusion** — why a gyroscope drifts away and a two-liner repairs it.

## Prior knowledge

Module 17 script, **section 2** (tracking, rotations, sensor fusion). Linear algebra
(matrix × vector), trigonometry. `scipy.spatial.transform` is introduced in the notebook.

> ### Without a VR headset — and why that is not a makeshift
> In this environment there is no VR hardware and no 3D engine. But that hardly affects this
> project: the intellectual core of XR is **mathematics and timing behaviour**. In an engine both
> disappear behind `transform.rotation = Quaternion.Slerp(...)` — here you recompute it yourself
> and **see** what happens there. For practical work afterwards: Unity + OpenXR.

## Task

Most of it is given (this is the basic project!). You fill in **five** `# TODO` blocks:

1. apply the two rotation orders (`(B*A)` vs. `(A*B)`),
2. compute the angular distance in the gimbal lock test,
3. build the quaternion **by hand** from $q=(\cos\frac\theta2,\ \hat{\mathbf n}\sin\frac\theta2)$,
4. **LERP** by hand (mix + normalize),
5. the three IMU estimators (gyro integration, accel, the **complementary filter**).

## What comes out at the end

**1. Rotations do not commute.** The same two 90° rotations applied to $\hat z$:
`A then B → [0,-1,0]`, `B then A → [1,0,0]`. **Different points.** That is why "Euler angles"
without stating the order (ZYX? XYZ?) is meaningless.

**2. Gimbal lock** (convention ZYX, pitch = 90°):

| yaw | roll | yaw − roll | quaternion | distance to the reference |
|---|---|---|---|---|
| 0° | 0° | 0 | `[0, 0.7071, 0, 0.7071]` | **0.000000°** |
| 40° | 40° | 0 | `[0, 0.7071, 0, 0.7071]` | **0.000000°** |
| 90° | 90° | 0 | `[0, 0.7071, 0, 0.7071]` | **0.000000°** |

Not *similar* — **identical**. Only the **difference** survives (yaw−roll = 0/20/40/80 → a
distance of 0°/20°/40°/80°). **3 DoF → 2 DoF.** The back-conversion `as_euler` of (0°, 90°, 40°)
returns `[-40, 90, 0]`: scipy folds everything into the yaw — it *cannot* do otherwise, the
information is gone. At pitch = 0 everything is normal (yaw=40/roll=40 → 55.98° from the
identity).

> **A widespread misconception is explicitly refuted:** "(yaw=0,roll=40) and (yaw=40,roll=0) are
> then equal" — **no**, those lie **80°** apart (differences of −40 vs. +40). It is the
> *difference* that survives, not the individual value. *(I fell for this myself while building —
> which is why the counter-check is in the notebook.)*

**3. Quaternions.** Built by hand = scipy ✓. And the trap: `q` and `-q` are **0.000000°** apart —
the same rotation, different numbers (the double cover, a consequence of the $\theta/2$). Whoever
compares poses with `==` has a bug; the correct way is $|q_1\cdot q_2|\approx1$.

**4. SLERP vs. LERP** (0° → 170°): SLERP has **exactly constant** angular steps (spread
**0.000**), LERP does not (**5.659**) — LERP runs the chord instead of the arc and is too fast in
the middle. Negligible for small angles (which is why LERP between dense frames is common).

**5. Sensor fusion** (60 s, 100 Hz, gyro bias 0.5 °/s):

| Estimator | RMSE | final error |
|---|---|---|
| gyro only (integrated) | 16.98° | **29.95°** ← drift |
| accel only | 3.01° | 1.22° |
| **complementary filter** | **0.42°** | **0.32°** |

The prediction is accurate to the decimal place: 0.5 °/s × 60 s = **30°** — the measured final
error is **29.95°**. And the filter (**one line**) is **better than either individual sensor**:
two poor sources become one good one, because their errors are complementary.

Runtime: **~3 s** in total.

> ### The connection to the script
> This is not a mathematical finger exercise. According to Slater (script 1.2), **presence** arises
> from **sensorimotor contingency** — the world has to react to the head movement *as it would in
> reality*. If the gyro drifts (5), the world tips; if a gimbal lock catches you (2), the view
> jerks; if you interpolate wrongly (4), it wobbles. **In every case the place illusion collapses —
> and the user gets sick.** Hence quaternions, hence sensor fusion.

## Running / setup

Repo `venv`. Open the notebook:
`/.../xtAI_Learn-Repository/.venv/bin/python -m jupyter lab` (or the `.venv` kernel in VS Code).
Only `numpy`, `scipy`, `matplotlib` — all available. **No download, no hardware, no GPU.**

## Solution

Fully solved and **executed** in
[`solution/head_tracking_solution.ipynb`](solution/head_tracking_solution.ipynb). Try it yourself
first! It contains five extension tasks (vary α — what happens at α=1.0?; multiply the bias by ten;
**the long way round** for SLERP 0°→350° and what that has to do with $q\equiv-q$; look for gimbal
lock in quaternions; why the accelerometer cannot correct **yaw** in principle).

## Transfer

Orientation, drift and interpolation are the building blocks of the **motion-to-photon chain**
(script 3) — prediction extrapolates exactly the pose you estimate here, and timewarp corrects it
shortly before display. That is the material of the following projects.

---

# Projekt 01 (basic) — Head-Tracking: die Mathematik hinter der Präsenz (deutsche Fassung)

**Format: Jupyter Notebook** (`head_tracking.ipynb`). **Warum?** Rotationen versteht man nur,
wenn man sie **ausprobiert**: eine Achse drehen, sehen wie ein Freiheitsgrad verschwindet, eine
Drift-Kurve davonlaufen sehen. Zahlen und Plots direkt neben der Herleitung — das ist ein Notebook.

---

## Ziel

Du baust die Mathematik nach, die darüber entscheidet, ob ein Headset **Präsenz** erzeugt oder
Übelkeit — die **Orientierung des Kopfes**:

1. **Rotationen kommutieren nicht** — die Quelle unzähliger XR-Bugs,
2. **Gimbal Lock** — wie Euler-Winkel einen Freiheitsgrad verlieren,
3. **Quaternionen** — warum XR sie benutzt, inkl. der $q \equiv -q$-Falle,
4. **SLERP vs. LERP** — gleichmäßig drehen,
5. **IMU-Sensorfusion** — warum ein Gyroskop wegdriftet und ein Zweizeiler das repariert.

## Vorwissen

Skript **Modul 17, Abschnitt 2** (Tracking, Rotationen, Sensorfusion). Lineare Algebra
(Matrix × Vektor), Trigonometrie. `scipy.spatial.transform` wird im Notebook eingeführt.

> ### Ohne VR-Brille — und warum das kein Notbehelf ist
> In dieser Umgebung gibt es keine VR-Hardware und keine 3D-Engine. Das trifft dieses Projekt
> aber kaum: Der intellektuelle Kern von XR ist **Mathematik und Zeitverhalten**. In einer Engine
> verschwindet beides hinter `transform.rotation = Quaternion.Slerp(...)` — hier rechnest du es
> selbst nach und **siehst**, was dort passiert. Für die Praxis danach: Unity + OpenXR.

## Aufgabe

Das meiste ist vorgegeben (basic!). Du füllst **fünf** `# TODO`-Blöcke:

1. Die zwei Rotations-Reihenfolgen anwenden (`(B*A)` vs. `(A*B)`),
2. den Winkelabstand im Gimbal-Lock-Test berechnen,
3. das Quaternion **von Hand** aus $q=(\cos\frac\theta2,\ \hat{\mathbf n}\sin\frac\theta2)$ bauen,
4. **LERP** von Hand (mischen + normieren),
5. die drei IMU-Schätzer (Gyro-Integration, Accel, **Komplementärfilter**).

## Was am Ende herauskommt

**1. Rotationen kommutieren nicht.** Dieselben zwei 90°-Drehungen auf $\hat z$:
`erst A dann B → [0,-1,0]`, `erst B dann A → [1,0,0]`. **Verschiedene Punkte.** Deshalb ist
„Euler-Winkel" ohne Angabe der Reihenfolge (ZYX? XYZ?) bedeutungslos.

**2. Gimbal Lock** (Konvention ZYX, Pitch = 90°):

| yaw | roll | yaw − roll | Quaternion | Abstand zur Referenz |
|---|---|---|---|---|
| 0° | 0° | 0 | `[0, 0.7071, 0, 0.7071]` | **0,000000°** |
| 40° | 40° | 0 | `[0, 0.7071, 0, 0.7071]` | **0,000000°** |
| 90° | 90° | 0 | `[0, 0.7071, 0, 0.7071]` | **0,000000°** |

Nicht *ähnlich* — **identisch**. Nur die **Differenz** überlebt (yaw−roll = 0/20/40/80 → Abstand
0°/20°/40°/80°). **3 DoF → 2 DoF.** Die Rückrechnung `as_euler` von (0°, 90°, 40°) liefert
`[-40, 90, 0]`: scipy faltet alles in den yaw — es *kann* nicht anders, die Information ist weg.
Bei Pitch = 0 ist alles normal (yaw=40/roll=40 → 55,98° von der Identität).

> **Ein verbreiteter Irrtum wird explizit widerlegt:** „(yaw=0,roll=40) und (yaw=40,roll=0) sind
> dann gleich" — **nein**, die liegen **80°** auseinander (Differenzen −40 vs. +40). Es ist die
> *Differenz*, die überlebt, nicht der Einzelwert. *(Ich bin beim Bauen selbst darauf
> reingefallen — deshalb steht die Gegenprobe im Notebook.)*

**3. Quaternionen.** Von Hand gebaut = scipy ✓. Und die Falle: `q` und `-q` sind **0,000000°**
auseinander — dieselbe Rotation, verschiedene Zahlen (doppelte Überdeckung, Folge des $\theta/2$).
Wer Posen mit `==` vergleicht, hat einen Bug; richtig ist $|q_1\cdot q_2|\approx1$.

**4. SLERP vs. LERP** (0° → 170°): SLERP hat **exakt konstante** Winkelschritte (Streuung
**0,000**), LERP nicht (**5,659**) — LERP läuft die Sehne statt des Bogens und ist in der Mitte
zu schnell. Bei kleinen Winkeln vernachlässigbar (deshalb ist LERP zwischen dichten Frames üblich).

**5. Sensorfusion** (60 s, 100 Hz, Gyro-Bias 0,5 °/s):

| Schätzer | RMSE | Endfehler |
|---|---|---|
| nur Gyro (integriert) | 16,98° | **29,95°** ← Drift |
| nur Accel | 3,01° | 1,22° |
| **Komplementärfilter** | **0,42°** | **0,32°** |

Die Vorhersage stimmt auf die Nachkommastelle: 0,5 °/s × 60 s = **30°** — gemessener Endfehler
**29,95°**. Und der Filter (**eine Zeile**) ist **besser als beide Einzelsensoren**: aus zwei
schlechten Quellen wird eine gute, weil ihre Fehler komplementär sind.

Laufzeit: **~3 s** komplett.

> ### Die Verbindung zum Skript
> Das ist keine Mathe-Fingerübung. Nach Slater (Skript 1.2) entsteht **Präsenz** aus
> **sensomotorischer Kontingenz** — die Welt muss auf die Kopfbewegung reagieren *wie in echt*.
> Driftet das Gyro (5), kippt die Welt; erwischt dich ein Gimbal Lock (2), ruckt der Blick;
> interpolierst du falsch (4), taumelt es. **In jedem Fall bricht die Place Illusion zusammen —
> und dem Nutzer wird schlecht.** Deshalb Quaternionen, deshalb Sensorfusion.

## Ausführen / Setup

Repo-`venv`. Notebook öffnen:
`/.../xtAI_Learn-Repository/.venv/bin/python -m jupyter lab` (oder `.venv`-Kernel in VS Code).
Nur `numpy`, `scipy`, `matplotlib` — alles vorhanden. **Kein Download, keine Hardware, kein GPU.**

## Lösung

Vollständig gelöst und **ausgeführt** in
[`solution/head_tracking_solution.ipynb`](solution/head_tracking_solution.ipynb). Erst selbst
probieren! Enthält fünf Erweiterungs-Aufgaben (α variieren — was passiert bei α=1,0?; Bias
verzehnfachen; **der lange Weg** bei SLERP 0°→350° und was das mit $q\equiv-q$ zu tun hat;
Gimbal Lock bei Quaternionen suchen; warum das Accelerometer **yaw** prinzipiell nicht
korrigieren kann).

## Transfer

Orientierung, Drift und Interpolation sind die Bausteine der **Motion-to-Photon-Kette**
(Skript 3) — Prediction extrapoliert genau die Pose, die du hier schätzt, und Timewarp
korrigiert sie kurz vor der Anzeige nach. Das ist der Stoff der folgenden Projekte.
