# Projekt 01 (basic) — Head-Tracking: die Mathematik hinter der Präsenz

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
