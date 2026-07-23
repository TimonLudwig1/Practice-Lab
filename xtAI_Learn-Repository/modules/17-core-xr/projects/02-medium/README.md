# Project 02 (medium) — motion-to-photon: the millisecond budget

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Format: Python project** (`.py` + tests). **Why?** This is about a **pipeline** with clear
inputs/outputs and numbers that have to be exactly right — and unit tests are the right tool for
that. The physics of latency can be secured precisely.

---

## Goal

Project 01 showed *how* the head is tracked. This project covers the **most critical quantity of
XR**: the **motion-to-photon latency** — the time from "the head moves" to "the matching photon
hits the retina". If it is too large, the world lags behind the movement and the user gets sick
(script 3 + 4.1).

You simulate the latency chain and the two tricks with which modern headsets **outwit** it:

1. break down the **latency budget** (why 90 Hz is tight),
2. measure **how strongly** the world lags at which latency,
3. **prediction** — extrapolating instead of lagging behind, including the **overshoot**,
4. **timewarp** — shifting the finished image shortly before display.

## Prior knowledge

Module 17 script, **section 3** (motion-to-photon, prediction, timewarp) and **4.1**
(cybersickness). Project 01 (orientation, tracking). NumPy.

## Files

| File | Role |
|---|---|
| `head_motion.py` | Produce realistic head motion. **Given** — infrastructure. |
| `latency.py` | The latency chain + countermeasures. **This is your work** (4 TODOs). |
| `run.py` | The experiment: budget, latency sweep, prediction sweep, timewarp. Given. |
| `test_latency.py` | Test suite (**13 tests**). |

## Task

Four functions in `latency.py` (the physics is yours, `LatencyBudget` and `ms_to_steps` are
given):

1. **`displayed_pose`** — the display shows the pose from `latency_ms` ago (shift the array by
   `lat` samples, hold the beginning with `np.clip`).
2. **`predicted_pose`** — prediction: `truth[i-lat] + velocity[i-lat] * horizon`. Mind the units
   (velocity in °/s, horizon in ms).
3. **`timewarped_pose`** — for a pure rotation the effective latency is simply the warp latency;
   the render latency is irrelevant (and does **not** appear in the formula — think about why).
4. **`angular_error`** — `|displayed − true|` per time point.

## What comes out at the end

`python test_latency.py` → **13/13 green** (<0.1 s). `python run.py` → the experiment (~1 s):

**1. The budget** (script 3.1): sensor 1.5 + fusion 1.0 + app 4.0 + **render 11.1** + scanout 3.0
+ display 2.0 = **22.6 ms** → above the 20 ms limit. A single frame at 90 Hz (11.1 ms) alone eats
half the budget.

**2. How strongly does the world lag?** (without countermeasures)

| Latency | mean error | max error |
|---|---|---|
| 5 ms | 0.56° | 4.21° |
| 20 ms | 2.22° | 16.42° |
| 50 ms | 5.43° | 35.35° |

**3. Prediction — the sweet spot is a U shape:**

| Horizon | mean error | vs. without |
|---|---|---|
| 0 ms | 2.22° | ±0 % |
| **20 ms** (= the latency) | **0.31°** | **−86 %** |
| 40 ms | 2.30° | −3 % |

The optimal prediction horizon is **exactly the latency** (you render for the moment of display).
**Too much** prediction (40 ms) is as bad as **none at all** — you extrapolate past the movement.
And the **overshoot**: at changes of direction the prediction error is **1.05°** vs. **0.22°**
during smooth movement (almost 5×) — there the extrapolation keeps running in the *old* direction.
That is why in practice one predicts only ~20–40 ms ahead, no more.

**4. Timewarp — making the latency invisible:**

A render latency of 20 ms → a mean error of **2.22°**. With timewarp (the warp fetches the pose at
2 ms) → **0.22°** (**−90 %**).

> **The central insight:** timewarp does **not make the latency smaller** — it makes it
> **invisible for the orientation**, and that is exactly what the vestibular system reacts to most
> sensitively (script 4.1). That is why the render latency does not appear in `timewarped_pose` at
> all. With **translation** the trick would not work: then information would be missing behind
> foreground objects that was never rendered (**disocclusion**, script 3.2).

## Running / setup

```bash
/.../xtAI_Learn-Repository/.venv/bin/python test_latency.py   # 13 tests, <0.1 s
/.../xtAI_Learn-Repository/.venv/bin/python run.py            # experiment + plots, ~1 s
```
Only `numpy` (+ `matplotlib`). CPU, no hardware. `pytest` optional (`__main__` runner).

## Solution

Complete in [`solution/`](solution/). Try it yourself first — one test recomputes the ideal case:
at a **constant** angular velocity linear prediction is **exact** (error < 1e-6), i.e. the entire
remaining error comes from **acceleration** (changes of direction).

## Going further

- **Prediction with quaternions** (project 01): instead of extrapolating numbers linearly, you
  extrapolate orientations via SLERP beyond the last frame. Build that with
  `scipy.spatial.transform`.
- **Noise**: put measurement noise on the angular velocity (as with the gyro in P01). Does that
  make prediction unstable? (It amplifies noise, because it uses the *derivative*.)
- **Stuttering instead of latency**: simulate a GPU that only manages 45 instead of 90 Hz and
  replace every second image by reprojection (**asynchronous spacewarp**). What does the error
  look like then?
- **An adaptive horizon**: reduce the prediction horizon when the acceleration is high (a change of
  direction detected). Does the overshoot disappear?

---

# Projekt 02 (medium) — Motion-to-Photon: das Millisekunden-Budget (deutsche Fassung)

**Format: Python-Projekt** (`.py` + Tests). **Warum?** Hier geht es um eine **Pipeline** mit
klaren Ein-/Ausgaben und Zahlen, die exakt stimmen müssen — dafür sind Unit-Tests das richtige
Werkzeug. Die Latenz-Physik lässt sich präzise absichern.

---

## Ziel

Projekt 01 hat gezeigt, *wie* der Kopf getrackt wird. Dieses Projekt behandelt die **kritischste
Größe von XR**: die **Motion-to-Photon-Latenz** — die Zeit von „Kopf bewegt sich" bis „passendes
Photon trifft die Netzhaut". Ist sie zu groß, hinkt die Welt der Bewegung hinterher, und dem
Nutzer wird übel (Skript 3 + 4.1).

Du simulierst die Latenzkette und die zwei Tricks, mit denen moderne Headsets sie **überlisten**:

1. das **Latenz-Budget** aufschlüsseln (warum 90 Hz eng ist),
2. messen, **wie stark** die Welt bei welcher Latenz nachhängt,
3. **Prediction** — extrapolieren statt hinterherhinken, inkl. des **Overshoots**,
4. **Timewarp** — das fertige Bild kurz vor der Anzeige nachschieben.

## Vorwissen

Skript **Modul 17, Abschnitt 3** (Motion-to-Photon, Prediction, Timewarp) und **4.1**
(Cybersickness). Projekt 01 (Orientierung, Tracking). NumPy.

## Dateien

| Datei | Rolle |
|---|---|
| `head_motion.py` | Realistische Kopfbewegung erzeugen. **Vorgegeben** — Infrastruktur. |
| `latency.py` | Die Latenzkette + Gegenmittel. **Hier ist deine Arbeit** (4 TODOs). |
| `run.py` | Das Experiment: Budget, Latenz-Sweep, Prediction-Sweep, Timewarp. Vorgegeben. |
| `test_latency.py` | Test-Suite (**13 Tests**). |

## Aufgabe

In `latency.py` vier Funktionen (die Physik ist deine, `LatencyBudget` und `ms_to_steps` sind
vorgegeben):

1. **`displayed_pose`** — das Display zeigt die Pose von vor `latency_ms` (Array um
   `lat` Samples verschieben, Anfang mit `np.clip` halten).
2. **`predicted_pose`** — Prediction: `truth[i-lat] + velocity[i-lat] * horizon`. Achte auf die
   Einheiten (velocity in °/s, horizon in ms).
3. **`timewarped_pose`** — für reine Rotation ist die effektive Latenz einfach die Warp-Latenz;
   die Render-Latenz ist egal (und taucht **nicht** in der Formel auf — überlege, warum).
4. **`angular_error`** — `|angezeigt − wahr|` je Zeitpunkt.

## Was am Ende herauskommt

`python test_latency.py` → **13/13 grün** (<0,1 s). `python run.py` → das Experiment (~1 s):

**1. Budget** (Skript 3.1): sensor 1,5 + fusion 1,0 + app 4,0 + **render 11,1** + scanout 3,0 +
display 2,0 = **22,6 ms** → über der 20-ms-Grenze. Allein ein Frame bei 90 Hz (11,1 ms) frisst
das halbe Budget.

**2. Wie stark hängt die Welt nach?** (ohne Gegenmittel)

| Latenz | mittl. Fehler | max Fehler |
|---|---|---|
| 5 ms | 0,56° | 4,21° |
| 20 ms | 2,22° | 16,42° |
| 50 ms | 5,43° | 35,35° |

**3. Prediction — der Sweet Spot ist eine U-Form:**

| Horizont | mittl. Fehler | vs. ohne |
|---|---|---|
| 0 ms | 2,22° | ±0 % |
| **20 ms** (= Latenz) | **0,31°** | **−86 %** |
| 40 ms | 2,30° | −3 % |

Der optimale Vorhersage-Horizont ist **exakt die Latenz** (man rendert für den Anzeige-Zeitpunkt).
**Zu viel** Vorhersage (40 ms) ist so schlecht wie **gar keine** — man extrapoliert an der Bewegung
vorbei. Und der **Overshoot**: An Richtungswechseln ist der Prediction-Fehler **1,05°** vs. **0,22°**
bei glatter Bewegung (fast 5×) — die Extrapolation läuft dort noch in die *alte* Richtung weiter.
Deshalb sagt man in der Praxis nur ~20–40 ms voraus, nicht mehr.

**4. Timewarp — die Latenz unsichtbar machen:**

Render-Latenz 20 ms → mittl. Fehler **2,22°**. Mit Timewarp (Warp holt die Pose bei 2 ms) →
**0,22°** (**−90 %**).

> **Die zentrale Einsicht:** Timewarp macht die Latenz **nicht kleiner** — es macht sie für die
> **Orientierung unsichtbar**, und genau darauf reagiert das Vestibularsystem am empfindlichsten
> (Skript 4.1). Deshalb taucht die Render-Latenz in `timewarped_pose` gar nicht auf. Bei
> **Translation** ginge der Trick nicht: dann fehlt hinter Vordergrundobjekten Information, die
> nie gerendert wurde (**Disokklusion**, Skript 3.2).

## Ausführen / Setup

```bash
/.../xtAI_Learn-Repository/.venv/bin/python test_latency.py   # 13 Tests, <0.1 s
/.../xtAI_Learn-Repository/.venv/bin/python run.py            # Experiment + Plots, ~1 s
```
Nur `numpy` (+ `matplotlib`). CPU, keine Hardware. `pytest` optional (`__main__`-Runner).

## Lösung

Vollständig in [`solution/`](solution/). Erst selbst versuchen — ein Test rechnet den Idealfall
nach: bei **konstanter** Winkelgeschwindigkeit ist lineare Prediction **exakt** (Fehler < 1e-6),
d. h. der ganze Restfehler kommt von **Beschleunigung** (Richtungswechseln).

## Weiterdenken

- **Prediction mit Quaternionen** (Projekt 01): Statt Zahlen linear zu extrapolieren, extrapoliert
  man Orientierungen per SLERP über den letzten Frame hinaus. Baue das mit
  `scipy.spatial.transform`.
- **Rauschen**: Leg auf die Winkelgeschwindigkeit Messrauschen (wie beim Gyro in P01). Wird
  Prediction dadurch instabil? (Sie verstärkt Rauschen, weil sie die *Ableitung* nutzt.)
- **Ruckeln statt Latenz**: Simuliere eine GPU, die nur 45 statt 90 Hz schafft, und ersetze jedes
  zweite Bild per Reprojection (**Asynchronous Spacewarp**). Wie sieht der Fehler dann aus?
- **Adaptiver Horizont**: Reduziere den Prediction-Horizont, wenn die Beschleunigung hoch ist
  (Richtungswechsel erkannt). Verschwindet der Overshoot?
