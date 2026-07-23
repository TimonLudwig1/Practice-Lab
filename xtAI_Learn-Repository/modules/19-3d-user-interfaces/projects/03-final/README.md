# P03 (final) — a comparative 3D selection study under clutter

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 19 — 3D User Interfaces** · Format: **Python project (free implementation, no given code)**

> The final project. **No given code** — you design and build the study yourself out of the tools from projects P01 (ray-object intersection) and P02 (the angular pointing model). The reference solution is in [`solution/`](solution/); **try it yourself first**. This README is the specification.

## What it is about

You carry out a **controlled evaluation study** comparing three selection techniques — **ray-casting**, **cone/flashlight** and the **bubble cursor** (script, ch. 5–7) — over various **target distances** and **scene densities**. The goal is the central insight of the module, empirically supported: **there is no universally best 3D selection technique** — the choice is a **trade-off between precision, reach/capture and robustness against clutter**, and this trade-off is the same disambiguation problem as reference resolution in module 18.

## Learning objective

You apply the entire module 19 toolbox: ray-object intersection (P01), the angular pointing model and Fitts (P02), and a sound **evaluation methodology** (ISO 9241-9 throughput + within-subject statistics as in module 17).

## Prior knowledge

The module 19 [script](../../README.md) completely, especially **ch. 5–7 (selection techniques), 10 (angular Fitts), 14 (ISO throughput)**. P01 (`ray_sphere`), P02 (angular size, Fitts). The statistical methodology from module 17 (Wilcoxon, effect size, multiplicity correction).

---

## Assignment (the specification)

### 1. The scene generator (disclosed, reproducible)

Build a seed-reproducible generator. A scene consists of:
- a **target object** at distance $L$ (a sphere, radius $r$),
- $N$ **distractors** that **angularly surround** the target (up to a spread angle), some of them **nearer** to the viewer (→ **occlusion** for ray-casting).

Why **synthetic**: only this way do you know the *ground truth* (which object is the target) and can set the density, distance and occlusion *independently* — with real VR logs none of these factors could be isolated.

### 2. The motor model

The pointer is a ray from the origin. The **intended** direction points at the target; the **real** direction has **angular Gaussian noise** $\sigma_\theta$ (hand tremor + tracking + Heisenberg, script ch. 12–13). Implement the noisy direction as a small 2D deflection in the tangent plane.

### 3. The three techniques

- **Ray-casting**: select the object **intersected** by the (noisy) ray with the **smallest $t$** (the nearest hit, via ray-sphere from P01); if nothing is hit → a mis-selection.
- **Cone**: among all objects in the cone (half angle $\alpha$) the one with the **smallest angle to the cone axis**.
- **Bubble**: the object with the **smallest angle to the surface** ($\text{the angle to the centre} - \text{the angular radius}$) — it always captures the angularly nearest one.

### 4. Evaluation (three experiments)

- **Experiment A — accuracy over the conditions**: measure the selection accuracy (many trials) in at least four conditions spanning *isolated/near*, *sparse/far*, *dense/near*, *dense/far*. Plus optionally a **distance sweep** (a sparse scene, $L=1\dots16$ m).
- **Experiment B — time & throughput**: for **isolated** targets, compute the selection time via the **angular Fitts' law** (from P02) with a technique-specific **effective capture width** $W_{\text{eff}}$ (ray-casting: the target's angular diameter; cone: $2\alpha$; bubble: a large Voronoi width). Report $MT$ **and** the throughput $TP=ID/MT$.
- **Experiment C — statistics**: simulate $\sim$16 "participants" (seeds), within-subject over the techniques. Compare pairwise with the **Wilcoxon signed-rank** test, **rank-biserial** as the effect size and the **Holm-Bonferroni** correction (from scratch, since statsmodels is missing).

The plots go to `results/` (gitignored), the test suite as a `__main__` runner.

---

## What should come out at the end (reference orders of magnitude)

Your numbers may vary with the parameters/seeds; the **story** has to hold:

**Experiment A** (accuracy):

| Condition | raycast | cone | bubble | the best |
|---|---|---|---|---|
| ISOLATED-NEAR | 0.96 | 0.95 | 0.97 | all good |
| SPARSE-FAR | **0.14** | 0.80 | 0.79 | cone/bubble |
| DENSE-NEAR | 0.26 | **0.51** | 0.46 | cone |
| DENSE-FAR | 0.10 | **0.30** | 0.28 | cone |

The distance sweep (sparse): ray-casting falls from ~0.72 ($L=1$) to **~0.04** ($L=16$); cone/bubble stay flat at ~0.87.

**Experiment B** (isolated targets): bubble is the **fastest** ($MT\approx0.31$ s vs. ray-casting $0.73$ s, ×2.4), because the large capture area lowers the $ID$ — but the **throughput** is *higher* for ray-casting ($\approx4.0$ vs. $2.6$ bit/s), because throughput rewards precision.

**Experiment C**: in SPARSE-FAR, cone/bubble beat ray-casting **highly significantly** (rank-biserial $=1.0$, Holm $p\approx0.001$); cone vs. bubble is n.s. In DENSE-NEAR, cone is **significantly better** than bubble ($p\approx0.0004$) — bubble **over-selects** the nearest neighbour.

> **The big lesson.** No technique wins everywhere:
> - **Ray-casting** is precise, but usable only for **near, large, isolated** targets — it collapses with **distance** (angular shrinkage, $\theta_W\approx W/L$) and with **occlusion/overlap**.
> - **Bubble** is **the fastest and most accurate** in sparse scenes (capture), but **over-selects** in **clutter** (it grabs the nearest neighbour).
> - **Cone** is the **most robust all-rounder**.
>
> Selecting in a crowd is a **disambiguation problem** — the same as multimodal reference resolution in module 18, only with angle/distance instead of time/semantics as the cues. Whoever chooses a technique implicitly chooses a position in the triangle **precision — speed — robustness**.

## Setup & running

```bash
cd modules/19-3d-user-interfaces/projects/03-final
# write your own implementation, then:
/Users/.../.venv/bin/python test_selection3d.py   # the test suite
/Users/.../.venv/bin/python run.py                 # 3 experiments + plots
```

Only `numpy`, `scipy` (for `wilcoxon`), `matplotlib`. Runtime ~3 s (pure geometry/statistics, no training).

## Solution

A complete reference is in [`solution/`](solution/): `selection3d.py` (the generator + techniques + throughput), `stats_tools.py` (rank-biserial, Holm from scratch), `run.py` (the three experiments + plots), `test_selection3d.py` (8 tests).

## Looking back & ahead

With this module 19 closes: from the **transformations + ray-casting** (P01) via the **angular pointing model + Go-Go** (P02) to the **comparative selection study** (P03). The 3D geometry and transformation mathematics is the direct foundation for **module 20 "3D Point Cloud Processing"** (registration/ICP, segmentation on point clouds).

---

# P03 (final) — Vergleichende 3D-Selektionsstudie unter Clutter (deutsche Fassung)

**Modul 19 — 3D User Interfaces** · Format: **Python-Projekt (freie Umsetzung, keine Code-Vorgabe)**

> Abschlussprojekt. **Kein vorgegebener Code** — du entwirfst und baust die Studie selbst aus den Werkzeugen der Projekte P01 (Ray-Objekt-Schnitt) und P02 (angulares Zeigemodell). Referenzlösung in [`solution/`](solution/); **erst selbst versuchen**. Diese README ist die Spezifikation.

## Worum es geht

Du führst eine **kontrollierte Evaluationsstudie** durch, die drei Selektionstechniken vergleicht — **Ray-Casting**, **Cone/Flashlight** und **Bubble Cursor** (Skript, Kap. 5–7) — über verschiedene **Zieldistanzen** und **Szenendichten**. Ziel ist die zentrale Erkenntnis des Moduls, empirisch belegt: **Es gibt keine universell beste 3D-Selektionstechnik** — die Wahl ist ein **Trade-off zwischen Präzision, Reichweite/Capture und Robustheit gegen Gedränge**, und dieser Trade-off ist dasselbe Disambiguierungsproblem wie die Referenzauflösung in Modul 18.

## Lernziel

Du wendest die gesamte Modul-19-Toolbox an: Ray-Objekt-Schnitt (P01), das angulare Zeigemodell und Fitts (P02), und eine saubere **Evaluationsmethodik** (ISO-9241-9-Throughput + within-subject-Statistik wie in Modul 17).

## Vorwissen

Modul-19-[Skript](../../README.md) komplett, besonders **Kap. 5–7 (Selektionstechniken), 10 (angulares Fitts), 14 (ISO-Throughput)**. P01 (`ray_sphere`), P02 (angulare Größe, Fitts). Statistik-Methodik aus Modul 17 (Wilcoxon, Effektstärke, Multiplizitätskorrektur).

---

## Aufgabenstellung (Spezifikation)

### 1. Szenen-Generator (offengelegt, reproduzierbar)

Baue einen geseedet-reproduzierbaren Generator. Eine Szene besteht aus:
- einem **Zielobjekt** in Distanz $L$ (Kugel, Radius $r$),
- $N$ **Distraktoren**, die das Ziel **angular umgeben** (bis zu einem Spread-Winkel), teils **näher** am Betrachter (→ **Verdeckung** für Ray-Casting).

Warum **synthetisch**: Nur so kennst du die *ground truth* (welches Objekt ist das Ziel) und kannst Dichte, Distanz und Verdeckung *unabhängig* einstellen — mit echten VR-Logs wäre keiner dieser Faktoren isolierbar.

### 2. Motor-Modell

Der Zeiger ist ein Strahl vom Ursprung. Die **intendierte** Richtung zeigt aufs Ziel; die **reale** Richtung hat **angulares Gauß-Rauschen** $\sigma_\theta$ (Hand-Tremor + Tracking + Heisenberg, Skript Kap. 12–13). Implementiere die verrauschte Richtung als kleine 2D-Auslenkung in der Tangentialebene.

### 3. Die drei Techniken

- **Ray-Casting**: selektiere das vom (verrauschten) Strahl **geschnittene** Objekt mit **kleinstem $t$** (nächster Treffer, via Ray-Kugel aus P01); nichts getroffen → Fehlselektion.
- **Cone**: unter allen Objekten im Kegel (Halbwinkel $\alpha$) das mit **kleinstem Winkel zur Kegelachse**.
- **Bubble**: das Objekt mit **kleinstem Winkel zur Oberfläche** ($\text{Winkel zum Zentrum} - \text{angularer Radius}$) — fängt immer das angular nächste ein.

### 4. Evaluation (drei Experimente)

- **Experiment A — Genauigkeit über Bedingungen**: Miss die Selektions-Genauigkeit (viele Trials) in mindestens vier Bedingungen, die *isoliert/nah*, *dünn/fern*, *dicht/nah*, *dicht/fern* aufspannen. Plus optional einen **Distanz-Sweep** (dünne Szene, $L=1\dots16$ m).
- **Experiment B — Zeit & Throughput**: Berechne für **isolierte** Ziele die Selektionszeit über das **angulare Fitts' Law** (aus P02) mit technik-spezifischer **effektiver Fangbreite** $W_{\text{eff}}$ (Ray-Casting: Ziel-Winkeldurchmesser; Cone: $2\alpha$; Bubble: große Voronoi-Breite). Berichte $MT$ **und** Throughput $TP=ID/MT$.
- **Experiment C — Statistik**: Simuliere $\sim$16 „Versuchspersonen" (Seeds), within-subject über die Techniken. Vergleiche paarweise mit **Wilcoxon signed-rank**, **rank-biserial** als Effektstärke und **Holm-Bonferroni**-Korrektur (from scratch, da statsmodels fehlt).

Plots nach `results/` (gitignored), Testsuite als `__main__`-Runner.

---

## Was am Ende herauskommen soll (Referenz-Größenordnungen)

Deine Zahlen dürfen mit Parametern/Seeds variieren; die **Geschichte** muss stimmen:

**Experiment A** (Genauigkeit):

| Bedingung | raycast | cone | bubble | bester |
|---|---|---|---|---|
| ISOLATED-NEAR | 0.96 | 0.95 | 0.97 | alle gut |
| SPARSE-FAR | **0.14** | 0.80 | 0.79 | Cone/Bubble |
| DENSE-NEAR | 0.26 | **0.51** | 0.46 | Cone |
| DENSE-FAR | 0.10 | **0.30** | 0.28 | Cone |

Distanz-Sweep (dünn): Ray-Casting fällt von ~0.72 ($L=1$) auf **~0.04** ($L=16$); Cone/Bubble bleiben flach bei ~0.87.

**Experiment B** (isolierte Ziele): Bubble ist am **schnellsten** ($MT\approx0.31$ s vs. Ray-Casting $0.73$ s, ×2.4), weil die große Fangfläche das $ID$ senkt — aber der **Throughput** ist bei Ray-Casting *höher* ($\approx4.0$ vs. $2.6$ bit/s), weil Throughput Präzision belohnt.

**Experiment C**: In SPARSE-FAR schlagen Cone/Bubble Ray-Casting **hochsignifikant** (rank-biserial $=1.0$, Holm $p\approx0.001$); Cone vs. Bubble n.s. In DENSE-NEAR ist Cone **signifikant besser** als Bubble ($p\approx0.0004$) — Bubble **über-selektiert** den nächsten Nachbarn.

> **Die große Lehre.** Keine Technik gewinnt überall:
> - **Ray-Casting** ist präzise, aber nur für **nahe, große, isolierte** Ziele brauchbar — es bricht mit der **Distanz** (angulare Schrumpfung, $\theta_W\approx W/L$) und mit **Verdeckung/Überlappung** ein.
> - **Bubble** ist im Dünnen **am schnellsten und am treffsichersten** (Capture), **über-selektiert** aber im **Gedränge** (greift den nächsten Nachbarn).
> - **Cone** ist der **robusteste Allrounder**.
>
> Das Selektieren im Gedränge ist ein **Disambiguierungsproblem** — dasselbe wie die multimodale Referenzauflösung in Modul 18, nur mit Winkel/Distanz statt Zeit/Semantik als Hinweisen. Wer eine Technik wählt, wählt implizit eine Position im Dreieck **Präzision — Geschwindigkeit — Robustheit**.

## Setup & Ausführen

```bash
cd modules/19-3d-user-interfaces/projects/03-final
# eigene Umsetzung schreiben, dann:
/Users/.../.venv/bin/python test_selection3d.py   # Testsuite
/Users/.../.venv/bin/python run.py                 # 3 Experimente + Plots
```

Nur `numpy`, `scipy` (für `wilcoxon`), `matplotlib`. Laufzeit ~3 s (reine Geometrie/Statistik, kein Training).

## Lösung

Vollständige Referenz in [`solution/`](solution/): `selection3d.py` (Generator + Techniken + Throughput), `stats_tools.py` (rank-biserial, Holm from scratch), `run.py` (drei Experimente + Plots), `test_selection3d.py` (8 Tests).

## Rückblick & Ausblick

Damit schließt Modul 19: von den **Transformationen + Ray-Casting** (P01) über das **angulare Zeigemodell + Go-Go** (P02) zur **vergleichenden Selektionsstudie** (P03). Die 3D-Geometrie und Transformationsmathematik ist die direkte Grundlage für **Modul 20 „3D Point Cloud Processing"** (Registrierung/ICP, Segmentierung auf Punktwolken).
