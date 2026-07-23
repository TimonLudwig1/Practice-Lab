# P01 (basic) — homogeneous transformations & ray-casting selection

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Module 19 — 3D User Interfaces** · Format: **Jupyter notebook**

## Goal

You build the two geometric foundations of every 3D interface from scratch:

1. **Homogeneous 4×4 transformations** — translation, rotation, scaling as chainable matrices; the transformation chain and its closed-form inverse. You see concretely *why* homogeneous coordinates are needed.
2. **Ray-casting selection** — the ray-sphere intersection mathematics (derived from the quadratic equation) and choosing the *nearest* object hit in a 3D scene.

## Why this format?

A **notebook**, because 3D geometry lives off numbers *and* visualization side by side: you want to see the transformed points and the ray pointing at the object it hit in a 3D plot.

## Why synthetic data?

This is about **laws of geometry**, not about a dataset. A hand-built mini scene of spheres allows every computation (the transformed point, the intersection distance $t$) to be checked against a target value you can follow by hand.

## Prior knowledge

Linear algebra (matrix × vector, the dot product). **Chapters 3 & 5** of the [module 19 script](../../README.md). The rotation mathematics from module 17 (Rodrigues/quaternion → rotation matrix) reappears.

## Assignment (step by step)

Open `transforms_raycasting.ipynb`. Most cells are given; at the `# TODO` places you fill in the core building blocks:

- **Part A** — `translation` and `scaling` as 4×4 matrices. Chain scale→rotate→translate and observe that the order counts.
- **Part B** — `rigid_inverse` with the closed-form formula $M^{-1}=\begin{psmallmatrix}\mathbf R^\top & -\mathbf R^\top\mathbf t\\0&1\end{psmallmatrix}$ (no `np.linalg.inv`).
- **Part C** — `ray_sphere`: the ray-sphere intersection equation (the discriminant, the nearer positive intersection).
- **Part D** — `select`: iterate over the scene and choose the object with the smallest $t>0$. The plot is given.

## What comes out at the end (expected values)

- Part A: the point $(1,0,0)$ → **$(1,4,3)$** after scaling by 2 / 90° about z / a translation by $(1,2,3)$.
- Part B: the deviation from `np.linalg.inv` is ~$10^{-16}$, $MM^{-1}=I$.
- Part C: a ray along $+x$ hits the sphere around $(5,0,0)$, $R=1$ at **$t=4$**; a ray along $+y$ misses it (`None`).
- Part D: the ray directed almost along $+x$ selects **A** (at $t\approx2.6$) — the *nearest* object, although further ones behind it would also be hit. A 3D plot with the ray, the hit point and the object marked in green.

## Setup

```bash
cd modules/19-3d-user-interfaces/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # open transforms_raycasting.ipynb
```

Only `numpy` + `matplotlib` (both in the `.venv`). Runtime a few seconds.

## Solution

The complete, executed solution is in [`solution/transforms_raycasting_solution.ipynb`](solution/transforms_raycasting_solution.ipynb) — **try it yourself first!**

## What comes next

- **P02 (medium)**: reality — hand tremor makes the pointing noisy; the angular Fitts' law (small distant targets are hard) and the Go-Go reach extension.
- **P03 (final)**: a complete comparison of selection techniques under clutter with ISO throughput.

---

# P01 (basic) — Homogene Transformationen & Ray-Casting-Selektion (deutsche Fassung)

**Modul 19 — 3D User Interfaces** · Format: **Jupyter-Notebook**

## Ziel

Du baust die zwei geometrischen Fundamente jedes 3D-Interfaces von Grund auf:

1. **Homogene 4×4-Transformationen** — Translation, Rotation, Skalierung als verkettbare Matrizen; die Transformationskette und ihre geschlossene Inverse. Du siehst konkret, *warum* man homogene Koordinaten braucht.
2. **Ray-Casting-Selektion** — die Ray-Kugel-Schnittmathematik (aus der quadratischen Gleichung hergeleitet) und die Auswahl des *nächsten* getroffenen Objekts in einer 3D-Szene.

## Warum dieses Format?

Ein **Notebook**, weil 3D-Geometrie von Zahlen *und* Visualisierung nebeneinander lebt: Man will die transformierten Punkte sehen und den Strahl im 3D-Plot auf das getroffene Objekt zeigen.

## Warum synthetische Daten?

Es geht um **Geometrie-Gesetze**, nicht um einen Datensatz. Eine handgebaute Mini-Szene aus Kugeln erlaubt, jede Rechnung (transformierter Punkt, Schnittdistanz $t$) gegen den von Hand nachvollziehbaren Sollwert zu prüfen.

## Vorwissen

Lineare Algebra (Matrix × Vektor, Skalarprodukt). **Kapitel 3 & 5** des [Modul-19-Skripts](../../README.md). Die Rotationsmathematik aus Modul 17 (Rodrigues/Quaternion → Rotationsmatrix) taucht wieder auf.

## Aufgabenstellung (Schritt für Schritt)

Öffne `transforms_raycasting.ipynb`. Die meisten Zellen sind vorgegeben; an den `# TODO`-Stellen füllst du die Kernbausteine ein:

- **Teil A** — `translation` und `scaling` als 4×4-Matrizen. Verkette skalieren→rotieren→translatieren und beobachte, dass die Reihenfolge zählt.
- **Teil B** — `rigid_inverse` mit der geschlossenen Formel $M^{-1}=\begin{psmallmatrix}\mathbf R^\top & -\mathbf R^\top\mathbf t\\0&1\end{psmallmatrix}$ (kein `np.linalg.inv`).
- **Teil C** — `ray_sphere`: die Ray-Kugel-Schnittgleichung (Diskriminante, näherer positiver Schnitt).
- **Teil D** — `select`: über die Szene iterieren und das Objekt mit kleinstem $t>0$ wählen. Plot vorgegeben.

## Was am Ende herauskommt (Erwartungswerte)

- Teil A: Punkt $(1,0,0)$ → **$(1,4,3)$** nach skalieren×2 / 90° um z / verschieben um $(1,2,3)$.
- Teil B: Abweichung von `np.linalg.inv` ~$10^{-16}$, $MM^{-1}=I$.
- Teil C: Strahl entlang $+x$ trifft Kugel um $(5,0,0)$, $R=1$ bei **$t=4$**; Strahl entlang $+y$ verfehlt (`None`).
- Teil D: Der fast entlang $+x$ gerichtete Strahl selektiert **A** (bei $t\approx2.6$) — das *nächste* Objekt, obwohl weiter hinten weitere getroffen würden. 3D-Plot mit Strahl, Trefferpunkt und grün markiertem Objekt.

## Setup

```bash
cd modules/19-3d-user-interfaces/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # transforms_raycasting.ipynb öffnen
```

Nur `numpy` + `matplotlib` (beide in der `.venv`). Laufzeit wenige Sekunden.

## Lösung

Vollständige, ausgeführte Lösung in [`solution/transforms_raycasting_solution.ipynb`](solution/transforms_raycasting_solution.ipynb) — **erst selbst probieren!**

## Weiter geht's

- **P02 (medium)**: Realität — Hand-Tremor macht das Zeigen verrauscht; angulares Fitts' Law (kleine ferne Ziele sind schwer) und die Go-Go-Reichweitenverlängerung.
- **P03 (final)**: kompletter Selektionstechnik-Vergleich unter Gedränge mit ISO-Throughput.
