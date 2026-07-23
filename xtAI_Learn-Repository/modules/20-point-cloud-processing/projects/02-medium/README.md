# P02 (medium) — ICP registration from scratch: Kabsch SVD & iteration

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 20 — 3D Point Cloud Processing** · Format: **Python module + test suite**

## Goal

You implement the **central procedure** of point cloud processing — **Iterative Closest Point (ICP)** — from the ground up and investigate its behaviour:

1. The **closed-form SVD solution** of the Procrustes problem (**Kabsch**, script ch. 7): $\mathbf R=\mathbf V\,\mathrm{diag}(1,1,\det(\mathbf V\mathbf U^\top))\,\mathbf U^\top$ — including the **determinant correction** against reflections.
2. The **ICP iteration** (script ch. 6–8): nearest correspondence (kd-tree) → Kabsch → apply, until convergence.
3. Three experiments: **convergence** (monotone), the **convergence basin** (ICP is only *local* — it needs a good init), and **robustness** (trimmed ICP against outliers/partial overlap).

## Why this format?

A **Python module with a test suite** — ICP is an algorithm with clearly testable building blocks (Kabsch exactly against a known transformation, monotone convergence), and the experiments systematically vary parameters (misalignment, outliers).

## Why synthetic data?

Only with a **known** ground-truth transformation can you check whether ICP converges *correctly* (rotation error against the truth). The test shape is an **anisotropic ellipsoid with a bump** — deliberately **asymmetric**, so that the rotation is *uniquely* determined (a sphere would be rotationally ambiguous!). The generator is disclosed, the seed is fixed.

## Prior knowledge

**P01** of this module (kd-tree), **ch. 6–8** of the [script](../../README.md), **Procrustes from module 10**, **rigid transformations from module 19**, SVD.

## Task

Open `icp.py`. `apply_transform`, the trimming scaffold, the data generator and the experiment/test scripts are given — you implement the **two cores** (`# TODO` / `NotImplementedError`):

1. **`kabsch(P, Q)`** — the 5 steps: centre, cross-covariance $\mathbf H=\mathbf P_c^\top\mathbf Q_c$, SVD, rotation with the det correction, translation.
2. **The ICP loop in `icp`** — solve Kabsch on the (trimmed) correspondences, update `src`, accumulate the total transformation ($\mathbf R\leftarrow d\mathbf R\,\mathbf R$, $\mathbf t\leftarrow d\mathbf R\,\mathbf t+d\mathbf t$), log the RMSE, stop on convergence.

Then:

```bash
cd modules/20-point-cloud-processing/projects/02-medium
/Users/.../.venv/bin/python test_icp.py   # 6 tests -> all PASS
/Users/.../.venv/bin/python run.py         # 3 experiments + plots
```

## What should come out (expected values)

- **Convergence**: the RMSE falls **monotonically** (~0.38 → ~0.017) in ~27 iterations; the residual rotation error is **~0.06°** (ICP rotates the misalignment cleanly back).
- **Convergence basin**: ICP converges correctly up to a **~90° misalignment**; beyond that it locks into a **wrong minimum** (residual error ~179°). → That is the proof that **ICP is only local** and needs a **good initialisation** (global registration).
- **Robustness**: with 10 % outliers + partial overlap, **vanilla ICP** sits at ~1.5° error, **trimmed ICP** (best 80 % of the pairs) at **~0.08°** — trimming discards the bad correspondences.

> **The lesson.** ICP is two alternating *optimal* steps (correspondence + Kabsch) → **guaranteed monotone convergence**, but only to the **nearest local** minimum. The determinant correction in Kabsch is not a detail but the difference between a proper rotation and a (physically impossible) reflection.

## Solution

The complete reference is in [`solution/`](solution/). Try it yourself first!

## What comes next

**P03 (final)**: a complete **segmentation pipeline** — RANSAC plane extraction + clustering, to decompose a scene into the ground and the objects. No code given.

---
---

# P02 (medium) — ICP-Registrierung from scratch: Kabsch-SVD & Iteration (deutsche Fassung)

**Modul 20 — 3D Point Cloud Processing** · Format: **Python-Modul + Testsuite**

## Ziel

Du implementierst das **zentrale Verfahren** der Punktwolken-Verarbeitung — **Iterative Closest Point (ICP)** — von Grund auf und untersuchst sein Verhalten:

1. Die **geschlossene SVD-Lösung** des Procrustes-Problems (**Kabsch**, Skript Kap. 7): $\mathbf R=\mathbf V\,\mathrm{diag}(1,1,\det(\mathbf V\mathbf U^\top))\,\mathbf U^\top$ — inklusive der **Determinanten-Korrektur** gegen Spiegelungen.
2. Die **ICP-Iteration** (Skript Kap. 6–8): nächste Korrespondenz (kd-Baum) → Kabsch → anwenden, bis Konvergenz.
3. Drei Experimente: **Konvergenz** (monoton), der **Konvergenz-Basin** (ICP ist nur *lokal* — braucht gute Init), und **Robustheit** (Trimmed ICP gegen Ausreißer/Teilüberlappung).

## Warum dieses Format?

Ein **Python-Modul mit Testsuite** — ICP ist ein Algorithmus mit klar testbaren Bausteinen (Kabsch exakt gegen bekannte Transformation, Konvergenz-Monotonie), und die Experimente variieren systematisch Parameter (Fehlstellung, Ausreißer).

## Warum synthetische Daten?

Nur mit einer **bekannten** ground-truth-Transformation kann man prüfen, ob ICP *korrekt* konvergiert (Rotationsfehler gegen die Wahrheit). Die Testform ist ein **anisotropes Ellipsoid mit Beule** — bewusst **asymmetrisch**, damit die Rotation *eindeutig* bestimmbar ist (eine Kugel wäre rotationsmehrdeutig!). Generator offengelegt, fester Seed.

## Vorwissen

**P01** dieses Moduls (kd-Baum), **Kap. 6–8** des [Skripts](../../README.md), **Procrustes aus Modul 10**, **Rigid-Transformationen aus Modul 19**, SVD.

## Aufgabenstellung

Öffne `icp.py`. `apply_transform`, das Trimming-Gerüst, der Datengenerator und die Experiment-/Test-Skripte sind vorgegeben — du implementierst die **zwei Kerne** (`# TODO` / `NotImplementedError`):

1. **`kabsch(P, Q)`** — die 5 Schritte: zentrieren, Kreuz-Kovarianz $\mathbf H=\mathbf P_c^\top\mathbf Q_c$, SVD, Rotation mit Det-Korrektur, Translation.
2. **Die ICP-Schleife in `icp`** — Kabsch auf den (getrimmten) Korrespondenzen lösen, `src` updaten, Gesamt-Transformation akkumulieren ($\mathbf R\leftarrow d\mathbf R\,\mathbf R$, $\mathbf t\leftarrow d\mathbf R\,\mathbf t+d\mathbf t$), RMSE protokollieren, bei Konvergenz abbrechen.

Dann:

```bash
cd modules/20-point-cloud-processing/projects/02-medium
/Users/.../.venv/bin/python test_icp.py   # 6 Tests -> alle PASS
/Users/.../.venv/bin/python run.py         # 3 Experimente + Plots
```

## Was am Ende herauskommt (Erwartungswerte)

- **Konvergenz**: RMSE fällt **monoton** (~0.38 → ~0.017) in ~27 Iterationen; Rest-Rotationsfehler **~0.06°** (ICP dreht die Fehlstellung sauber zurück).
- **Konvergenz-Basin**: ICP konvergiert korrekt bis **~90° Fehlstellung**; darüber rastet es in einem **falschen Minimum** ein (Rest-Fehler ~179°). → Das ist der Beweis, dass **ICP nur lokal** ist und eine **gute Initialisierung** (globale Registrierung) braucht.
- **Robustheit**: mit 10 % Ausreißern + Teilüberlappung liegt **Vanilla-ICP** bei ~1.5° Fehler, **Trimmed ICP** (beste 80 % Paare) bei **~0.08°** — Trimming verwirft die schlechten Korrespondenzen.

> **Die Lehre.** ICP ist zwei abwechselnde *optimale* Schritte (Korrespondenz + Kabsch) → **garantierte monotone Konvergenz**, aber nur zum **nächsten lokalen** Minimum. Die Determinanten-Korrektur in Kabsch ist kein Detail, sondern der Unterschied zwischen einer echten Rotation und einer (physikalisch unmöglichen) Spiegelung.

## Lösung

Vollständige Referenz in [`solution/`](solution/). Erst selbst versuchen!

## Weiter geht's

**P03 (final)**: eine vollständige **Segmentierungspipeline** — RANSAC-Ebenenextraktion + Clustering, um eine Szene in Boden und Objekte zu zerlegen. Keine Code-Vorgabe.
