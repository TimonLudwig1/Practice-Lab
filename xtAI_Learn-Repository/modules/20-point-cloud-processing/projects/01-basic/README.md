# P01 (basic) — point cloud basics: neighbourhoods, downsampling & normals

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 20 — 3D Point Cloud Processing** · Format: **Jupyter notebook**

## Goal

You build the three basic operations of every point cloud pipeline:

1. **kd-tree neighbourhoods** (kNN & radius) with `scipy.spatial.cKDTree` — the engine underneath everything that follows.
2. **Voxel downsampling** — reduce the number of points, even out the density (centroid per voxel).
3. **Normals + curvature via a local PCA** — the eigendecomposition of the local covariance matrix (script ch. 5).

The neat part: we sample a **sphere** whose true normals are known (radially outwards) — so you validate your estimate against the **ground truth** (angular error).

## Why this format?

A **notebook**, because point cloud work thrives on having the 3D visualisation next to the computation: you want to see the cloud, the downsampling result and the normal arrows.

## Why synthetic data?

A sampled sphere provides **analytic normals** as a yardstick — with real scan data you would not know the true normal and could not validate the estimate quantitatively. A fixed seed makes everything reproducible.

## Prior knowledge

PCA / eigendecomposition (**module 05**), **ch. 3–5** of the [module 20 script](../../README.md).

## Task (step by step)

Open `pointcloud_basics.ipynb`. Much is given; at the `# TODO` spots you implement the learning cores:

- **Part A/B** (given): generate the sphere point cloud; kd-tree kNN and radius search.
- **Part C** — `voxel_downsample`: group points by voxel index and average within each voxel.
- **Part D** — normals: the **covariance matrix** of the neighbourhood, the **normal = eigenvector of the smallest eigenvalue**, the **curvature** $\sigma=\lambda_0/\sum\lambda$. Validation against the sphere normals.
- **Part E** (given): visualise the normals as arrows.

## What should come out (expected values)

- kNN/radius return plausible neighbours (the nearest one is the point itself, distance 0).
- Voxel downsampling: 2500 → ~680 points (voxel 0.15), the sphere shape preserved.
- **Normal angular error**: median ~**2.8°** (at $k=16$, noise 0.01) — the local PCA reconstructs the sphere normals well.
- **Curvature** small (~0.009) and roughly constant across the smooth sphere.
- 3D plot: the normals point cleanly radially outwards.

Experiment: more noise or a smaller $k$ → a larger error. Normal estimation is a **bias-variance trade-off in the neighbourhood size**.

## Setup

```bash
cd modules/20-point-cloud-processing/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # pointcloud_basics.ipynb
```

Only `numpy`, `scipy`, `matplotlib`. Runtime a few seconds.

## Solution

The complete, executed solution is in [`solution/pointcloud_basics_solution.ipynb`](solution/pointcloud_basics_solution.ipynb) — **try it yourself first!**

## What comes next

- **P02 (medium)**: **ICP registration** from scratch — Kabsch SVD + iteration, bringing two scans into alignment.
- **P03 (final)**: **RANSAC + clustering** — segmenting a scene into the ground plane and objects.

---
---

# P01 (basic) — Punktwolken-Grundlagen: Nachbarschaften, Downsampling & Normalen (deutsche Fassung)

**Modul 20 — 3D Point Cloud Processing** · Format: **Jupyter-Notebook**

## Ziel

Du baust die drei Grundoperationen jeder Punktwolken-Pipeline:

1. **kd-Baum-Nachbarschaften** (kNN & Radius) mit `scipy.spatial.cKDTree` — der Motor unter allem Folgenden.
2. **Voxel-Downsampling** — Punktzahl reduzieren, Dichte vergleichmäßigen (Schwerpunkt je Voxel).
3. **Normalen + Krümmung via lokaler PCA** — die Eigenzerlegung der lokalen Kovarianzmatrix (Skript Kap. 5).

Der Clou: Wir sampeln eine **Kugel**, deren wahre Normalen bekannt sind (radial nach außen) — so validierst du deine Schätzung gegen die **ground truth** (Winkelfehler).

## Warum dieses Format?

Ein **Notebook**, weil Punktwolken-Arbeit von der 3D-Visualisierung neben der Rechnung lebt: Man will die Wolke, das Downsampling-Resultat und die Normalen-Pfeile sehen.

## Warum synthetische Daten?

Eine gesampelte Kugel liefert **analytische Normalen** als Prüfmaß — mit echten Scandaten kennte man die wahre Normale nicht und könnte die Schätzung nicht quantitativ validieren. Fester Seed macht alles reproduzierbar.

## Vorwissen

PCA / Eigenzerlegung (**Modul 05**), **Kap. 3–5** des [Modul-20-Skripts](../../README.md).

## Aufgabenstellung (Schritt für Schritt)

Öffne `pointcloud_basics.ipynb`. Vieles ist vorgegeben; an den `# TODO`-Stellen implementierst du die Lernkerne:

- **Teil A/B** (vorgegeben): Kugel-Punktwolke erzeugen; kd-Baum-kNN und Radius-Suche.
- **Teil C** — `voxel_downsample`: Punkte nach Voxel-Index gruppieren und je Voxel mitteln.
- **Teil D** — Normalen: die **Kovarianzmatrix** der Nachbarschaft, die **Normale = Eigenvektor zum kleinsten Eigenwert**, die **Krümmung** $\sigma=\lambda_0/\sum\lambda$. Validierung gegen die Kugelnormalen.
- **Teil E** (vorgegeben): Normalen als Pfeile visualisieren.

## Was am Ende herauskommt (Erwartungswerte)

- kNN/Radius geben plausible Nachbarn (nächster ist der Punkt selbst, Distanz 0).
- Voxel-Downsampling: 2500 → ~680 Punkte (Voxel 0.15), Kugelform erhalten.
- **Normalen-Winkelfehler**: Median ~**2.8°** (bei $k=16$, Rauschen 0.01) — die lokale PCA rekonstruiert die Kugelnormalen gut.
- **Krümmung** klein (~0.009) und über die glatte Kugel etwa konstant.
- 3D-Plot: Normalen zeigen sauber radial nach außen.

Experimentiere: mehr Rauschen oder kleineres $k$ → größerer Fehler. Normalen-Schätzung ist ein **Bias-Varianz-Tradeoff in der Nachbarschaftsgröße**.

## Setup

```bash
cd modules/20-point-cloud-processing/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # pointcloud_basics.ipynb
```

Nur `numpy`, `scipy`, `matplotlib`. Laufzeit wenige Sekunden.

## Lösung

Vollständige, ausgeführte Lösung in [`solution/pointcloud_basics_solution.ipynb`](solution/pointcloud_basics_solution.ipynb) — **erst selbst probieren!**

## Weiter geht's

- **P02 (medium)**: **ICP-Registrierung** from scratch — Kabsch-SVD + Iteration, zwei Scans zur Deckung bringen.
- **P03 (final)**: **RANSAC + Clustering** — eine Szene in Bodenebene und Objekte segmentieren.
