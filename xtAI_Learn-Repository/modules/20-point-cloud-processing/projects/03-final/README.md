# P03 (final) — tabletop segmentation: RANSAC plane + clustering

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 20 — 3D Point Cloud Processing** · Format: **Python project (free implementation, no code given)**

> Final project. **No code is given** — you build the pipeline yourself from the tools of P01 (neighbourhoods/normals) and the script. The reference solution is in [`solution/`](solution/); **try it yourself first**. This README is the specification.

## What it is about

You build the classic **"tabletop" segmentation pipeline** of robotic perception (script ch. 9–10): a 3D scene of **ground + objects + outliers** is decomposed by
1. finding and separating off the dominant **plane via RANSAC** (the floor/table),
2. separating the remaining points into individual objects via **Euclidean clustering**.

This is the standard preprocessing step before a robot grasps or recognises objects.

## Learning objective

You apply **RANSAC** (including the iteration-count theory) and **clustering**, build a **multi-stage pipeline** and evaluate it against ground-truth labels — including the **parameter trade-offs** that decide success or failure in practice.

## Prior knowledge

The module 20 [script](../../README.md), especially **ch. 9 (RANSAC) & 10 (clustering)**. P01 (kd-tree, `query_ball_point`). DBSCAN/clustering from module 05. Plane geometry (normal = cross product).

---

## Task (specification)

### 1. Scene generator (disclosed, reproducible)

Build a seeded, reproducible generator. A scene contains:
- a **ground** — a large, **slightly tilted** plane (so that RANSAC has to find a non-trivial normal), with some noise;
- $k$ **objects** (spheres/boxes) sitting **on** the ground (with a minimum separation so they do not merge);
- **outliers** — random points in the volume (mismeasurements).

Keep the **ground-truth labels** (0 = ground, $1..k$ = objects, $-1$ = outliers) and the **true ground normal** for the evaluation.

Why **synthetic**: only this way do you know the true labels and the ground normal, and can set the outlier fraction/tilt *independently*.

### 2. RANSAC plane estimation

- **Plane from 3 points**: normal = normalised cross product of two edge vectors, offset $d = \mathbf n\cdot\mathbf p_0$.
- **RANSAC loop**: draw 3 points many times, count the inliers ($|\mathbf n\cdot\mathbf x - d| < \tau$), keep the best model. At the end refine the plane by **PCA over all inliers**.
- Implement the **iteration-count formula** $N=\log(1-p)/\log(1-w^s)$ (with $s=3$) and **verify it empirically**: is the predicted $N$ enough to hit the plane reliably?

### 3. Euclidean clustering

Region growing via the kd-tree: points whose $\varepsilon$-neighbourhoods touch form a cluster; clusters smaller than `min_size` are discarded as noise ($-1$) (at its core this is **DBSCAN**, module 05).

### 4. Pipeline & evaluation (three experiments)

- **Experiment A — RANSAC plane**: find the ground; normal error, precision/recall; verify the **iteration-count formula**.
- **Experiment B — full pipeline**: remove the plane → cluster the rest; evaluate with the **object ARI** (over the true object points only, `adjusted_rand_score`), the object count, the object-point recall; 3D plot of the segmentation.
- **Experiment C — parameter trade-off**: sweep `min_size` (with many outliers) → show the **U-shaped trade-off**: too small → outliers form noise clusters; too large → real objects drop out.

Plots go to `results/` (gitignored), the test suite is a `__main__` runner.

---

## What should come out (reference orders of magnitude)

**Experiment A**: RANSAC finds the ground with a **normal error of ~0.02°**, precision ~0.97, recall ~0.99. The iteration-count formula at $w\approx0.66$: $N\approx7$ ($p=0.9$) gives an empirical success rate of ~0.78, $N\approx13$ ($p=0.99$) ~0.95, $N\approx20$ ($p=0.999$) ~0.98 — the measured success rate tracks the requested confidence, so the formula **holds empirically** (it is a lower bound on the effort, not a guarantee per run).

**Experiment B**: 4/4 objects found, **object ARI ~0.99** (clean separation), object-point recall ~0.93 (the object bottoms inside the plane tolerance band are absorbed along with it).

**Experiment C** (`min_size` sweep at 30 % outliers):

| min_size | clusters found | objects detected (/4) |
|---|---|---|
| 5 | ~12 | 4 |
| 30–150 | **4** | **4** |
| 300 | ~2.8 | ~2.8 |
| 450 | ~0 | ~0 |

> **The lesson.** A segmentation pipeline is a **chain** whose links each contribute one robust building block: **RANSAC** is immune to outliers (a clean 3-point sample suffices, and the iteration formula guarantees it gets drawn), **clustering** separates the objects. But the **parameters** ($\tau$, $\varepsilon$, `min_size`) carry a real trade-off: `min_size` too small turns outliers into objects, too large deletes small objects. There is no universally correct value — it depends on the point density and the object size. Exactly this trade-off thinking is the practical master-level competence.

## Setup & running

```bash
cd modules/20-point-cloud-processing/projects/03-final
# write your own implementation, then:
/Users/.../.venv/bin/python test_pointcloud_seg.py   # test suite
/Users/.../.venv/bin/python run.py                    # 3 experiments + plots
```

Only `numpy`, `scipy`, `sklearn` (for the ARI metric only), `matplotlib`. Runtime ~2 s.

## Solution

The complete reference is in [`solution/`](solution/): `pointcloud_seg.py` (generator + RANSAC + clustering), `run.py` (3 experiments + plots), `test_pointcloud_seg.py` (6 tests).

## Looking back & ahead

This closes module 20: from **neighbourhoods + normals** (P01) via **ICP registration** (P02) to the **segmentation pipeline** (P03) — all from scratch. Point cloud perception (registration, segmentation) and the 3D transformations (module 19) are central building blocks of **robot perception** — the topic of **module 21 "Robotics 1"**.

---
---

# P03 (final) — Tabletop-Segmentierung: RANSAC-Ebene + Clustering (deutsche Fassung)

**Modul 20 — 3D Point Cloud Processing** · Format: **Python-Projekt (freie Umsetzung, keine Code-Vorgabe)**

> Abschlussprojekt. **Kein vorgegebener Code** — du baust die Pipeline selbst aus den Werkzeugen von P01 (Nachbarschaften/Normalen) und dem Skript. Referenzlösung in [`solution/`](solution/); **erst selbst versuchen**. Diese README ist die Spezifikation.

## Worum es geht

Du baust die klassische **„tabletop"-Segmentierungspipeline** der Robotik-Perzeption (Skript Kap. 9–10): Eine 3D-Szene aus **Boden + Objekten + Ausreißern** wird zerlegt, indem man
1. die dominante **Ebene per RANSAC** findet und abtrennt (den Boden/Tisch),
2. die verbleibenden Punkte per **Euclidean Clustering** in einzelne Objekte separiert.

Das ist der Standard-Vorverarbeitungsschritt, bevor ein Roboter Objekte greift oder erkennt.

## Lernziel

Du wendest **RANSAC** (inklusive der Iterationszahl-Theorie) und **Clustering** an, baust eine **mehrstufige Pipeline** und evaluierst sie gegen ground-truth-Labels — inklusive der **Parameter-Trade-offs**, die in der Praxis über Erfolg oder Misserfolg entscheiden.

## Vorwissen

Modul-20-[Skript](../../README.md), besonders **Kap. 9 (RANSAC) & 10 (Clustering)**. P01 (kd-Baum, `query_ball_point`). DBSCAN/Clustering aus Modul 05. Ebenengeometrie (Normale = Kreuzprodukt).

---

## Aufgabenstellung (Spezifikation)

### 1. Szenen-Generator (offengelegt, reproduzierbar)

Baue einen geseedet-reproduzierbaren Generator. Eine Szene enthält:
- einen **Boden** — eine große, **leicht gekippte** Ebene (damit RANSAC eine nicht-triviale Normale finden muss), mit etwas Rauschen;
- $k$ **Objekte** (Kugeln/Quader), die **auf** dem Boden sitzen (mit Mindestabstand, damit sie nicht verschmelzen);
- **Ausreißer** — zufällige Punkte im Volumen (Fehlmessungen).

Halte **ground-truth-Labels** (0 = Boden, $1..k$ = Objekte, $-1$ = Ausreißer) und die **wahre Bodennormale** für die Evaluation.

Warum **synthetisch**: Nur so kennst du die wahren Labels und die Bodennormale und kannst Ausreißer-Anteil/Neigung *unabhängig* einstellen.

### 2. RANSAC-Ebenenschätzung

- **Ebene aus 3 Punkten**: Normale = normiertes Kreuzprodukt zweier Kantenvektoren, Offset $d = \mathbf n\cdot\mathbf p_0$.
- **RANSAC-Schleife**: viele Male 3 Punkte ziehen, Inlier zählen ($|\mathbf n\cdot\mathbf x - d| < \tau$), das beste Modell behalten. Am Ende die Ebene per **PCA über alle Inlier** verfeinern.
- **Iterationszahl-Formel** $N=\log(1-p)/\log(1-w^s)$ (mit $s=3$) implementieren und **empirisch verifizieren**: Reicht das vorhergesagte $N$, um die Ebene zuverlässig zu treffen?

### 3. Euclidean Clustering

Region Growing per kd-Baum: Punkte, deren $\varepsilon$-Nachbarschaften sich berühren, bilden ein Cluster; Cluster kleiner als `min_size` werden als Rauschen ($-1$) verworfen (das ist im Kern **DBSCAN**, Modul 05).

### 4. Pipeline & Evaluation (drei Experimente)

- **Experiment A — RANSAC-Ebene**: Boden finden; Normalen-Fehler, Precision/Recall; **Iterationszahl-Formel** verifizieren.
- **Experiment B — volle Pipeline**: Ebene entfernen → Rest clustern; Auswertung mit **Objekt-ARI** (nur über echte Objektpunkte, `adjusted_rand_score`), Objektzahl, Objektpunkt-Recall; 3D-Plot der Segmentierung.
- **Experiment C — Parameter-Trade-off**: Sweep über `min_size` (bei viel Ausreißern) → zeige den **U-förmigen Trade-off**: zu klein → Ausreißer bilden Rausch-Cluster; zu groß → echte Objekte fallen weg.

Plots nach `results/` (gitignored), Testsuite als `__main__`-Runner.

---

## Was am Ende herauskommen soll (Referenz-Größenordnungen)

**Experiment A**: RANSAC findet den Boden mit **Normalen-Fehler ~0.02°**, Precision ~0.97, Recall ~0.99. Iterationszahl-Formel bei $w\approx0.66$: $N\approx7$ ($p=0.9$) ergibt eine empirische Erfolgsrate von ~0.78, $N\approx13$ ($p=0.99$) ~0.95, $N\approx20$ ($p=0.999$) ~0.98 — die gemessene Erfolgsrate folgt der geforderten Konfidenz, die Formel **stimmt empirisch** (sie ist eine Untergrenze für den Aufwand, keine Garantie pro Lauf).

**Experiment B**: 4/4 Objekte gefunden, **Objekt-ARI ~0.99** (saubere Trennung), Objektpunkt-Recall ~0.93 (die Objektböden im Ebenen-Toleranzband werden mit-absorbiert).

**Experiment C** (`min_size`-Sweep bei 30 % Ausreißern):

| min_size | Cluster gefunden | Objekte erkannt (/4) |
|---|---|---|
| 5 | ~12 | 4 |
| 30–150 | **4** | **4** |
| 300 | ~2.8 | ~2.8 |
| 450 | ~0 | ~0 |

> **Die Lehre.** Eine Segmentierungspipeline ist eine **Kette**, deren Glieder je einen robusten Baustein liefern: **RANSAC** ist gegen Ausreißer immun (ein sauberes 3-Punkt-Sample genügt, und die Iterationsformel garantiert, dass es gezogen wird), **Clustering** trennt die Objekte. Aber die **Parameter** ($\tau$, $\varepsilon$, `min_size`) tragen einen echten Trade-off: `min_size` zu klein macht aus Ausreißern Objekte, zu groß löscht kleine Objekte. Es gibt keinen universell richtigen Wert — er hängt von Punktdichte und Objektgröße ab. Genau dieses Trade-off-Denken ist die praktische Master-Kompetenz.

## Setup & Ausführen

```bash
cd modules/20-point-cloud-processing/projects/03-final
# eigene Umsetzung schreiben, dann:
/Users/.../.venv/bin/python test_pointcloud_seg.py   # Testsuite
/Users/.../.venv/bin/python run.py                    # 3 Experimente + Plots
```

Nur `numpy`, `scipy`, `sklearn` (nur für die ARI-Metrik), `matplotlib`. Laufzeit ~2 s.

## Lösung

Vollständige Referenz in [`solution/`](solution/): `pointcloud_seg.py` (Generator + RANSAC + Clustering), `run.py` (3 Experimente + Plots), `test_pointcloud_seg.py` (6 Tests).

## Rückblick & Ausblick

Damit schließt Modul 20: von den **Nachbarschaften + Normalen** (P01) über die **ICP-Registrierung** (P02) zur **Segmentierungspipeline** (P03) — alles from scratch. Die Punktwolken-Perzeption (Registrierung, Segmentierung) und die 3D-Transformationen (Modul 19) sind zentrale Bausteine der **Roboter-Wahrnehmung** — Thema von **Modul 21 „Robotics 1"**.
