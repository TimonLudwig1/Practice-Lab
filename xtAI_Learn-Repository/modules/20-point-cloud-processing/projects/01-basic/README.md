# P01 (basic) — Punktwolken-Grundlagen: Nachbarschaften, Downsampling & Normalen

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
