# P03 (final) — Tabletop-Segmentierung: RANSAC-Ebene + Clustering

**Modul 20 — 3D Point Cloud Processing** · Format: **Python-Projekt (freie Umsetzung, keine Code-Vorgabe)**

> Abschlussprojekt. **Kein vorgegebener Code** — du baust die Pipeline selbst aus den Werkzeugen von P01 (Nachbarschaften/Normalen) und dem Skript. Referenzlösung in [`loesung/`](loesung/); **erst selbst versuchen**. Diese README ist die Spezifikation.

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

Plots nach `ergebnisse/` (gitignored), Testsuite als `__main__`-Runner.

---

## Was am Ende herauskommen soll (Referenz-Größenordnungen)

**Experiment A**: RANSAC findet den Boden mit **Normalen-Fehler ~0.01°**, Precision ~0.98, Recall ~0.98. Iterationszahl-Formel bei $w\approx0.65$: $N\approx7$ ($p=0.9$) reicht für ~90 % Erfolg, $N\approx14$ ($p=0.99$) für ~100 % — die Formel **stimmt empirisch**.

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
cd module/20-point-cloud-processing/projekte/03-final
# eigene Umsetzung schreiben, dann:
/Users/.../.venv/bin/python test_pointcloud_seg.py   # Testsuite
/Users/.../.venv/bin/python run.py                    # 3 Experimente + Plots
```

Nur `numpy`, `scipy`, `sklearn` (nur für die ARI-Metrik), `matplotlib`. Laufzeit ~2 s.

## Lösung

Vollständige Referenz in [`loesung/`](loesung/): `pointcloud_seg.py` (Generator + RANSAC + Clustering), `run.py` (3 Experimente + Plots), `test_pointcloud_seg.py` (6 Tests).

## Rückblick & Ausblick

Damit schließt Modul 20: von den **Nachbarschaften + Normalen** (P01) über die **ICP-Registrierung** (P02) zur **Segmentierungspipeline** (P03) — alles from scratch. Die Punktwolken-Perzeption (Registrierung, Segmentierung) und die 3D-Transformationen (Modul 19) sind zentrale Bausteine der **Roboter-Wahrnehmung** — Thema von **Modul 21 „Robotics 1"**.
