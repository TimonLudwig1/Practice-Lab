# P02 (medium) — Inverse Kinematik: analytisch, numerisch und die Singularitätsfalle

**Modul 21 — Robotics 1** · Format: **Python-Modul + Testsuite**

## Ziel

Du drehst die Kinematik aus P01 um — von der Pose zu den Gelenkwinkeln — und lernst dabei die zentrale numerische Falle der Robotik kennen:

1. **Analytische IK** für den 2-Gelenk-Arm (Kosinussatz) — mit **beiden** Lösungen (Ellbogen oben/unten), korrekter Erreichbarkeitsprüfung und `atan2`.
2. **Numerische IK** über die Jacobi-Matrix in drei Varianten: **Transponierte**, **Pseudoinverse** und **Damped Least Squares**.
3. Die **Singularitätsfalle**: Warum die Pseudoinverse nahe $\det\mathbf J = 0$ **explodiert** (Gelenksprünge $\sim 1/\det\mathbf J$) und DLS sie zähmt — plus der $\lambda$-Trade-off.
4. **Redundanz**: Beim 3-Gelenk-Arm gibt es einen **Nullraum** — Gelenkbewegungen, die den Endeffektor nicht bewegen.

## Warum dieses Format?

Ein **Python-Modul mit Testsuite**: Die IK-Varianten sind klar testbare Funktionen (analytische Lösung muss das Ziel *exakt* treffen, Jacobi gegen numerische Ableitung, Nullraum-Drift ≈ 0), und die Experimente variieren systematisch Parameter ($\lambda$, Singularitätsnähe).

## Warum synthetische Daten?

Es geht um **Algorithmen-Eigenschaften**, nicht um Messdaten. Ein selbst definierter Arm erlaubt, jede Lösung per Vorwärtskinematik **zurückzurechnen** und exakt zu prüfen — und die Singularitätsnähe *gezielt* einzustellen (was mit echten Roboterdaten nicht ginge).

## Vorwissen

**P01** dieses Moduls (FK, Jacobi, Singularitäten), **Kap. 6** des [Skripts](../../README.md), Pseudoinverse/Least Squares.

## Aufgabenstellung

Öffne `ik.py`. FK, Gelenkpositionen und Jacobi sind vorgegeben (aus P01) — du implementierst die **drei Kerne** (`# TODO` / `NotImplementedError`):

1. **`analytic_ik_2link(target, lengths)`** — Erreichbarkeitsprüfung, Kosinussatz, **beide** Vorzeichen von $q_2$, $q_1$ per `atan2`, Duplikat an der Grenze entfernen.
2. **Die drei Update-Regeln in `numeric_ik`** — `transpose` (mit optimaler Schrittweite), `pinv`, `dls`.
3. **`nullspace_step(q, lengths, z)`** — die Projektion $(\mathbf I - \mathbf J^{+}\mathbf J)\,\mathbf z$.

Dann:

```bash
cd module/21-robotics-1/projekte/02-medium
/Users/.../.venv/bin/python test_ik.py   # 8 Tests -> alle PASS
/Users/.../.venv/bin/python run.py        # 4 Experimente + Plots
```

## Was am Ende herauskommt (Erwartungswerte)

**Experiment 1 — analytische IK.** Ziel $(1,1)$ hat **zwei** Lösungen: $q=(0°, +90°)$ und $q=(90°, -90°)$ — beide treffen exakt. Ziel $(0,2)$ (gestreckt, Arbeitsraumgrenze) hat **eine**; $(2.5, 0)$ ist **unerreichbar**.

**Experiment 2 — Methodenvergleich** (200 Zufallsziele):

| Methode | Erfolg | Iterationen (Median) | max &#124;Δq&#124; |
|---|---|---|---|
| transpose | 1.00 | 26 | 12 |
| pinv | 0.87 | **7** | **1194** |
| dls | 0.89 | 9 | 16 |

Die Transponierte ist am **robustesten**, aber langsam (Gradientenabstieg). Pseudoinverse ist am schnellsten — bezahlt das aber mit gewaltigen Einzelschritten.

**Experiment 3 — die Singularitätsfalle.** Startet man immer näher am gestreckten Arm ($q_2\to0$, also $\det\mathbf J\to0$), skalieren die Pseudoinverse-Schritte **exakt wie $1/\det\mathbf J$**:

| $q_2^{\text{start}}$ | $\det\mathbf J$ | pinv max&#124;Δq&#124; | DLS max&#124;Δq&#124; |
|---|---|---|---|
| 0.5 | 0.479 | 4.4 | 3.7 |
| 0.1 | 0.0998 | 26.5 | 4.4 |
| 0.01 | 0.0100 | 274.7 | 2.7 |
| 0.001 | 0.00100 | 2756 | 4.9 |
| 0.0001 | 0.000100 | **27573** | **4.7** |

Jede Zehnerpotenz näher an der Singularität = **zehnfach größere** Gelenksprünge bei pinv; DLS bleibt bei ~3–5. Der $\lambda$-Sweep zeigt den Trade-off: kleines $\lambda$ genau aber sprunghaft, großes $\lambda$ sanft aber langsam/ungenauer.

**Experiment 4 — Redundanz.** Beim 3-Gelenk-Arm ist $\mathbf J$ eine $2\times3$-Matrix mit **Nullraum-Dimension 1**. 200 zufällige Nullraum-Schritte bewegen den Endeffektor um maximal $1.7\cdot10^{-5}$ — die Gelenke bewegen sich, die Hand steht still.

> **Die Lehre.** Die Pseudoinverse ist mathematisch „optimal" (kleinste Gelenkänderung) und praktisch **gefährlich**: Sie kennt keine Grenze für $\|\Delta\mathbf q\|$. Die Dämpfung in DLS ist kein Schönheitsfehler, sondern die Bedingung dafür, dass ein realer Roboter nicht in die Singularität hineinschlägt — man **tauscht bewusst Genauigkeit gegen beschränkte Gelenkgeschwindigkeiten**.

## Lösung

Vollständige Referenz in [`loesung/`](loesung/). Erst selbst versuchen!

## Weiter geht's

**P03 (final)**: der volle **Sense-Plan-Act-Zyklus** eines mobilen Roboters — RRT-Planung, Partikelfilter-Lokalisierung und PID-Pfadverfolgung. Keine Code-Vorgabe.
