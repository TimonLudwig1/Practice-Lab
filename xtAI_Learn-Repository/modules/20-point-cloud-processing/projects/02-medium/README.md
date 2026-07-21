# P02 (medium) — ICP-Registrierung from scratch: Kabsch-SVD & Iteration

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
