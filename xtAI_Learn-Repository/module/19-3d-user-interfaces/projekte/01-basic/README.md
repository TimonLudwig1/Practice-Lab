# P01 (basic) — Homogene Transformationen & Ray-Casting-Selektion

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
cd module/19-3d-user-interfaces/projekte/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # transforms_raycasting.ipynb öffnen
```

Nur `numpy` + `matplotlib` (beide in der `.venv`). Laufzeit wenige Sekunden.

## Lösung

Vollständige, ausgeführte Lösung in [`loesung/transforms_raycasting_loesung.ipynb`](loesung/transforms_raycasting_loesung.ipynb) — **erst selbst probieren!**

## Weiter geht's

- **P02 (medium)**: Realität — Hand-Tremor macht das Zeigen verrauscht; angulares Fitts' Law (kleine ferne Ziele sind schwer) und die Go-Go-Reichweitenverlängerung.
- **P03 (final)**: kompletter Selektionstechnik-Vergleich unter Gedränge mit ISO-Throughput.
