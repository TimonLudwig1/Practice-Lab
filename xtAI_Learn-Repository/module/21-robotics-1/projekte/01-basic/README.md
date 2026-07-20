# P01 (basic) — Vorwärtskinematik, Arbeitsraum und die Jacobi-Matrix

**Modul 21 — Robotics 1** · Format: **Jupyter-Notebook**

## Ziel

Du baust die kinematische Grundlage jedes Roboterarms:

1. Die allgemeine **DH-Transformationsmatrix** (Skript Kap. 4).
2. Die **kinematische Kette** als Matrixprodukt — verifiziert gegen die von Hand hergeleitete geschlossene Formel.
3. Den **Arbeitsraum** durch Sampling (Kreisring, mit Loch bei ungleichen Gliedlängen).
4. Die **Jacobi-Matrix** analytisch, gegengeprüft per numerischer Ableitung — und ihre **Singularitäten** $\det\mathbf J = l_1l_2\sin q_2$.

## Warum dieses Format?

Ein **Notebook**, weil Kinematik vom Zusammenspiel aus Formel, Zahl und Zeichnung lebt: Man will den Arbeitsraum sehen, die Armposen zeichnen und die Manipulierbarkeit über dem Konfigurationsraum als Heatmap betrachten.

## Warum synthetische Daten?

Es geht um **Geometrie-Gesetze**, nicht um einen Datensatz. Ein selbst definierter Arm (bekannte Gliedlängen) erlaubt, jede Rechnung gegen einen analytisch bekannten Sollwert zu prüfen — genau das macht das Notebook durchgehend (Kette vs. geschlossene Formel, analytische vs. numerische Jacobi, gemessene vs. theoretische Reichweite).

## Vorwissen

Homogene $4\times4$-Transformationen (**Modul 19**), partielle Ableitungen, **Kap. 3–5** des [Modul-21-Skripts](../../README.md).

## Aufgabenstellung (Schritt für Schritt)

Öffne `kinematik.ipynb`. Vieles ist vorgegeben; an den `# TODO`-Stellen baust du die Kerne:

- **Teil A** — `dh_matrix(theta, d, a, alpha)`: die $4\times4$-DH-Matrix.
- **Teil B** — `fk_joints(q, lengths)`: die Kette aufmultiplizieren und dabei die Gelenkpositionen sammeln. Verifikation gegen $x=l_1\cos q_1+l_2\cos(q_1{+}q_2)$ etc.
- **Teil C** (vorgegeben) — Arbeitsraum sampeln und plotten.
- **Teil D** — `jacobian_analytic(q, lengths)`: die $2\times2$-Jacobi eintragen; automatischer Vergleich mit der numerischen Ableitung, dann Determinante und Singularitäten.
- **Teil E** (vorgegeben) — Manipulierbarkeit über dem C-Space + Armposen zeichnen.

## Was am Ende herauskommt (Erwartungswerte)

- **Teil B**: `gleich=True` für alle Testkonfigurationen; bei $q=(0,0)$ steht der Endeffektor bei $(2,0)$.
- **Teil C**: bei $l_1=l_2$ eine volle Scheibe (Radius 2); bei $l_1=1.2, l_2=0.5$ ein **Ring**: gemessene max. Reichweite **1.700** ($=l_1+l_2$), min. **0.700** ($=|l_1-l_2|$).
- **Teil D**: `uebereinstimmend: True`; $\det\mathbf J = 0.891207$ **exakt** gleich $l_1l_2\sin q_2$; bei $q_2=0$ und $q_2=\pi$ fällt $\det\mathbf J$ auf ~$10^{-17}$ → **singulär**.
- **Teil E**: Die Manipulierbarkeits-Heatmap zeigt **senkrechte Streifen** — $w$ hängt nur von $q_2$ ab, konsistent mit $\det\mathbf J=l_1l_2\sin q_2$.

> **Die physikalische Lesart der Singularität:** Im gestreckten Zustand kann sich der Endeffektor **nicht weiter radial nach außen** bewegen — eine Bewegungsrichtung geht verloren. In der Nähe braucht man immer größere Gelenkgeschwindigkeiten für dieselbe Handbewegung. Genau das sprengt in P02 die naive Pseudoinverse.

## Setup

```bash
cd module/21-robotics-1/projekte/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # kinematik.ipynb
```

Nur `numpy` + `matplotlib`. Laufzeit wenige Sekunden.

## Lösung

Vollständige, ausgeführte Lösung in [`loesung/kinematik_loesung.ipynb`](loesung/kinematik_loesung.ipynb) — **erst selbst probieren!**

## Weiter geht's

- **P02 (medium)**: **Inverse Kinematik** — analytisch (beide Ellbogen-Lösungen) und numerisch über die Jacobi-Matrix, inklusive **Damped Least Squares** gegen die Singularitäten aus Teil D.
- **P03 (final)**: die vollständige **Sense-Plan-Act-Navigation** (RRT + Partikelfilter + PID).
