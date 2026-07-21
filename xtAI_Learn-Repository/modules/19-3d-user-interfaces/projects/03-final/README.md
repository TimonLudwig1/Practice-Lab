# P03 (final) — Vergleichende 3D-Selektionsstudie unter Clutter

**Modul 19 — 3D User Interfaces** · Format: **Python-Projekt (freie Umsetzung, keine Code-Vorgabe)**

> Abschlussprojekt. **Kein vorgegebener Code** — du entwirfst und baust die Studie selbst aus den Werkzeugen der Projekte P01 (Ray-Objekt-Schnitt) und P02 (angulares Zeigemodell). Referenzlösung in [`solution/`](solution/); **erst selbst versuchen**. Diese README ist die Spezifikation.

## Worum es geht

Du führst eine **kontrollierte Evaluationsstudie** durch, die drei Selektionstechniken vergleicht — **Ray-Casting**, **Cone/Flashlight** und **Bubble Cursor** (Skript, Kap. 5–7) — über verschiedene **Zieldistanzen** und **Szenendichten**. Ziel ist die zentrale Erkenntnis des Moduls, empirisch belegt: **Es gibt keine universell beste 3D-Selektionstechnik** — die Wahl ist ein **Trade-off zwischen Präzision, Reichweite/Capture und Robustheit gegen Gedränge**, und dieser Trade-off ist dasselbe Disambiguierungsproblem wie die Referenzauflösung in Modul 18.

## Lernziel

Du wendest die gesamte Modul-19-Toolbox an: Ray-Objekt-Schnitt (P01), das angulare Zeigemodell und Fitts (P02), und eine saubere **Evaluationsmethodik** (ISO-9241-9-Throughput + within-subject-Statistik wie in Modul 17).

## Vorwissen

Modul-19-[Skript](../../README.md) komplett, besonders **Kap. 5–7 (Selektionstechniken), 10 (angulares Fitts), 14 (ISO-Throughput)**. P01 (`ray_sphere`), P02 (angulare Größe, Fitts). Statistik-Methodik aus Modul 17 (Wilcoxon, Effektstärke, Multiplizitätskorrektur).

---

## Aufgabenstellung (Spezifikation)

### 1. Szenen-Generator (offengelegt, reproduzierbar)

Baue einen geseedet-reproduzierbaren Generator. Eine Szene besteht aus:
- einem **Zielobjekt** in Distanz $L$ (Kugel, Radius $r$),
- $N$ **Distraktoren**, die das Ziel **angular umgeben** (bis zu einem Spread-Winkel), teils **näher** am Betrachter (→ **Verdeckung** für Ray-Casting).

Warum **synthetisch**: Nur so kennst du die *ground truth* (welches Objekt ist das Ziel) und kannst Dichte, Distanz und Verdeckung *unabhängig* einstellen — mit echten VR-Logs wäre keiner dieser Faktoren isolierbar.

### 2. Motor-Modell

Der Zeiger ist ein Strahl vom Ursprung. Die **intendierte** Richtung zeigt aufs Ziel; die **reale** Richtung hat **angulares Gauß-Rauschen** $\sigma_\theta$ (Hand-Tremor + Tracking + Heisenberg, Skript Kap. 12–13). Implementiere die verrauschte Richtung als kleine 2D-Auslenkung in der Tangentialebene.

### 3. Die drei Techniken

- **Ray-Casting**: selektiere das vom (verrauschten) Strahl **geschnittene** Objekt mit **kleinstem $t$** (nächster Treffer, via Ray-Kugel aus P01); nichts getroffen → Fehlselektion.
- **Cone**: unter allen Objekten im Kegel (Halbwinkel $\alpha$) das mit **kleinstem Winkel zur Kegelachse**.
- **Bubble**: das Objekt mit **kleinstem Winkel zur Oberfläche** ($\text{Winkel zum Zentrum} - \text{angularer Radius}$) — fängt immer das angular nächste ein.

### 4. Evaluation (drei Experimente)

- **Experiment A — Genauigkeit über Bedingungen**: Miss die Selektions-Genauigkeit (viele Trials) in mindestens vier Bedingungen, die *isoliert/nah*, *dünn/fern*, *dicht/nah*, *dicht/fern* aufspannen. Plus optional einen **Distanz-Sweep** (dünne Szene, $L=1\dots16$ m).
- **Experiment B — Zeit & Throughput**: Berechne für **isolierte** Ziele die Selektionszeit über das **angulare Fitts' Law** (aus P02) mit technik-spezifischer **effektiver Fangbreite** $W_{\text{eff}}$ (Ray-Casting: Ziel-Winkeldurchmesser; Cone: $2\alpha$; Bubble: große Voronoi-Breite). Berichte $MT$ **und** Throughput $TP=ID/MT$.
- **Experiment C — Statistik**: Simuliere $\sim$16 „Versuchspersonen" (Seeds), within-subject über die Techniken. Vergleiche paarweise mit **Wilcoxon signed-rank**, **rank-biserial** als Effektstärke und **Holm-Bonferroni**-Korrektur (from scratch, da statsmodels fehlt).

Plots nach `results/` (gitignored), Testsuite als `__main__`-Runner.

---

## Was am Ende herauskommen soll (Referenz-Größenordnungen)

Deine Zahlen dürfen mit Parametern/Seeds variieren; die **Geschichte** muss stimmen:

**Experiment A** (Genauigkeit):

| Bedingung | raycast | cone | bubble | bester |
|---|---|---|---|---|
| ISOLATED-NEAR | 0.96 | 0.95 | 0.97 | alle gut |
| SPARSE-FAR | **0.14** | 0.80 | 0.79 | Cone/Bubble |
| DENSE-NEAR | 0.26 | **0.51** | 0.46 | Cone |
| DENSE-FAR | 0.10 | **0.30** | 0.28 | Cone |

Distanz-Sweep (dünn): Ray-Casting fällt von ~0.72 ($L=1$) auf **~0.04** ($L=16$); Cone/Bubble bleiben flach bei ~0.87.

**Experiment B** (isolierte Ziele): Bubble ist am **schnellsten** ($MT\approx0.31$ s vs. Ray-Casting $0.73$ s, ×2.4), weil die große Fangfläche das $ID$ senkt — aber der **Throughput** ist bei Ray-Casting *höher* ($\approx4.0$ vs. $2.6$ bit/s), weil Throughput Präzision belohnt.

**Experiment C**: In SPARSE-FAR schlagen Cone/Bubble Ray-Casting **hochsignifikant** (rank-biserial $=1.0$, Holm $p\approx0.001$); Cone vs. Bubble n.s. In DENSE-NEAR ist Cone **signifikant besser** als Bubble ($p\approx0.0004$) — Bubble **über-selektiert** den nächsten Nachbarn.

> **Die große Lehre.** Keine Technik gewinnt überall:
> - **Ray-Casting** ist präzise, aber nur für **nahe, große, isolierte** Ziele brauchbar — es bricht mit der **Distanz** (angulare Schrumpfung, $\theta_W\approx W/L$) und mit **Verdeckung/Überlappung** ein.
> - **Bubble** ist im Dünnen **am schnellsten und am treffsichersten** (Capture), **über-selektiert** aber im **Gedränge** (greift den nächsten Nachbarn).
> - **Cone** ist der **robusteste Allrounder**.
>
> Das Selektieren im Gedränge ist ein **Disambiguierungsproblem** — dasselbe wie die multimodale Referenzauflösung in Modul 18, nur mit Winkel/Distanz statt Zeit/Semantik als Hinweisen. Wer eine Technik wählt, wählt implizit eine Position im Dreieck **Präzision — Geschwindigkeit — Robustheit**.

## Setup & Ausführen

```bash
cd modules/19-3d-user-interfaces/projects/03-final
# eigene Umsetzung schreiben, dann:
/Users/.../.venv/bin/python test_selection3d.py   # Testsuite
/Users/.../.venv/bin/python run.py                 # 3 Experimente + Plots
```

Nur `numpy`, `scipy` (für `wilcoxon`), `matplotlib`. Laufzeit ~3 s (reine Geometrie/Statistik, kein Training).

## Lösung

Vollständige Referenz in [`solution/`](solution/): `selection3d.py` (Generator + Techniken + Throughput), `stats_tools.py` (rank-biserial, Holm from scratch), `run.py` (drei Experimente + Plots), `test_selection3d.py` (8 Tests).

## Rückblick & Ausblick

Damit schließt Modul 19: von den **Transformationen + Ray-Casting** (P01) über das **angulare Zeigemodell + Go-Go** (P02) zur **vergleichenden Selektionsstudie** (P03). Die 3D-Geometrie und Transformationsmathematik ist die direkte Grundlage für **Modul 20 „3D Point Cloud Processing"** (Registrierung/ICP, Segmentierung auf Punktwolken).
