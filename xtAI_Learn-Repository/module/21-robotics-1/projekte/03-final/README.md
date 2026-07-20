# P03 (final) — Sense-Plan-Act: RRT-Planung, Partikelfilter-Lokalisierung, Pfadverfolgung

**Modul 21 — Robotics 1** · Format: **Python-Projekt (freie Umsetzung, keine Code-Vorgabe)**

> Abschlussprojekt. **Kein vorgegebener Code** — du baust den kompletten Navigationsstack selbst. Referenzlösung in [`loesung/`](loesung/); **erst selbst versuchen**. Diese README ist die Spezifikation.

## Worum es geht

Du baust den **vollständigen Sense-Plan-Act-Zyklus** eines mobilen Roboters (Skript Kap. 1) — und zeigst, dass er nur als **Ganzes** funktioniert:

- **PLAN**: ein **RRT** findet einen kollisionsfreien Weg durch eine Hindernislandschaft, eine **Shortcut-Glättung** kürzt ihn.
- **SENSE**: Die Odometrie **driftet** (Dead Reckoning); ein **Partikelfilter** korrigiert sie mit Entfernungsmessungen zu bekannten Landmarken.
- **ACT**: Ein **Pure-Pursuit-Regler** folgt dem Pfad — und zwar auf der **geschätzten**, nicht der wahren Pose.

Der Kernbefund, den du empirisch belegst: **Ohne das SENSE bricht alles zusammen.** Ein Roboter, der nur seiner Odometrie glaubt, verlässt den perfekt geplanten Pfad und kollidiert.

## Lernziel

Du integrierst die drei Säulen des Moduls — **Planung** (Kap. 7–8), **Zustandsschätzung** (Kap. 9–11) und **Regelung** (Kap. 12) — zu einem System und evaluierst es quantitativ.

## Vorwissen

Modul-21-[Skript](../../README.md), besonders **Kap. 7–12**. A\*/Suche aus Modul 06, Bayes aus Modul 07, Sensorfusion aus Modul 17/18.

---

## Aufgabenstellung (Spezifikation)

### 1. Welt und Roboter

- **Welt**: 2D-Bereich (z. B. $10\times10$) mit kreisförmigen **Hindernissen**, Kollisionstest für Punkte *und* Strecken (dichte Abtastung entlang der Kante). Ein **Sicherheitsabstand** (Roboterradius) muss einstellbar sein.
- **Roboter**: **Einspurmodell (unicycle)** mit Zustand $(x, y, \theta)$:
  $$x \mathrel{+}= v\cos\theta\,\Delta t,\quad y \mathrel{+}= v\sin\theta\,\Delta t,\quad \theta \mathrel{+}= \omega\,\Delta t$$
  Die *wahre* Bewegung bekommt **Rauschen** auf $v$ und $\omega$; die **Odometrie** rechnet mit den reinen Sollbefehlen (und driftet dadurch).
- **Sensor**: **Entfernungen zu bekannten Landmarken**, verrauscht.

Warum **synthetisch**: Nur so kennst du die *wahre* Pose und kannst den Lokalisierungsfehler überhaupt messen — bei einem echten Roboter gibt es keine ground truth.

### 2. PLAN — RRT + Glättung

Implementiere **RRT** (Skript Kap. 8): Stichprobe (mit **Goal-Bias**) → nächster Baumknoten → Schritt der Länge $\varepsilon$ → **Kollisionstest der Kante** → einfügen; bei Zielnähe Pfad durch Rückverfolgen der Eltern. Danach **Shortcut-Glättung** (wiederholt zwei Pfadpunkte direkt verbinden, wenn frei).

> **Zwei Fallstricke, die du treffen wirst:** (1) Nach der Glättung besteht der Pfad aus **wenigen weit auseinanderliegenden Ecken** — der Pfadverfolger braucht dann **verdichtete** Stützpunkte, sonst springt sein Zielpunkt. (2) Der Regler **schneidet Kurven**; plane deshalb mit **Sicherheitsabstand**, sonst streift der Roboter Hindernisse, obwohl der Pfad formal frei war.

### 3. SENSE — Partikelfilter

Implementiere die Monte-Carlo-Lokalisierung (Skript Kap. 11):
1. **Prädiktion**: Bewegungsmodell + Rauschen auf jedes Partikel.
2. **Gewichtung**: $w^{[m]} \propto p(\mathbf z\mid \mathbf x^{[m]})$ — Gauß-Likelihood der Entfernungen. *Rechne in **log**-Likelihoods und ziehe das Maximum ab, sonst unterlaufen die Gewichte numerisch.*
3. **Resampling**: **systematisch**, und **nur wenn** $N_{\text{eff}} = 1/\sum_m (w^{[m]})^2$ unter $M/2$ fällt (gegen **Partikelverarmung**).
4. **Schätzung**: gewichteter Mittelwert; die **Orientierung** über $\operatorname{atan2}(\sum w\sin\theta, \sum w\cos\theta)$ mitteln (nicht naiv arithmetisch!).

### 4. ACT — Pfadverfolgung

**Pure Pursuit**: Suche den am weitesten fortgeschrittenen Pfadpunkt im Vorausschau-Radius, drehe proportional zum Winkelfehler dorthin, fahre bei großem Winkelfehler langsamer. **Begrenze $\omega$** (Sättigung) — sonst dreht der Roboter pro Zeitschritt zu weit und schwingt auf.

### 5. Evaluation (drei Experimente)

- **A — Planung**: Erfolgsrate, Baumgröße und Pfadlänge über verschiedene **Goal-Bias**-Werte; Wirkung der **Glättung**.
- **B — Lokalisierung**: Fehlerverlauf **Partikelfilter vs. reine Odometrie** über die Fahrt; **Partikelzahl** $M$ variieren.
- **C — Geschlossener Kreis**: Navigation mit Regelung auf der **Filter-Schätzung** vs. auf **reiner Odometrie** — Zielerreichung, Endabstand, Kollisionen (je 10 Läufe).

Plots nach `ergebnisse/` (gitignored), Testsuite als `__main__`-Runner.

---

## Was am Ende herauskommen soll (Referenz-Größenordnungen)

**A — Planung**: RRT findet in dieser Welt immer eine Lösung. Der Goal-Bias senkt den Suchaufwand spürbar (~200 → 120–140 Knoten), ändert die Pfadlänge kaum. Der große Gewinn kommt vom **Glätten**: ~16.3 → ~13.6 (rund **17 % kürzer**) — RRT ist eben probabilistisch vollständig, aber **nicht optimal**.

**B — Lokalisierung** (ein Lauf über ~160 Schritte):

| | Fehler Mittel | Fehler Ende |
|---|---|---|
| Partikelfilter | **0.068 m** | **0.089 m** |
| Odometrie (dead reckoning) | 0.541 m | **1.370 m** |

Der Odometriefehler **wächst unbeschränkt** (Random Walk, v. a. der Orientierungsfehler), der Filter bleibt durch die Landmarken **beschränkt**. Partikelzahl: $M=10$ → 0.494 m, $M=50$ → 0.092 m, $M=200$ → 0.082 m, $M=1000$ → 0.070 m — der Gewinn **sättigt** ab einigen Hundert Partikeln.

**C — Geschlossener Kreis** (10 Läufe):

| Regelung auf | Ziel erreicht | Endabstand | Kollisionen |
|---|---|---|---|
| **Partikelfilter** | **10/10** | 0.37 m | **0/10** |
| nur Odometrie | 2/10 | 1.39 m | 7/10 |

> **Die große Lehre.** Ein perfekter Plan nützt nichts, wenn der Roboter nicht weiß, wo er ist. Die Odometrie ist ein **integrierender** Prozess — jeder kleine Fehler bleibt für immer im Zustand, der Fehler wächst unbeschränkt. Erst der **Korrekturschritt des Bayes-Filters** (externe Messung) bricht diesen Drift. Sense, Plan und Act sind keine drei getrennten Module, sondern ein **Kreislauf**: Die Regelung wirkt auf die *Schätzung*, und die Qualität der Schätzung entscheidet über Erfolg oder Kollision.

## Setup & Ausführen

```bash
cd module/21-robotics-1/projekte/03-final
# eigene Umsetzung schreiben, dann:
/Users/.../.venv/bin/python test_navigation.py   # Testsuite
/Users/.../.venv/bin/python run.py                # 3 Experimente + Plots
```

Nur `numpy` + `matplotlib`. Laufzeit ~10 s (reine Simulation, kein Training).

## Lösung

Vollständige Referenz in [`loesung/`](loesung/): `navigation.py` (Welt, RRT, Glättung/Verdichtung, Einspurmodell, Partikelfilter, Pure Pursuit), `run.py` (3 Experimente + Plots), `test_navigation.py` (9 Tests).

## Rückblick & Ausblick

Damit schließt Modul 21: von der **Vorwärtskinematik und Jacobi** (P01) über die **inverse Kinematik mit DLS** (P02) zum **vollständigen Navigationsstack** (P03). Weiter geht es in **Modul 22 „Robotics 2"** mit Dynamik (Kräfte statt nur Geometrie), **SLAM** (Karte *und* Pose gleichzeitig schätzen — die Fortsetzung von Experiment B) und fortgeschrittener Regelung.
