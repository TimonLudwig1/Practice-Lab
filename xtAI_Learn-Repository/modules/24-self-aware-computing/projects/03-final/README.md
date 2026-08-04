# P03 (final) — the complete self-aware system: models at run-time, drift and self-healing

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 24 — Self-aware Computing** · Format: **Python project (free implementation, no code given)**

> Final project. **No code is given** — you build the whole system yourself. The reference solution is in [`solution/`](solution/); **try it yourself first**. This README is the specification.

## What it is about

P02's auto-scaler had a fixed, correct model. Reality does not. This project builds a **genuinely self-aware system** in Kounev's sense: it carries a **model of itself at run-time**, **re-estimates that model online** as the system drifts beneath it, and **heals** replicas it believes have failed — which forces an honest confrontation with the **base-rate fallacy**.

Two things go wrong that no static design can survive: a **deployment** silently changes the true service rate, and **replicas fall sick** at random. The system must notice both, using nothing but its own measurements.

## Learning objective

You implement all three of Kounev's properties concretely: **self-reflective** (monitor load, capacity, latency, per-replica health), **self-predictive** (carry an M/M/c model and invert it to plan capacity), **self-adaptive** (scale, and repair). Above all you implement the thing that makes it *self-aware* rather than merely adaptive: **keeping the model true**, and **monitoring your own prediction error** to notice when it is not.

## Prior knowledge

**P01** (the queueing self-model) and **P02** (the MAPE-K loop and model inversion), the module 24 script ch. 3, 7 and 8 — and the **base-rate fallacy** from **module 15**, which is the trap this project is built around.

---

## Task (specification)

### 1. The managed element (the plant)

A replicated service whose internals are **hidden from the manager**:
- $n$ replica slots; the manager provisions $c$ of them. Each healthy replica serves at the **true** rate $\mu_{\text{true}}$.
- **Drift**: a `deploy(factor)` event changes $\mu_{\text{true}}$ (e.g. by $\times0.7$) at a chosen interval. Nothing announces it.
- **Failure**: each interval, a replica may **fall sick** with small probability and then serves at a fraction (e.g. 30 %) of the nominal rate.
- **Repair**: restarting a replica makes it healthy but **unavailable for a few intervals** — so a *false* restart actively removes healthy capacity. This is what makes the base-rate question bite.
- The observable **p95 latency** follows from the capacity actually serving (use the M/M/c formula of P01/P02).

The manager may only use: a **noisy measurement of the average service rate**, a **noisy per-replica health signal**, the load, and the latency.

### 2. The model at run-time

One parameter: the estimated service rate $\hat\mu$. Two modes:
- **static** — keep the design-time value forever (the naive baseline);
- **re-estimated online** — update from the monitor reading with an EWMA, $\hat\mu \leftarrow 0.85\hat\mu + 0.15\,\mu_{\text{meas}}$.

Also compute, every interval, the **prediction error**: |predicted latency − observed latency|. This is the self-diagnostic — the signal that the model has gone stale.

### 3. The MAPE-K loop

**Monitor** (load, rate, health, latency) → **Analyze** (predict latency with the model; compare against reality) → **Plan** (invert the model for the required capacity; decide which replicas to restart) → **Execute** (capacity changes take effect only after a **provisioning delay**; restarts take a repair time) — over the **Knowledge** of $\hat\mu$ and the health streaks.

**The healing rule and its defence:** flag a replica when its health signal is anomalous, but only restart it after **$k$ consecutive** anomalous intervals (the *persistence* defence from script ch. 7).

### 4. Evaluation (three experiments)

- **A — model drift**: static vs. re-estimated model across a deployment (failures switched *off*, to isolate the effect). Report SLO violations before/after and the prediction error.
- **B — self-healing and the base-rate trap**: no healing vs. healing at persistence $k\in\{1,2,3,5\}$. Report SLO violations, restarts, **true/false positives and precision**.
- **C — the whole system**: drift *and* failures, with each capability switched on in turn.

Plots go to `results/` (gitignored), the test suite is a `__main__` runner.

---

## What should come out (reference orders of magnitude)

**Experiment A — model drift** (true $\mu$ drops 30 % at interval 150):

| model | SLO violations before | after | \|prediction error\| after |
|---|---|---|---|
| static (design-time) | 6.0 % | **97.3 %** | 3.696 |
| re-estimated online | 8.0 % | **13.3 %** | 0.234 |

The adaptive estimate converges to $\hat\mu=7.05$ against a true 7.00. A stale model does not merely become inaccurate — it becomes **dangerous**, provisioning for capacity that no longer exists.

**Experiment B — self-healing and the base-rate trap:**

| strategy | SLO violations | restarts | TP | FP | precision |
|---|---|---|---|---|---|
| no healing | 20.3 % | — | — | — | — |
| heal, $k=1$ (naive) | **60.0 %** | 149 | 6 | **143** | **4.0 %** |
| heal, $k=2$ | 18.0 % | 23 | 11 | 12 | 47.8 % |
| heal, $k=3$ | **12.3 %** | 12 | 11 | 1 | **91.7 %** |
| heal, $k=5$ | 13.7 % | 11 | 11 | 0 | 100.0 % |

The naive healer is not merely wasteful: at 4 % precision it makes the SLO **3× worse than not healing at all** — it causes the outage it exists to prevent. Persistence $k=3$ turns it into a genuine win.

**Experiment C — the whole system** (drift *and* failures):

| configuration | SLO violations | \|pred. error\| |
|---|---|---|
| static model, no healing | 89.0 % | 3.653 |
| static model, healing (k=3) | 60.0 % | 2.133 |
| run-time model, no healing | 40.0 % | 0.409 |
| **run-time model + healing (k=3)** | **14.3 %** | **0.254** |

> **The lesson.** This is what separates *self-aware* from *adaptive*. First, **a model is only useful while it is true** — and nothing announces when it stops being true, so the system must monitor **its own prediction error**, not just its metrics. A stale model is worse than a crude one, because it fails *confidently*. Second, **acting on rare-event detection is a statistical trap, not an engineering detail**: at a realistic base rate even a decent detector yields 4 % precision, and because repair has a cost, naive self-healing tripled the damage. Persistence — demanding that the evidence *persist* before acting — is what converts a detector into a controller. Third, **the capabilities are complementary, not substitutable**: healing cannot rescue a wrong capacity plan, and a perfect model cannot rescue sick replicas. Self-awareness is the whole loop — know yourself, keep the model true, predict, and act on every knob you have.

## Setup & running

```bash
cd modules/24-self-aware-computing/projects/03-final
# write your own implementation, then:
/Users/.../.venv/bin/python test_self_aware.py   # test suite
/Users/.../.venv/bin/python run.py                # 3 experiments + plots
```

Only `numpy` + `matplotlib`. Runtime under a second.

## Solution

The complete reference is in [`solution/`](solution/): `self_aware.py` (the M/M/c self-model, the `Service` plant with drift and sickness, the MAPE-K manager with online re-estimation and healing), `run.py` (3 experiments + plots), `test_self_aware.py` (8 tests).

## Looking back & ahead

This closes module 24 — the arc from the **self-model** (P01: queueing theory verified against a simulator), through the **adaptation loop** (P02: three elasticity policies on the elasticity triple), to the **complete self-aware system** (P03: models at run-time, drift, and healing). The pattern generalises far beyond auto-scaling: any system that must run itself needs a model of itself, a way to keep that model true, and the statistical honesty to know when its own alarms are lying.

---
---

# P03 (final) — das vollständige self-aware System: Modelle zur Laufzeit, Drift und Selbstheilung (deutsche Fassung)

**Modul 24 — Self-aware Computing** · Format: **Python-Projekt (freie Umsetzung, keine Code-Vorgabe)**

> Abschlussprojekt. **Kein vorgegebener Code** — du baust das ganze System selbst. Die Referenzlösung liegt in [`solution/`](solution/); **erst selbst versuchen**. Diese README ist die Spezifikation.

## Worum es geht

Der Auto-Scaler aus P02 hatte ein festes, korrektes Modell. Die Realität hat das nicht. Dieses Projekt baut ein **echtes self-aware System** in Kounevs Sinne: Es trägt ein **Modell von sich selbst zur Laufzeit**, **schätzt dieses Modell online nach**, während das System darunter driftet, und **heilt** Repliken, die es für ausgefallen hält — was eine ehrliche Konfrontation mit der **Base-Rate-Fallacy** erzwingt.

Zwei Dinge gehen schief, die kein statischer Entwurf überlebt: Ein **Deployment** ändert stillschweigend die wahre Servicerate, und **Repliken werden zufällig krank**. Das System muss beides bemerken — mit nichts als seinen eigenen Messungen.

## Lernziel

Du implementierst alle drei Eigenschaften Kounevs konkret: **self-reflective** (Last, Kapazität, Latenz, Replikengesundheit überwachen), **self-predictive** (ein M/M/c-Modell tragen und für die Kapazitätsplanung invertieren), **self-adaptive** (skalieren und reparieren). Vor allem implementierst du das, was es *self-aware* statt bloß adaptiv macht: **das Modell wahr halten** und **den eigenen Prädiktionsfehler überwachen**, um zu merken, wenn es das nicht mehr ist.

## Vorwissen

**P01** (das Warteschlangen-Selbstmodell) und **P02** (die MAPE-K-Schleife und die Modellinvertierung), das Modul-24-Skript Kap. 3, 7 und 8 — und die **Base-Rate-Fallacy** aus **Modul 15**, die Falle, um die herum dieses Projekt gebaut ist.

---

## Aufgabenstellung (Spezifikation)

### 1. Das Managed Element (die Anlage)

Ein replizierter Dienst, dessen Interna dem Manager **verborgen** sind:
- $n$ Replikenplätze; der Manager provisioniert $c$ davon. Jede gesunde Replik bedient mit der **wahren** Rate $\mu_{\text{true}}$.
- **Drift**: Ein `deploy(factor)`-Ereignis ändert $\mu_{\text{true}}$ (z. B. um $\times0{,}7$) zu einem gewählten Intervall. Nichts kündigt es an.
- **Ausfall**: Jedes Intervall kann eine Replik mit kleiner Wahrscheinlichkeit **krank werden** und bedient dann mit einem Bruchteil (z. B. 30 %) der Nennrate.
- **Reparatur**: Ein Neustart macht eine Replik gesund, aber für einige Intervalle **nicht verfügbar** — ein *falscher* Neustart entfernt also aktiv gesunde Kapazität. Genau das lässt die Base-Rate-Frage beißen.
- Die beobachtbare **p95-Latenz** folgt aus der tatsächlich bedienenden Kapazität (M/M/c-Formel aus P01/P02).

Der Manager darf nur nutzen: eine **verrauschte Messung der mittleren Servicerate**, ein **verrauschtes Gesundheitssignal je Replik**, die Last und die Latenz.

### 2. Das Modell zur Laufzeit

Ein Parameter: die geschätzte Servicerate $\hat\mu$. Zwei Modi:
- **statisch** — den Entwurfszeit-Wert für immer behalten (die naive Baseline);
- **online nachgeschätzt** — aus der Monitor-Messung per EWMA aktualisieren, $\hat\mu \leftarrow 0{,}85\hat\mu + 0{,}15\,\mu_{\text{meas}}$.

Berechne außerdem jedes Intervall den **Prädiktionsfehler**: |prädizierte Latenz − beobachtete Latenz|. Das ist die Selbstdiagnose — das Signal, dass das Modell schal geworden ist.

### 3. Die MAPE-K-Schleife

**Monitor** (Last, Rate, Gesundheit, Latenz) → **Analyze** (Latenz mit dem Modell prädizieren; gegen die Realität halten) → **Plan** (das Modell für die nötige Kapazität invertieren; entscheiden, welche Repliken neu starten) → **Execute** (Kapazitätsänderungen wirken erst nach einer **Bereitstellungsverzögerung**; Neustarts brauchen eine Reparaturzeit) — über der **Knowledge** aus $\hat\mu$ und den Gesundheits-Streaks.

**Die Heilungsregel und ihre Absicherung:** Markiere eine Replik, wenn ihr Gesundheitssignal anomal ist, aber starte sie erst nach **$k$ aufeinanderfolgenden** anomalen Intervallen neu (die *Persistenz*-Absicherung aus Skript Kap. 7).

### 4. Evaluation (drei Experimente)

- **A — Modell-Drift**: statisches vs. nachgeschätztes Modell über ein Deployment hinweg (Ausfälle *aus*, um den Effekt zu isolieren). SLO-Verletzungen vorher/nachher und den Prädiktionsfehler berichten.
- **B — Selbstheilung und die Base-Rate-Falle**: keine Heilung vs. Heilung mit Persistenz $k\in\{1,2,3,5\}$. SLO-Verletzungen, Neustarts, **True/False Positives und Precision** berichten.
- **C — das Gesamtsystem**: Drift *und* Ausfälle, mit jeder Fähigkeit einzeln zugeschaltet.

Plots nach `results/` (gitignored), die Testsuite als `__main__`-Runner.

---

## Was am Ende herauskommen soll (Referenz-Größenordnungen)

**Experiment A — Modell-Drift** (wahres $\mu$ fällt bei Intervall 150 um 30 %):

| Modell | SLO-Verletzungen vorher | nachher | \|Prädiktionsfehler\| nachher |
|---|---|---|---|
| statisch (Entwurfszeit) | 6,0 % | **97,3 %** | 3,696 |
| online nachgeschätzt | 8,0 % | **13,3 %** | 0,234 |

Die adaptive Schätzung konvergiert auf $\hat\mu=7{,}05$ gegen ein wahres 7,00. Ein schales Modell wird nicht bloß ungenau — es wird **gefährlich**, weil es für eine Kapazität provisioniert, die es nicht mehr gibt.

**Experiment B — Selbstheilung und die Base-Rate-Falle:**

| Strategie | SLO-Verletzungen | Neustarts | TP | FP | Precision |
|---|---|---|---|---|---|
| keine Heilung | 20,3 % | — | — | — | — |
| Heilung, $k=1$ (naiv) | **60,0 %** | 149 | 6 | **143** | **4,0 %** |
| Heilung, $k=2$ | 18,0 % | 23 | 11 | 12 | 47,8 % |
| Heilung, $k=3$ | **12,3 %** | 12 | 11 | 1 | **91,7 %** |
| Heilung, $k=5$ | 13,7 % | 11 | 11 | 0 | 100,0 % |

Der naive Heiler ist nicht bloß verschwenderisch: Mit 4 % Precision macht er das SLO **3× schlechter als gar nicht zu heilen** — er verursacht den Ausfall, den zu verhindern er existiert. Persistenz $k=3$ macht daraus einen echten Gewinn.

**Experiment C — das Gesamtsystem** (Drift *und* Ausfälle):

| Konfiguration | SLO-Verletzungen | \|Präd.-Fehler\| |
|---|---|---|
| statisches Modell, keine Heilung | 89,0 % | 3,653 |
| statisches Modell, Heilung (k=3) | 60,0 % | 2,133 |
| Laufzeit-Modell, keine Heilung | 40,0 % | 0,409 |
| **Laufzeit-Modell + Heilung (k=3)** | **14,3 %** | **0,254** |

> **Die Lehre.** Das ist es, was *self-aware* von *adaptiv* trennt. Erstens: **Ein Modell nützt nur, solange es wahr ist** — und nichts kündigt an, wann es das nicht mehr ist, das System muss also **seinen eigenen Prädiktionsfehler** überwachen, nicht nur seine Metriken. Ein schales Modell ist schlimmer als ein grobes, weil es *selbstsicher* versagt. Zweitens: **Auf Seltene-Ereignis-Erkennung zu handeln ist eine statistische Falle, kein Implementierungsdetail** — bei realistischer Basisrate liefert selbst ein ordentlicher Detektor 4 % Precision, und weil Reparatur kostet, hat naive Selbstheilung den Schaden verdreifacht. Persistenz — zu verlangen, dass die Evidenz *anhält*, bevor gehandelt wird — verwandelt einen Detektor in einen Regler. Drittens: **Die Fähigkeiten sind komplementär, nicht austauschbar**: Heilung kann keinen falschen Kapazitätsplan retten, und ein perfektes Modell keine kranken Repliken. Selbstwahrnehmung ist die ganze Schleife — sich kennen, das Modell wahr halten, prädizieren und an jedem Hebel handeln, den man hat.

## Setup & Ausführen

```bash
cd modules/24-self-aware-computing/projects/03-final
# eigene Umsetzung schreiben, dann:
/Users/.../.venv/bin/python test_self_aware.py   # Testsuite
/Users/.../.venv/bin/python run.py                # 3 Experimente + Plots
```

Nur `numpy` + `matplotlib`. Laufzeit unter einer Sekunde.

## Lösung

Die vollständige Referenz liegt in [`solution/`](solution/): `self_aware.py` (das M/M/c-Selbstmodell, die `Service`-Anlage mit Drift und Krankheit, der MAPE-K-Manager mit Online-Nachschätzung und Heilung), `run.py` (3 Experimente + Plots), `test_self_aware.py` (8 Tests).

## Rückblick & Ausblick

Damit schließt Modul 24 — der Bogen vom **Selbstmodell** (P01: Warteschlangentheorie gegen einen Simulator verifiziert) über die **Adaptionsschleife** (P02: drei Elastizitätsstrategien am Elastizitäts-Tripel) zum **vollständigen self-aware System** (P03: Modelle zur Laufzeit, Drift und Heilung). Das Muster verallgemeinert weit über Auto-Scaling hinaus: Jedes System, das sich selbst fahren muss, braucht ein Modell von sich, einen Weg, dieses Modell wahr zu halten, und die statistische Ehrlichkeit zu erkennen, wann die eigenen Alarme lügen.
