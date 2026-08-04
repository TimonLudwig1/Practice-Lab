# P02 (medium) — the MAPE-K auto-scaler: reactive vs. control-theoretic vs. predictive elasticity

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 24 — Self-aware Computing** · Format: **Python module + test suite**

## Goal

You put the self-model of P01 inside a **MAPE-K loop** and let it run the system. A service with `c` replicas faces a daily load pattern plus a flash crowd; an autonomic manager must keep the p95 latency under an SLO without wasting capacity — while every capacity change only takes effect after a **provisioning delay**.

1. the **M/M/c** model and its **inversion** `min_replicas` — "how much capacity would this load need?" (script ch. 5),
2. three **elasticity policies**: **reactive** (threshold + hysteresis), **control-theoretic** (HPA-style proportional + integral), **model-based predictive** (forecast + model inversion — MPC applied to the system itself, script ch. 6),
3. honest evaluation with the **elasticity triple**: SLO violations (under-provisioning), cost (over-provisioning), adaptations (instability/flapping).

## Why this format?

A **Python module with a test suite** — the policies are small, sharply testable functions (the model inversion must be *minimal and sufficient*; a rising trend must produce more capacity than a steady one), and the experiments sweep parameters systematically.

## Why synthetic data?

The evaluation needs a **ground-truth "needed capacity"** for every interval, which only a generated workload gives — that is what makes the under/over-provisioning metrics and the *oracle* baseline computable at all. The load profile (daily sinusoid + flash crowd + noise) is the standard shape auto-scalers are benchmarked on.

## Prior knowledge

**P01** of this module (the queueing self-model), the module 24 script ch. 5–6, and — for the predictive policy — the **MPC** idea from **module 23** (predict a horizon, act on the first step, repeat).

## Task

Open `scaling.py`. The M/M/c model, the workload, the MAPE-K loop, the metrics and `policy_control` (worked out as an example) are given — you implement the **three cores** (`# TODO` / `NotImplementedError`):

1. **`min_replicas(lam, mu, slo)`** — the smallest replica count whose predicted p95 meets the SLO. This is the **model inversion**, the self-predictive step.
2. **`policy_reactive(state, knob)`** — scale out above `knob`, in below `knob - gap`; the gap *is* the hysteresis.
3. **`policy_predictive(state, knob)`** — extrapolate the smoothed load over the provisioning delay, then invert the model for that forecast (times a safety margin).

Then:

```bash
cd modules/24-self-aware-computing/projects/02-medium
/Users/.../.venv/bin/python test_scaling.py   # 8 tests -> all PASS
/Users/.../.venv/bin/python run.py             # 3 experiments + plots
```

## What should come out (expected values)

**Experiment 1 — the elasticity triple** (SLO = p95 < 500 ms, provisioning delay 3 intervals):

| policy | SLO violations | avg replicas | adaptations | over-provisioning |
|---|---|---|---|---|
| reactive | **1.0 %** | 8.79 | 24 | 2.38 |
| control | 13.0 % | 7.78 | 55 | 1.51 |
| predictive | 10.0 % | **6.97** | 51 | **0.69** |
| *oracle* | *0.0 %* | *6.42* | — | *0.00* |

Read the **whole row**: `reactive` wins on SLO only because it **buys it with capacity** (2.38 replicas wasted on average). `predictive` runs closest to the oracle's cost.

**Experiment 2 — the Pareto front.** Each policy has a knob trading cost against violations, so a single-setting comparison proves nothing. Sweeping them:

- at **equal cost ≈ 8.79 replicas**: reactive 1.0 % vs. **predictive 0.5 %** violations;
- at **equal violations ≈ 10–11 %**: reactive 7.62 vs. **predictive 6.97** replicas;
- `control` is **dominated throughout**;
- `predictive` at its cheapest setting reaches **6.39 replicas — below the oracle's 6.42** (it trades a few violations for cost, landing on the other side of the oracle point).

**Experiment 3 — flapping.** Hysteresis and cooldown are independent, and both matter:

| configuration | adaptations |
|---|---|
| gap 0.40, cooldown 2, serialised | **24** |
| gap 0.05 (weak hysteresis) | 58 |
| gap 0.40, no cooldown, concurrent | 60 |
| gap 0.05, no cooldown, concurrent | **156** |

Removing both multiplies the adaptation count by **6.5×**.

> **The lesson.** Three things. First, **no single metric characterises an auto-scaler** — the elasticity triple exists because each number alone is trivially gamed (provision 10× and never violate the SLO). Second, **a model beats a threshold**: the predictive policy provisions almost exactly what is needed because it *inverts* the queueing model for a *forecast* load, so it can start capacity before the demand arrives — the only way to hide the provisioning delay. That is precisely MPC (module 23) with the software system as the plant, and it is what "self-predictive" means operationally. Third, **adaptation itself has a cost**: without hysteresis and cooldown the manager chases its own not-yet-visible corrections, and since every change carries a warm-up, the loop degrades the very system it is improving.

## Solution

The complete reference is in [`solution/`](solution/). Try it yourself first!

## What comes next

**P03 (final)**: the complete **self-aware system** — a MAPE-K loop with a **model at run-time** whose parameters are **re-estimated online** as the system drifts, plus **anomaly detection and self-healing** (and the base-rate trap that makes naive healing dangerous). No code given.

---
---

# P02 (medium) — der MAPE-K-Auto-Scaler: reaktive vs. regelungstheoretische vs. prädiktive Elastizität (deutsche Fassung)

**Modul 24 — Self-aware Computing** · Format: **Python-Modul + Testsuite**

## Ziel

Du setzt das Selbstmodell aus P01 in eine **MAPE-K-Schleife** und lässt es das System fahren. Ein Dienst mit `c` Repliken trifft auf ein Tageslastmuster plus einen Flash Crowd; ein Autonomic Manager muss die p95-Latenz unter einem SLO halten, ohne Kapazität zu verschwenden — wobei jede Kapazitätsänderung erst nach einer **Bereitstellungsverzögerung** wirkt.

1. das **M/M/c**-Modell und seine **Invertierung** `min_replicas` — „wie viel Kapazität bräuchte diese Last?" (Skript Kap. 5),
2. drei **Elastizitätsstrategien**: **reaktiv** (Schwelle + Hysterese), **regelungstheoretisch** (HPA-artig proportional + integral), **modellbasiert-prädiktiv** (Prognose + Modellinvertierung — MPC auf das System selbst angewandt, Skript Kap. 6),
3. ehrliche Bewertung mit dem **Elastizitäts-Tripel**: SLO-Verletzungen (Under-Provisioning), Kosten (Over-Provisioning), Adaptionen (Instabilität/Flapping).

## Warum dieses Format?

Ein **Python-Modul mit Testsuite** — die Strategien sind kleine, scharf testbare Funktionen (die Modellinvertierung muss *minimal und hinreichend* sein; ein steigender Trend muss mehr Kapazität liefern als ein konstanter), und die Experimente sweepen Parameter systematisch.

## Warum synthetische Daten?

Die Bewertung braucht für jedes Intervall eine **ground-truth-„benötigte Kapazität"**, die nur eine generierte Last liefert — erst dadurch werden die Under-/Over-Provisioning-Metriken und die *Oracle*-Baseline überhaupt berechenbar. Das Lastprofil (Tages-Sinus + Flash Crowd + Rauschen) ist die Standardform, an der Auto-Scaler gemessen werden.

## Vorwissen

**P01** dieses Moduls (das Warteschlangen-Selbstmodell), das Modul-24-Skript Kap. 5–6 und — für die prädiktive Strategie — die **MPC**-Idee aus **Modul 23** (einen Horizont prädizieren, den ersten Schritt anwenden, wiederholen).

## Aufgabenstellung

Öffne `scaling.py`. Das M/M/c-Modell, die Last, die MAPE-K-Schleife, die Metriken und `policy_control` (als Beispiel ausgearbeitet) sind vorgegeben — du implementierst die **drei Kerne** (`# TODO` / `NotImplementedError`):

1. **`min_replicas(lam, mu, slo)`** — die kleinste Replikenzahl, deren prädizierte p95 das SLO hält. Das ist die **Modellinvertierung**, der self-predictive-Schritt.
2. **`policy_reactive(state, knob)`** — hochskalieren über `knob`, runter unter `knob - gap`; die Lücke *ist* die Hysterese.
3. **`policy_predictive(state, knob)`** — die geglättete Last über die Bereitstellungsverzögerung extrapolieren, dann das Modell für diese Prognose invertieren (mal einer Sicherheitsmarge).

Dann:

```bash
cd modules/24-self-aware-computing/projects/02-medium
/Users/.../.venv/bin/python test_scaling.py   # 8 Tests -> alle PASS
/Users/.../.venv/bin/python run.py             # 3 Experimente + Plots
```

## Was am Ende herauskommt (Erwartungswerte)

**Experiment 1 — das Elastizitäts-Tripel** (SLO = p95 < 500 ms, Bereitstellungsverzögerung 3 Intervalle):

| Strategie | SLO-Verletzungen | Ø Repliken | Adaptionen | Over-Provisioning |
|---|---|---|---|---|
| reaktiv | **1,0 %** | 8,79 | 24 | 2,38 |
| control | 13,0 % | 7,78 | 55 | 1,51 |
| prädiktiv | 10,0 % | **6,97** | 51 | **0,69** |
| *Oracle* | *0,0 %* | *6,42* | — | *0,00* |

Lies die **ganze Zeile**: `reaktiv` gewinnt beim SLO nur, weil es dieses **mit Kapazität erkauft** (im Mittel 2,38 Repliken verschwendet). `prädiktiv` läuft am nächsten an den Oracle-Kosten.

**Experiment 2 — die Pareto-Front.** Jede Strategie hat einen Knopf, der Kosten gegen Verletzungen tauscht, ein Vergleich bei einer Einstellung beweist also nichts. Beim Sweep:

- bei **gleichen Kosten ≈ 8,79 Repliken**: reaktiv 1,0 % vs. **prädiktiv 0,5 %** Verletzungen;
- bei **gleichen Verletzungen ≈ 10–11 %**: reaktiv 7,62 vs. **prädiktiv 6,97** Repliken;
- `control` ist **durchgehend dominiert**;
- `prädiktiv` erreicht in seiner billigsten Einstellung **6,39 Repliken — unter den 6,42 des Oracle** (es tauscht ein paar Verletzungen gegen Kosten und landet jenseits des Oracle-Punkts).

**Experiment 3 — Flapping.** Hysterese und Cooldown sind unabhängig, und beide zählen:

| Konfiguration | Adaptionen |
|---|---|
| Lücke 0,40, Cooldown 2, serialisiert | **24** |
| Lücke 0,05 (schwache Hysterese) | 58 |
| Lücke 0,40, kein Cooldown, nebenläufig | 60 |
| Lücke 0,05, kein Cooldown, nebenläufig | **156** |

Beides zu entfernen vervielfacht die Adaptionszahl um **6,5×**.

> **Die Lehre.** Drei Dinge. Erstens: **Keine einzelne Metrik charakterisiert einen Auto-Scaler** — das Elastizitäts-Tripel existiert, weil jede Zahl für sich trivial manipulierbar ist (10-fach provisionieren und das SLO nie verletzen). Zweitens: **Ein Modell schlägt eine Schwelle** — die prädiktive Strategie provisioniert fast exakt das Nötige, weil sie das Warteschlangenmodell für eine *prognostizierte* Last *invertiert* und Kapazität so starten kann, bevor die Nachfrage eintrifft — der einzige Weg, die Bereitstellungsverzögerung zu verbergen. Genau das ist MPC (Modul 23) mit dem Softwaresystem als Anlage, und genau das heißt „self-predictive" operativ. Drittens: **Adaption selbst kostet** — ohne Hysterese und Cooldown jagt der Manager seinen eigenen, noch nicht sichtbaren Korrekturen hinterher, und da jede Änderung ein Aufwärmen trägt, verschlechtert die Schleife genau das System, das sie verbessern soll.

## Lösung

Die vollständige Referenz liegt in [`solution/`](solution/). Erst selbst versuchen!

## Weiter geht's

**P03 (final)**: das vollständige **self-aware System** — eine MAPE-K-Schleife mit einem **Modell zur Laufzeit**, dessen Parameter **online nachgeschätzt** werden, während das System driftet, plus **Anomalieerkennung und Selbstheilung** (und die Base-Rate-Falle, die naives Heilen gefährlich macht). Keine Code-Vorgabe.
