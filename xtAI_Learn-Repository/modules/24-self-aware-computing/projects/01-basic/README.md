# P01 (basic) — the self-model: queueing theory verified against a simulator

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 24 — Self-aware Computing** · Format: **Jupyter notebook**

## Goal

A self-aware system needs a **model of itself** — something it can ask *"if the load doubles, will I still meet my SLO?"* **before** acting. This project builds that model. You simulate a service as an **M/M/1 queue**, measure it the way a monitoring agent would, and check the analytic predictions against the simulation.

1. a **discrete-event simulation** of a single-server queue,
2. **measurement** as a monitor would do it ($X$, $U$, $R$) and verification of **Little's Law** $N=X\cdot R$ (script ch. 4),
3. the analytic **M/M/1** predictions, held against the simulation (script ch. 5),
4. **response-time quantiles** ($R_{95}$, $R_{99}$) — why SLOs are percentiles,
5. the **$1/(1-\rho)$ explosion**, used to answer a capacity question *before* acting.

## Why this format?

A **notebook** — the whole point is putting a *predicted* number next to a *measured* one, with the plot of the explosion beside it.

## Why synthetic data?

The model has to be **validated against ground truth**, and only a simulator gives you that: you set $\lambda$ and $\mu$, so you know exactly what the right answer is. On a real service you would never know whether a mismatch came from the model or from something you failed to measure.

## Prior knowledge

The module 24 script ch. 4–5, exponential distributions, means and quantiles.

## Task (step by step)

Open `queueing_self_model.ipynb`. The simulator is given; at the `# TODO` spots you build the cores:

- **Part A** (given) — the discrete-event simulator of the queue.
- **Part B** — `measure(sim)`: throughput $X=C/T$, utilisation $U=B/T$, mean response time $R$; then the Little's Law cross-check.
- **Part C** — `mm1_predict(lam, mu)`: the analytic $\rho$, $N$, $R$.
- **Part D** — `response_quantile(lam, mu, p)`: the quantiles of the exponential response-time distribution.
- **Part E** (given) — the explosion table/plot and the capacity question.

## What should come out (expected values)

- **Part B**: Little's Law holds to **machine precision** (relative difference 0) — it is an identity, not an approximation.
- **Part C**: prediction vs. simulation — $\rho=0.5$: 1.999 vs 2.000 · $\rho=0.8$: 4.998 vs 5.000 · $\rho=0.9$: 10.44 vs 10.00 (**~4 %**). The error grows near saturation because the simulation is still converging, not because the model is wrong.
- **Part D**: $R_{95}/R \approx 2.96$ (theory $\ln 20=3.00$), $R_{99}/R\approx4.4$ (theory $\ln100=4.61$ — the deepest tail needs the most samples).
- **Part E**: the explosion $\rho=0.5\to R/S=2$, $0.9\to10$, $0.99\to100$. Capacity question at $\lambda=90$/s, $\mu=100$/s: mean latency a comfortable **100 ms**, but $R_{95}=\mathbf{300}$ ms → the 200 ms SLO is **violated**; the required capacity is $\mu\ge\mathbf{105}$/s — only **1.05×**.

> **The lesson.** Three things worth carrying forward. **Little's Law** ($N=XR$) is exact for any system, so a self-aware system can measure two quantities and *know* the third. **The mean lies**: at $\lambda=90,\mu=100$ the mean latency looks healthy while the 95th percentile breaks the SLO — which is why SLOs are percentiles and why a system must model its **tail**. And the **$1/(1-\rho)$ non-linearity** cuts both ways: it is why utilisation is a treacherous control signal, but also why a small, well-timed capacity increase (1.05×) can cut tail latency by a third. Above all, the model let the system answer a capacity question **by computing, not by trying** — that is the *self-predictive* property in miniature.

## Setup

```bash
cd modules/24-self-aware-computing/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # queueing_self_model.ipynb
```

Only `numpy` + `matplotlib`. Runtime a few seconds.

## Solution

The complete, executed solution is in [`solution/queueing_self_model_solution.ipynb`](solution/queueing_self_model_solution.ipynb) — **try it yourself first!**

## What comes next

- **P02 (medium)**: the **MAPE-K auto-scaler** — three elasticity policies (reactive, control-theoretic, model-based predictive) measured on SLO violations, cost and flapping. The predictive one uses exactly the formulas you just verified.
- **P03 (final)**: the complete **self-aware system** — model at run-time, online re-estimation under drift, and self-healing.

---
---

# P01 (basic) — das Selbstmodell: Warteschlangentheorie gegen einen Simulator verifiziert (deutsche Fassung)

**Modul 24 — Self-aware Computing** · Format: **Jupyter-Notebook**

## Ziel

Ein self-aware System braucht ein **Modell von sich selbst** — etwas, das es fragen kann *„Wenn sich die Last verdoppelt, halte ich dann noch mein SLO?"*, **bevor** es handelt. Dieses Projekt baut dieses Modell. Du simulierst einen Dienst als **M/M/1-Warteschlange**, misst ihn so, wie es ein Monitoring-Agent täte, und prüfst die analytischen Vorhersagen gegen die Simulation.

1. eine **Discrete-Event-Simulation** einer Einzelserver-Warteschlange,
2. **Messung** wie durch einen Monitor ($X$, $U$, $R$) und Verifikation von **Little's Law** $N=X\cdot R$ (Skript Kap. 4),
3. die analytischen **M/M/1**-Vorhersagen, gegen die Simulation gehalten (Skript Kap. 5),
4. **Antwortzeit-Quantile** ($R_{95}$, $R_{99}$) — warum SLOs Perzentile sind,
5. die **$1/(1-\rho)$-Explosion**, genutzt, um eine Kapazitätsfrage *vor* dem Handeln zu beantworten.

## Warum dieses Format?

Ein **Notebook** — der ganze Sinn ist, eine *prädizierte* Zahl neben eine *gemessene* zu stellen, mit dem Plot der Explosion daneben.

## Warum synthetische Daten?

Das Modell muss **gegen ground truth validiert** werden, und nur ein Simulator liefert die: Du setzt $\lambda$ und $\mu$, kennst also die richtige Antwort exakt. An einem echten Dienst wüsstest du nie, ob eine Abweichung vom Modell kommt oder von etwas, das du nicht gemessen hast.

## Vorwissen

Das Modul-24-Skript Kap. 4–5, Exponentialverteilungen, Mittelwerte und Quantile.

## Aufgabenstellung (Schritt für Schritt)

Öffne `queueing_self_model.ipynb`. Der Simulator ist vorgegeben; an den `# TODO`-Stellen baust du die Kerne:

- **Teil A** (vorgegeben) — der Discrete-Event-Simulator der Warteschlange.
- **Teil B** — `measure(sim)`: Durchsatz $X=C/T$, Auslastung $U=B/T$, mittlere Antwortzeit $R$; dann der Little's-Law-Gegencheck.
- **Teil C** — `mm1_predict(lam, mu)`: die analytischen $\rho$, $N$, $R$.
- **Teil D** — `response_quantile(lam, mu, p)`: die Quantile der exponentiellen Antwortzeitverteilung.
- **Teil E** (vorgegeben) — die Explosionstabelle/-plot und die Kapazitätsfrage.

## Was am Ende herauskommt (Erwartungswerte)

- **Teil B**: Little's Law gilt bis auf **Maschinengenauigkeit** (relative Differenz 0) — es ist eine Identität, keine Näherung.
- **Teil C**: Vorhersage vs. Simulation — $\rho=0{,}5$: 1,999 vs 2,000 · $\rho=0{,}8$: 4,998 vs 5,000 · $\rho=0{,}9$: 10,44 vs 10,00 (**~4 %**). Der Fehler wächst nahe der Sättigung, weil die Simulation noch konvergiert, nicht weil das Modell falsch wäre.
- **Teil D**: $R_{95}/R \approx 2{,}96$ (Theorie $\ln 20=3{,}00$), $R_{99}/R\approx4{,}4$ (Theorie $\ln100=4{,}61$ — der tiefste Schwanz braucht die meisten Stichproben).
- **Teil E**: die Explosion $\rho=0{,}5\to R/S=2$, $0{,}9\to10$, $0{,}99\to100$. Kapazitätsfrage bei $\lambda=90$/s, $\mu=100$/s: mittlere Latenz komfortable **100 ms**, aber $R_{95}=\mathbf{300}$ ms → das 200-ms-SLO ist **verletzt**; nötige Kapazität $\mu\ge\mathbf{105}$/s — nur **1,05×**.

> **Die Lehre.** Drei Dinge zum Mitnehmen. **Little's Law** ($N=XR$) ist exakt für jedes System, ein self-aware System kann also zwei Größen messen und die dritte *wissen*. **Der Mittelwert lügt**: bei $\lambda=90,\mu=100$ sieht die mittlere Latenz gesund aus, während das 95-%-Perzentil das SLO reißt — deshalb sind SLOs Perzentile, und deshalb muss ein System seinen **Schwanz** modellieren. Und die **$1/(1-\rho)$-Nichtlinearität** wirkt in beide Richtungen: Sie macht die Auslastung zu einem tückischen Regelsignal, aber sie ist auch der Grund, warum eine kleine, gut getimte Kapazitätserhöhung (1,05×) die Tail-Latenz um ein Drittel senken kann. Vor allem aber konnte das Modell eine Kapazitätsfrage **durch Rechnen statt durch Ausprobieren** beantworten — das ist die *self-predictive*-Eigenschaft im Kleinen.

## Setup

```bash
cd modules/24-self-aware-computing/projects/01-basic
/Users/.../.venv/bin/python -m jupyter lab   # queueing_self_model.ipynb
```

Nur `numpy` + `matplotlib`. Laufzeit wenige Sekunden.

## Lösung

Die vollständige, ausgeführte Lösung liegt in [`solution/queueing_self_model_solution.ipynb`](solution/queueing_self_model_solution.ipynb) — **erst selbst probieren!**

## Weiter geht's

- **P02 (medium)**: der **MAPE-K-Auto-Scaler** — drei Elastizitätsstrategien (reaktiv, regelungstheoretisch, modellbasiert-prädiktiv), gemessen an SLO-Verletzungen, Kosten und Flapping. Die prädiktive nutzt genau die Formeln, die du gerade verifiziert hast.
- **P03 (final)**: das vollständige **self-aware System** — Modell zur Laufzeit, Online-Nachschätzung unter Drift und Selbstheilung.
