# P01 (basic) — inverse-variance fusion: how the brain merges two senses

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Module 18 — Multimodal Interfaces** · Format: **Jupyter notebook**

## Goal

You recompute the mathematical heart of multimodal fusion yourself — the **inverse-variance weighting** (maximum likelihood integration) — and thereby reproduce the three core findings of the famous experiment by **Ernst & Banks (2002)**:

1. **A gain in precision**: the fusion of two noisy modalities is *always* more precise than either individual one ($\sigma_{fus}^2 \le \min$).
2. **A reliability-adaptive perceptual shift**: under a sensory conflict the perception moves towards the *more reliable* modality — the actual punchline.
3. **A gain in discrimination**: the discrimination threshold (JND) drops.

This is at the same time the direct continuation of the **complementary filter from module 17** (gyro + accel): the same formula, different senses.

## Why this format?

A **notebook**, because the formula, the simulation, the numbers and the plots belong together here: you see the distributions become narrower and the perception tip over. That is the core of the intuition.

## Why synthetic data?

The concept is a **law about noise and reliability**, not a dataset phenomenon. Simulated Gaussian measurements with *known* variances allow the theory (the inverse-variance formula) to be checked directly against the empirical result (the sample spread) — with real sensor data you would not know the true variances and could not run the comparison cleanly. A fixed seed (`np.random.default_rng(42)`) makes everything reproducible.

## Prior knowledge

The Gaussian distribution, variance, some probability theory. Read **chapters 8–9** of the [module 18 script](../../README.md) (inverse-variance weighting, Ernst & Banks).

## Assignment (step by step)

Open `cue_integration.ipynb`. Most cells are given; at the `# TODO` places you fill in the core formulas:

- **Part A** — the inverse-variance **weights** $w_V, w_H$, the **fused estimate** and $\sigma_{fus}$. Show that the fusion is more precise than the better sensor.
- **Part B** — the **predicted perceptual value** under a conflict, from the weights. Observe how the same conflict (53 vs. 57 mm) is perceived *differently* depending on whether the seeing is sharp or noisy.
- **Part C** (given) — the **gain in discrimination** (2AFC, JND).
- **Part D** (text) — the arc back to module 17.

## What should work at the end

The executed notebook shows:
- weights $w_V=0.8$, $w_H=0.2$; $\sigma_{fus}\approx 0.894$ mm $<$ 1 mm (the empirical `std(fused)` matches it).
- a perceptual shift from **53.8 → 56.2 mm** purely by making the seeing noisy.
- the JND of the fusion (**0.85 mm**) smaller than that of the best individual sense (seeing, 0.95 mm).
- two plots: the narrower fusion distribution; the perception tipping from seeing to touching.

## Setup

From the repo root directory (the venv set up as in the repo):

```bash
.venv/bin/python -m jupyter lab
```

Then open `cue_integration.ipynb` and execute it cell by cell. It needs only `numpy`, `scipy`, `matplotlib` (all present in the `.venv`). Runtime: a few seconds.

## Solution

The complete, executed solution is in [`solution/cue_integration_solution.ipynb`](solution/cue_integration_solution.ipynb) — **try it yourself first!**

## What comes next

- **P02 (medium)**: early vs. late fusion in classification + mutual disambiguation — including the trap of correlated errors.
- **P03 (final)**: a complete "Put-that-there" interpreter with temporal fusion.

---

# P01 (basic) — Inverse-Varianz-Fusion: Wie das Gehirn zwei Sinne verschmilzt (deutsche Fassung)

**Modul 18 — Multimodal Interfaces** · Format: **Jupyter-Notebook**

## Ziel

Du rechnest das mathematische Herzstück multimodaler Fusion selbst nach — die **inverse-Varianz-Gewichtung** (Maximum-Likelihood-Integration) — und reproduzierst damit die drei Kernbefunde des berühmten Experiments von **Ernst & Banks (2002)**:

1. **Präzisionsgewinn**: Die Fusion zweier verrauschter Modalitäten ist *immer* präziser als jede einzelne ($\sigma_{fus}^2 \le \min$).
2. **Reliabilitätsadaptiver Wahrnehmungs-Shift**: Bei einem Sinneskonflikt wandert die Wahrnehmung zur *zuverlässigeren* Modalität — der eigentliche Clou.
3. **Diskriminationsgewinn**: Die Unterscheidungsschwelle (JND) sinkt.

Das ist zugleich die direkte Fortsetzung des **Komplementärfilters aus Modul 17** (Gyro + Accel): dieselbe Formel, andere Sinne.

## Warum dieses Format?

Ein **Notebook**, weil Formel, Simulation, Zahlen und Plots hier zusammengehören: Man sieht die Verteilungen schmaler werden und die Wahrnehmung kippen. Das ist der Kern der Anschauung.

## Warum synthetische Daten?

Das Konzept ist ein **Gesetz über Rauschen und Reliabilität**, kein Datensatz-Phänomen. Simulierte Gauß-Messungen mit *bekannten* Varianzen erlauben, die Theorie (inverse-Varianz-Formel) direkt gegen die Empirie (Stichproben-Streuung) zu prüfen — mit echten Sensordaten kennte man die wahren Varianzen nicht und könnte den Abgleich nicht sauber führen. Fester Seed (`np.random.default_rng(42)`) macht alles reproduzierbar.

## Vorwissen

Gauß-Verteilung, Varianz, etwas Wahrscheinlichkeitsrechnung. Lies **Kapitel 8–9** des [Modul-18-Skripts](../../README.md) (inverse-Varianz-Gewichtung, Ernst & Banks).

## Aufgabenstellung (Schritt für Schritt)

Öffne `cue_integration.ipynb`. Die meisten Zellen sind vorgegeben; an den `# TODO`-Stellen füllst du die Kernformeln ein:

- **Teil A** — die inverse-Varianz-**Gewichte** $w_V, w_H$, die **fusionierte Schätzung** und $\sigma_{fus}$. Zeige, dass die Fusion präziser ist als der bessere Sensor.
- **Teil B** — der **vorhergesagte Wahrnehmungswert** bei Konflikt aus den Gewichten. Beobachte, wie derselbe Konflikt (53 vs. 57 mm) *anders* wahrgenommen wird, je nachdem, ob das Sehen scharf oder verrauscht ist.
- **Teil C** (vorgegeben) — der **Diskriminationsgewinn** (2AFC, JND).
- **Teil D** (Text) — der Bogen zurück zu Modul 17.

## Was am Ende funktionieren soll

Das ausgeführte Notebook zeigt:
- Gewichte $w_V=0.8$, $w_H=0.2$; $\sigma_{fus}\approx 0.894$ mm $<$ 1 mm (empirische `std(fused)` trifft das).
- Wahrnehmungs-Shift von **53.8 → 56.2 mm** allein durch Verrauschen des Sehens.
- JND der Fusion (**0.85 mm**) kleiner als die des besten Einzelsinns (Sehen, 0.95 mm).
- Zwei Plots: schmalere Fusionsverteilung; Wahrnehmung, die vom Sehen zum Tasten kippt.

## Setup

Aus dem Repo-Wurzelverzeichnis (venv wie im Repo eingerichtet):

```bash
.venv/bin/python -m jupyter lab
```

Dann `cue_integration.ipynb` öffnen und Zelle für Zelle ausführen. Benötigt nur `numpy`, `scipy`, `matplotlib` (alle in der `.venv` vorhanden). Laufzeit: wenige Sekunden.

## Lösung

Die vollständige, ausgeführte Lösung liegt in [`solution/cue_integration_solution.ipynb`](solution/cue_integration_solution.ipynb) — **erst selbst probieren!**

## Weiter geht's

- **P02 (medium)**: early vs. late Fusion in der Klassifikation + Mutual Disambiguation — inklusive der Falle korrelierter Fehler.
- **P03 (final)**: ein vollständiger „Put-that-there"-Interpreter mit zeitlicher Fusion.
