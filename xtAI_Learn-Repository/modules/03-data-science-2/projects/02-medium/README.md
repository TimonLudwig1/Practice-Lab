# Project 02 (medium) — Bootstrap and permutation test: statistics from the computer

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format:** Jupyter notebook (`bootstrap_permutation.ipynb`).
**Why this format?** Resampling procedures are understood through the histograms of the bootstrap and null distributions — code and figure have to sit right next to each other.

**Data: synthetic, with a known truth — that is the didactic core here.** An A/B test (2 x 4,000 visitors, conversion + order values, fixed seed 42) is generated in the first cell. Because the true effects are written in the generator, you can check at the end whether your tests found the truth — and you experience, under controlled conditions, the most important case in practice: **a genuine effect that the test fails to find for lack of power.**

## Goal

Implement bootstrap confidence intervals (median, difference of medians) and a permutation test by hand, apply them to both A/B questions and interpret the results correctly: the order value effect is significant (p about 0.005), the conversion effect is not detectable despite really existing (p about 0.19) → "not significant is not the same as no effect".

## Prior knowledge

- Sections 3.1 and 3.2 of the module script; module 02 section 3.1 (p-values, CIs)
- numpy basics (`choice`, `permutation`, `percentile`)

## Tasks

1. Work through the notebook; to be implemented are `bootstrap_distribution`, the confidence interval of the difference, `permutation_test` and its application to the conversion rates.
2. Mini checks must evaluate to `True`.
3. To finish: formulate a three-sentence recommendation for the product team (a reference version can be unfolded).

## What should work in the end

- The bootstrap 95 % CI of the B median is about [53, 68] EUR; the CI of the difference is about [3.5, 20.5] EUR (it does not contain 0, but it does contain the true value of 6 EUR).
- Permutation test on the order value: p below 0.02; on conversion: p above 0.05.
- You can explain why the conversion test fails despite a genuine effect (power).

## Solution

Fully executed reference solution: [`solution/solution.ipynb`](solution/solution.ipynb).

---
---

# Projekt 02 (medium) — Bootstrap & Permutationstest: Statistik aus dem Computer (deutsche Fassung)

**Format:** Jupyter Notebook (`bootstrap_permutation.ipynb`).
**Warum dieses Format?** Resampling-Verfahren versteht man über die Histogramme der Bootstrap- und Nullverteilungen — Code und Grafik müssen direkt nebeneinander stehen.

**Daten: synthetisch, mit bekannter Wahrheit — das ist hier der Kern der Didaktik.** Ein A/B-Test (2×4.000 Besucher, Konversion + Bestellwerte, fester Seed 42) wird in der ersten Zelle erzeugt. Weil die wahren Effekte im Generator stehen, kannst du am Ende prüfen, ob deine Tests die Wahrheit gefunden haben — und erlebst kontrolliert den wichtigsten Fall der Praxis: **einen echten Effekt, den der Test mangels Power nicht findet.**

## Ziel

Bootstrap-Konfidenzintervalle (Median, Median-Differenz) und einen Permutationstest von Hand implementieren, auf beide A/B-Fragen anwenden und die Ergebnisse korrekt interpretieren: Bestellwert-Effekt signifikant (p ≈ 0,005), Konversions-Effekt trotz realer Existenz nicht nachweisbar (p ≈ 0,19) → „nicht signifikant ≠ kein Effekt".

## Vorwissen

- Modul-Skript Abschnitte 3.1 und 3.2; Modul 02 Abschnitt 3.1 (p-Werte, KIs)
- numpy-Grundlagen (`choice`, `permutation`, `percentile`)

## Aufgaben

1. Notebook durcharbeiten; zu implementieren sind `bootstrap_verteilung`, das Differenz-KI, `permutationstest` und dessen Anwendung auf die Konversionsraten.
2. Mini-Checks müssen `True` ergeben.
3. Abschluss: 3-Sätze-Empfehlung ans Produktteam formulieren (Muster ausklappbar).

## Was am Ende funktionieren soll

- Bootstrap-95%-KI des B-Medians ≈ [53, 68] €; KI der Differenz ≈ [3,5, 20,5] € (enthält die 0 nicht, enthält aber den wahren Wert 6 €).
- Permutationstest Bestellwert: p < 0,02; Konversion: p > 0,05.
- Du kannst erklären, warum der Konversionstest trotz echten Effekts scheitert (Power).

## Lösung

Vollständig ausgeführte Musterlösung: [`solution/solution.ipynb`](solution/solution.ipynb).
