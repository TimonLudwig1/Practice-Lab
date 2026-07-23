# P02 (medium) — early vs. late fusion, mutual disambiguation & the correlation trap

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 18 — Multimodal Interfaces** · Format: **Python module + test suite**

## Goal

You implement the **fusion mechanics** yourself and investigate empirically the three central statements of the script:

1. **Complementarity**: two modalities, *each of which alone is ambiguous* (only ~50 % accuracy), become almost perfect together (~95 %). You compare **early** (feature concatenation) and **late** fusion (the Bayes product of the posteriors).
2. **Robustness against missing modalities**: what happens if a modality fails at test time? Late fusion degrades *gracefully*, early fusion needs an imputation hack.
3. **The correlation trap** (the most important learning point): the fusion gain — and the *mutual disambiguation* $P(\text{both wrong}) \approx \varepsilon_A\varepsilon_B$ — holds **only for independent errors**. If you correlate the noise of the modalities, **the gain disappears**.

## Why this format?

A **Python module with a test suite**, because here the *logic* (the fusion functions, the experiment setup) is in the foreground and we want to test it cleanly and vary it systematically over a parameter (the correlation $\rho$) — that is clearer in a module/script structure than in a notebook.

## Why synthetic data?

Both effects are **structural laws**, not dataset accidents. Only with self-constructed modalities can you *guarantee* that each one alone is ambiguous (experiment 1) resp. set the error correlation exactly (experiment 2). With real data the effect could not be isolated. Both generators are disclosed and reproducible with a fixed seed (`multimodal.py`).

## Prior knowledge

- **P01** of this module (inverse-variance fusion) and **chapters 6–10** of the [script](../../README.md) (the Bayes product rule, early/late, mutual disambiguation).
- Naive Bayes / `predict_proba` (modules 04/08), the basic idea of conditional independence (module 07).

## Assignment

Open `multimodal.py`. The data generators and classifiers are given — you implement the two **fusion functions** (the ones with `# TODO` / `NotImplementedError`):

1. **`late_fusion_proba(proba_list, prior)`** — the Bayes product rule:
   $$P(y\mid z_A,z_B,\dots) \propto \frac{\prod_m P(y\mid z_m)}{P(y)^{M-1}}.$$
   Multiply element-wise, divide out the shared prior $M{-}1$ times, normalize row-wise.
2. **`early_fusion_fit_predict(...)`** — concatenate the feature vectors with `np.hstack`, train *one* `GaussianNB`, return the test predictions.

After that:

```bash
cd modules/18-multimodal-interfaces/projects/02-medium
/Users/.../.venv/bin/python test_multimodal.py   # 7 tests -> all PASS
/Users/.../.venv/bin/python run.py                # both experiments + plots
```

`run.py` prints the comparison tables and puts two plots into `results/` (gitignored).

## What comes out at the end (expected values)

**Experiment 1 — complementarity:**
- modality A / B only: **~0.47** (chance would be 0.25, but each modality cannot resolve the second class of its pair → a ceiling at ~0.5).
- **late fusion ~0.95**, **early fusion ~0.95** — both resolve the ambiguity, because the modalities are *complementary*.
- missing modality (B drops out): late simply uses A's posterior (~0.47, cleanly); early needs imputation.

**Experiment 2 — the correlation trap** (a table over $\rho$):
- $\rho=0$: fusion halves the error (~0.16 → **0.08**); $P(\text{both wrong})=$ **0.025** $\approx \varepsilon_A\varepsilon_B=0.026$ (the independence signature matches).
- $\rho=0.98$: the fusion gain is ~**0.002** (practically zero); $P(\text{both wrong})$ rises to **0.14** $\approx$ the individual error — the modalities make *the same* error.

> **The lesson.** Fusion is **not a free gain**. It lives off the **diversity of the errors** — exactly like ensembles (modules 04/05). Whoever builds in fusion has to **measure the error correlation**, not assume independence.

## Solution

A complete, runnable reference is in [`solution/`](solution/) (an identical `run.py`/`test_multimodal.py`, plus a filled-in `multimodal.py`). Try it yourself first!

## What comes next

**P03 (final)**: the grand arc — a complete **"Put-that-there" interpreter** with asynchronous speech and gesture streams, time-weighted reference resolution and quantified mutual disambiguation. No given code.

---

# P02 (medium) — Early vs. Late Fusion, Mutual Disambiguation & die Korrelations-Falle (deutsche Fassung)

**Modul 18 — Multimodal Interfaces** · Format: **Python-Modul + Testsuite**

## Ziel

Du implementierst die **Fusionsmechanik** selbst und untersuchst empirisch die drei zentralen Aussagen des Skripts:

1. **Komplementarität**: Zwei Modalitäten, von denen *jede allein mehrdeutig* ist (nur ~50 % Genauigkeit), werden zusammen fast perfekt (~95 %). Du vergleichst **early** (Feature-Konkatenation) und **late** Fusion (Bayes-Produkt der Posteriors).
2. **Robustheit gegen fehlende Modalitäten**: Was passiert, wenn zur Testzeit eine Modalität ausfällt? Late Fusion degradiert *sauber*, Early Fusion braucht einen Imputations-Hack.
3. **Die Korrelations-Falle** (der wichtigste Lernpunkt): Der Fusionsgewinn — und die *Mutual Disambiguation* $P(\text{beide falsch}) \approx \varepsilon_A\varepsilon_B$ — gilt **nur bei unabhängigen Fehlern**. Korrelierst du das Rauschen der Modalitäten, **verschwindet der Gewinn**.

## Warum dieses Format?

Ein **Python-Modul mit Testsuite**, weil hier die *Logik* (Fusionsfunktionen, Experiment-Setup) im Vordergrund steht und wir sie sauber testen und über Parameter (Korrelation $\rho$) systematisch variieren wollen — das ist in einer Modul-/Skript-Struktur klarer als in einem Notebook.

## Warum synthetische Daten?

Beide Effekte sind **strukturelle Gesetze**, keine Datensatz-Zufälle. Nur mit selbst konstruierten Modalitäten kann man *garantieren*, dass jede allein mehrdeutig ist (Experiment 1) bzw. die Fehler-Korrelation exakt einstellen (Experiment 2). Mit echten Daten ließe sich der Effekt nicht isolieren. Beide Generatoren sind offengelegt und mit festem Seed reproduzierbar (`multimodal.py`).

## Vorwissen

- **P01** dieses Moduls (inverse-Varianz-Fusion) und **Kapitel 6–10** des [Skripts](../../README.md) (Bayes-Produktregel, early/late, Mutual Disambiguation).
- Naive Bayes / `predict_proba` (Modul 04/08), Grundidee der bedingten Unabhängigkeit (Modul 07).

## Aufgabenstellung

Öffne `multimodal.py`. Die Datengeneratoren und Klassifikatoren sind vorgegeben — du implementierst die beiden **Fusionsfunktionen** (die mit `# TODO` / `NotImplementedError`):

1. **`late_fusion_proba(proba_list, prior)`** — die Bayes-Produktregel:
   $$P(y\mid z_A,z_B,\dots) \propto \frac{\prod_m P(y\mid z_m)}{P(y)^{M-1}}.$$
   Elementweise multiplizieren, den geteilten Prior $M{-}1$-mal herausdividieren, zeilenweise normalisieren.
2. **`early_fusion_fit_predict(...)`** — Feature-Vektoren mit `np.hstack` konkatenieren, *ein* `GaussianNB` trainieren, Testvorhersagen zurückgeben.

Danach:

```bash
cd modules/18-multimodal-interfaces/projects/02-medium
/Users/.../.venv/bin/python test_multimodal.py   # 7 Tests -> alle PASS
/Users/.../.venv/bin/python run.py                # beide Experimente + Plots
```

`run.py` druckt die Vergleichstabellen und legt zwei Plots in `results/` ab (gitignored).

## Was am Ende herauskommt (Erwartungswerte)

**Experiment 1 — Komplementarität:**
- nur Modalität A / B: **~0.47** (Zufall wäre 0.25, aber jede Modalität kann die zweite Klasse ihres Paares nicht auflösen → Deckel bei ~0.5).
- **late Fusion ~0.95**, **early Fusion ~0.95** — beide lösen die Mehrdeutigkeit, weil die Modalitäten *komplementär* sind.
- Missing-Modality (B fällt aus): late nutzt einfach A's Posterior (~0.47, sauber); early braucht Imputation.

**Experiment 2 — Korrelations-Falle** (Tabelle über $\rho$):
- $\rho=0$: Fusion halbiert den Fehler (~0.16 → **0.08**); $P(\text{beide falsch})=$ **0.025** $\approx \varepsilon_A\varepsilon_B=0.026$ (Unabhängigkeits-Signatur trifft).
- $\rho=0.98$: Fusionsgewinn ~**0.002** (praktisch null); $P(\text{beide falsch})$ steigt auf **0.14** $\approx$ Einzelfehler — die Modalitäten machen *denselben* Fehler.

> **Die Lehre.** Fusion ist **kein Gratis-Gewinn**. Sie lebt von der **Diversität der Fehler** — genau wie Ensembles (Modul 04/05). Wer Fusion einbaut, muss die **Fehlerkorrelation messen**, nicht Unabhängigkeit annehmen.

## Lösung

Vollständige, lauffähige Referenz in [`solution/`](solution/) (identische `run.py`/`test_multimodal.py`, plus ausgefüllte `multimodal.py`). Erst selbst versuchen!

## Weiter geht's

**P03 (final)**: Der große Bogen — ein vollständiger **„Put-that-there"-Interpreter** mit asynchronen Sprach- und Gesten-Strömen, zeitgewichteter Referenzauflösung und quantifizierter Mutual Disambiguation. Keine Code-Vorgabe.
