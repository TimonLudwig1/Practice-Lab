# P02 (medium) — Early vs. Late Fusion, Mutual Disambiguation & die Korrelations-Falle

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
