# Project 03 (final) — Explaining and predicting demand (regression and time series)

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format:** Jupyter notebook (`regression_forecasting.ipynb`) + download script.
**Why this format?** Model diagnostics live on plots right next to the code (residuals, forecast vs. reality) — and the project tells one continuous analytical story that should be readable as a report.

**Data: real.** The same bike sharing data (UCI, Capital Bikeshare Washington D.C.) as in the final project of module 02, this time at the **daily level** (`day.csv`, 731 days). The deliberate connection to the familiar scenario is part of the design: "describing" (module 02) becomes "explaining and predicting" (module 03). A bonus of the real data: the test period contains **Hurricane Sandy** — the perfect lesson about the limits of a model.

## Goal

1. **Explain:** simple → quadratic → multiple regression on daily demand; interpret coefficients ceteris paribus; find the non-linearity via the residual plot (comfort temperature about 29 degrees C).
2. **Predict:** build lag features, split temporally, compete against naive/seasonal baselines (model MAE about 859 vs. 952/1458).
3. **Leakage demo:** a random split apparently delivers MAE about 625 — and you can explain why that is a lie.

## Prior knowledge

- Sections 1.1–1.2, 2.1, 2.2 of the module script (mandatory)
- Module 02 in full (especially the bike sharing final project)
- scikit-learn is used here for the first time; the necessary API (`fit`/`predict`/`score`) is described in the task hints of the notebook (you write the code yourself)

## Tasks

1. Fetch the data: `python datasets/download_data.py` (or copy `day.csv` from module 02).
2. Work through the notebook: simple regression → residual plot → quadratic term → multiple regression (interpret every coefficient in one sentence!) → lag features + temporal split + baselines → leakage demo → conclusion for management.
3. Look at the three worst forecast days and explain them (Sandy!).

## What should work in the end

- $R^2$: 0.394 (linear) → 0.453 (quadratic) → 0.794 (multiple); the vertex at 25–31 degrees C.
- MAE ranking: model (about 859) < naive (about 952) < seasonal naive (about 1458); your self-checks (the value ranges named in the tasks) hold.
- A random-split MAE of about 625 — with your explanation of why that value does not count.
- A four- to five-point conclusion in everyday language.

## Solution

Fully executed reference solution: [`solution/solution.ipynb`](solution/solution.ipynb).

---
---

# Projekt 03 (final) — Nachfrage erklären und vorhersagen (Regression & Zeitreihen) (deutsche Fassung)

**Format:** Jupyter Notebook (`regression_forecasting.ipynb`) + Download-Skript.
**Warum dieses Format?** Modelldiagnose lebt von Plots direkt am Code (Residuen, Prognose vs. Realität) — und das Projekt erzählt eine durchgehende Analyse-Geschichte, die man als Bericht lesen können soll.

**Daten: echt.** Dieselben Bike-Sharing-Daten (UCI, Capital Bikeshare Washington D.C.) wie im Final-Projekt von Modul 02, diesmal auf **Tagesebene** (`day.csv`, 731 Tage). Der bewusste Anschluss ans bekannte Szenario ist Teil des Designs: Aus „beschreiben" (Modul 02) wird „erklären und vorhersagen" (Modul 03). Bonus der echten Daten: Der Testzeitraum enthält **Hurrikan Sandy** — die perfekte Lektion über Modellgrenzen.

## Ziel

1. **Erklären:** Einfache → quadratische → multiple Regression auf die Tagesnachfrage; Koeffizienten ceteris paribus interpretieren; per Residuenplot die Nichtlinearität finden (Wohlfühltemperatur ≈ 29 °C).
2. **Vorhersagen:** Lag-Features bauen, zeitlich splitten, gegen naive/saisonale Baselines antreten (Modell-MAE ≈ 859 vs. 952/1458).
3. **Leakage-Demo:** Zufalls-Split liefert scheinbar MAE ≈ 625 — und du kannst erklären, warum das gelogen ist.

## Vorwissen

- Modul-Skript Abschnitte 1.1–1.2, 2.1, 2.2 (Pflicht)
- Modul 02 komplett (v. a. das Bike-Sharing-Final-Projekt)
- scikit-learn wird hier zum ersten Mal benutzt; die nötige API (`fit`/`predict`/`score`) ist in den Aufgaben-Hinweisen des Notebooks beschrieben (Code schreibst du selbst)

## Aufgaben

1. Daten holen: `python datasets/download_data.py` (oder `day.csv` aus Modul 02 kopieren).
2. Notebook durcharbeiten: einfache Regression → Residuenplot → Quadratterm → multiple Regression (jeden Koeffizienten in einem Satz interpretieren!) → Lag-Features + zeitlicher Split + Baselines → Leakage-Demo → Fazit an die Betriebsleitung.
3. Die drei schlechtesten Prognosetage anschauen und erklären (Sandy!).

## Was am Ende funktionieren soll

- $R^2$: 0,394 (linear) → 0,453 (quadratisch) → 0,794 (multiple); Scheitel bei 25–31 °C.
- MAE-Rangfolge: Modell (≈859) < naiv (≈952) < saisonal-naiv (≈1458); deine Selbstchecks (in den Aufgaben genannte Wertebereiche) treffen zu.
- Zufalls-Split-MAE ≈ 625 — mit deiner Erklärung, warum dieser Wert nicht zählt.
- Ein 4–5-Punkte-Fazit in Alltagssprache.

## Lösung

Vollständig ausgeführte Musterlösung: [`solution/solution.ipynb`](solution/solution.ipynb).
