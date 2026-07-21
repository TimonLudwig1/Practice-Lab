# Project 03 (final) — EDA: who cycles when, and why?

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format:** Jupyter notebook (`bikesharing_eda.ipynb`) + download script.
**Why this format?** An EDA *is* a notebook: question → code → figure → sentence of interpretation, in one document that you can hand to someone at the end.

**Data: real.** The **bike sharing** data set (UCI ML Repository): 17,379 real hourly records of the Capital Bikeshare system in Washington D.C. 2011–2012, including weather and calendar attributes. Real data are the core of the practical connection here — the data set has exactly the quirks on which you really learn EDA: normalised columns that you only understand from the documentation (`temp*41` = degrees Celsius!), 165 hidden missing hours and a weather category with only n = 3.

## Goal

A complete EDA following the workflow from script 2.5, with a business scenario: management wants to know what the demand depends on. Besides the analyses, the result is a **five-point conclusion in everyday language**.

Core findings that you will work out (yourself):

- the commuter double peak (8 a.m. / 5–6 p.m.) on working days vs. the afternoon hump at the weekend,
- registered users = commuters, casual users = leisure,
- a temperature effect of r about 0.41 at the hourly level, but 0.63 at the daily level — the aggregation lesson,
- "no NaN" is not "nothing missing": 165 complete hours are missing as rows.

## Prior knowledge

- The module script in full (especially 2.4, 2.5, 3.2)
- Projects 01 and 02 of this module

## Tasks

1. Fetch the data: `python datasets/download_data.py` (the CSV files are not committed).
2. Work through the notebook. The guiding questions are given, but you write the code **entirely yourself** (hints are given as prose in the tasks; the code cells are empty). **Every figure gets a sentence of interpretation** — the reference interpretations are given below the cells for comparison.
3. Finally, formulate the conclusion for management (reference conclusion can be unfolded).

## What should work in the end

- The quality check reproduced: 0 NaN, 0 duplicates, `cnt` consistency, **165 missing hours**.
- A daily-profile plot with a double peak (working days) vs. a hump (days off).
- r(temp, cnt) about **0.405** hourly and about **0.627** daily.
- A five-point conclusion of your own.

## Solution

Fully executed reference solution with all plots and interpretations: [`solution/solution.ipynb`](solution/solution.ipynb).

---
---

# Projekt 03 (final) — EDA: Wer fährt wann Fahrrad, und warum? (deutsche Fassung)

**Format:** Jupyter Notebook (`bikesharing_eda.ipynb`) + Download-Skript.
**Warum dieses Format?** Eine EDA *ist* ein Notebook: Frage → Code → Grafik → Interpretationssatz, in einem Dokument, das man am Ende jemandem geben kann.

**Daten: echt.** Der **Bike-Sharing-Datensatz** (UCI ML Repository): 17.379 echte Stundenwerte des Capital-Bikeshare-Systems Washington D.C. 2011–2012, inklusive Wetter- und Kalendermerkmalen. Echte Daten sind hier der Kern des Praxisbezugs — der Datensatz hat genau die Eigenheiten, an denen man EDA wirklich lernt: normierte Spalten, die man nur per Dokumentation versteht (`temp*41` = °C!), 165 versteckt fehlende Stunden und eine Wetterkategorie mit nur n = 3.

## Ziel

Eine vollständige EDA nach dem Workflow aus Skript 2.5, mit einem Business-Szenario: Die Betriebsleitung will wissen, wovon die Nachfrage abhängt. Ergebnis ist neben den Analysen ein **5-Punkte-Fazit in Alltagssprache**.

Kernbefunde, die du (selbst) herausarbeiten wirst:

- Pendler-Doppelspitze (8 / 17–18 Uhr) an Arbeitstagen vs. Nachmittagsbuckel am Wochenende,
- registrierte Nutzer = Pendler, Gelegenheitsnutzer = Freizeit,
- Temperatureffekt r ≈ 0,41 auf Stunden-, aber 0,63 auf Tagesebene — die Aggregations-Lektion,
- „keine NaN" ≠ „nichts fehlt": 165 komplette Stunden fehlen als Zeilen.

## Vorwissen

- Modul-Skript komplett (v. a. 2.4, 2.5, 3.2)
- Projekte 01 und 02 dieses Moduls

## Aufgaben

1. Daten holen: `python datasets/download_data.py` (CSV-Dateien werden nicht committet).
2. Notebook durcharbeiten. Die Leitfragen sind vorgegeben, den Code schreibst du **vollständig selbst** (Hinweise stehen in Prosa in den Aufgaben; die Code-Zellen sind leer). **Jede Grafik bekommt einen Interpretationssatz** — die Musterinterpretationen stehen als Vergleich unter den Zellen.
3. Zum Schluss das Fazit an die Betriebsleitung formulieren (Musterfazit ausklappbar).

## Was am Ende funktionieren soll

- Qualitätscheck reproduziert: 0 NaN, 0 Duplikate, `cnt`-Konsistenz, **165 fehlende Stunden**.
- Tagesprofil-Plot mit Doppelspitze (Arbeitstage) vs. Buckel (frei).
- r(Temp, cnt) ≈ **0,405** stündlich und ≈ **0,627** täglich.
- Ein eigenes 5-Punkte-Fazit.

## Lösung

Vollständig ausgeführte Musterlösung mit allen Plots und Interpretationen: [`solution/solution.ipynb`](solution/solution.ipynb).
