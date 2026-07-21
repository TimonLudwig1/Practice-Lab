# Project 02 (medium) — Rescuing a dirty data set

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook and the generator are English only; the German block below still names the original German columns where it describes the old state.

**Format:** Jupyter notebook (`data_cleaning.ipynb`) + data generator (`generate_data.py`).
**Why this format?** Cleaning is iterative inspecting and checking with a constant eye on intermediate results — notebook territory. The generator is deliberately a separate, readable script: you should be able to see how the data come about.

**Data: synthetic — on purpose.** 500 orders of a fictional online shop, into which `generate_data.py` (fixed seed 42, reproducible) builds **eight documented, realistic problems**: price as text ("49.99 EUR"), missing values, decimal-point outliers (x100), the special code −999, an impossible age, inconsistent city names, two date formats, exact duplicates. Synthetic because here there is a **verifiable truth**: at the end you can read in the generator whether you really found everything — no real data set can offer that.

## Goal

Find all the data problems yourself, treat them with justification (every decision is documented in one sentence) and prove with a given **acceptance test** (assert cell) that the data set is clean. A core lesson along the way: the IQR outlier rule fails globally (41 false hits) and only works **per category** (exactly 3 genuine hits).

## Prior knowledge

- Section 2.1 of the module script (cleaning — read it first!), 1.4, 2.3
- Project 01-basic (basic pandas patterns)

## Tasks

1. Generate the data: `python generate_data.py` (creates `datasets/orders_raw.csv` — the CSV is not committed, only the generator).
2. Work through the notebook. The route: inspect and write a list of problems → duplicates → price type conversion → outliers (global vs. per group!) → cities → age → date → acceptance test → mini analysis.
3. **Search yourself first, then read `generate_data.py`** — the generator is the answer key.

## What should work in the end

The acceptance cell runs through without an `AssertionError` and reports "ACCEPTANCE PASSED":
500 unique rows, price as float (12 NaN, median about 71.6 EUR, maximum below 500 EUR), 5 normalised cities, age at most 100 (26 NaN), date as `datetime64` without gaps.

## Solution

Fully executed reference solution including all justifications for the decisions: [`solution/solution.ipynb`](solution/solution.ipynb).

---
---

# Projekt 02 (medium) — Einen schmutzigen Datensatz retten (deutsche Fassung)

**Format:** Jupyter Notebook (`data_cleaning.ipynb`) + Daten-Generator (`generate_data.py`).
**Warum dieses Format?** Bereinigung ist iteratives Sichten und Prüfen mit ständigem Blick auf Zwischenergebnisse — Notebook-Territorium. Der Generator ist bewusst ein separates, lesbares Skript: Du sollst sehen (können), wie die Daten entstehen.

**Daten: synthetisch — mit Absicht.** 500 Bestellungen eines fiktiven Online-Shops, in die `generate_data.py` (fester Seed 42, reproduzierbar) **acht dokumentierte, realistische Probleme** einbaut: Preis als Text („49,99 EUR"), fehlende Werte, Kommafehler-Ausreißer (×100), Sondercode −999, unmögliches Alter, inkonsistente Städtenamen, zwei Datumsformate, exakte Duplikate. Synthetisch deshalb, weil es hier eine **überprüfbare Wahrheit** gibt: Am Ende kannst du im Generator nachlesen, ob du wirklich alles gefunden hast — das kann kein echter Datensatz bieten.

## Ziel

Alle Datenprobleme selbst finden, begründet behandeln (jede Entscheidung wird in einem Satz dokumentiert) und mit einem vorgegebenen **Abnahmetest** (assert-Zelle) nachweisen, dass der Datensatz sauber ist. Kernlektion nebenbei: Die IQR-Ausreißerregel schlägt global fehl (41 falsche Treffer) und funktioniert erst **pro Kategorie** (genau 3 echte Treffer).

## Vorwissen

- Modul-Skript Abschnitt 2.1 (Bereinigung — vorher lesen!), 1.4, 2.3
- Projekt 01-basic (pandas-Grundmuster)

## Aufgaben

1. Daten erzeugen: `python generate_data.py` (legt `datasets/bestellungen_roh.csv` an — die CSV wird nicht committet, nur der Generator).
2. Notebook durcharbeiten. Ablauf: sichten & Problemliste schreiben → Duplikate → Preis-Typkonvertierung → Ausreißer (global vs. pro Gruppe!) → Städte → Alter → Datum → Abnahmetest → Mini-Analyse.
3. **Erst selbst suchen, dann `generate_data.py` lesen** — der Generator ist die Auflösung.

## Was am Ende funktionieren soll

Die Abnahmezelle läuft ohne `AssertionError` durch und meldet „ABNAHME BESTANDEN":
500 eindeutige Zeilen, Preis als float (12 NaN, Median ≈ 71,6 €, Maximum < 500 €), 5 normalisierte Städte, Alter ≤ 100 (26 NaN), Datum als `datetime64` ohne Lücken.

## Lösung

Vollständig ausgeführte Musterlösung inkl. aller Entscheidungs-Begründungen: [`solution/solution.ipynb`](solution/solution.ipynb).
