# Project 01 (basic) — pandas basics with real penguins

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format:** Jupyter notebook (`pandas_basics.ipynb`).
**Why this format?** You only learn pandas by trying things directly with an immediate look at the tables and plots — that is exactly what notebooks are for.

**Data: real.** The **Palmer Penguins** data set (344 penguins, 3 species, Antarctic research data 2007–2009) — directly available via `seaborn.load_dataset("penguins")` (downloaded automatically on first call and cached, internet required). Real data chosen deliberately: the data set has genuine missing values and genuine group structure, yet is small and manageable.

## Goal

Apply the core pandas tools confidently: first inspection (`head`/`info`/`describe`), counting missing values, boolean filters, vectorised new columns, `groupby` aggregation, the three basic plots (histogram, box plot, scatter) and a correlation — each with an interpretation.

## Prior knowledge

- Sections 1.3–1.5 and 2.3–2.4 of the module script
- Python basics; notebook handling from module 01

## Tasks

1. Open the notebook and work through it from the top; fill in the `TODO` cells yourself.
2. Every mini check below a task must print `True`.
3. Do not skip the questions below the plots — a figure without a statement is decoration (script 2.5).

## What should work in the end

- All mini checks `True` (19 missing values; 124 Gentoo; 61 penguins over 5 kg; Gentoo as the heaviest species; $r > 0.85$).
- You can explain the bimodal mass distribution in the histogram using the `groupby` result (Gentoo is markedly heavier than the other two species).

## Solution

Fully executed reference solution with all outputs and plots: [`solution/solution.ipynb`](solution/solution.ipynb).

---
---

# Projekt 01 (basic) — pandas-Grundlagen mit echten Pinguinen (deutsche Fassung)

**Format:** Jupyter Notebook (`pandas_basics.ipynb`).
**Warum dieses Format?** pandas lernt man nur durch direktes Ausprobieren mit sofortigem Blick auf die Tabellen und Plots — genau dafür sind Notebooks da.

**Daten: echt.** Der **Palmer-Penguins**-Datensatz (344 Pinguine, 3 Arten, Antarktis-Forschungsdaten 2007–2009) — über `seaborn.load_dataset("penguins")` direkt verfügbar (wird beim ersten Aufruf automatisch geladen und gecacht, Internet nötig). Echte Daten bewusst gewählt: Der Datensatz hat echte fehlende Werte und echte Gruppenstruktur, ist aber klein und überschaubar.

## Ziel

Die pandas-Kernwerkzeuge sicher anwenden: Erstinspektion (`head`/`info`/`describe`), fehlende Werte zählen, boolesche Filter, vektorisierte neue Spalten, `groupby`-Aggregation, die drei Grundplots (Histogramm, Boxplot, Scatter) und eine Korrelation — jeweils mit Interpretation.

## Vorwissen

- Modul-Skript Abschnitte 1.3–1.5 und 2.3–2.4
- Python-Basics; Notebook-Bedienung aus Modul 01

## Aufgaben

1. Notebook öffnen und von oben durcharbeiten; die `TODO`-Zellen selbst füllen.
2. Jeder Mini-Check unter einer Aufgabe muss `True` ausgeben.
3. Die Fragen unter den Plots nicht überspringen — eine Grafik ohne Aussage ist Deko (Skript 2.5).

## Was am Ende funktionieren soll

- Alle Mini-Checks `True` (19 fehlende Werte; 124 Gentoo; 61 Pinguine > 5 kg; Gentoo als schwerste Art; $r > 0{,}85$).
- Du kannst die bimodale Massenverteilung im Histogramm mit dem `groupby`-Ergebnis erklären (Gentoo ist deutlich schwerer als die anderen beiden Arten).

## Lösung

Vollständig ausgeführte Musterlösung mit allen Outputs und Plots: [`solution/solution.ipynb`](solution/solution.ipynb).
