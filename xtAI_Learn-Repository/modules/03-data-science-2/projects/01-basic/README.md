# Project 01 (basic) — SQL: fetching data where they live

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook and the generator are English only; the customer names in the data stay German on purpose (it is a German shop).

**Format:** Jupyter notebook (`sql_basics.ipynb`) + database generator (`generate_db.py`).
**Why this format?** You learn SQL by writing a query and looking at the result, second by second — with `pd.read_sql` in a notebook that is exactly what happens, and the transition SQL → pandas (the workflow in practice) becomes directly visible.

**Data: synthetic — on purpose.** A small SQLite database (60 customers, 25 products, 800 orders, fixed seed 7): for SQL basics you need a clean, manageable schema with foreign keys and **known answers** for the mini checks. Deliberately built in are 5 customers without orders (the LEFT JOIN lesson) and a December peak (the GROUP BY lesson).

## Goal

Apply the basic SQL building blocks confidently: `SELECT`/`WHERE`/`ORDER BY`/`LIMIT`, aggregation with `GROUP BY`, `JOIN` across two and three tables, `LEFT JOIN` for "who is missing?", `WHERE` vs. `HAVING` — and the division of labour between SQL and pandas.

## Prior knowledge

- Section 2.4 of the module script (SQL) — read it first
- Module 02 (pandas: merge/groupby as the mental counterpart)
- Recommended: 15–30 minutes of https://sqlbolt.com as a warm-up

## Tasks

1. Create the database: `python generate_db.py` (the `.db` file is not committed, only the generator).
2. Work through the notebook; write the SQL queries in the `TODO` cells yourself. Mini checks must evaluate to `True`.
3. In section 6, deliberately produce the `WHERE COUNT(*)` error and read the error message.

## What should work in the end

- All mini checks `True`: 6 products above 50 EUR, Electronics as the category with the highest revenue, "Emma Schulz" as the top customer, exactly 5 customers without an order, 4 cities with at least 150 orders.
- The monthly plot shows the December peak, the pivot table shows revenue by city and category.

## Solution

Fully executed reference solution: [`solution/solution.ipynb`](solution/solution.ipynb).

---
---

# Projekt 01 (basic) — SQL: Daten holen, wo sie wohnen (deutsche Fassung)

**Format:** Jupyter Notebook (`sql_basics.ipynb`) + DB-Generator (`generate_db.py`).
**Warum dieses Format?** SQL lernt man durch Abfrage-schreiben-Ergebnis-anschauen im Sekundentakt — mit `pd.read_sql` im Notebook ist genau das möglich, und der Übergang SQL → pandas (der Praxis-Workflow) wird direkt sichtbar.

**Daten: synthetisch — mit Absicht.** Eine kleine SQLite-Datenbank (60 Kunden, 25 Produkte, 800 Bestellungen, fester Seed 7): Für SQL-Grundlagen braucht es ein sauberes, überschaubares Schema mit Fremdschlüsseln und **bekannten Antworten** für die Mini-Checks. Eingebaut sind gezielt 5 Kunden ohne Bestellungen (LEFT-JOIN-Lektion) und ein Dezember-Peak (GROUP-BY-Lektion).

## Ziel

Die SQL-Grundbausteine sicher anwenden: `SELECT`/`WHERE`/`ORDER BY`/`LIMIT`, Aggregation mit `GROUP BY`, `JOIN` über zwei und drei Tabellen, `LEFT JOIN` für „wer fehlt?", `WHERE` vs. `HAVING` — und die Arbeitsteilung SQL ↔ pandas.

## Vorwissen

- Modul-Skript Abschnitt 2.4 (SQL) — vorher lesen
- Modul 02 (pandas: merge/groupby als mentales Gegenstück)
- Empfohlen: 15–30 Minuten https://sqlbolt.com als Warm-up

## Aufgaben

1. Datenbank erzeugen: `python generate_db.py` (die `.db`-Datei wird nicht committet, nur der Generator).
2. Notebook durcharbeiten; die SQL-Abfragen in den `TODO`-Zellen selbst schreiben. Mini-Checks müssen `True` ergeben.
3. In Abschnitt 6 absichtlich den `WHERE COUNT(*)`-Fehler produzieren und die Fehlermeldung lesen.

## Was am Ende funktionieren soll

- Alle Mini-Checks `True`: 6 Produkte über 50 €, Elektronik als umsatzstärkste Kategorie, „Emma Schulz" als Top-Kundin, genau 5 Kunden ohne Bestellung, 4 Städte mit ≥ 150 Bestellungen.
- Der Monats-Plot zeigt den Dezember-Peak, die Pivot-Tabelle Umsatz je Stadt × Kategorie.

## Lösung

Vollständig ausgeführte Musterlösung: [`solution/solution.ipynb`](solution/solution.ipynb).
