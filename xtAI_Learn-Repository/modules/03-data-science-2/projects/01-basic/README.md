# Projekt 01 (basic) — SQL: Daten holen, wo sie wohnen

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
