# Projekt 02 (medium) — Einen schmutzigen Datensatz retten

**Format:** Jupyter Notebook (`datenbereinigung.ipynb`) + Daten-Generator (`generate_data.py`).
**Warum dieses Format?** Bereinigung ist iteratives Sichten und Prüfen mit ständigem Blick auf Zwischenergebnisse — Notebook-Territorium. Der Generator ist bewusst ein separates, lesbares Skript: Du sollst sehen (können), wie die Daten entstehen.

**Daten: synthetisch — mit Absicht.** 500 Bestellungen eines fiktiven Online-Shops, in die `generate_data.py` (fester Seed 42, reproduzierbar) **acht dokumentierte, realistische Probleme** einbaut: Preis als Text („49,99 EUR"), fehlende Werte, Kommafehler-Ausreißer (×100), Sondercode −999, unmögliches Alter, inkonsistente Städtenamen, zwei Datumsformate, exakte Duplikate. Synthetisch deshalb, weil es hier eine **überprüfbare Wahrheit** gibt: Am Ende kannst du im Generator nachlesen, ob du wirklich alles gefunden hast — das kann kein echter Datensatz bieten.

## Ziel

Alle Datenprobleme selbst finden, begründet behandeln (jede Entscheidung wird in einem Satz dokumentiert) und mit einem vorgegebenen **Abnahmetest** (assert-Zelle) nachweisen, dass der Datensatz sauber ist. Kernlektion nebenbei: Die IQR-Ausreißerregel schlägt global fehl (41 falsche Treffer) und funktioniert erst **pro Kategorie** (genau 3 echte Treffer).

## Vorwissen

- Modul-Skript Abschnitt 2.1 (Bereinigung — vorher lesen!), 1.4, 2.3
- Projekt 01-basic (pandas-Grundmuster)

## Aufgaben

1. Daten erzeugen: `python generate_data.py` (legt `daten/bestellungen_roh.csv` an — die CSV wird nicht committet, nur der Generator).
2. Notebook durcharbeiten. Ablauf: sichten & Problemliste schreiben → Duplikate → Preis-Typkonvertierung → Ausreißer (global vs. pro Gruppe!) → Städte → Alter → Datum → Abnahmetest → Mini-Analyse.
3. **Erst selbst suchen, dann `generate_data.py` lesen** — der Generator ist die Auflösung.

## Was am Ende funktionieren soll

Die Abnahmezelle läuft ohne `AssertionError` durch und meldet „ABNAHME BESTANDEN":
500 eindeutige Zeilen, Preis als float (12 NaN, Median ≈ 71,6 €, Maximum < 500 €), 5 normalisierte Städte, Alter ≤ 100 (26 NaN), Datum als `datetime64` ohne Lücken.

## Lösung

Vollständig ausgeführte Musterlösung inkl. aller Entscheidungs-Begründungen: [`loesung/loesung.ipynb`](loesung/loesung.ipynb).
