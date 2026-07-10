# Projekt 03 (final) — EDA: Wer fährt wann Fahrrad, und warum?

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

1. Daten holen: `python daten/download_daten.py` (CSV-Dateien werden nicht committet).
2. Notebook durcharbeiten. Die Leitfragen sind vorgegeben, den Code schreibst du **vollständig selbst** (Hinweise stehen in Prosa in den Aufgaben; die Code-Zellen sind leer). **Jede Grafik bekommt einen Interpretationssatz** — die Musterinterpretationen stehen als Vergleich unter den Zellen.
3. Zum Schluss das Fazit an die Betriebsleitung formulieren (Musterfazit ausklappbar).

## Was am Ende funktionieren soll

- Qualitätscheck reproduziert: 0 NaN, 0 Duplikate, `cnt`-Konsistenz, **165 fehlende Stunden**.
- Tagesprofil-Plot mit Doppelspitze (Arbeitstage) vs. Buckel (frei).
- r(Temp, cnt) ≈ **0,405** stündlich und ≈ **0,627** täglich.
- Ein eigenes 5-Punkte-Fazit.

## Lösung

Vollständig ausgeführte Musterlösung mit allen Plots und Interpretationen: [`loesung/loesung.ipynb`](loesung/loesung.ipynb).
