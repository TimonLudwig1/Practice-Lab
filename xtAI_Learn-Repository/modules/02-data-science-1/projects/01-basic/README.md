# Projekt 01 (basic) — pandas-Grundlagen mit echten Pinguinen

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
