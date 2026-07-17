# Projekt 03-final: Job-Queue-Simulation

## Ziel

Eine Analyseplattform besitzt einen Worker, aber viele eingehende Jobs. Dieses
Projekt untersucht, wie sich die Queue Policy auf Wartezeiten auswirkt. Derselbe
synthetische Ankunftsstrom wird einmal strikt nach FIFO und einmal mit einer
stabilen Prioritätsregel verarbeitet. Anschließend werden Gesamtverteilung und
die drei Klassen `urgent`, `standard` und `batch` getrennt ausgewertet.

Das Projekt verwendet Python-Skripte, CSV und eine PNG-Visualisierung. Skripte
sind hier geeigneter als ein Notebook, weil Datengenerierung, Simulation und
Reporting als reproduzierbare Pipeline ausführbar und automatisch testbar
bleiben. Matplotlib dient nur der Darstellung; die Queue-Simulation selbst nutzt
ausschließlich die Standardbibliothek.

## Aufgabenstellung

1. Erzeuge in `data/generate_data.py` einen reproduzierbaren Job-Strom. Jeder Job
   besitzt ID, Ankunftszeit, Bearbeitungsdauer, Priorität und Typ.
2. Implementiere eine ereignisgetriebene Simulation mit genau einem Worker. Die
   Uhr springt direkt zur nächsten Ankunft oder Fertigstellung; unnötige
   Zeitschritte werden nicht simuliert.
3. Vergleiche zwei nicht-präemptive Regeln:
   - **FIFO:** Der am längsten wartende Job startet zuerst.
   - **Priority:** Die kleinste Prioritätszahl startet zuerst; bei Gleichstand
     bleibt die Ankunftsreihenfolge erhalten.
4. Ermittle Mittelwert, Median, P95 und Maximum der Wartezeit sowie Turnaround,
   Durchsatz und Auslastung. Werte Prioritätsklassen zusätzlich getrennt aus.
5. Exportiere Detaildaten, Zusammenfassung, Ergebnisbericht und Vergleichsplot.

Eine vollständige Referenzimplementierung ist enthalten. Der Lernweg wird
klarer, wenn zunächst an einem kleinen Job-Strom Start- und Endzeiten von Hand
berechnet und erst dann die Simulation implementiert wird.

## Simulation an einem kleinen Beispiel

| Job | Ankunft | Dauer | Priorität |
|---|---:|---:|---:|
| A | 0 | 4 | 2 |
| B | 1 | 3 | 3 |
| C | 1 | 1 | 1 |

Job A beginnt bei Zeit 0. B und C treffen während seiner Bearbeitung ein. Da die
Simulation nicht-präemptiv ist, läuft A in beiden Strategien bis Zeit 4.

- FIFO verarbeitet anschließend B und C. Die Wartezeiten sind `0, 3, 6`.
- Priority verarbeitet anschließend C und B. Die Wartezeiten sind `0, 3, 4`.

Dieses Beispiel zeigt zugleich eine wichtige Grenze: Käme C erst bei Zeit 5,
könnte C einen bereits gestarteten Job B nicht mehr unterbrechen.

## Invarianten und Modellgrenzen

Während der Simulation befindet sich jeder Job in genau einem Zustand: noch
nicht angekommen, wartend, in Bearbeitung oder abgeschlossen. Die simulierte Uhr
läuft niemals rückwärts. Vor jeder Auswahl werden alle Jobs mit
`arrival_time <= current_time` in die Waiting Queue aufgenommen.

FIFO nutzt `collections.deque` mit O(1) für Einfügen und Entnehmen. Die Priority
Queue verwendet einen Heap mit O(log n) pro Einfügen und Entnehmen. Ein laufender
Job wird nicht verdrängt. Mehrere Worker, Abbrüche, Deadlines und Aging sind
bewusst nicht Teil des Modells.

Die stabile Prioritätsregel senkt typischerweise die Wartezeit dringender Jobs,
kann aber Batch-Jobs benachteiligen. Deshalb reichen Gesamtmittelwerte nicht:
P95, Maximum und getrennte Klassenmetriken machen Fairness und mögliches
Starvation-Risiko sichtbar.

## Projektstruktur

```text
03-final/
├── data/
│   ├── generate_data.py
│   └── jobs.csv
├── output/
│   ├── fifo_results.csv
│   ├── priority_results.csv
│   ├── summary.csv
│   ├── REPORT.md
│   └── wait_time_comparison.png
├── simulation.py
├── run_analysis.py
└── test_simulation.py
```

## Ausführen

Im Projektordner:

```bash
python data/generate_data.py
python run_analysis.py
python -m pytest -q
```

`run_analysis.py` erzeugt den Standarddatensatz automatisch, falls er fehlt. Der
Seed `20260717` bleibt absichtlich konstant, damit beide Strategien exakt dieselbe
Last sehen und alle Ergebniswerte reproduzierbar sind.

## Hinweise

- Sortiere die Eingabeliste nicht in-place; die beiden Strategien müssen
  unabhängig mit identischen Jobs laufen können.
- Bei einer Priority Queue braucht der Heap neben der Priorität eine stabile
  Sequenznummer. Sonst müsste er bei Gleichstand `Job`-Objekte vergleichen.
- Wartezeit ist `start - arrival`, Turnaround ist `finish - arrival`.
- Ein geringer Mittelwert kann einzelne sehr lange Wartezeiten verdecken.
- Eine Prioritätsregel erzeugt keine zusätzliche Rechenkapazität. Sie verteilt
  die vorhandene Wartezeit lediglich anders zwischen Job-Klassen.

## Fertig, wenn …

- derselbe per Seed erzeugte Job-Strom beide Strategien durchläuft,
- FIFO und stabile Priority Queue an Handbeispielen die erwartete Reihenfolge
  liefern,
- Leerlaufphasen, gleiche Ankunftszeiten und nicht-präemptive Verarbeitung
  korrekt behandelt werden,
- Detail-CSV und segmentierte Kennzahlen für beide Strategien entstehen,
- der Ergebnisbericht Fairness statt nur Gesamtmittelwerte diskutiert,
- die PNG-Grafik Verteilungen und mittlere Wartezeiten je Klasse vergleicht,
- die vollständige Pipeline ohne Fehler läuft und
- alle Tests mit `python -m pytest -q` erfolgreich sind.
