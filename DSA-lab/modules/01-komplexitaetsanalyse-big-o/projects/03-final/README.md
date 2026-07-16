# 03-final — Performance-Audit einer Datenpipeline

## Szenario

Ein E-Commerce-Team aggregiert Ereignisdaten zu einer Kundenübersicht. Die
bestehende Pipeline liefert fachlich korrekte Resultate, wird bei größeren
CSV-Dateien jedoch schnell langsam. Deine Aufgabe ist ein vollständiges
Performance-Audit:

1. Engpässe im bestehenden Code lokalisieren,
2. deren Zeit- und Speicherkomplexität herleiten,
3. eine effizientere Implementierung entwerfen,
4. Gleichheit der Resultate beweisen,
5. Vorher und Nachher über wachsende Datenmengen messen,
6. den Speedup nachvollziehbar dokumentieren.

Der Datensatz ist synthetisch und wird mit einem festen Seed erzeugt. Es werden
keine externen Daten heruntergeladen.

## Warum Python-Skripte?

Dieses Projekt simuliert eine kleine produktive Datenpipeline. Getrennte Skripte
für Datengenerierung, Pipeline-Logik, Audit und Tests machen Datenfluss,
Verantwortlichkeiten und Messgrenzen explizit. Ein Notebook würde die
Ausführungsreihenfolge und den Unterschied zwischen Setup und gemessenem
Pipeline-Code leichter verwischen.

## Projektstruktur

~~~text
03-final/
├── README.md
├── LOESUNG.md
├── audit_pipeline.py
├── run_audit.py
├── test_audit_pipeline.py
├── requirements.txt
├── data/
│   ├── __init__.py
│   ├── generate_data.py
│   └── events.csv                  # generiert, nicht versioniert
└── results/                        # generiert, nicht versioniert
    ├── customer_summary.csv
    ├── performance_audit.csv
    ├── performance_audit.png
    └── AUDIT_REPORT.md
~~~

## Datenmodell

Jede CSV-Zeile beschreibt ein Ereignis:

| Feld | Bedeutung |
|---|---|
| event_id | eindeutige Ereignis-ID |
| customer_id | Kunden-ID |
| category | Produktkategorie |
| amount_cents | Bruttobetrag in Cent |
| discount_cents | Rabatt in Cent |
| status | completed, pending oder cancelled |
| event_timestamp | ISO-Zeitstempel |

Die Pipeline berücksichtigt nur **completed**-Ereignisse und erzeugt pro Kunde:

- Anzahl abgeschlossener Ereignisse,
- Nettoumsatz in Cent,
- ganzzahliger durchschnittlicher Nettobetrag,
- Anzahl unterschiedlicher Kategorien,
- Zeitstempel des letzten abgeschlossenen Ereignisses.

Ganzzahlige Cent-Beträge verhindern Rundungsdifferenzen zwischen
Implementierungen.

## Vorbereitung

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
~~~

## Phase 1: Baseline analysieren

Öffne in **audit_pipeline.py** zunächst nur **inefficient_pipeline**. Lies die
optimierte Funktion noch nicht.

Dokumentiere:

1. Welche Größe beschreibt \(n\)?
2. Welche zweite Größe \(u\) ist für die Anzahl unterschiedlicher Kunden
   sinnvoll?
3. Wie teuer ist **customer_id not in customer_ids** bei einer Liste?
4. Wie oft wird die vollständige Ereignisliste erneut gescannt?
5. Welche Worst-Case-Laufzeit folgt in \(n\) und \(u\)?
6. Welche temporären Container werden angelegt?
7. Welche Arbeit ist fachlich notwendig, welche entsteht nur durch den
   gewählten Ablauf?

Formuliere vor der Messung eine Hypothese:

| Pipeline | erwartete Zeit | erwarteter Auxiliary Space |
|---|---|---|
| Baseline |  |  |
| Ziel für Refactoring |  |  |

## Phase 2: Refactoring entwerfen

Entwirf eine Pipeline, die jedes Ereignis möglichst nur einmal besucht. Überlege
dabei:

- Welche Struktur erlaubt direkten Zugriff auf den Akkumulator eines Kunden?
- Welche Kennzahlen können inkrementell aktualisiert werden?
- Welche Struktur eignet sich für eindeutige Kategorien?
- Welche Kosten bleiben für eine deterministisch sortierte Ausgabe?

Vergleiche deinen Entwurf danach mit **optimized_pipeline**. Beide Funktionen
müssen exakt dieselbe Liste unveränderlicher **CustomerSummary**-Objekte
zurückgeben.

## Phase 3: Tests

~~~bash
python3 -m unittest -v test_audit_pipeline.py
~~~

Die Tests prüfen unter anderem:

- reproduzierbare Datengenerierung mit festem Seed,
- korrektes CSV-Laden,
- identische Resultate beider Pipelines auf konstruierten und generierten Daten,
- Filterung nicht abgeschlossener Ereignisse,
- Benchmark-Validierung,
- CSV- und Markdown-Berichtsexport.

## Phase 4: Vorher/Nachher messen

~~~bash
python3 run_audit.py
~~~

Der Standardlauf erzeugt 8.000 Ereignisse mit Seed 20260716 und misst beide
Pipelines für fünf wachsende Präfixe. Datengenerierung und CSV-Laden liegen
bewusst außerhalb der Zeitmessung. Vor jedem Messpunkt wird die vollständige
Ergebnisgleichheit geprüft.

Der Lauf erzeugt:

- Rohzeiten und Speedups in **results/performance_audit.csv**,
- eine Kundenübersicht aus der optimierten Pipeline,
- einen Plot mit Laufzeitkurven und Speedup,
- **results/AUDIT_REPORT.md** mit Konfiguration, Ergebnistabelle und
  Interpretationsleitfaden.

Ein schneller Probelauf:

~~~bash
python3 run_audit.py \
  --rows 1000 \
  --sizes 250 500 1000 \
  --repeats 2
~~~

Eine vorhandene CSV statt einer Neugenerierung verwenden:

~~~bash
python3 run_audit.py --reuse-data
~~~

Alle Optionen:

~~~bash
python3 run_audit.py --help
~~~

## Phase 5: Audit-Bericht interpretieren

Ergänze den automatisch erzeugten Bericht um deine eigene Bewertung:

1. Wie verändern sich Baseline- und optimierte Laufzeit bei Verdopplung?
2. Wächst der Speedup mit \(n\)? Warum ist das zu erwarten?
3. Ab welcher getesteten Größe ist der Unterschied praktisch relevant?
4. Welche Optimierung bringt den größten asymptotischen Gewinn?
5. Welche Kosten bleiben in der optimierten Pipeline?
6. Weshalb ist Ergebnisgleichheit wichtiger als Speedup allein?
7. Welche Grenzen besitzt dieser synthetische Benchmark gegenüber einer realen
   produktiven Pipeline?

Nutze **LOESUNG.md** erst nach deiner eigenen Analyse.

## Messregeln

- Absolute Zeiten gelten nur für den ausführenden Rechner.
- Der Median mehrerer Wiederholungen reduziert kurze Störungen.
- Beide Varianten erhalten dieselben unveränderten Event-Objekte.
- CSV-Lesen und Datengenerierung werden nicht der Aggregationslogik
  zugerechnet.
- Der Speedup ist
  \[
  \text{Speedup} =
  \frac{\text{Baseline-Zeit}}{\text{optimierte Zeit}}.
  \]
- Ein schnelleres Ergebnis gilt nur dann als Erfolg, wenn alle fachlichen
  Resultate identisch bleiben.

## Fertig, wenn …

- du die Baseline in \(O(nu)\) und im Worst Case in \(O(n^2)\) hergeleitet hast,
- du die versteckten Kosten von Listen-Membership und wiederholten Vollscans
  erklären kannst,
- alle Unit Tests erfolgreich laufen,
- der Generator denselben Datensatz für denselben Seed erzeugt,
- beide Pipelines für alle Benchmark-Größen identische Ergebnisse liefern,
- CSV, Kundenübersicht, Plot und Audit-Bericht erzeugt werden,
- du Vorher/Nachher-Zeiten und Speedup über mehrere Größen dokumentiert hast,
- du begründet hast, warum die optimierte Aggregation plus Ausgabesortierung
  \(O(n+u\log u)\) benötigt,
- du die Ergebnisse und Grenzen des Audits in eigenen Worten festgehalten hast.
