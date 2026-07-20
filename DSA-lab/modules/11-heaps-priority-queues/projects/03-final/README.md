# Projekt 03-final: Task-Scheduler mit Prioritäten

## Ziel

Dieses Projekt überführt die Priority Queue in eine realistische
Single-Server-Simulation. 300 Jobs besitzen Ankunftszeit, Dauer und eine von drei
Prioritätsklassen. Zwei nicht-präemptive Scheduler verarbeiten exakt dieselben
Jobs:

- **FIFO** nimmt den ältesten bereiten Job aus einer Queue.
- **Priority** nimmt den wichtigsten bereiten Job aus einem Min-Heap.

Anschließend werden Warte- und Durchlaufzeiten nach Prioritätsklasse verglichen.
Das Projekt zeigt damit sowohl den Nutzen der Priority Queue als auch ihren
zentralen Trade-off: Dringende Arbeit wartet kürzer, niedrige Prioritäten können
stark benachteiligt werden.

## Dateien

- `../../data/generate_data.py`: reproduzierbarer Jobgenerator, Seed `20260720`
- `../../data/scheduler_jobs.csv`: 300 generierte Jobs
- `scheduler.py`: Simulation, Invarianten und Kennzahlen
- `run_simulation.py`: vollständiger Vergleich und Ausgabeerzeugung
- `test_scheduler.py`: Unit-, Property-, Daten- und End-to-End-Tests
- `output/schedule_results.csv`: alle 600 Job-/Policy-Ergebnisse
- `output/waiting_time_summary.csv`: Kennzahlen je Policy und Klasse
- `output/waiting_time_comparison.png`: Mean-/P95-Vergleich

## Simulationsmodell

Die Simulation besitzt genau einen Prozessor. Wenn kein Job bereit ist, springt
die Uhr zur nächsten Ankunft. Andernfalls wählt die Policy einen Job und führt ihn
vollständig aus. Währenddessen eintreffende Jobs werden erst vor der nächsten
Auswahl in die Ready Queue aufgenommen.

```text
Zeit 0: Job A (batch, Dauer 8) startet
Zeit 2: Job B (critical, Dauer 1) kommt an
Zeit 8: A endet, erst jetzt kann B starten
```

Der Priority Scheduler ist also **nicht-präemptiv**. Priorität entscheidet nur
zwischen bereits bereiten Jobs; sie unterbricht keinen laufenden Job.

## Ordnungsschlüssel

FIFO sortiert implizit nach:

```text
(arrival_time, job_id)
```

Der Priority Heap speichert:

```text
(priority, arrival_time, job_id, job)
```

Kleinere Prioritätszahlen sind wichtiger. Ankunftszeit und Job-ID liefern einen
deterministischen FIFO-Tie-Break innerhalb derselben Klasse und verhindern, dass
Python die Jobobjekte vergleichen muss.

## Kennzahlen

Für jeden Job gelten:

```text
waiting_time    = start_time  - arrival_time
turnaround_time = finish_time - arrival_time
finish_time     = start_time  + duration
```

Je Prioritätsklasse werden Anzahl, Mean, Median, P95 und Maximum der Wartezeit
sowie die mittlere Turnaround Time berechnet. Mean zeigt den Gesamteffekt, P95
macht die Erfahrung langsam wartender Jobs sichtbar und Maximum warnt vor
Starvation-ähnlichen Ausreißern.

Der reproduzierbare Standardlauf ergibt:

| Klasse | Policy | Mean Wait | Median | P95 | Maximum |
|---|---|---:|---:|---:|---:|
| critical | FIFO | 552,88 | 556,50 | 1.038,30 | 1.108 |
| critical | Priority | 10,00 | 8,00 | 29,10 | 34 |
| standard | FIFO | 572,99 | 531,00 | 1.064,05 | 1.112 |
| standard | Priority | 500,19 | 533,00 | 648,55 | 664 |
| batch | FIFO | 583,23 | 571,50 | 1.069,35 | 1.124 |
| batch | Priority | 1.105,00 | 1.125,50 | 1.156,10 | 1.162 |

Priority Scheduling reduziert den Mean der kritischen Klasse um 542,88
Zeiteinheiten und den kritischen P95 um 1.009,20. Standard-Jobs profitieren
moderater. Batch-Jobs tragen den Preis: Ihre mittlere Wartezeit steigt um 521,77.
Der Vergleich macht damit sichtbar, dass eine Priority Queue Wartezeit nicht
verschwinden lässt, sondern entsprechend der Policy zwischen Klassen verschiebt.

## Daten

Der feste Seed erzeugt genau 60 `critical`-, 150 `standard`- und 90 `batch`-Jobs.
Ganzzahlige Ankunftslücken von meist null bis drei Zeiteinheiten erzeugen Bursts;
Jobdauern liegen zwischen eins und zehn. Da die mittlere Arbeitslast schneller
eintrifft als sie verarbeitet werden kann, entsteht eine echte Ready Queue und
die Auswahlpolicy wird sichtbar.

## Komplexität

Bei `n` Jobs und maximal `q` gleichzeitig bereiten Jobs gilt:

| Policy | Enqueue | Auswahl | Gesamtsimulation |
|---|---:|---:|---:|
| FIFO mit `deque` | `O(1)` | `O(1)` | `O(n log n)` inklusive Eingangssortierung |
| Priority Queue | `O(log q)` | `O(log q)` | `O(n log n)` inklusive Eingangssortierung |

Die Priority Queue bezahlt logarithmische Verwaltungskosten, um stets den
wichtigsten Job auszuwählen. Für die kleine Simulation sind Laufzeiten nicht das
Ziel; entscheidend sind Semantik und Wartezeitverteilung.

## Aufgabenstellung

1. Simuliere vier Jobs von Hand, sodass FIFO und Priority unterschiedliche
   Reihenfolgen erzeugen. Berechne alle Wartezeiten.
2. Begründe jedes Feld des Priority-Tupels. Entferne testweise einen Tie-Breaker
   und beschreibe die Folge.
3. Führe Generator und Simulation aus. Vergleiche Mean, P95 und Maximum je
   Klasse in CSV und Plot.
4. Erkläre, warum beide Policies jeden Job genau einmal verarbeiten und warum
   keine Jobs überlappen.
5. Konstruiere einen Fall, in dem ein kritischer Job trotz Priority Scheduler
   lange wartet. Nutze die Nicht-Präemption.
6. Entwirf Aging als Erweiterung: Welche effektive Priorität soll ein lange
   wartender Batch-Job erhalten? Welche Heap-Probleme entstehen durch eine sich
   mit der Zeit ändernde Priorität?
7. Ergänze mindestens einen Test für Idle Time, Tie-Breaking und ungültige Daten.

## Interpretation

Priority Scheduling optimiert keine universelle Fairnessmetrik. Es verschiebt
Wartezeit gezielt zwischen Klassen. Ein Produktionssystem braucht deshalb meist
zusätzliche Regeln wie Aging, Quoten oder maximale Wartezeiten. Der Heap setzt
die gewählte Policy effizient um; er entscheidet nicht, ob diese Policy gerecht
ist.

## Ausführen

Im Projektordner:

```bash
python3 ../../data/generate_data.py
python3 run_simulation.py
python3 -m pytest -q
```

## Fertig, wenn …

- beide Policies exakt dieselben Jobs nicht-präemptiv verarbeiten,
- FIFO und Priority Queue ihre jeweiligen Tie-Break-Regeln einhalten,
- kein Job vor Ankunft startet, verloren geht, doppelt läuft oder überlappt,
- Warte- und Turnaround-Zeiten für jeden Job korrekt sind,
- Kennzahlen nach Prioritätsklasse reproduzierbar berechnet werden,
- CSV-Berichte und PNG-Visualisierung vollständig entstehen,
- der Trade-off zwischen Dringlichkeit und Fairness erklärt wird,
- Generator, Simulation, Syntaxprüfung und alle Tests fehlerfrei durchlaufen.
