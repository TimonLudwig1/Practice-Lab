# Projekt 03-final: Hash-basierte Log-Analyse-Engine

## Ziel

Dieses Projekt verarbeitet einen synthetischen Serverlog-Strom vollständig
reproduzierbar. Es entfernt doppelte Event-IDs, zählt Zugriffe je IP, weist
Ereignisse anhand eines Inaktivitätsfensters Sessions zu und liefert die Top-K
IPs. Eine zweite, sortierbasierte Engine berechnet dasselbe Ergebnis als
unabhängige Referenz und Performance-Vergleich.

Python-Skripte, CSV-Ausgaben und ein Markdown-Bericht bilden eine nachvollziehbare
End-to-End-Pipeline. Ein Notebook wäre für Exploration geeignet, die Skriptform
macht Datengenerierung, Äquivalenzprüfung, Benchmark und Tests jedoch mit einem
Aufruf reproduzierbar.

## Aufgabenstellung

1. Erzeuge chronologische Logs mit Event-ID, Zeitstempel, IP, Pfad und Status.
   Nutze einen festen Seed und füge gezielt wiederholte Event-IDs ein.
2. Filtere Duplikate in Eingabereihenfolge mit einem Set. Das erste Auftreten
   bleibt erhalten.
3. Zähle eindeutige Ereignisse pro IP mit einer Hash Map und ermittle Top-K.
4. Halte für jede IP den letzten Zeitstempel und die aktuelle Sessionnummer. Ein
   Abstand größer als 30 Sekunden eröffnet eine neue Session.
5. Implementiere eine sortierbasierte Referenz ohne Set-/Dictionary-Aggregation:
   Sortieren nach Event-ID, IP und Zeit sowie Run-Length-Gruppierung.
6. Vergleiche beide Engines über wachsende Präfixe und brich ab, falls auch nur
   eine Session-Zuordnung oder Kennzahl abweicht.

## Datenfluss

```text
server_logs.csv
      |
      v
Event-ID-Deduplizierung ----> duplicate_count
      |
      +----> IP-Häufigkeiten ----> Top-K
      |
      +----> letzter Zustand je IP ----> sessionized_events.csv
      |
      +----> Benchmark gegen Sortier-Engine ----> REPORT.md
```

## Session-Simulation

Bei einem Timeout von 30 Sekunden:

| Zeit | IP | vorherige Zeit dieser IP | Ergebnis |
|---:|---|---:|---|
| 0 | A | – | `A-s0001` |
| 5 | B | – | `B-s0001` |
| 30 | A | 0 | `A-s0001` |
| 40 | B | 5 | `B-s0002` |
| 60,1 | A | 30 | `A-s0002` |

Die Zustände der IPs sind unabhängig. Genau 30 Sekunden gehören noch zur
bisherigen Session; erst ein größerer Abstand startet eine neue.

## Invarianten

- Die Eingabe ist chronologisch nicht fallend sortiert.
- Jede Event-ID erscheint nach der Filterung höchstens einmal.
- `input_count = unique_count + duplicate_count`.
- Jede eindeutige Zeile erhält genau eine Session-ID.
- Die Sessionnummer einer IP beginnt bei 1 und steigt nur bei Timeoutüberschreitung.
- Beide Engines liefern abgesehen vom Methodennamen exakt dasselbe Ergebnis.

## Komplexität

Seien `n` die Logzeilen und `u` die verschiedenen IPs.

| Phase | Hash-Engine | Sortier-Engine |
|---|---:|---:|
| Deduplizierung | erwartet O(n) | O(n log n) |
| IP-Aggregation | erwartet O(n) | O(n log n) |
| Sessionisierung | erwartet O(n) | O(n log n) |
| Top-K-Rangfolge | O(u log u) | O(u log u) |
| Zusatzspeicher | O(n + u) | O(n) |

Die Hash-Engine arbeitet im Kern in einem Durchlauf. Die abschließende Rangfolge
sortiert nur die `u` unterschiedlichen IPs, nicht alle `n` Ereignisse.

## Projektstruktur

```text
03-final/
├── data/
│   ├── generate_data.py
│   └── server_logs.csv
├── output/
│   ├── sessionized_events.csv
│   ├── top_ips.csv
│   ├── benchmark.csv
│   └── REPORT.md
├── log_engine.py
├── run_analysis.py
└── test_log_engine.py
```

## Ausführen

Im Projektordner:

```bash
python data/generate_data.py
python run_analysis.py
python -m pytest -q
```

Der Seed `60603` sorgt dafür, dass beide Engines dieselben 4.000 Zeilen,
Duplikate und IP-Verteilungen sehen. `run_analysis.py` erzeugt die Eingabedatei
automatisch, falls sie fehlt.

## Hinweise

- Dedupliziere vor Häufigkeitszählung und Sessionisierung; sonst verfälschen
  Wiederholungen beide Resultate.
- Ein Duplikat wird über die Event-ID definiert, nicht über identische übrige
  Felder.
- Verwende einen deterministischen Tie-Break für Top-K: bei gleicher Häufigkeit
  lexikografisch nach IP.
- Die Hash-Lösung setzt chronologische Eingabe voraus. Eine unsortierte Quelle
  muss vor der Pipeline geordnet werden.
- Absolute Benchmarkzeiten schwanken. Entscheidend sind Äquivalenz und
  Wachstumsklasse.

## Fertig, wenn …

- der Generator mit festem Seed einen chronologischen Log-Strom erzeugt,
- Duplikate entfernt werden, bevor sie weitere Kennzahlen beeinflussen,
- Top-K und Session-Zuordnungen für verschachtelte IP-Aktivität korrekt sind,
- Timeoutgrenze und lexikografischer Tie-Break explizit getestet werden,
- Hash- und Sortier-Engine für alle Benchmarkgrößen exakt übereinstimmen,
- Detail-CSV, Top-K, Benchmark und Bericht erzeugt werden,
- die vollständige 4.000-Zeilen-Pipeline läuft und
- alle Tests mit `python -m pytest -q` erfolgreich sind.
