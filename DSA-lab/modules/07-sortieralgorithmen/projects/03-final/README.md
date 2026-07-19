# 03-final — Externe Sortierung

## Ziel

Dieses Projekt überträgt Sortieralgorithmen auf einen typischen Engpass aus
Datenpipelines: Die Eingabedatei ist größer als der für das Sortieren erlaubte
Arbeitsspeicher. Statt alle Zeilen gleichzeitig zu laden, zerlegt die Pipeline
die Datei in beschränkte Chunks, sortiert jeden Chunk und verbindet die
entstandenen Runs mit einem stabilen k-Way-Merge.

Das Projekt verwendet Python-Skripte und CSV-Dateien, weil sich Speichergrenzen,
Zwischendateien und I/O-Verhalten darin transparenter beobachten und testen
lassen als in einem Notebook.

## Aufgabenstellung

Baue eine externe Sortierung für CSV-Dateien mit folgenden Eigenschaften:

1. Lies höchstens eine festgelegte Anzahl Datensätze in einen Chunk.
2. Sortiere jeden Chunk stabil nach frei wählbaren Spalten und schreibe einen
   temporären Run.
3. Verbinde höchstens `merge_fan_in` Runs gleichzeitig mit einem Min-Heap.
4. Wiederhole den Merge, bis genau ein global sortierter Run übrig bleibt.
5. Erhalte alle CSV-Spalten, Unicode und korrekt gequotete Felder.
6. Prüfe die Ausgabe stromorientiert auf Sortierung und Datenerhalt.
7. Dokumentiere die tatsächlich beobachteten Ressourcengrenzen.

Die Fallstudie in `run_pipeline.py` erzeugt 5.000 Sensordatensätze und sortiert
sie stabil nach `timestamp` und `sensor_id`. Für gleiche Schlüssel muss die
ursprüngliche Eingabereihenfolge erhalten bleiben.

## Dateien

- `external_sort.py`: Chunk-Erzeugung, mehrstufiger k-Way-Merge, Metriken und
  strombasierte Integritätsprüfung
- `generate_data.py`: reproduzierbarer, zeilenweise arbeitender Datengenerator
- `run_pipeline.py`: vollständige Fallstudie samt Bericht
- `test_external_sort.py`: Unit-, Property- und Fehlerfalltests
- `data/unsorted_events.csv`: generierte Eingabe der Standardfallstudie
- `output/sorted_events.csv`: sortierte Ausgabe
- `output/run_metrics.csv`: maschinenlesbare Pipeline-Metriken
- `output/RUN_REPORT.md`: Interpretation des vollständigen Laufs

## Hinweise

Der öffentliche Sortierschlüssel allein reicht für Stabilität nicht aus, sobald
gleiche Schlüssel in unterschiedlichen Chunks liegen. Dekoriere deshalb intern
jeden Datensatz mit seiner ursprünglichen Zeilennummer. Sie darf in der finalen
CSV nicht erscheinen.

Beim k-Way-Merge muss nicht jeder Run vollständig geladen werden. Im Heap liegt
nur der aktuelle Kopf jedes geöffneten Runs. Nach dem Entfernen des kleinsten
Elements wird genau ein Nachfolger aus demselben Run nachgeladen.

Ein begrenzter Fan-in ist praxisnäher als das gleichzeitige Öffnen beliebig
vieler Dateien. Er erzwingt bei vielen Chunks mehrere Merge-Pässe und macht den
Trade-off zwischen I/O, Heap-Größe und Dateideskriptoren sichtbar.

## Ausführen

```bash
python3 -m pytest -q
python3 run_pipeline.py
```

## Fertig, wenn …

- die Ausgabe für unterschiedliche Chunk-Größen exakt einem stabilen
  In-Memory-Referenzsort entspricht,
- kein sortierter Chunk die künstliche Datensatzgrenze überschreitet,
- der Merge-Heap nie größer als der konfigurierte Fan-in wird,
- gleiche Schlüssel auch über Chunk-Grenzen stabil bleiben,
- leere Dateien, ein einzelner Run, fehlende Spalten und fehlerhafte CSV-Zeilen
  sinnvoll behandelt werden,
- die strombasierte Prüfung sowohl falsche Reihenfolge als auch veränderte
  Datensätze erkennt,
- alle Tests erfolgreich sind und der Standardlauf 5.000 Datensätze vollständig
  sortiert sowie `RUN_REPORT.md` und `run_metrics.csv` erzeugt.
