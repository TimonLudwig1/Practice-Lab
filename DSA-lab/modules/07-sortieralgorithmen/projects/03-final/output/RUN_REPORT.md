# Laufbericht: Externe Sortierung

## Szenario

Die Pipeline sortiert 5.000 künstlich erzeugte Sensorereignisse
stabil nach `timestamp` und `sensor_id`. Der Generator arbeitet mit Seed
`70703`. Die CSV ist absichtlich nicht vorsortiert und enthält gleiche
Sortierschlüssel sowie Felder mit Kommata.

## Künstliche Ressourcengrenzen

- Höchstens **250 Datensätze** dürfen gleichzeitig als Chunk
  sortiert werden.
- Ein Merge verbindet höchstens **8 Runs** gleichzeitig.
- Die beobachteten Maxima lagen bei 250 Datensätzen pro
  Chunk und 8 Heap-Einträgen im k-Way-Merge.

## Ablauf und Ergebnis

- Eingabedatei: 0.38 MB
- Ausgabedatei: 0.38 MB
- Initiale sortierte Runs: 20
- Runs je Stufe: 20 → 3 → 1
- Merge-Pässe: 2
- Sortierreihenfolge korrekt: ja
- Datensätze vollständig und unverändert: ja

Die versteckte ursprüngliche Zeilennummer ist der letzte Sortierschlüssel. Darum
bleibt die Reihenfolge gleicher öffentlicher Schlüssel auch über Chunk-Grenzen
hinweg stabil. Die Nummer wird nicht in die Ausgabe geschrieben.

## Einordnung

Das Verfahren ersetzt die unmögliche Annahme „alle Daten passen in den RAM“ durch
zwei beschränkte Schritte: lokal sortierte Runs und wiederholte k-Way-Merges. Für
`r` gleichzeitig geöffnete Runs enthält der Heap nur `r` Köpfe; jede Ausgabezeile
verursacht damit `O(log r)` Heap-Arbeit. Die I/O-Kosten wachsen mit der Anzahl der
Merge-Pässe. Ein größerer Fan-in spart Pässe, benötigt aber mehr Dateideskriptoren
und Heap-Einträge. Dieses Muster bildet die Grundlage externer Datenbank-Sorts
und verteilter Sortierphasen in Datenpipelines.
