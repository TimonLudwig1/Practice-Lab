# 03-final — Streaming-Anomalieerkennung

## Ziel

Dieses Projekt überträgt Sliding Window auf eine realistische Zeitreihenpipeline.
Ein simulierter Metrik-Stream wird online gegen Mittelwert und
Standardabweichung des vorherigen Fensters geprüft. Die optimierte Variante
pflegt Queue, Summe und Quadratsumme in `O(1)` pro Messwert; eine naive Referenz
berechnet jedes Fenster vollständig neu.

Python-Skripte, CSV-Artefakte und ein Plot machen Datenherkunft, Online-Zustand,
Ergebnisgleichheit, Laufzeit und Erkennungsqualität separat überprüfbar.

## Aufgabenstellung

1. Erzeuge einen Seed-basierten Stream mit langsamem Signal, Rauschen und
   markierten Ausreißern.
2. Pflege ein festes Rolling Window mit begrenzter Queue, Summe und Quadratsumme.
3. Bewerte jeden neuen Wert gegen die vorherigen `k` Werte per z-Score.
4. Implementiere dieselbe Logik als naive `O(nk)`-Referenz.
5. Vergleiche Flags und Statistiken innerhalb enger Gleitkommatoleranzen.
6. Berichte Laufzeit, Precision/Recall und Confusion Counts in CSV, Markdown und
   einer Zeitreihengrafik.

## Dateien

- `anomaly_detector.py`: Rolling-Zustand, beide Detektoren, Metriken und CSV-Reader
- `generate_stream.py`: reproduzierbarer Stream mit Ground Truth
- `run_pipeline.py`: vollständiger Standardlauf und Artefakterzeugung
- `test_anomaly_detector.py`: Zustands-, Referenz-, Fehler- und Generatortests
- `data/metric_stream.csv`: 12.000 Minutenmesswerte
- `output/detections.csv`: Rolling-Statistiken und Flags nach Warm-up
- `output/benchmark_metrics.csv`: Laufzeit- und Qualitätskennzahlen
- `output/anomaly_detection.png`: Messwerte, Rolling Mean und erkannte Ausreißer
- `output/RUN_REPORT.md`: Interpretation und numerische Einschränkungen

## Zustandsinvariante

Unmittelbar vor der Bewertung eines neuen Messwerts enthält `RollingStats`
genau die vorherigen `window_size` Werte. `total` und `square_total` sind deren
Summe beziehungsweise Quadratsumme. Der neue Wert gehört noch nicht zur Baseline
und wird erst nach seiner Bewertung angefügt; bei vollem Fenster verlässt dabei
genau der älteste Wert den Zustand.

Damit verwendet jeder Schritt ausschließlich Vergangenheit, benötigt konstante
Updatearbeit und speichert höchstens `window_size` Werte.

## Numerischer Hinweis

Die Populationsvarianz wird als `E[x²] - E[x]²` berechnet. Das ermöglicht
konstante Updates, kann aber bei sehr großen nahezu gleichen Zahlen
Gleitkommaauslöschung verursachen. Winzige negative Rundungsreste werden auf null
begrenzt. Für extreme Produktionsdaten wären stabilere Online-Algorithmen oder
regelmäßige Neuberechnungen zu erwägen.

## Ausführen

```bash
python3 -m pytest -q
python3 run_pipeline.py
```

## Fertig, wenn …

- Queue, Summe und Quadratsumme nach Eintritt und Austritt konsistent sind,
- der aktuelle Wert niemals Teil seiner eigenen Baseline ist,
- konstante Fenster und Nullvarianz sinnvoll behandelt werden,
- Streaming- und Referenzdetektor auf 300 Seed-Fällen dieselben Flags liefern,
- Generator und Reader fehlerhafte Konfigurationen beziehungsweise CSVs abweisen,
- der Standardlauf alle 20 injizierten Ausreißer bewertet und Ergebnisgleichheit
  vor dem Schreiben bestätigt,
- CSVs, Plot und deutscher Laufbericht erzeugt werden,
- alle Tests und die vollständige 12.000-Zeilen-Pipeline erfolgreich laufen.
