# 03-final — Schwellenwert-Optimierung für einen Klassifikator

## Ziel

Dieses Projekt überträgt Binary Search auf eine realistische Data-Science-
Entscheidung: Aus Modell-Scores soll ein Klassifikationsschwellenwert gewählt
werden, der eine fachliche Nebenbedingung erfüllt und unter den zulässigen
Schwellen den Recall maximiert.

Die Lösung besteht aus Python-Skripten, CSV-Artefakten und einer Visualisierung.
So bleiben Datengenerierung, Suchlogik, maschinenlesbare Ergebnisse und fachliche
Interpretation getrennt und vollständig reproduzierbar.

## Aufgabenstellung

1. Erzeuge mit festem Seed binäre Labels und überlappende Klassifikator-Scores.
2. Werte für einen Schwellenwert die Confusion Counts, Recall, False Positive
   Rate und Precision aus.
3. Finde per Binary Search den kleinsten exakten Schwellenwert mit
   `False Positive Rate ≤ 5 %`.
4. Begründe, warum diese kleinste zulässige Schwelle den Recall maximiert.
5. Vergleiche das Ergebnis mit einer gleichmäßigen Rastersuche und einer
   erschöpfenden Suche über alle exakten Kandidaten.
6. Schreibe Suchspur, Methodenvergleich, Diagramm und Laufbericht.

## Warum FPR statt Precision?

Die im Modulplan genannte Precision-Nebenbedingung ist ein anschauliches Ziel,
aber Precision ist auf einem endlichen Datensatz im Allgemeinen **nicht
monoton**. Beim Entfernen einer Vorhersage kann je nach Label sowohl Zähler als
auch Nenner anders reagieren. Eine gewöhnliche Binary Search wäre ohne weitere
Annahmen nicht korrekt.

Die False Positive Rate ist dagegen monoton nicht steigend: Wenn der
Schwellenwert steigt, können keine neuen False Positives hinzukommen. Recall ist
ebenfalls nicht steigend. Deshalb liefert die erste Schwelle, die die FPR-Grenze
einhält, unter allen zulässigen Schwellen den maximalen Recall. Diese fachliche
Begründung ist Teil der Lösung, nicht nur ein Implementierungsdetail.

## Dateien

- `threshold_optimizer.py`: Datentypen, Score-Index, Metriken und drei
  Optimierungsverfahren
- `generate_data.py`: reproduzierbarer Score-Datensatz mit Seed
- `run_pipeline.py`: vollständiger Standardlauf samt CSV, Plot und Bericht
- `test_threshold_optimizer.py`: Metrik-, Monotonie-, Referenz- und Fehlerfalltests
- `data/scores.csv`: 5.000 generierte Modellbewertungen
- `output/binary_search_trace.csv`: exakte Intervallentscheidungen
- `output/method_comparison.csv`: Binary Search, Raster und erschöpfende Referenz
- `output/threshold_metrics.png`: Metrikkurven mit gewählter Schwelle
- `output/RUN_REPORT.md`: fachliche und algorithmische Interpretation

## Algorithmus

Positive und negative Scores werden getrennt sortiert. Für einen Schwellenwert
liefert `bisect_left` jeweils die Zahl der Scores darunter; daraus folgen die
Confusion Counts in `O(log n)`.

Die exakten Kandidaten sind alle beobachteten Scorewerte plus eine Schwelle
oberhalb von `1.0`, die keine positive Vorhersage erzeugt und damit sicher die
FPR-Grenze erfüllt. Binary Search sucht auf den Kandidatenindizes nach dem ersten
zulässigen Wert.

Bei `u` Kandidaten benötigt sie `O(log u)` Metrikauswertungen. Die erschöpfende
Referenz benötigt `O(u)`, die naive Rastersuche eine Auswertung pro Rasterpunkt.
Die einmalige Sortierung kostet `O(n log n)`.

## Hinweise

Die Klassifikationsregel ist inklusiv: `score >= threshold` bedeutet positiv.
Darum gehören beobachtete Scores selbst zu den exakten Entscheidungsgrenzen.

Die Rastersuche ist nur eine Näherung. Ein feineres Raster erhöht die Kosten,
garantiert aber weiterhin nicht, einen beobachteten Score exakt zu treffen.

Ein korrektes Ergebnis muss nicht nur zulässig sein. Die erschöpfende Referenz
muss denselben exakten Schwellenwert und dieselben Confusion Counts liefern, und
der direkte kleinere Kandidat muss die Nebenbedingung verletzen.

## Ausführen

```bash
python3 -m pytest -q
python3 run_pipeline.py
```

## Fertig, wenn …

- die Metrikberechnung für handprüfbare Confusion Matrices stimmt,
- FPR und Recall entlang wachsender Kandidatenschwellen nicht steigen,
- die Binary Search für reproduzierbare Zufallsdaten exakt mit der
  erschöpfenden Kandidatensuche übereinstimmt,
- die gewählte Schwelle die FPR-Grenze erfüllt und maximalen Recall besitzt,
- leere, doppelte oder ungültige Datensätze und Konfigurationen abgewiesen werden,
- die Binary Search deutlich weniger Metrikauswertungen als Raster und
  erschöpfende Suche benötigt,
- alle Tests erfolgreich sind und der 5.000-Zeilen-Standardlauf Suchspur,
  Vergleich, Diagramm und deutschen Laufbericht erzeugt.
