# 03-final — Zeitreihen-Toolkit ohne Pandas

## Ziel

In diesem Abschlussprojekt von Modul 02 baust du eine kleine, aber vollständige
Analyse-Pipeline für Sensordaten. Die Algorithmen arbeiten ausschließlich mit
Python-Listen beziehungsweise allgemeinen Sequenzen. Pandas ist nicht erlaubt.
NumPy kommt erst am Ende als unabhängige Referenz und Performance-Vergleich hinzu.

Das Projekt verbindet drei Array-Patterns mit einem realistischen
Data-Science-Ablauf:

- ein Rolling Window für gleitende Mittelwerte,
- Prefix Sums für viele Bereichssummen,
- zwei vollständige Array-Durchläufe für Z-Score-Ausreißer.

Python-Skripte sind hier das passende Format: Generator, Toolkit, Tests und
Benchmark sind getrennt ausführbar, reproduzierbar und leichter automatisiert zu
prüfen als versteckter Notebook-Zustand.

## Szenario

Ein Temperatursensor liefert eine eindimensionale Zeitreihe. Die synthetischen
Daten enthalten:

- einen langsamen linearen Drift,
- einen periodischen Tagesverlauf,
- normalverteiltes Messrauschen,
- drei absichtlich injizierte positive oder negative Ausreißer.

Der feste Standard-Seed `20260716` sorgt dafür, dass Tests und Messungen bei
gleichen Parametern dieselben Eingaben erhalten. Die Ausreißerpositionen werden
als Ground Truth separat gespeichert; der Erkennungsalgorithmus kennt sie nicht.

## Projektstruktur

```text
03-final/
├── README.md
├── LOESUNG.md
├── generate_sensor_data.py
├── sensor_toolkit.py
├── benchmark.py
├── test_sensor_toolkit.py
├── requirements.txt
├── data/                         # beim Lauf erzeugt
│   └── sensor_readings.csv
└── results/                      # beim Lauf erzeugt
    ├── benchmark_results.csv
    └── sensor_and_runtime_comparison.png
```

Generierte CSV- und Bilddateien sind bewusst ignoriert. Sie lassen sich mit dem
festen Seed jederzeit identisch inhaltlich neu erzeugen; Laufzeiten selbst hängen
vom jeweiligen Rechner ab.

## Aufgabe 1 — Reproduzierbare Sensordaten

Implementiere in `generate_sensor_data.py`:

```python
generate_sensor_data(size, seed, anomaly_indices) -> SensorDataset
write_sensor_csv(dataset, output_path) -> None
```

Anforderungen:

- Verwende eine lokale `random.Random`-Instanz, damit kein globaler
  Zufallszustand verändert wird.
- Modelliere Baseline, periodische Komponente und Rauschen getrennt.
- Injiziere bekannte Spikes erst nach Erzeugung der normalen Zeitreihe.
- Prüfe Größe, Seed und Ausreißerindizes explizit.
- Schreibe Index, Messwert und Ground-Truth-Markierung in die CSV-Datei.

## Aufgabe 2 — Gleitender Mittelwert in O(n)

Implementiere:

```python
moving_average(values, window) -> list[float]
```

Für `n` Werte und Fensterbreite `w` soll die Ergebnislänge `n - w + 1` sein.
Berechne nur das erste Fenster vollständig. Danach wird beim Verschieben ein Wert
abgezogen und ein Wert addiert:

```text
neue Summe = alte Summe - austretender Wert + eintretender Wert
```

Eine Lösung, die jedes Fenster erneut mit `sum` berechnet, kostet O(n · w) und
erfüllt das Ziel nicht. Ungültige Fenster sowie nichtnumerische oder nichtendliche
Messwerte müssen abgewiesen werden.

## Aufgabe 3 — Bereichssummen via Prefix Sums

Implementiere:

```python
index = PrefixSumIndex.from_readings(values)
index.range_sum(start, end) -> float
index.batch_range_sums(ranges) -> list[float]
```

Die Abfragegrenzen sind halboffen: `[start, end)`. Der Aufbau kostet O(n), eine
einzelne Abfrage O(1) und ein Batch mit `q` Abfragen O(q). Der gespeicherte Index
soll unveränderlich und von späteren Änderungen der Eingabeliste unabhängig sein.

## Aufgabe 4 — Ausreißererkennung

Implementiere:

```python
detect_zscore_outliers(values, threshold=4.0) -> list[Outlier]
```

Nutze Mittelwert und Populationsstandardabweichung:

```text
z_i = (x_i - Mittelwert) / Standardabweichung
```

Ein Wert gilt als Ausreißer, wenn `abs(z_i) >= threshold`. Das Ergebnis enthält
Index, Messwert und vorzeichenbehafteten Z-Score. Definiere das Verhalten für
leere oder konstante Reihen und prüfe den Schwellwert.

Wichtig: Ein globaler Z-Score ist eine bewusst einfache Methode. Bei starkem
Trend, saisonal wechselnder Varianz oder vielen Ausreißern wäre eine lokale oder
robuste Methode fachlich geeigneter. Hier steht das lineare Array-Verfahren im
Mittelpunkt.

## Aufgabe 5 — Pipeline zusammensetzen

Implementiere:

```python
analyze_sensor_readings(
    values,
    window=...,
    ranges=...,
    outlier_threshold=...,
) -> AnalysisResult
```

Die Pipeline soll die drei Analysen verbinden und einen stabilen Snapshot aus
Tuples zurückgeben. Ein Generator von Bereichsabfragen darf dabei nur einmal
konsumiert werden.

## Aufgabe 6 — NumPy-Vergleich

`benchmark.py` vergleicht vier getrennte Operationen:

| Operation | Listenimplementierung | NumPy-Äquivalent |
|---|---|---|
| Moving Average | Rolling Sum | `np.convolve` |
| Prefix-Aufbau | sequenzieller Aufbau | `np.cumsum` |
| Bereichsabfragen | Python-Schleife | vektorisierte Indexierung |
| Ausreißer | zwei Python-Durchläufe | vektorisierte Reduktionen/Maskierung |

Vor jeder Laufzeitinterpretation müssen die Ergebnisse übereinstimmen:

- Gleitende Mittelwerte und Summen werden mit `allclose` verglichen.
- Form und Reihenfolge müssen identisch sein.
- Erkannte Ausreißerindizes müssen exakt gleich sein.
- Der größte absolute numerische Fehler wird protokolliert.

Gemessen wird der Median mehrerer Wiederholungen. Die Messung verspricht keinen
bestimmten Speedup: Hardware, Python-/NumPy-Version, Eingabegröße und Operation
beeinflussen das Ergebnis. Sie soll zeigen, wo contiguous memory und in C
ausgeführte Schleifen praktisch relevant werden.

## Installation und Ausführung

Abhängigkeiten installieren:

```bash
python3 -m pip install -r requirements.txt
```

Alle Tests ausführen:

```bash
python3 -m unittest -v test_sensor_toolkit.py
```

Nur einen Datensatz erzeugen:

```bash
python3 generate_sensor_data.py --size 10000
```

Den vollständigen Referenzlauf starten:

```bash
python3 benchmark.py
```

Für einen schnellen Kontrolllauf:

```bash
python3 benchmark.py --size 10000 --queries 2000 --repetitions 2
```

Alle Befehle werden aus dem Projektordner ausgeführt.

## Ergebnisse lesen

`benchmark_results.csv` enthält pro Operation:

- Medianlaufzeit der Listenimplementierung,
- Medianlaufzeit des NumPy-Äquivalents,
- Faktor `Python-Zeit / NumPy-Zeit`,
- größten absoluten Ergebnisfehler.

Ein Faktor größer als `1` bedeutet, dass NumPy in diesem Lauf schneller war. Ein
Faktor kleiner als `1` ist kein Widerspruch: Bei kleinen Eingaben kann der
Aufruf- und Allokations-Overhead einer Bibliotheksoperation dominieren.

Die Grafik verbindet die fachliche und technische Sicht. Oben stehen Rohsignal,
geglättete Reihe und erkannte Ausreißer. Unten werden die Laufzeiten auf einer
logarithmischen Achse verglichen.

## Fertig, wenn …

- der Generator mit identischem Seed exakt reproduzierbare Reihen erzeugt,
- injizierte Ausreißer als getrennte Ground Truth erhalten bleiben,
- der gleitende Mittelwert in O(n) statt O(n · w) berechnet wird,
- Prefix-Aufbau O(n) und einzelne Bereichsabfragen O(1) kosten,
- die Z-Score-Erkennung leere, konstante und ungültige Reihen korrekt behandelt,
- die kombinierte Pipeline alle drei Analysen liefert,
- normale Fälle, Randfälle und Fehlerfälle automatisiert getestet sind,
- der vollständige Benchmark alle Python-Ergebnisse gegen NumPy validiert,
- CSV-Dateien und Vergleichsgrafik erfolgreich erzeugt werden,
- du Performance-Unterschiede mit Datenlayout, Interpreter-Schleifen und
  Vektorisierung erklären kannst.
