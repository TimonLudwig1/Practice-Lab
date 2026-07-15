# 01-basic — Laufzeit-Labor

## Ziel

In diesem Projekt untersuchst du fünf Funktionen, deren Aufwand von konstantem
bis quadratischem Wachstum reicht. Du leitest ihre Komplexitätsklassen zuerst
aus dem Code ab und vergleichst deine Vorhersagen anschließend mit reproduzierbar
erhobenen Laufzeitmessungen.

Das Lernziel ist nicht, aus einer einzelnen Zeitangabe Big-O „abzulesen“.
Stattdessen verbindest du drei Perspektiven:

1. die Anzahl der ausgeführten Operationen im Code,
2. die Form der Kurve im Log-Log-Plot,
3. die Veränderung der Laufzeit beim Verdoppeln der Eingabegröße.

## Warum ein Python-Skript?

Das Projekt ist als Python-Skript statt als Notebook umgesetzt. Benchmarks sollen
mehrfach unter denselben Bedingungen ausführbar sein, Ergebnisse als CSV
speichern und ohne manuelle Zellreihenfolge einen Plot erzeugen. Ein separates
Testskript prüft die deterministischen Teile unabhängig von schwankenden
Zeitmessungen.

## Projektstruktur

~~~text
01-basic/
├── README.md
├── LOESUNG.md
├── requirements.txt
├── runtime_lab.py
├── test_runtime_lab.py
└── results/              # wird beim Lauf erzeugt und nicht versioniert
    ├── measurements.csv
    └── runtime_growth.png
~~~

## Vorbereitung

Wechsle in diesen Projektordner und installiere die einzige externe
Abhängigkeit:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
~~~

## Aufgabenstellung

### 1. Vorhersagen aus dem Code ableiten

Öffne **runtime_lab.py** und lies die Funktionen **curve_a** bis **curve_e**.
Führe das Skript noch nicht aus. Trage für jede Kurve eine Vermutung ein:

| Kurve | vermutete Klasse | Begründung aus dem Code |
|---|---|---|
| A |  |  |
| B |  |  |
| C |  |  |
| D |  |  |
| E |  |  |

Zur Auswahl stehen genau diese Klassen:

- \(O(1)\)
- \(O(\log n)\)
- \(O(n)\)
- \(O(n\log n)\)
- \(O(n^2)\)

Markiere für jede Funktion, welche Schleife von \(n\) abhängt, wie sich eine
Schleifenvariable verändert und ob Schleifen nacheinander oder geschachtelt
ausgeführt werden.

### 2. Das Labor ausführen

~~~bash
python3 runtime_lab.py
~~~

Das Skript

- misst jede Funktion für sechs wachsende Eingabegrößen,
- kalibriert automatisch, wie oft sehr schnelle Funktionen pro Messung
  wiederholt werden,
- verwendet den Median aus mehreren Wiederholungen,
- schreibt Rohdaten nach **results/measurements.csv**,
- erzeugt **results/runtime_growth.png** mit logarithmischen Achsen,
- gibt pro Kurve eine empirische Log-Log-Steigung aus.

### 3. Messungen interpretieren

Ergänze nach dem Lauf:

| Kurve | gemessene Steigung | Verhalten bei Verdopplung | endgültige Klasse |
|---|---:|---|---|
| A |  |  |  |
| B |  |  |  |
| C |  |  |  |
| D |  |  |  |
| E |  |  |  |

Beantworte anschließend in eigenen Worten:

1. Welche Kurve bleibt nahezu flach?
2. Welche Kurve wächst bei einer Verdopplung von \(n\) ungefähr um Faktor vier?
3. Warum besitzt \(O(\log n)\) im Log-Log-Plot keine feste positive Steigung wie
   \(O(n)\) oder \(O(n^2)\)?
4. Warum dürfen die gemessenen Steigungen von den theoretischen Idealwerten
   abweichen?
5. Welche Aussage stammt aus der Code-Analyse und welche nur aus dem Experiment?

Öffne **LOESUNG.md** erst, nachdem du deine Zuordnung und Begründungen
aufgeschrieben hast.

## Messmethodik

Sehr schnelle Funktionen liegen nahe an der Auflösung einer einzelnen
Zeitmessung. Das Labor ruft sie deshalb innerhalb eines Messsamples mehrfach auf,
bis eine Mindestdauer erreicht ist. Die gemessene Gesamtdauer wird anschließend
durch die Anzahl der Aufrufe geteilt.

Für jede Kombination aus Funktion und Eingabegröße werden mehrere solcher
Samples aufgenommen. Der Median ist robuster gegen kurze Störungen durch andere
Prozesse als ein einzelner Wert oder der Mittelwert.

Die Standardgrößen sind Zweierpotenzen. Dadurch lässt sich das
Verdopplungsverhalten direkt vergleichen. Eigene Größen können über die
Kommandozeile gesetzt werden:

~~~bash
python3 runtime_lab.py --sizes 64 128 256 512 1024
~~~

Für eine schnellere Proberunde:

~~~bash
python3 runtime_lab.py --sizes 32 64 128 256 --repeats 3 --min-sample-ms 2
~~~

Alle Optionen:

~~~bash
python3 runtime_lab.py --help
~~~

## Tests

Die Tests beurteilen keine absoluten Laufzeiten, weil diese von Rechner und
Systemlast abhängen. Sie prüfen stattdessen die exakt erwartete Anzahl abstrakter
Operationen, die Konfiguration, die Steigungsberechnung, eine kleine
Benchmark-Messreihe und den CSV-Export.

~~~bash
python3 -m unittest -v test_runtime_lab.py
~~~

## Hinweise

- Eine verschachtelte Schleife ist nur dann quadratisch, wenn beide
  Ausführungshäufigkeiten proportional zu \(n\) wachsen.
- Wiederholtes Halbieren deutet auf logarithmisches Wachstum.
- Bei \(n\log n\) steckt ein linearer und ein logarithmischer Faktor in der
  Arbeit.
- Kleine Eingaben werden häufig von konstantem Funktions- und Messaufwand
  dominiert.
- Der Plot plausibilisiert eine theoretische Analyse, beweist sie aber nicht.

## Fertig, wenn …

- du vor der Messung für alle fünf Funktionen eine Komplexitätsklasse
  vorhergesagt und aus dem Code begründet hast,
- alle Unit Tests erfolgreich laufen,
- das Messskript ohne Fehler CSV und PNG erzeugt,
- du alle fünf Kurven anhand von Code und Messdaten zugeordnet hast,
- du Abweichungen zwischen empirischer Steigung und theoretischer Klasse
  erklären kannst,
- du schriftlich begründet hast, warum \(O(n\log n)\) für große \(n\)
  \(O(n^2)\) schlägt.
