# 02-medium — Komplexitäts-Detektiv

## Ziel

In diesem Projekt analysierst du zehn kurze Python-Funktionen. Einige sehen
harmlos aus, verbergen aber Kosten in List Slicing, Membership-Tests,
String-Konkatenation, Sortierung oder Listenoperationen. Andere besitzen
verschachtelte Schleifen, sind trotzdem nicht quadratisch.

Du arbeitest wie bei einem Performance-Audit:

1. Eingabegröße und relevante Operationen benennen,
2. Laufzeit- und Speicherkomplexität aus dem Code herleiten,
3. die Vorhersage durch kontrollierte Messungen plausibilisieren,
4. Abweichungen zwischen Theorie und Empirie erklären.

## Warum ein Python-Skript?

Die zehn Fälle sollen unter identischen Bedingungen wiederholt gemessen werden.
Ein Skript eignet sich dafür besser als ein Notebook: Es besitzt einen
eindeutigen Einstiegspunkt, kalibriert kurze Messungen automatisch und erzeugt
CSV sowie Plot in einem reproduzierbaren Lauf. Unit Tests prüfen die
funktionalen Ergebnisse getrennt vom schwankenden Timing.

## Projektstruktur

~~~text
02-medium/
├── README.md
├── LOESUNG.md
├── complexity_detective.py
├── test_complexity_detective.py
├── requirements.txt
└── results/                    # wird erzeugt und nicht versioniert
    ├── measurements.csv
    └── normalized_growth.png
~~~

## Vorbereitung

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
~~~

## Teil 1: Vorhersagen

Öffne **complexity_detective.py** und lies ausschließlich die Funktionen
**case_01** bis **case_10**. Führe das Benchmark noch nicht aus.

Fülle für jeden Fall die Tabelle aus:

| Fall | Zeitklasse | Auxiliary Space | dominante Operation | Annahmen |
|---|---|---|---|---|
| 01 |  |  |  |  |
| 02 |  |  |  |  |
| 03 |  |  |  |  |
| 04 |  |  |  |  |
| 05 |  |  |  |  |
| 06 |  |  |  |  |
| 07 |  |  |  |  |
| 08 |  |  |  |  |
| 09 |  |  |  |  |
| 10 |  |  |  |  |

Gehe dabei systematisch vor:

1. \(n\) ist jeweils der Parameter **size**.
2. Markiere sichtbare Schleifen.
3. Ermittle die Kosten aufgerufener Operationen wie Slicing, **in**,
   **insert** und **sorted**.
4. Addiere aufeinanderfolgende Phasen und summiere abhängige Schleifenlängen.
5. Trenne gleichzeitig belegten Speicher von der über die Laufzeit insgesamt
   allokierten Datenmenge.

## Teil 2: Benchmark ausführen

~~~bash
python3 complexity_detective.py
~~~

Der Standardlauf misst alle zehn Fälle über sieben verdoppelte Eingabegrößen.
Sehr schnelle Funktionen werden innerhalb eines Samples automatisch mehrfach
aufgerufen. Pro Messpunkt wird der Median mehrerer Samples gespeichert.

Erzeugte Artefakte:

- **results/measurements.csv** enthält die Rohmessungen und Batch-Größen.
- **results/normalized_growth.png** zeigt jede Laufzeit relativ zum ersten
  Messpunkt. Durch die Normalisierung beginnen alle Kurven bei 1 und ihr
  Wachstum wird trotz stark verschiedener absoluter Konstanten vergleichbar.

Die Konsole zeigt zusätzlich die empirische Log-Log-Steigung und den gesamten
Wachstumsfaktor jeder Funktion.

## Teil 3: Detektivbericht

Ergänze nach der Messung:

| Fall | vorhergesagte Klasse | empirische Steigung | bestätigt? | Erklärung |
|---|---|---:|---|---|
| 01 |  |  |  |  |
| 02 |  |  |  |  |
| 03 |  |  |  |  |
| 04 |  |  |  |  |
| 05 |  |  |  |  |
| 06 |  |  |  |  |
| 07 |  |  |  |  |
| 08 |  |  |  |  |
| 09 |  |  |  |  |
| 10 |  |  |  |  |

Beantworte in eigenen Worten:

1. Welche drei Fälle enthalten eine sichtbar oder versteckt quadratische
   Datenverschiebung beziehungsweise Kopierarbeit?
2. Warum ist der Fall mit zwei verschachtelten Schleifen nicht automatisch
   quadratisch?
3. Wie verändert der Container-Typ die Kosten des Operators **in**?
4. Warum kann eine eingebaute C-implementierte Operation trotz schlechterer
   asymptotischer Klasse bei kleinen \(n\) schneller aussehen?
5. Weshalb ist die gemessene Steigung von \(n\log n\) in diesem begrenzten
   Bereich oft nur wenig größer als 1?
6. Welcher Fall tauscht lineare erwartete Laufzeit gegen linearen Zusatzspeicher?
7. Bei welchem Fall wächst auch der gleichzeitig belegte Zusatzspeicher
   quadratisch und warum?

Vergleiche deine Analyse erst danach mit **LOESUNG.md**.

## Eigene Messbereiche

Ein kurzer Diagnoselauf:

~~~bash
python3 complexity_detective.py \
  --sizes 32 64 128 256 \
  --repeats 3 \
  --min-sample-ms 2
~~~

Einzelne Fälle messen:

~~~bash
python3 complexity_detective.py --cases 05 06 07
~~~

Alle Optionen:

~~~bash
python3 complexity_detective.py --help
~~~

## Tests

Absolute Laufzeiten sind nicht teststabil. Die Tests prüfen stattdessen die
Ergebnisse aller zehn Fälle, ungültige Benchmark-Konfigurationen, die
Steigungsberechnung, eine kleine Messreihe und den CSV-Export.

~~~bash
python3 -m unittest -v test_complexity_detective.py
~~~

## Hinweise

- Eine einzelne Python-Zeile kann intern linear oder linearithmisch arbeiten.
- Zwei aufeinanderfolgende lineare Schleifen ergeben \(O(n+n)=O(n)\).
- Bei einer geometrisch schrumpfenden inneren Arbeit kann
  \(n+n/2+n/4+\dots\) insgesamt linear bleiben.
- Ein Set bietet Membership-Tests nur unter den üblichen Hashing-Annahmen in
  erwarteter konstanter Zeit.
- Timsort nutzt vorhandene Ordnung. Der Sortierfall erzeugt deshalb bewusst eine
  deterministische, aber unregelmäßige Eingabe.
- Ein Benchmark stützt eine Herleitung, ersetzt sie jedoch nicht.

## Fertig, wenn …

- du für alle zehn Fälle Zeit- und Speicherkomplexität vor der Messung
  hergeleitet hast,
- du versteckte Kosten von Slicing, String-Konkatenation,
  **list.insert(0, x)**, **in** und **sorted** erklären kannst,
- alle Unit Tests erfolgreich laufen,
- CSV und normalisierter Log-Log-Plot ohne Fehler erzeugt werden,
- du jede empirische Kurve mit deiner Vorhersage abgeglichen hast,
- du mindestens drei Messabweichungen anhand von Konstanten,
  Interpreterkosten oder begrenztem Größenbereich erklärst,
- dein Detektivbericht Theorie und Beobachtung sprachlich klar trennt.
