# 01-basic — Dynamisches Array selbst bauen

## Ziel

In diesem Projekt implementierst und untersuchst du die zentrale Mechanik hinter
Python-Listen: ein dynamisches Array auf Basis eines festen, zusammenhängenden
Puffers.

Die Klasse unterstützt:

- direkten Indexzugriff,
- Append mit Kapazitätsverdopplung,
- Insert mit Rechtsverschiebung,
- Delete mit Linksverschiebung,
- internes Resize samt vollständiger Kopie,
- ein unveränderliches Protokoll aller Kapazitätswechsel.

Ein separates Experiment macht sichtbar, weshalb einzelne appends teuer sein
können, der amortisierte Aufwand über eine lange Folge aber konstant bleibt.

## Warum Python-Skripte?

Die Datenstruktur lebt von Zustandsänderungen, Invarianten und automatischen
Kantenfalltests. Python-Skripte trennen Implementierung, Experiment und Tests
klar voneinander. Das Wachstumsexperiment kann außerdem ohne manuelle
Notebook-Schritte identisch wiederholt und als CSV sowie PNG dokumentiert
werden.

## Projektstruktur

~~~text
01-basic/
├── README.md
├── LOESUNG.md
├── dynamic_array.py
├── growth_experiment.py
├── test_dynamic_array.py
├── requirements.txt
└── results/                         # generiert, nicht versioniert
    ├── growth_log.csv
    └── capacity_and_costs.png
~~~

## Technische Grundlage

Der Elementpuffer ist keine Python-Liste. **DynamicArray** reserviert über
**ctypes.py_object * capacity** ein Array fester Länge. Dieser Puffer kann nicht
erweitert werden. Für ein Resize wird tatsächlich ein neuer größerer Puffer
angelegt und jedes aktive Element einzeln kopiert.

Eine kleine Python-Liste wird ausschließlich für Diagnoseereignisse verwendet.
Sie speichert keine Nutzdaten und ist nicht Teil der Array-Operationen.

## Vorbereitung

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
~~~

## Aufgabenstellung

### 1. Invarianten formulieren

Notiere vor der Codelektüre:

1. Welche Beziehung muss zwischen Länge und Kapazität immer gelten?
2. Welche Pufferpositionen enthalten gültige Elemente?
3. Was muss nach einem Resize unverändert bleiben?
4. Warum darf Delete die Reihenfolge der verbleibenden Werte nicht verändern?

Prüfe deine Aussagen anschließend an **dynamic_array.py**.

### 2. Append und Resize tracen

Starte mit Kapazität 1 und füge A bis I ein. Protokolliere nach jedem append:

| append | Länge | Kapazität | Resize? | kopierte Elemente |
|---|---:|---:|---|---:|
| A |  |  |  |  |
| B |  |  |  |  |
| C |  |  |  |  |
| D |  |  |  |  |
| E |  |  |  |  |
| F |  |  |  |  |
| G |  |  |  |  |
| H |  |  |  |  |
| I |  |  |  |  |

Kontrolliere deine Tabelle mit der Eigenschaft **growth_events**.

### 3. Insert von Hand simulieren

Gegeben sei:

~~~text
buffer   = [A, B, C, D, _, _, _, _]
length   = 4
capacity = 8
~~~

Führe **insert(1, X)** aus. Notiere die Reihenfolge der Verschiebungen und
erkläre, warum von rechts nach links kopiert werden muss.

### 4. Delete von Hand simulieren

Führe danach **delete(2)** aus. Notiere:

- den zurückgegebenen Wert,
- jede Linksverschiebung,
- die neue Länge,
- den Inhalt der freigewordenen letzten aktiven Pufferposition.

### 5. Implementierung nachvollziehen

Analysiere diese Methoden in Reihenfolge:

1. **_make_buffer**
2. **_resize**
3. **_ensure_capacity**
4. **append**
5. **insert**
6. **delete**
7. **_normalize_index**

Erkläre für jede Methode Zeit- und Zusatzspeicherkomplexität.

## Wachstumsexperiment

~~~bash
python3 growth_experiment.py
~~~

Der Standardlauf führt 64 appends aus und erzeugt:

- **results/growth_log.csv** mit Länge, Kapazität, Kopierkosten und
  kumuliertem Durchschnitt,
- **results/capacity_and_costs.png** mit Kapazitätstreppe und Kostenspitzen.

Das Kostenmodell zählt:

- 1 Einheit für das Schreiben des neuen Elements,
- 1 Einheit für jedes bei einem Resize kopierte alte Element.

Damit besitzt ein normaler append Kosten 1. Ein append von Länge 32 auf 33
kostet bei Verdopplung 33 Einheiten: 32 Kopien plus eine neue Zuweisung.

Eigene Konfiguration:

~~~bash
python3 growth_experiment.py --appends 256 --initial-capacity 2
~~~

## Tests

~~~bash
python3 -m unittest -v test_dynamic_array.py
~~~

Die Tests prüfen:

- den festen ctypes-Puffer,
- Append und exakte Verdopplungen,
- Indexzugriff, Zuweisung und negative Indizes,
- Insert an Anfang, Mitte und Ende,
- Delete samt Reihenfolge und Rückgabewert,
- alle Bounds-Fehler,
- gemischte Python-Objekte,
- exakte Resize-Ereignisse,
- die geometrische Schranke der Gesamtkopien,
- Experimentdaten und CSV-Export.

## Erwartete Komplexitäten

Fülle die Tabelle vor dem Blick in **LOESUNG.md**:

| Operation | Best Case | Worst Case | amortisiert | Auxiliary Space |
|---|---:|---:|---:|---:|
| Indexzugriff |  |  |  |  |
| append |  |  |  |  |
| insert(0, x) |  |  |  |  |
| insert(length, x) |  |  |  |  |
| delete(0) |  |  |  |  |
| delete(length - 1) |  |  |  |  |
| Resize |  |  |  |  |

## Interpretation

Beantworte nach Experiment und Tests:

1. An welchen append-Nummern treten Resizes auf?
2. Warum besitzen die Kostenspitzen Zweierpotenzen als Abstand?
3. Gegen welchen Wert bewegt sich der kumulierte Durchschnitt?
4. Weshalb ist amortisierte Analyse keine Average-Case-Analyse?
5. Warum schrumpft die Implementierung nach Delete nicht automatisch?
6. Was wäre problematisch, wenn Insert von links nach rechts verschöbe?
7. Welche Unterschiede zu einer vollständigen Python-Liste bleiben bestehen?

## Hinweise

- Kopiere beim Resize nur die aktiven Positionen von 0 bis length - 1.
- Insert verschiebt von rechts nach links, Delete von links nach rechts.
- Aktualisiere length erst, nachdem die notwendige Verschiebung abgeschlossen
  ist.
- Setze die frei gewordene Position nach Delete auf None, damit keine
  überflüssige Referenz gehalten wird.
- Kapazität und logische Länge sind verschiedene Größen.
- growth_events wird als Tupel zurückgegeben, damit Aufrufer das interne
  Protokoll nicht verändern können.

## Fertig, wenn …

- du die vier Kerninvarianten des Arrays in eigenen Worten formuliert hast,
- du Append, Insert, Delete und Resize ohne Python-Listen als Elementspeicher
  nachvollziehen kannst,
- alle Unit Tests erfolgreich laufen,
- das Wachstumsexperiment CSV und Plot erzeugt,
- du die Resize-Punkte und Kopiermengen für mindestens 16 appends korrekt
  vorhersagen kannst,
- du aus der geometrischen Summe erklärst, warum append amortisiert
  \(\Theta(1)\) kostet,
- du die linearen Verschiebungskosten von Insert und Delete begründen kannst.
