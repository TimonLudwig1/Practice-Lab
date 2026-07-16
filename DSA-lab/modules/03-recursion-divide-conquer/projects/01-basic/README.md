# 01-basic — Rekursions-Visualizer

## Ziel

Dieses Projekt macht den unsichtbaren Call Stack sichtbar. Du implementierst vier
kleine rekursive Funktionen und einen Decorator, der jeden Eintritt, jede Rückgabe
und jeden Fehler mit der aktuellen Rekursionstiefe protokolliert. Das Ergebnis ist
ein Call Tree im Terminal, den du Zeile für Zeile mit einem manuellen Stack-Trace
vergleichen kannst.

Python-Skripte mit `unittest` sind hier das passende Format: Terminalausgabe lässt
sich exakt abfangen und prüfen, während die Algorithmen weiterhin direkt
ausführbar bleiben.

## Aufgaben

### 1. Universellen Trace-Decorator bauen

Implementiere:

```python
@trace_calls
def example(...):
    ...
```

Der Decorator soll:

- den Funktionsnamen und alle Argumente beim Eintritt ausgeben,
- zwei Leerzeichen pro aktiver Aufrufebene einrücken,
- den Rückgabewert beim Aufstieg ausgeben,
- Exceptions sichtbar machen und unverändert weiterwerfen,
- nach Fehlern die Tiefe zuverlässig zurücksetzen,
- Metadaten wie `__name__` und `__doc__` erhalten,
- auch korrekt einrücken, wenn verschiedene dekorierte Funktionen einander
  aufrufen.

Beispiel:

```text
→ factorial(2)
  → factorial(1)
    → factorial(0)
    ← factorial(0) = 1
  ← factorial(1) = 1
← factorial(2) = 2
```

Hinweis: Ein `ContextVar` kann die Tiefe kapseln, ohne eine globale veränderliche
Ganzzahl manuell verwalten zu müssen.

### 2. Fakultät

```python
factorial(n) -> int
```

- `0!` ist `1`.
- Der Rekursionsfall reduziert `n` um eins.
- Negative, nichtganzzahlige und boolesche Eingaben werden abgewiesen.

### 3. Fibonacci

```python
fibonacci(n) -> int
```

Implementiere bewusst die direkte verzweigte Rekurrenz. Sie ist für große Werte
ineffizient, zeigt aber im Trace wiederholte Teilprobleme besonders deutlich.

### 4. Rekursive Summe

```python
recursive_sum(values, index=0) -> Number
```

Verwende einen fortschreitenden Index statt `values[1:]`. Slicing würde in jedem
Frame eine neue Sequenz erzeugen und die Gesamtkosten unnötig von O(n) auf O(n²)
erhöhen.

### 5. Rekursive Potenz

```python
power(base, exponent) -> Number
```

Diese Einstiegsvariante reduziert den Betrag eines ganzzahligen Exponenten linear.
Negative Exponenten werden als Kehrwert der positiven Potenz behandelt. Die
schnelle binäre Variante folgt im nächsten Projekt.

## Ausführen

Alle Tests:

```bash
python3 -m unittest -v test_recursion_visualizer.py
```

Alle vier Call Trees:

```bash
python3 demo.py
```

Es werden ausschließlich Module der Python-Standardbibliothek benötigt.

## Beobachtungsaufgaben

1. Markiere im Fakultäts-Trace den tiefsten Zeitpunkt des Call Stacks.
2. Zähle bei `fibonacci(4)`, wie oft `fibonacci(2)` ausgeführt wird.
3. Vergleiche die Reihenfolge der Eintritts- und Rückgabezeilen.
4. Erkläre, warum die Ergebniszeilen von innen nach außen erscheinen.
5. Provoziere eine ungültige Eingabe und prüfe, ob der nächste Top-Level-Aufruf
   wieder ohne Einrückung beginnt.

## Projektstruktur

```text
01-basic/
├── README.md
├── LOESUNG.md
├── recursion_visualizer.py
├── demo.py
└── test_recursion_visualizer.py
```

## Fertig, wenn …

- der Decorator Eintritt, Rückgabe und Exceptions korrekt visualisiert,
- verschachtelte Aufrufe exakt zwei Leerzeichen pro Ebene erhalten,
- Fakultät, Fibonacci, Summe und Potenz korrekte Ergebnisse liefern,
- jede Funktion einen erreichbaren Basisfall und messbaren Fortschritt besitzt,
- ungültige Eingaben klar definierte Exceptions auslösen,
- alle Tests und die vollständige Demo fehlerfrei laufen,
- du einen erzeugten Trace als Call-Stack-Abstieg und -Aufstieg erklären kannst.
