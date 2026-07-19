# 01-basic — Muster-Grundübungen

## Ziel

Dieses Projekt trainiert drei Two-Pointers- und drei Sliding-Window-Aufgaben. Zu
jeder Musterlösung existiert eine bewusst einfache Referenz. Ergebnisvergleiche
und Laufzeitmessungen zeigen, welche Struktur die wiederholte Arbeit vermeidet.

Python-Skripte sind hier sinnvoll, weil dieselben Implementierungen für kleine
Randfalltests, reproduzierbare Property-Fälle und größere Benchmarks verwendet
werden können.

## Aufgabenstellung

### Two Pointers

1. Finde eine Zielsumme im sortierten Array mit gegenläufigen Zeigern.
2. Bestimme den Wassercontainer mit maximaler Fläche.
3. Filtere eine Liste stabil und in-place mit Lese-/Schreibzeiger.

### Sliding Window

1. Berechne alle Summen eines Fensters fester Breite.
2. Finde den frühesten längsten Substring ohne Duplikate.
3. Finde das kürzeste Fenster positiver Zahlen mit Summe mindestens `target`.

Implementiere jeweils eine Referenzlösung, prüfe Ergebnisgleichheit und vergleiche
die Laufzeit auf einem Seed-basierten Standardfall.

## Dateien

- `patterns.py`: sechs Referenz-/Musterpaare
- `test_patterns.py`: Randfall- und 2.400 reproduzierbare Property-Fälle
- `run_benchmark.py`: sechs vollständige Median-Benchmarks
- `output/benchmark.csv`: maschinenlesbare Messwerte
- `output/REPORT.md`: Interpretation der Messungen

## Hinweise

Die Paarsummenlösung setzt eine nicht absteigend sortierte Eingabe voraus. Eine
lineare Sortierungsprüfung ist absichtlich nicht Teil jeder Suche, weil sie die
Messung und den Schnittstellenvertrag vermischen würde.

Beim Wassercontainer darf nur die kleinere Begrenzung verworfen werden. Eine
Bewegung der größeren Seite verkürzt den Abstand, ohne die begrenzende Höhe zu
verbessern.

Das variable Summenfenster ist nur für positive Werte implementiert. Dadurch
sinkt die Summe beim Schrumpfen garantiert. Negative Werte würden dieses
Monotonieargument zerstören.

Die In-Place-Filterung ist wie ihre Kopierreferenz `O(n)`. Ihr Vorteil ist
`O(1)` statt `O(n)` zusätzlicher Ergebnisspeicher, nicht zwingend eine kürzere
Python-Laufzeit.

## Ausführen

```bash
python3 -m pytest -q
python3 run_benchmark.py
```

## Fertig, wenn …

- alle sechs Musterlösungen für bekannte Randfälle korrekt sind,
- jede optimierte Lösung mit ihrer Referenz auf Seed-basierten Zufallsfällen
  übereinstimmt,
- Paarsummenindizes verschieden sind und wirklich das Ziel ergeben,
- die In-Place-Filterung Listenidentität und Reihenfolge erhält,
- ungültige Höhen, Fensterbreiten und positive Summenverträge abgewiesen werden,
- der Benchmark vor jeder Messung Ergebnisgleichheit bestätigt,
- `benchmark.csv` und `REPORT.md` alle sechs Aufgaben enthalten,
- alle Tests und der vollständige Benchmark erfolgreich laufen.
