# 02-medium — Suche auf der Antwort

## Ziel

Dieses Projekt löst drei äußerlich verschiedene Aufgaben mit derselben
algorithmischen Idee: Der Suchraum besteht aus möglichen Antworten, und eine
monotone Ja-Nein-Funktion trennt unmögliche von möglichen Kandidaten.

Python-Skripte eignen sich hier, weil die gemeinsame Suchschablone, die drei
Prädikate und ihre Intervall-Traces direkt nebeneinander getestet werden können.

## Aufgabenstellung

Implementiere eine wiederverwendbare Suche nach dem ersten wahren ganzzahligen
Kandidaten und löse damit:

1. **Minimale Versandkapazität:** Pakete bleiben in Eingabereihenfolge und müssen
   innerhalb einer vorgegebenen Zahl Tage verschickt werden.
2. **Ganzzahlige Quadratwurzel:** Bestimme `floor(sqrt(n))` ohne `math.sqrt` und
   ohne Gleitkommaarithmetik.
3. **k-kleinstes Matrixelement:** Die Zeilen und Spalten einer Matrix sind
   aufsteigend sortiert; gesucht ist ein Rang, ohne die Matrix zu flatten und
   vollständig zu sortieren.

Jede Lösung soll strukturierte Schritte der Antwortsuche ausgeben können. Die
Tests müssen neben dem Ergebnis auch die Minimalität der Antwort und den
strikten Fortschritt des Suchintervalls prüfen.

## Dateien

- `answer_search.py`: gemeinsame `first_true`-Schablone und Trace-Datensatz
- `problems.py`: drei Problemreduktionen und ihre Machbarkeitsprüfungen
- `demo.py`: vollständige Suchspuren der Standardszenarien
- `test_answer_search.py`: Referenz-, Randfall-, Monotonie- und Property-Tests

## Warum sind die Prädikate monoton?

### Versandkapazität

Wenn Kapazität `c` ausreicht, kann jede größere Kapazität dieselbe Tageseinteilung
verwenden. Größere Kapazität kann die Zahl benötigter Tage daher nicht erhöhen.
Gesucht ist die erste Kapazität mit `required_days <= day_limit`.

### Quadratwurzel

Für nichtnegative ganzzahlige Kandidaten ist `candidate² > n` zunächst falsch
und ab einem Punkt dauerhaft wahr. Die Suche liefert den ersten zu großen Wert;
sein Vorgänger ist `floor(sqrt(n))`.

### Matrixrang

Mit wachsendem Kandidaten kann die Anzahl der Matrixwerte `<= candidate` nie
sinken. Der erste Kandidat, für den mindestens `k` Werte gezählt werden, ist das
k-kleinste Element. Die Zählung startet links unten und benötigt aufgrund der
Zeilen- und Spaltenordnung nur `O(rows + columns)` Schritte.

## Komplexität

- Versand: `O(n · log(sum(weights) - max(weights) + 1))`
- Quadratwurzel: `O(log n)`
- Matrixrang: `O((rows + columns) · log(max_value - min_value + 1))`

Die einmalige Matrixvalidierung kostet zusätzlich `O(rows · columns)`. Sie ist
Teil der robusten öffentlichen Schnittstelle; die eigentliche Prädikatsprüfung
bleibt linear in den Matrixdimensionen.

## Hinweise

`first_true` verlangt eine garantiert machbare obere Grenze. Für den Versand ist
die Summe aller Gewichte machbar, für die Wurzel ist `n + 1` sicher zu groß und
für den Matrixrang ist das größte Matrixelement immer ausreichend.

Ein Ergebnis ist nicht allein dadurch korrekt, dass sein Prädikat wahr ist. Für
eine minimale Antwort muss der direkte Vorgänger falsch sein, sofern er noch im
Suchraum liegt.

## Ausführen

```bash
python3 -m pytest -q
python3 demo.py
```

## Fertig, wenn …

- die gemeinsame Suche den ersten wahren Kandidaten an beiden Intervallrändern
  und im Inneren findet,
- die Versandlösung gegen eine erschöpfende Kapazitätssuche geprüft ist,
- die Quadratwurzel für große Ganzzahlen mit `math.isqrt` übereinstimmt,
- das Matrixergebnis mit einer nur im Test verwendeten sortierten Flachliste
  übereinstimmt,
- jede Problem-README-Begründung tatsächlich ein False→True-Prädikat beschreibt,
- Trace-Tests strikten Fortschritt, Minimalität und Prädikatsergebnisse bestätigen,
- ungültige Ressourcengrenzen, Zahlen, Matrizen und Ränge abgewiesen werden,
- alle Tests und die vollständige Drei-Probleme-Demo erfolgreich laufen.
