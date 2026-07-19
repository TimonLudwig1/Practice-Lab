# 02-medium — Fenster mit Zustand

## Ziel

Dieses Projekt vertieft variable und feste Sliding Windows mit drei
unterschiedlichen Zustandsmodellen: offene Zeichenbedarfe, Anzahl ungültiger
Elemente und eine Frequenzdifferenz. Jede Invariante wird dokumentiert, über
strukturierte Traces sichtbar gemacht und gegen eine unabhängige Brute-Force-
Referenz geprüft.

Python-Skripte sind geeignet, weil Counter-/Hash-Map-Zustände, Randfälle und
reproduzierbare Property-Tests direkt ausführbar bleiben.

## Aufgabenstellung

1. Finde das kürzeste Fenster, das alle Zielzeichen einschließlich
   Mehrfachvorkommen enthält.
2. Finde das längste Binärfenster, das mit höchstens `k` Null-zu-Eins-Flips nur
   Einsen enthalten könnte.
3. Finde alle Startindizes fester Fenster, die Anagramme eines Musters sind.

Implementiere für jede Aufgabe eine Brute-Force-Referenz, eine lineare
Fensterlösung und optionale unveränderliche Zustands-Traces.

## Dateien

- `stateful_windows.py`: drei Referenz-/Fensterpaare und Trace-Datentyp
- `demo.py`: Zustandsübergänge der drei Standardbeispiele
- `test_stateful_windows.py`: Randfall-, Invarianten- und Property-Tests

## Zustandsinvarianten

### Minimales Deckungsfenster

`need[c]` ist Sollhäufigkeit minus aktuelle Fensterhäufigkeit. Negative Werte
bedeuten Überschuss. `missing` ist die Summe aller positiven Defizite. Genau
wenn `missing == 0`, deckt das Fenster das Ziel vollständig ab. Während dieses
Zustands wird links geschrumpft und vor jedem Entfernen das beste Fenster geprüft.

### Maximale Einsen mit Flips

Nach der inneren `while`-Schleife enthält `[left, right]` höchstens `k` Nullen.
Es ist der längste gültige Suffix mit aktuellem `right`; daher darf seine Länge
direkt in das Maximum eingehen. Jeder Index verlässt das Fenster höchstens einmal.

### Anagrammfenster

`balance[c]` ist Fensterhäufigkeit minus Sollhäufigkeit. `nonzero` zählt Zeichen
mit einer Abweichung. Ein Fenster der festen Musterbreite ist genau dann ein
Anagramm, wenn `nonzero == 0`. Ein- und Austritt verändern jeweils nur einen
Map-Eintrag in erwarteter `O(1)`-Zeit.

## Hinweise

Beim Minimalfenster reicht ein Set nicht aus: Das Ziel `AABC` benötigt zwei
Vorkommen von `A`. Häufigkeiten sind Teil des Vertrags.

Beim Flip-Fenster muss `while zeros > k` verwendet werden. Ein einzelnes
Entfernen kann mehrere überschüssige Nullen nicht sicher beseitigen.

Beim Anagrammfenster werden Nullsalden aus der Map entfernt. `nonzero` vermeidet
einen vollständigen Countervergleich in jedem Schritt.

## Ausführen

```bash
python3 -m pytest -q
python3 demo.py
```

## Fertig, wenn …

- doppelte Zielzeichen und nicht erfüllbare Minimalfenster korrekt behandelt
  werden,
- das Flip-Fenster für `k = 0`, große `k` und leere Eingaben stimmt,
- überlappende und Unicode-Anagramme gefunden werden,
- alle optimierten Ergebnisse auf 2.200 Seed-basierten Fällen mit ihren
  Referenzen übereinstimmen,
- Trace-Tests die dokumentierten Zustandsinvarianten direkt nachrechnen,
- ungültige Binärwerte und Flipbudgets abgewiesen werden,
- alle Tests und die vollständige Drei-Zustände-Demo erfolgreich laufen.
