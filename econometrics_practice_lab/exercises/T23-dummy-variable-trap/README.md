# T23 – Die Dummy-Variable-Trap als Rangproblem

## Ausgangslage

Pendler nutzen Bus, Fahrrad oder Auto. Ein Modell soll Zufriedenheit mit Verkehrsmittel und Pendeldauer erklären. Wenn ein Intercept und alle drei Verkehrsmittel-Dummies gleichzeitig aufgenommen werden, ist eine Spalte exakt aus den anderen konstruierbar. Die Dummy-Variable-Trap ist damit ein lineares Algebra- und Identifikationsproblem.

## Lernziele

Nach dem Projekt kannst du:

- perfekte Multikollinearität in einer Designmatrix erkennen,
- Matrixrang, Singulärwerte und Condition Number diagnostizieren,
- die lineare Abhängigkeit `const = dummy_bus + dummy_bike + dummy_car` erklären,
- zwei gültige Dummy-Parametrisierungen vergleichen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Erzeuge alle drei Verkehrsmittel-Dummies und füge einen Intercept hinzu.
2. Prüfe explizit, dass sich die Dummyzeilen zu eins summieren.
3. Vergleiche Spaltenzahl und Matrixrang; berechne Singulärwerte und Condition Number.
4. Schätze das rangdefiziente Modell mit einer Pseudoinversen. Erkläre, weshalb eine ausgegebene Lösung nicht eindeutige Koeffizienten bedeutet.
5. Addiere eine Konstante zum Intercept und subtrahiere sie von allen Dummy-Koeffizienten. Prüfe, dass die Fits unverändert bleiben.
6. Schätze ein gültiges Modell mit Intercept und `Bus` als ausgelassener Referenz.
7. Schätze ein zweites gültiges Modell ohne Intercept, aber mit allen drei Dummies.
8. Zeige, dass beide gültigen Modelle identische Fits und dieselben adjustierten Gruppenmittel liefern, obwohl ihre Koeffizienten anders heißen.

## Ausführen

```bash
python3 exercises/T23-dummy-variable-trap/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/design_diagnostics.csv`, `results/parameterizations.csv`, `results/adjusted_mode_means.csv` und `results/dummy_trap.png`:

```bash
python3 exercises/T23-dummy-variable-trap/solution.py
```

## Denkfragen

- Warum löst das Entfernen des Intercepts die Trap ebenfalls?
- Welche Bedeutung besitzen die drei Dummy-Koeffizienten im Modell ohne Intercept?
- Ist beinahe perfekte Multikollinearität dasselbe wie die exakte Dummy-Variable-Trap?
