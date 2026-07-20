# T48 – Reduced Form einer Jobberatung

## Ausgangslage

Eine zufällige Einladung erhöht die Zahl wahrgenommener Jobberatungssitzungen. Sitzungen sind endogen, weil besonders engagierte Personen häufiger teilnehmen und zugleich bessere Bewerbungsresultate erzielen. Die Reduced Form misst zunächst nur den Gesamteffekt der Einladung auf das Outcome – unabhängig davon, über wie viele Sitzungen dieser Effekt vermittelt wird.

## Lernziele

- First Stage und Reduced Form inhaltlich und in ihren Einheiten unterscheiden,
- die Reduced Form als Intention-to-Treat-Effekt des Instruments interpretieren,
- den Wald-Schätzer als Reduced Form geteilt durch First Stage bilden,
- die Identität `Reduced Form = First Stage × IV` nachvollziehen,
- Unsicherheit der beiden Komponenten per Bootstrap visualisieren.

## Aufgaben

1. Berechne Sitzungs- und Outcome-Mittelwerte für eingeladene und nicht eingeladene Personen.
2. Schätze die First Stage `sessions ~ invitation`.
3. Schätze die Reduced Form `job_score ~ invitation`.
4. Interpretiere beide Koeffizienten mit ihren jeweiligen Einheiten.
5. Berechne den Wald-IV-Schätzer und prüfe die Produktidentität.
6. Ziehe Bootstrap-Stichproben und speichere First Stage, Reduced Form und Ratio.
7. Erkläre, warum ein Reduced-Form-Effekt allein noch kein Effekt je Sitzung ist.

## Ausführen

```bash
python3 exercises/T48-reduced-form/starter.py
python3 exercises/T48-reduced-form/solution.py
```

Die Lösung erzeugt `data/job_counseling_data.csv`, Ergebnistabellen und `results/reduced_form_decomposition.png`.

## Denkfragen

- Kann die Reduced Form null sein, obwohl die Behandlung kausal wirkt?
- Was bedeutet ein negatives Vorzeichen in der First Stage für die Vorzeichen des Ratios?
- Welche direkten Einladungseffekte würden die Vermittlungsinterpretation zerstören?
