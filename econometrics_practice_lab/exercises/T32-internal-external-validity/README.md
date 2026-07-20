# T32 – Übertragbarkeit einer Gesundheitsstudie

## Ausgangslage

Ein Gesundheitsprogramm wird innerhalb einer freiwillig rekrutierten Studiengruppe randomisiert. Die Randomisierung erlaubt eine intern glaubwürdige Wirkungsschätzung für die Teilnehmenden. Die Teilnehmenden unterscheiden sich jedoch in Alter, Ausgangsbelastung und Wohnort von der Zielpopulation, und die Wirkung ist heterogen. Dadurch kann die Übertragung des Studienergebnisses verzerrt sein.

## Lernziele

- interne Validität von externer Validität unterscheiden,
- den Sample ATE vom Population ATE trennen,
- erkennen, dass Randomisierung innerhalb der Studie die Stichprobenselektion nicht beseitigt,
- Effekt-Heterogenität als Voraussetzung eines External-Validity-Problems verstehen,
- inverse Teilnahmewahrscheinlichkeiten zur Generalisierung einsetzen und kritisch prüfen.

## Aufgaben

1. Vergleiche Zielpopulation und Studienteilnehmende hinsichtlich Alter, Belastung und ländlichem Wohnort.
2. Berechne den wahren Population ATE und den wahren Sample ATE aus den simulierten potenziellen Outcomes.
3. Schätze den Trial-Effekt mit der randomisierten Mittelwertsdifferenz.
4. Zerlege den Fehler gegenüber dem Population ATE in internen Schätzfehler und externe Übertragungslücke.
5. Generalisiere die randomisierte Schätzung mit Gewichten \(1/P(S=1|X)\).
6. Untersuche die Gewichtsverteilung und erkläre das Positivitätsproblem.
7. Zeige grafisch, welche Gruppen überrepräsentiert sind und wie der Treatment-Effekt mit der Ausgangsbelastung variiert.
8. Ordne Attrition, Noncompliance, Messfehler und selektive Rekrutierung jeweils interner oder externer Validität zu.

## Ausführen

```bash
python3 exercises/T32-internal-external-validity/starter.py
python3 exercises/T32-internal-external-validity/solution.py
```

Die Lösung erzeugt `data/health_trial_population.csv`, Ergebnistabellen und `results/internal_external_validity.png`.

## Denkfragen

- Kann eine Studie intern valide und gleichzeitig schlecht generalisierbar sein?
- Wann wären unterschiedliche Kovariatenverteilungen für die Übertragbarkeit harmlos?
- Weshalb können sehr große Generalisierungsgewichte die Schätzung instabil machen?
