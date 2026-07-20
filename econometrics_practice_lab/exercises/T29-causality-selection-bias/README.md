# T29 – Selbstselektion in ein Weiterbildungsprogramm

## Ausgangslage

Beschäftigte entscheiden selbst, ob sie an einer Weiterbildung teilnehmen. Motivierte und höher qualifizierte Personen melden sich häufiger an und hätten auch ohne Programm bessere Leistungsergebnisse. Ein einfacher Vergleich von Teilnehmenden und Nichtteilnehmenden vermischt deshalb Treatment-Effekt und Selektionsbias.

Der synthetische Datensatz enthält zu Lernzwecken beide potenziellen Outcomes. In realen Daten wäre pro Person nur eines davon beobachtbar.

## Lernziele

- Korrelation und kausalen Effekt unterscheiden,
- die Fundamental Problem of Causal Inference mit \(Y(0)\) und \(Y(1)\) erklären,
- ATE und ATT berechnen und unterscheiden,
- den naiven Gruppenvergleich in ATT und Selektionsbias zerlegen,
- Richtung und Größe des Bias aus dem Auswahlmechanismus ableiten.

## Aufgaben

1. Erzeuge `observed_performance = treatment*Y(1) + (1-treatment)*Y(0)` und prüfe diese Konsistenz.
2. Berechne ATE, ATT und die naive Mittelwertsdifferenz beobachteter Outcomes.
3. Berechne \(E[Y(0)|D=1]-E[Y(0)|D=0]\) als Selektionsbias.
4. Verifiziere die Identität: naive Differenz = ATT + Selektionsbias.
5. Untersuche Teilnahmequote, Motivation und Bildungsjahre nach Treatmentstatus.
6. Stelle dar, wie die Teilnahme mit Motivation und der individuelle Treatment-Effekt mit Motivation steigen.
7. Erkläre, warum ein statistisch präziser naiver Vergleich trotzdem keine glaubwürdige kausale Schätzung ist.

## Ausführen

```bash
python3 exercises/T29-causality-selection-bias/starter.py
python3 exercises/T29-causality-selection-bias/solution.py
```

Die Lösung erzeugt `data/voluntary_training.csv`, drei Ergebnistabellen und `results/selection_bias.png`.

## Denkfragen

- Welches unbeobachtete Counterfactual fehlt bei einer behandelten Person?
- Weshalb sind ATE und ATT bei heterogenen Effekten verschieden?
- In welche Richtung wäre der Bias, wenn vor allem Personen mit schlechten unbehandelten Outcomes teilnehmen?
