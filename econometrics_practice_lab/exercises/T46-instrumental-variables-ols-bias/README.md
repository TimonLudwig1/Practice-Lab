# T46 – Warum OLS bei Weiterbildung verzerrt ist

## Ausgangslage

Beschäftigte wählen ihre Weiterbildungsstunden teilweise nach unbeobachteter Motivation. Motivation erhöht zugleich das Einkommen. OLS schreibt dadurch einen Teil des Motivationseffekts fälschlich der Weiterbildung zu. Eine zufällig versandte Einladung verändert die Weiterbildungsstunden, aber nicht direkt das Einkommen und dient als Instrument.

## Lernziele

- Endogenität und ihre Wirkung auf OLS erklären,
- Relevanz und Exogenität eines Instruments empirisch und kausal unterscheiden,
- OLS, Oracle-Regression und einfachen IV-Schätzer vergleichen,
- erkennen, warum ein relevantes, aber nicht exogenes Instrument nicht genügt,
- die Richtung des OLS-Bias mit der ausgelassenen Variable begründen.

## Aufgaben

1. Schätze Einkommen auf Weiterbildungsstunden mit OLS.
2. Nutze die in der Simulation beobachtbare Motivation in einer Oracle-Regression.
3. Prüfe, ob die zufällige Einladung die Weiterbildungsstunden verändert.
4. Berechne den IV-Schätzer als Reduced-Form-Koeffizient geteilt durch First-Stage-Koeffizient.
5. Vergleiche Mittelwertsunterschiede in Weiterbildung und Motivation nach Instrumentstatus.
6. Wiederhole die IV-Schätzung mit einem absichtlich ungültigen Instrument, das mit Motivation zusammenhängt.
7. Formuliere für beide Instrumente eine Begründung zu Relevanz, Exogenität und Exclusion Restriction.

## Ausführen

```bash
python3 exercises/T46-instrumental-variables-ols-bias/starter.py
python3 exercises/T46-instrumental-variables-ols-bias/solution.py
```

Die Lösung erzeugt `data/training_iv_data.csv`, Ergebnistabellen und `results/iv_vs_ols_bias.png`.

## Denkfragen

- Warum kontrolliert eine große Stichprobe Endogenität nicht automatisch weg?
- Kann ein Instrument stark mit dem Treatment korrelieren und trotzdem ungültig sein?
- Welche direkten Wirkungswege der Einladung würden die Exclusion Restriction verletzen?
