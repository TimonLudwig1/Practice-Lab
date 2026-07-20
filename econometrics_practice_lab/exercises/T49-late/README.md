# T49 – LATE bei freiwilliger Programmteilnahme

## Ausgangslage

Eine zufällige Einladung zu einem Beschäftigungsprogramm führt nicht bei allen Menschen zur Teilnahme. Always-Takers nehmen unabhängig von der Einladung teil, Never-Takers nie und Compliers nur bei Einladung. Da die Treatment-Wirkung zwischen diesen Typen variiert, identifiziert der Wald-Schätzer nicht automatisch den Population ATE.

## Lernziele

- Always-Takers, Never-Takers, Compliers und Defiers definieren,
- Compliance-Anteile aus Take-up-Raten herleiten,
- Random Assignment, Exclusion Restriction, Relevanz und Monotonie für LATE einordnen,
- den Wald-Schätzer als Local Average Treatment Effect interpretieren,
- LATE von ATE und naivem Treatmentvergleich unterscheiden,
- verstehen, für welche Personen der IV-Effekt lokal ist.

## Aufgaben

1. Berechne `P(D=1|Z=0)` und `P(D=1|Z=1)`.
2. Leite unter Monotonie die Anteile der drei beobachtbaren Compliance-Typen her.
3. Schätze First Stage, Reduced Form und Wald-Ratio.
4. Vergleiche die Ratio mit dem wahren mittleren Effekt der Compliers.
5. Berechne in der Simulation außerdem Population ATE und naiven Treatmentvergleich.
6. Prüfe mit den potenziellen Treatmentzuständen, ob Defiers existieren.
7. Erkläre, warum Wirkungen für Always- und Never-Takers nicht durch das Instrument identifiziert werden.

## Ausführen

```bash
python3 exercises/T49-late/starter.py
python3 exercises/T49-late/solution.py
```

Die Lösung erzeugt `data/compliance_experiment.csv`, Ergebnistabellen und `results/late_compliance_types.png`.

## Denkfragen

- Warum ist LATE trotz seiner lokalen Interpretation ein kausaler Parameter?
- Wann wäre der LATE gleich dem Population ATE?
- Was geht schief, wenn manche Menschen gerade wegen einer Einladung nicht teilnehmen würden?
