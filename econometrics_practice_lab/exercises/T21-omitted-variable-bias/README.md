# T21 – Omitted Variable Bias bei Bildungsrenditen

## Ausgangslage

In synthetischen Lohndaten beeinflusst eine Fähigkeitsskala sowohl die erreichten Bildungsjahre als auch den späteren Stundenlohn. Ein kurzes Modell ohne Fähigkeit verwechselt daher einen Teil des Fähigkeitseffekts mit dem Bildungskoeffizienten. Da der wahre Datengenerierungsprozess bekannt ist, lässt sich die Verzerrung exakt zerlegen.

## Lernziele

Nach dem Projekt kannst du:

- die zwei notwendigen Bedingungen für Omitted Variable Bias nennen,
- die Richtung des Bias aus Vorzeichenargumenten ableiten,
- kurzes und langes Regressionsmodell vergleichen,
- die OVB-Formel empirisch mit einer Hilfsregression prüfen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Schätze das kurze Modell `log_hourly_wage ~ education_years`.
2. Schätze das lange Modell zusätzlich mit `ability_score`.
3. Vergleiche den Bildungskoeffizienten mit dem wahren Datengenerierungswert von 0,08.
4. Prüfe beide OVB-Bedingungen: Fähigkeit beeinflusst den Lohn und korreliert mit Bildung.
5. Leite vor der Rechnung die erwartete Bias-Richtung aus den beiden Vorzeichen ab.
6. Schätze die Hilfsregression `ability_score ~ education_years` und nenne ihre Steigung \(\hat\delta\).
7. Prüfe die endliche Stichprobenidentität
   \(\hat\beta_{short}-\hat\beta_{long}=\hat\gamma_{ability}\hat\delta\).
8. Erkläre, warum „mehr Kontrollvariablen“ nicht automatisch besser ist und warum Post-Treatment-Variablen problematisch sein können.

## Ausführen

```bash
python3 exercises/T21-omitted-variable-bias/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/model_comparison.csv`, `results/ovb_decomposition.csv` und `results/omitted_variable_bias.png`:

```bash
python3 exercises/T21-omitted-variable-bias/solution.py
```

## Denkfragen

- Wie würde sich die Bias-Richtung ändern, wenn Fähigkeit den Lohn positiv, Bildung aber negativ beeinflusste?
- Muss eine ausgelassene Variable beobachtbar sein, damit sie Bias erzeugt?
- Weshalb ist der lange Bildungskoeffizient nur dann kausal, wenn keine weiteren relevanten Endogenitätsprobleme verbleiben?
