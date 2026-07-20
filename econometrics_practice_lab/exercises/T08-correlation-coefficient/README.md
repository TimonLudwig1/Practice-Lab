# T08 – Der Korrelationskoeffizient unter Stress

## Ausgangslage

Du erhältst drei synthetische Datenszenarien: einen verrauschten linearen Zusammenhang, einen starken U-förmigen Zusammenhang und zwei ansonsten unabhängige Variablen mit einem gemeinsamen Extremwert. Die Szenarien zeigen, was der Pearson-Korrelationskoeffizient zuverlässig misst und wo eine einzelne Zahl wichtige Struktur verdeckt.

## Lernziele

Nach dem Projekt kannst du:

- den Pearson-Korrelationskoeffizienten aus standardisierten Werten berechnen,
- seine Begrenzung auf das Intervall \([-1,1]\) und sein Vorzeichen interpretieren,
- seine Invarianz gegenüber Verschiebung und positiver Skalierung demonstrieren,
- Nichtlinearität und Ausreißer als zentrale Diagnoseprobleme erkennen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Erzeuge die drei Szenarien und zeichne zuerst alle Scatterplots, ohne Korrelationen zu berechnen. Notiere deine qualitative Erwartung.
2. Berechne Pearson-\(r\) im linearen Szenario manuell als mittleres Produkt der z-standardisierten Werte.
3. Zeige numerisch, dass dies identisch zu \(\operatorname{Cov}(X,Y)/(s_Xs_Y)\) ist, und kontrolliere mit `scipy.stats.pearsonr`.
4. Ersetze im linearen Szenario \(X\) durch \(1000X+50\). Vergleiche Kovarianz und Korrelation vor und nach der Transformation.
5. Multipliziere \(X\) anschließend mit \(-1\). Erkläre, weshalb sich nur das Vorzeichen von \(r\) ändert.
6. Berechne \(r\) für den U-förmigen Zusammenhang. Begründe anhand des Scatterplots, warum ein Wert nahe null hier nicht „kein Zusammenhang“ bedeutet.
7. Berechne \(r\) im Ausreißerszenario mit und ohne die markierte letzte Beobachtung. Erkläre die Differenz.
8. Formuliere vier Grenzen einer rein numerischen Korrelationsanalyse, darunter zwingend Nichtlinearität, Ausreißer und fehlende Kausalitätsaussage.

## Ausführen

```bash
python3 exercises/T08-correlation-coefficient/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/correlation_summary.csv` und `results/correlation_pitfalls.png`:

```bash
python3 exercises/T08-correlation-coefficient/solution.py
```

## Denkfragen

- Kann \(r=0\) mit perfekter Vorhersagbarkeit von \(Y\) aus \(X\) vereinbar sein?
- Was bleibt von \(r\) unverändert, wenn Euro in Cent umgerechnet werden?
- Warum sollte vor jeder Korrelationsinterpretation ein Scatterplot stehen?
