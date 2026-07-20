# T09 – Produktivität und einfache OLS-Regression

## Ausgangslage

Für 250 synthetische Betriebe werden jährliche Trainingsstunden pro Beschäftigten und ein Produktivitätsindex beobachtet. Da die Daten simuliert sind, ist zusätzlich der wahre strukturelle Fehler sichtbar. Damit kannst du OLS nicht nur schätzen, sondern die Gauss–Markov-Annahmen kontrolliert untersuchen.

## Lernziele

Nach dem Projekt kannst du:

- Achsenabschnitt und Steigung einer einfachen OLS-Regression manuell berechnen,
- eine Regressionsgerade als beste lineare Anpassung interpretieren,
- die Gauss–Markov-Annahmen auf Datengenerierung und Stichprobe beziehen,
- erklären, warum Normalität keine Gauss–Markov-Annahme ist.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Visualisiere Produktivität gegen Trainingsstunden und formuliere eine erwartete Richtung der Steigung.
2. Berechne \(\hat\beta_1=\operatorname{Cov}(X,Y)/\operatorname{Var}(X)\) und \(\hat\beta_0=\bar Y-\hat\beta_1\bar X\) manuell.
3. Kontrolliere die Koeffizienten mit `statsmodels.OLS` und interpretiere beide mit ihren Einheiten.
4. Berechne Fits und Residuen. Prüfe Summe und Mittelwert der Residuen sowie ihre Stichprobenkovarianz mit \(X\).
5. Ordne folgende Gauss–Markov-Annahmen dem Experiment zu: linear in Parametern, Zufallsstichprobe, Variation in \(X\), \(E[u\mid X]=0\) und Homoskedastizität.
6. Teile die Trainingsstunden in fünf Gruppen. Untersuche Gruppenmittel und Gruppenvarianz des in der Simulation sichtbaren strukturellen Fehlers.
7. Erkläre, welche Annahmen aus echten Beobachtungsdaten nicht bewiesen werden können.
8. Begründe, warum OLS unter den Annahmen BLUE ist und warum dafür keine normalverteilten Fehler nötig sind.

## Ausführen

```bash
python3 exercises/T09-simple-linear-regression/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/model_summary.csv`, `results/gm_by_training_quintile.csv` und `results/gm_diagnostics.png`:

```bash
python3 exercises/T09-simple-linear-regression/solution.py
```

## Denkfragen

- Was bedeutet „linear“ in „linear in Parametern“?
- Ist \(\hat\beta_1\) automatisch ein kausaler Trainingseffekt?
- Welche Gauss–Markov-Annahme wäre verletzt, wenn motivierte Betriebe zugleich mehr Training und höhere unbeobachtete Produktivität hätten?
