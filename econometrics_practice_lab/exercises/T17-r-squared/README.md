# T17 – R² als Varianzzerlegung

## Ausgangslage

Bei 240 Lernenden wird ein Testscore durch wöchentliche Lernstunden erklärt. Zusätzlich steht eine vollständig irrelevante Zufallsvariable zur Verfügung. Du zerlegst die Gesamtvariation des Scores und beobachtest, warum gewöhnliches R² beim Hinzufügen eines Regressors nicht sinkt.

## Lernziele

Nach dem Projekt kannst du:

- TSS, ESS und RSS aus Beobachtungen, Fits und Residuen berechnen,
- \(R^2=1-RSS/TSS=ESS/TSS\) herleiten und prüfen,
- R² und adjustiertes R² unterscheiden,
- Modellgüte von Kausalität und Vorhersage außerhalb der Stichprobe abgrenzen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Schätze `exam_score ~ study_hours` mit Intercept.
2. Berechne \(TSS=\sum_i(y_i-\bar y)^2\), \(RSS=\sum_i(y_i-\hat y_i)^2\) und \(ESS=\sum_i(\hat y_i-\bar y)^2\).
3. Prüfe numerisch \(TSS=ESS+RSS\) und beide Formeln für R².
4. Interpretiere R² als Anteil der Stichprobenvariation von `exam_score`, der durch den linearen Fit zu `study_hours` erklärt wird.
5. Schätze zusätzlich ein Intercept-only-Modell und ein Modell mit `study_hours` plus `irrelevant_noise`.
6. Vergleiche gewöhnliches und adjustiertes R². Erkläre, weshalb gewöhnliches R² durch den irrelevanten Regressor nicht sinken kann.
7. Prüfe, ob das adjustierte R² die zusätzliche Variable in dieser Stichprobe belohnt oder bestraft.
8. Erkläre, warum ein hohes R² weder Kausalität noch gute Out-of-Sample-Prognosen garantiert.

## Ausführen

```bash
python3 exercises/T17-r-squared/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/r_squared_decomposition.csv` und `results/r_squared_visual.png`:

```bash
python3 exercises/T17-r-squared/solution.py
```

## Denkfragen

- Welches R² hat ein Intercept-only-Modell in derselben Stichprobe?
- Kann R² negativ werden, wenn ein Modell ohne Intercept geschätzt oder außerhalb der Schätzstichprobe bewertet wird?
- Ist ein niedriges R² bei der Schätzung eines kausalen Effekts automatisch ein Problem?
