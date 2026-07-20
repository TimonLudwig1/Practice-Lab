# T22 – Kategoriale Variablen im Filialmodell

## Ausgangslage

400 Filialen liegen in vier Standorttypen: Zentrum, Vorstadt, Universitätsviertel und Industriegebiet. Du codierst den Standort als Dummy-Variablen, wählst eine Referenzkategorie und vergleichst erwartete Umsätze bei denselben Marketingausgaben.

## Lernziele

Nach dem Projekt kannst du:

- eine kategoriale Variable mit \(K-1\) Dummies codieren,
- Intercept und Dummy-Koeffizienten relativ zur Referenz interpretieren,
- eine Referenzkategorie bewusst wechseln,
- erkennen, dass Reparametrisierung Fits und Vorhersagen nicht verändert.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Prüfe Häufigkeiten und Rohmittelwerte der vier Standorttypen.
2. Zentriere `marketing_thousand_eur` bei 20 und verwende zunächst `Suburb` als Referenz.
3. Erzeuge drei Dummies und schätze Umsatz auf zentriertes Marketing plus Standortdummies.
4. Interpretiere Intercept, Marketingkoeffizient und alle Dummy-Koeffizienten.
5. Berechne für jeden Standort den erwarteten Umsatz bei 20.000 Euro Marketing samt 95%-Konfidenzintervall.
6. Wiederhole die Schätzung mit `Center` als Referenz.
7. Zeige, wie sich Intercept und Dummy-Koeffizienten ändern, und prüfe, dass Fits und standortspezifische Vorhersagen identisch bleiben.
8. Erkläre, warum die Wahl der Referenz die beantworteten Koeffizientenfragen, aber nicht den Modellinhalt verändert.

## Ausführen

```bash
python3 exercises/T22-categorical-variables/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/coefficients_by_reference.csv`, `results/adjusted_location_means.csv` und `results/categorical_variables.png`:

```bash
python3 exercises/T22-categorical-variables/solution.py
```

## Denkfragen

- Welcher Standortvergleich steckt im Koeffizienten `location_University`, wenn `Suburb` Referenz ist?
- Weshalb ist Zentrierung für die Bedeutung des Intercepts hilfreich?
- Wie würdest du direkt Zentrum gegen Universitätsviertel testen, ohne die Referenz neu zu codieren?
