# T35 – Fixed Effects im Filialpanel

## Ausgangslage

Eine Einzelhandelskette beobachtet Werbung und Umsatz ihrer Filialen über zehn Monate. Attraktive Filialstandorte investieren dauerhaft mehr in Werbung und erzielen unabhängig davon höhere Umsätze. Pooled OLS verwechselt diesen zeitinvarianten Standortvorteil mit dem Werbeeffekt. Filial-Fixed-Effects vergleichen stattdessen jede Filiale mit sich selbst.

## Lernziele

- zeitinvariante unbeobachtete Heterogenität als Confounder erkennen,
- Pooled OLS, Between- und Fixed-Effects-Schätzung vergleichen,
- Fixed Effects als einheitsspezifische Intercepts verstehen,
- die FE-Schätzung als Nutzung von Within-Variation interpretieren,
- erkennen, warum zeitinvariante Regressoren mit Entity-FE nicht separat identifiziert sind.

## Aufgaben

1. Schätze den Werbeeffekt mit Pooled OLS.
2. Aggregiere auf Filialmittelwerte und schätze eine Between-Regression.
3. Ergänze Filialdummies und schätze das Fixed-Effects-Modell mit geclusterten Standardfehlern.
4. Vergleiche alle Koeffizienten mit dem wahren Effekt von 2,5 Umsatzpunkten je Werbeeinheit.
5. Prüfe die Korrelation zwischen Filialqualität und durchschnittlicher Werbung.
6. Erkläre anhand der Rohdaten- und Within-Grafik, warum sich die Schätzungen unterscheiden.
7. Zeige über den Matrixrang, weshalb der zeitinvariante `downtown_location`-Indikator neben vollständigen Filial-FE nicht separat identifiziert ist.

## Ausführen

```bash
python3 exercises/T35-fixed-effects/starter.py
python3 exercises/T35-fixed-effects/solution.py
```

Die Lösung erzeugt `data/store_panel.csv`, Ergebnistabellen und `results/fixed_effects.png`.

## Denkfragen

- Welche unbeobachteten Filialmerkmale werden durch Fixed Effects absorbiert?
- Warum kann FE den Effekt einer Variable ohne zeitliche Veränderung nicht schätzen?
- Ist eine große Zahl von Filialdummies selbst das kausale Identifikationsargument?
