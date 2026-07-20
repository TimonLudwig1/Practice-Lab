# T19 – Gemeinsame Hypothesen im Gehaltsmodell

## Ausgangslage

Ein Gehaltsmodell enthält Berufserfahrung, Bildungsjahre, Zertifikate und die Teilnahme an einem Führungstraining. Du testest einzelne Koeffizienten, eine Gleichheitsrestriktion und die gemeinsame Nullhypothese, dass Zertifikate und Training zusammen keinen Erklärungsbeitrag leisten.

## Lernziele

Nach dem Projekt kannst du:

- einzelne Koeffizienten mit t-Tests prüfen,
- lineare Restriktionen über Koeffizientendifferenzen formulieren,
- einen gemeinsamen F-Test aus restringiertem und unrestringiertem Modell berechnen,
- Einzel- und Gemeinschaftssignifikanz unterscheiden.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Schätze `annual_salary_eur` auf alle vier Regressoren und erstelle eine Koeffiziententabelle.
2. Teste zweiseitig \(H_0:\beta_{training}=0\) mit einem t-Test.
3. Teste \(H_0:\beta_{experience}=\beta_{education}\) als lineare Restriktion.
4. Schätze ein restringiertes Modell ohne `certifications` und `leadership_training`.
5. Berechne
   \(F=((RSS_R-RSS_U)/q)/(RSS_U/(n-k_U-1))\) für die gemeinsame Nullhypothese beider Koeffizienten gleich null.
6. Bestimme kritischen F-Wert und p-Wert und kontrolliere mit `model.f_test`.
7. Vergleiche den gemeinsamen Test mit den beiden individuellen t-Tests. Erkläre, warum die Aussagen nicht identisch sein müssen.
8. Formuliere jede Testentscheidung inhaltlich und vermeide „die Nullhypothese ist bewiesen“.

## Ausführen

```bash
python3 exercises/T19-multiple-hypothesis-testing/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/coefficients.csv`, `results/hypothesis_tests.csv` und `results/multiple_tests.png`:

```bash
python3 exercises/T19-multiple-hypothesis-testing/solution.py
```

## Denkfragen

- Warum besitzt der gemeinsame Test im Zähler zwei Freiheitsgrade?
- Kann ein gemeinsamer Test signifikant sein, obwohl keiner der Einzeltests bei 5% signifikant ist?
- Welche Rolle spielt die Kovarianz der Koeffizientenschätzer bei einer Gleichheitsrestriktion?
