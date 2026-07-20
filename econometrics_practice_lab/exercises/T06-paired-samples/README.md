# T06 – Gepaarte Vorher-Nachher-Daten bei Haushalten

## Ausgangslage

Bei 60 Haushalten wird der tägliche Stromverbrauch vor und nach Installation eines intelligenten Verbrauchsdisplays gemessen. Beide Werte derselben Zeile gehören zum selben Haushalt. Du untersuchst, ob sich der mittlere Verbrauch verändert hat und was verloren geht, wenn diese Paarung ignoriert wird.

## Lernziele

Nach dem Projekt kannst du:

- eine gepaarte Datenstruktur anhand der Erhebungseinheit erkennen,
- einen gepaarten t-Test als Ein-Stichproben-Test der individuellen Differenzen formulieren,
- den Standardfehler aus der Streuung der Differenzen berechnen,
- gepaarte und fälschlich unabhängige Analysen vergleichen.

## Aufgaben

Definiere durchgehend \(D_i=\text{nach}_i-\text{vor}_i\); negative Werte stehen damit für eine Reduktion. Bearbeite dann `starter.py`.

1. Erzeuge die Daten und prüfe, ob jeder Haushalt genau ein vollständiges Wertepaar besitzt.
2. Visualisiere Vorher- und Nachher-Werte mit verbundenen Linien und berechne ihre Korrelation.
3. Erzeuge die individuellen Differenzen und beschreibe deren Verteilung.
4. Teste \(H_0:\mu_D=0\) zweiseitig mit einem gepaarten t-Test, einschließlich 95%-Konfidenzintervall.
5. Verifiziere das Ergebnis mit `scipy.stats.ttest_rel`.
6. Behandle die beiden Spalten absichtlich als unabhängige Gruppen und führe einen Welch-Test aus.
7. Vergleiche Schätzwert, Standardfehler und p-Wert beider Analysen. Erkläre, warum der Punktschätzer gleich, die Unsicherheit aber verschieden ist.
8. Mische die Nachher-Werte zufällig zwischen Haushalten und untersuche, was mit der Streuung der Differenzen geschieht.

## Ausführen

```bash
python3 exercises/T06-paired-samples/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/paired_results.csv` und `results/paired_structure.png`:

```bash
python3 exercises/T06-paired-samples/solution.py
```

## Denkfragen

- Was ist in dieser Studie die statistisch unabhängige Beobachtungseinheit?
- Wann bringt Pairing besonders viel Präzision?
- Könnte ein Vorher-Nachher-Unterschied trotz sauberem gepaarten Test kausal falsch interpretiert werden?
