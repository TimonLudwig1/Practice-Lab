# T05 – Mittelwertsunterschied im A/B-Test

## Ausgangslage

Eine Lernplattform randomisiert neue Nutzerinnen und Nutzer auf zwei unabhängig zusammengesetzte Gruppen. Gruppe A verwendet das bisherige Interface, Gruppe B ein neues Interface. Nach einer Lerneinheit wird derselbe Wissenstest geschrieben. Du sollst den mittleren Score-Unterschied \(\mu_B-\mu_A\) schätzen und testen.

## Lernziele

Nach dem Projekt kannst du:

- unabhängige Gruppen in Daten erkennen,
- Mittelwertsdifferenz und Standardfehler berechnen,
- den Welch-t-Test bei ungleichen Varianzen und Gruppengrößen anwenden,
- Konfidenzintervall, p-Wert und praktische Effektgröße gemeinsam interpretieren.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Erzeuge die Daten und prüfe, ob jede `student_id` genau einmal und damit nur in einer Gruppe vorkommt.
2. Berichte je Gruppe \(n\), Mittelwert und Standardabweichung; visualisiere beide Verteilungen.
3. Schätze \(\bar Y_B-\bar Y_A\). Achte durchgehend auf dieselbe Vorzeichenkonvention.
4. Berechne den Welch-Standardfehler und die Welch–Satterthwaite-Freiheitsgrade.
5. Teste zweiseitig \(H_0:\mu_B-\mu_A=0\) bei \(\alpha=0{,}05\).
6. Konstruiere ein 95%-Konfidenzintervall für die Mittelwertsdifferenz.
7. Verifiziere Teststatistik und p-Wert mit `scipy.stats.ttest_ind(..., equal_var=False)`.
8. Diskutiere statistische und praktische Relevanz getrennt. Der Testscore reicht von 0 bis 100.

## Ausführen

```bash
python3 exercises/T05-difference-in-means/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/welch_test_results.csv` und `results/group_comparison.png`:

```bash
python3 exercises/T05-difference-in-means/solution.py
```

## Denkfragen

- Warum darf dieselbe Person in diesem Design nicht in beiden Gruppen vorkommen?
- Weshalb ist der Welch-Test ein sinnvoller Standard für zwei unabhängige Mittelwerte?
- Was würde sich an der Unsicherheit ändern, wenn der beobachtete Unterschied gleich bliebe, beide Gruppen aber viermal so groß wären?
