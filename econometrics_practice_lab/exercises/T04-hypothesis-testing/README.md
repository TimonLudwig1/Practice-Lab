# T04 – Hypothesentest für Lieferzeiten

## Ausgangslage

Ein Lieferdienst verspricht eine durchschnittliche Lieferzeit von 30 Minuten. Für 80 zufällig ausgewählte Lieferungen liegen Zeiten vor. Du prüfst zweiseitig, ob der Populationsmittelwert von 30 Minuten abweicht.

## Lernziele

Nach dem Projekt kannst du:

- Null- und Alternativhypothese präzise formulieren,
- eine t-Teststatistik aus Schätzwert und Standardfehler konstruieren,
- kritischen Wert, Ablehnungsbereich und p-Wert miteinander verbinden,
- eine Testentscheidung von ihrer inhaltlichen Interpretation unterscheiden.

## Aufgaben

Verwende \(H_0:\mu=30\), \(H_1:\mu\neq30\) und \(\alpha=0{,}05\). Bearbeite anschließend `starter.py`.

1. Erzeuge und visualisiere den Datensatz. Prüfe Stichprobengröße, Mittelwert, Standardabweichung und auffällige Beobachtungen.
2. Berechne Standardfehler und Teststatistik von Hand im Code.
3. Bestimme Freiheitsgrade sowie die beiden kritischen t-Werte.
4. Entscheide allein anhand des Ablehnungsbereichs über \(H_0\).
5. Berechne den zweiseitigen p-Wert und treffe dieselbe Entscheidung erneut.
6. Berechne ein 95%-Konfidenzintervall und erkläre seine Verbindung zum Test.
7. Prüfe deine Rechnung mit `scipy.stats.ttest_1samp`.
8. Formuliere das Ergebnis in Alltagssprache, ohne „\(H_0\) ist bewiesen“ oder „mit 95% Wahrscheinlichkeit“ zu schreiben.

## Ausführen

```bash
python3 exercises/T04-hypothesis-testing/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/test_results.csv` und `results/test_decision.png`:

```bash
python3 exercises/T04-hypothesis-testing/solution.py
```

## Denkfragen

- Was genau wäre unter \(H_0\) selten: die beobachteten Rohdaten oder eine mindestens so extreme Teststatistik?
- Ändert ein kleiner p-Wert automatisch die praktische Relevanz des Unterschieds?
- Warum ist „\(H_0\) nicht ablehnen“ nicht dasselbe wie „\(H_0\) bestätigen“?
