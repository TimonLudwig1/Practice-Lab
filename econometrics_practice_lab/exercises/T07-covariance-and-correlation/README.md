# T07 – Kovarianz und Maßeinheiten

## Ausgangslage

Für 300 synthetische Haushalte liegen monatliches Nettoeinkommen, Freizeitausgaben und Pendelzeit vor. Du untersuchst, wie gemeinsame Abweichungen von den jeweiligen Mittelwerten die Kovarianz erzeugen und warum ihr Zahlenwert von den Maßeinheiten abhängt.

## Lernziele

Nach dem Projekt kannst du:

- Stichprobenkovarianz aus zentrierten Beobachtungen berechnen,
- das Vorzeichen der Kovarianz aus einem Scatterplot erklären,
- Kovarianz und Korrelation klar voneinander abgrenzen,
- zeigen, wie eine Einheitenumrechnung die Kovarianz, aber nicht die Korrelation verändert.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Erzeuge den Haushaltsdatensatz und prüfe Verteilungen, Wertebereiche und fehlende Werte.
2. Berechne die Stichprobenkovarianz von Einkommen und Freizeitausgaben manuell mit
   \(\operatorname{Cov}(X,Y)=\sum_i (x_i-\bar x)(y_i-\bar y)/(n-1)\).
3. Wiederhole die Rechnung für Einkommen und Pendelzeit. Erkläre beide Vorzeichen anhand der Datenpunkte relativ zu den Mittelwertlinien.
4. Vergleiche deine Ergebnisse mit `pandas.DataFrame.cov()`.
5. Rechne das Einkommen von Euro in Tausend Euro um. Berechne Kovarianz und Korrelation mit den Freizeitausgaben erneut.
6. Erkläre rechnerisch, warum sich die Kovarianz um den Faktor 1.000 ändert, die Korrelation jedoch gleich bleibt.
7. Erzeuge eine Kovarianzmatrix und eine Korrelationsmatrix für alle numerischen Variablen.
8. Formuliere, warum die Größe einer Kovarianz ohne Kenntnis der Einheiten kaum vergleichbar ist.

## Ausführen

```bash
python3 exercises/T07-covariance-and-correlation/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/covariance_summary.csv`, zwei Matrizen und `results/covariance_intuition.png`:

```bash
python3 exercises/T07-covariance-and-correlation/solution.py
```

## Denkfragen

- Welche Beobachtungen liefern einen positiven Beitrag zur Kovarianz?
- Kann eine große positive Kovarianz trotzdem zu einer eher schwachen Korrelation gehören?
- Was passiert mit Kovarianz und Korrelation, wenn eine Variable mit \(-1\) multipliziert wird?
