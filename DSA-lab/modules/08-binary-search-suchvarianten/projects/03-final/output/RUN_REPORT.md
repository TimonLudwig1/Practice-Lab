# Laufbericht: Schwellenwert-Optimierung

## Szenario

Der Generator erzeugt mit Seed `80803` insgesamt 5.000
Klassifikationsbeispiele mit überlappenden Score-Verteilungen. Davon sind
1.508 positiv und 3.492 negativ. Ein Score
ab dem Schwellenwert wird als positive Vorhersage gewertet.

## Nebenbedingung und Monotonie

Gesucht wird maximaler Recall unter der Nebenbedingung
`False Positive Rate ≤ 5%`. Mit steigendem Schwellenwert
kann die Menge positiver Vorhersagen nur kleiner werden. Daher können weder False
Positives noch True Positives zunehmen: FPR und Recall sind beide monoton nicht
steigend. Die **kleinste zulässige Schwelle** maximiert somit den Recall unter der
FPR-Grenze.

Precision wurde bewusst nicht als Suchprädikat verwendet. Sie ist auf endlichen
Datensätzen im Allgemeinen nicht monoton und würde die Korrektheitsvoraussetzung
der Binary Search verletzen.

## Exaktes Ergebnis

- Schwellenwert: `0.57983508`
- Recall: 0.7977
- False Positive Rate: 0.0498
- Precision: 0.8736
- True Positives: 1203
- False Positives: 174
- Exakte Kandidatenschwellen: 5.001
- Metrikauswertungen der Binary Search: 13

Der direkte erschöpfende Lauf über alle exakten Kandidaten liefert denselben
Schwellenwert und dieselben Confusion Counts. Er benötigt
5.001 Auswertungen und damit 384.7-mal so
viele wie die Binary Search.

## Vergleich mit der Rastersuche

Die naive Rastersuche prüft 1.002 gleichmäßig verteilte
Schwellenwerte. Sie benötigt 77.1-mal so viele Auswertungen wie die
Binary Search und ist nur auf ihre Rasterauflösung genau.

- Raster-Schwellenwert: `0.58000000`
- Raster-Recall: 0.7971
- Raster-FPR: 0.0495

Die Vorverarbeitung sortiert positive und negative Scores einmalig. Eine
Metrikauswertung zählt danach die Werte oberhalb einer Schwelle mit zwei
`bisect_left`-Suchen. Die Optimierung benötigt `O(log u)` solcher Auswertungen für
`u` exakte Kandidatenschwellen; eine vollständige Rastersuche benötigt dagegen
eine Auswertung pro Rasterpunkt.
