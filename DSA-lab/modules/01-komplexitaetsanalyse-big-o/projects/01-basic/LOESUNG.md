# Lösung und Interpretation

Lies diese Datei erst, nachdem du alle fünf Funktionen selbst analysiert und das
Labor ausgeführt hast.

## Zuordnung

| Kurve | Klasse | Begründung |
|---|---|---|
| A | \(\Theta(1)\) | Die Schleife läuft immer genau 32-mal, unabhängig von \(n\). |
| B | \(\Theta(\log n)\) | Die Problemgröße wird pro äußerem Schritt halbiert; die innere Schleife hat konstante Länge 32. |
| C | \(\Theta(n)\) | Pro Eingabeeinheit läuft ein Block mit konstanter Länge acht. |
| D | \(\Theta(n\log n)\) | Für jedes der \(n\) Elemente wird die Problemgröße wiederholt halbiert. |
| E | \(\Theta(n^2)\) | Die inneren Längen summieren sich zu \(0+1+\dots+(n-1)=n(n-1)/2\). |

Die Funktionen geben die Anzahl ihrer abstrakten Arbeitsschritte zurück. Dadurch
lässt sich die Herleitung mit kleinen Eingaben exakt prüfen. Gemessen wird
dennoch reale Python-Laufzeit: Schleifenverwaltung, Funktionsaufrufe und
Interpreterkosten beeinflussen die beobachteten Werte.

## Erwartete Form der Messungen

Kurve A sollte annähernd flach bleiben. Kurve C nähert sich im Log-Log-Plot einer
Steigung von 1, Kurve E einer Steigung von 2. Kurve D liegt zwischen den beiden
und besitzt wegen des zusätzlichen Logarithmus eine leicht größere Steigung als
die lineare Kurve.

Für Kurve B ist Vorsicht nötig. Bei \(T(n)=\log n\) gilt im Log-Log-Plot

\[
\log T(n)=\log(\log n).
\]

Das ist keine Gerade mit einer festen positiven Steigung. Die Kurve wächst
langsam und wird für größere \(n\) relativ flacher. Im begrenzten Messbereich
können konstante Interpreterkosten den logarithmischen Anteil zusätzlich
überdecken.

## Verdopplungsquotienten

Theoretisch ist bei einer Verdopplung ungefähr Folgendes zu erwarten:

- A: Faktor 1,
- B: knapp über Faktor 1,
- C: Faktor 2,
- D: etwas über Faktor 2,
- E: Faktor 4.

Die gemessenen Quotienten schwanken. Besonders bei den schnellen Kurven sind
Scheduling, Timerauflösung, CPU-Takt und Cache-Effekte relativ groß. Deshalb
werden mehrere Aufrufe gebündelt und mehrere Samples über den Median
zusammengeführt.

## Warum n log n langfristig n² schlägt

Vergleiche das Verhältnis:

\[
\frac{n^2}{n\log n}=\frac{n}{\log n}.
\]

Der Zähler wächst wesentlich schneller als der Logarithmus im Nenner. Das
Verhältnis wächst daher ohne Grenze: Für hinreichend große Eingaben benötigt der
quadratische Algorithmus beliebig viel mehr Arbeit als der linearithmische.
Konstante Faktoren können lediglich verschieben, ab welcher Eingabegröße dieser
Vorteil sichtbar wird.

## Was das Experiment aussagen darf

Die Code-Analyse begründet die asymptotischen Klassen. Die Messreihe zeigt, dass
das beobachtete Wachstum im untersuchten Größenbereich damit vereinbar ist. Sie
beweist die Klassen nicht, denn endlich viele Messpunkte schließen anderes
Verhalten außerhalb des Messbereichs nicht aus.
