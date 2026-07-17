# Ergebnisbericht: FIFO versus Priority Queue

Der Vergleich verwendet denselben synthetischen Strom aus 250
Jobs für beide Strategien. Die Verarbeitung ist nicht-präemptiv: Ein gestarteter
Job wird nie von einem später eintreffenden dringenden Job unterbrochen.

| Strategie | Mittelwert Warten | Median | P95 | Maximum | Auslastung |
|---|---:|---:|---:|---:|---:|
| FIFO | 8.11 | 8.48 | 16.34 | 18.25 | 96.9% |
| Priorität | 7.94 | 3.70 | 32.84 | 47.73 | 96.9% |

## Interpretation

Für dringende Jobs verändert die Prioritätsregel die mittlere Wartezeit um
-7.85 Zeiteinheiten gegenüber FIFO. Für Batch-Jobs beträgt die
Änderung +13.75. Die Strategie verteilt Wartezeit somit gezielt
zwischen Klassen um; ein einzelner Gesamtmittelwert reicht zur Bewertung nicht
aus. Insbesondere Maximum und P95 machen sichtbar, ob niedrig priorisierte Jobs
sehr lange warten.

Durchsatz und Auslastung sind nahezu identisch, weil beide Strategien dieselben
Jobs auf demselben einzelnen Worker ohne Leerlaufentscheidung verarbeiten. Die
Queue Policy verändert primär die Reihenfolge und damit die Fairness, nicht die
vorhandene Rechenkapazität.
