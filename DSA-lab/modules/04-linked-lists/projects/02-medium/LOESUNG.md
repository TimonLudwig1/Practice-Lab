# Lösung und Auswertung

## In-place-Reversal

Der kritische Moment ist `current.next = previous`. Danach ist der alte
Nachfolger über `current` nicht mehr erreichbar. Deshalb muss er vorher in
`next_node` gesichert sein.

Die Schleifeninvariante partitioniert alle ursprünglichen Knoten in zwei
disjunkte Ketten:

- umgedrehter Prefix ab `previous`,
- unveränderter Suffix ab `current`.

Jeder Schritt verschiebt genau einen Knoten vom Suffix in den Prefix. Bei
`current is None` enthält der Prefix alle Knoten und `previous` ist der neue Head.
Jeder Knoten wird einmal verarbeitet: O(n) Zeit und O(1) Arbeitszustand.

## Runner-Mitte

Nach `k` Runden hat `slow` k und `fast` 2k Kanten zurückgelegt. Wenn `fast` das
Ende erreicht, liegt `slow` deshalb bei der Hälfte. Die Schleifenbedingung

```text
fast is not None and fast.next is not None
```

erlaubt nur vollständige Doppelschritte. Starten beide am Head, landet `slow`
bei gerader Länge auf dem zweiten mittleren Knoten. Das ist eine API-Entscheidung,
die Tests ausdrücklich festhalten.

## Floyd vollständig hergeleitet

Innerhalb eines Zyklus gewinnt `fast` pro Runde eine Position relativ zu `slow`.
Bei Zykluslänge `c` nimmt der relative Abstand Werte modulo `c` an und muss daher
null werden: Es gibt einen Treffpunkt.

Sei `p` die Prefix-Länge und `m` die Entfernung vom Eintritt bis zum Treffpunkt.
Beim Treffen ist die von `fast` zusätzlich zurückgelegte Strecke ein Vielfaches
der Zykluslänge. Daraus folgt, dass die Entfernung vom Treffpunkt bis zum Eintritt
modulo Zyklus genau zur Entfernung vom Head bis zum Eintritt passt. Bewegen sich
beide Zeiger danach gleich schnell, treffen sie am Eintritt.

Eine Runde vom Treffpunkt zurück zu ihm zählt `cycle_length`. Die gemeinsamen
Schritte von Head und Treffpunkt bis zum Eintritt zählen `prefix_length`.
Gesamtlaufzeit bleibt O(n), Speicher O(1).

## Stabiler Merge

Invariante vor jeder Auswahl:

> Von `dummy.next` bis `tail` liegt der vollständig sortierte Prefix aller bereits
> entfernten Quellknoten; `left` und `right` zeigen auf die kleinsten noch nicht
> verwendeten Knoten ihrer Ketten.

Der kleinere aktuelle Wert ist damit global der nächste. Bei Gleichheit wird
links gewählt, wodurch die ursprüngliche Reihenfolge zwischen den Eingaben
definiert bleibt. Sobald eine Quelle leer ist, ist die gesamte andere Quelle
bereits sortiert und kann mit einer einzigen Linkzuweisung angehängt werden.

Der lokale Dummy vereinfacht den ersten Schritt, gehört aber nicht zum Resultat.
Die Tests vergleichen die Menge aller Knotenidentitäten vor und nach dem Merge;
damit fällt eine versteckte Kopierlösung sofort auf.

## Komplexitätsübersicht

| Algorithmus | Zeit | Extra-Speicher | Verändert Links? |
|---|---:|---:|---|
| Reversal | O(n) | O(1) | ja |
| Mitte | O(n) | O(1) | nein |
| Floyd | O(n) | O(1) | nein |
| Merge | O(n + m) | O(1) | ja |

Die optionalen Trace-Listen benötigen zusätzlichen Speicher proportional zur
Zahl protokollierter Schritte. Sie sind Lerninstrumente und werden in der
algorithmischen O(1)-Angabe nicht mitgerechnet, wenn `trace=None` ist.
