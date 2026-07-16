# Lösung — Dynamisches Array

## Repräsentationsinvarianten

Die Implementierung hält zu jedem Zeitpunkt:

1. \(0\le length\le capacity\).
2. Die Positionen 0 bis length - 1 enthalten die logischen Elemente in ihrer
   Reihenfolge.
3. Der ctypes-Puffer besitzt exakt capacity feste Positionen.
4. Ein Resize verändert weder Länge noch Werte noch deren Reihenfolge.
5. Außerhalb des aktiven Bereichs wird kein Element über die öffentliche API
   gelesen.

Diese Aussagen sind die Grundlage aller Methoden. Append erweitert den aktiven
Bereich um eine Position. Insert schafft vorher eine Lücke. Delete schließt eine
Lücke und verkleinert danach den aktiven Bereich.

## Append und Resize

Bei freier Kapazität schreibt append an Index length und erhöht length. Das
kostet \(\Theta(1)\).

Ist length gleich capacity, erzeugt _resize einen Puffer doppelter Größe und
kopiert length Referenzen. Dieser einzelne append kostet dadurch
\(\Theta(n)\).

Für Startkapazität 1 treten Resizes beim Einfügen der Elemente 2, 3, 5, 9, 17
und so weiter auf:

| append-Nummer | alte Kapazität | neue Kapazität | Kopien | tatsächliche Modellkosten |
|---:|---:|---:|---:|---:|
| 2 | 1 | 2 | 1 | 2 |
| 3 | 2 | 4 | 2 | 3 |
| 5 | 4 | 8 | 4 | 5 |
| 9 | 8 | 16 | 8 | 9 |
| 17 | 16 | 32 | 16 | 17 |

Bis zu \(n\) appends werden weniger als

\[
1+2+4+\dots < 2n
\]

alte Referenzen kopiert. Hinzu kommen \(n\) Schreiboperationen für die neuen
Werte. Die gesamten Modellkosten sind kleiner als \(3n\), also amortisiert
kleiner als drei Einheiten pro append. Damit ist append amortisiert
\(\Theta(1)\).

Amortisiert bedeutet eine Garantie über jede lange Operationsfolge. Es wird
keine Wahrscheinlichkeitsverteilung über Eingaben angenommen und ist deshalb
nicht dasselbe wie Average Case.

## Insert

Für insert(index, value) müssen alle aktiven Elemente ab index eine Position
nach rechts rücken. Die Schleife läuft von rechts nach links.

Würde sie links beginnen, würde der Wert an index + 1 überschrieben, bevor er an
index + 2 kopiert werden kann. Die ursprünglichen Daten gingen verloren.

Bei Länge \(n\) sind:

- Insert am Ende: \(\Theta(1)\) ohne Resize,
- Insert am Anfang: \(\Theta(n)\),
- Worst Case einschließlich Resize: \(\Theta(n)\).

Der Auxiliary Space ist normalerweise \(\Theta(1)\). Wenn ein Resize nötig ist,
existieren alter und neuer Puffer kurz gleichzeitig, also \(\Theta(n)\).

## Delete

Delete speichert zunächst den Rückgabewert. Danach werden alle späteren Elemente
von links nach rechts um eine Position verschoben. Die letzte ehemals aktive
Position wird auf None gesetzt, damit keine unnötige Referenz erhalten bleibt.

Die Kosten betragen:

- Delete am Ende: \(\Theta(1)\),
- Delete am Anfang: \(\Theta(n)\),
- Auxiliary Space: \(\Theta(1)\).

Die Kapazität schrumpft nicht automatisch. Ein sofortiges Schrumpfen nach jedem
Delete könnte bei abwechselndem Append und Delete zu ständigem Kopieren führen.
Reale dynamische Arrays verwenden, falls sie schrumpfen, eine deutliche
Hysterese: Wachstum und Schrumpfen passieren bei verschiedenen Schwellen.

## Komplexitätstabelle

| Operation | Best Case | Worst Case | amortisiert | Auxiliary Space |
|---|---:|---:|---:|---:|
| Indexzugriff | \(\Theta(1)\) | \(\Theta(1)\) | \(\Theta(1)\) | \(\Theta(1)\) |
| append | \(\Theta(1)\) | \(\Theta(n)\) | \(\Theta(1)\) | bis \(\Theta(n)\) beim Resize |
| insert(0, x) | \(\Theta(n)\) | \(\Theta(n)\) | \(\Theta(n)\) | bis \(\Theta(n)\) beim Resize |
| insert(length, x) | \(\Theta(1)\) | \(\Theta(n)\) | \(\Theta(1)\) | bis \(\Theta(n)\) beim Resize |
| delete(0) | \(\Theta(n)\) | \(\Theta(n)\) | \(\Theta(n)\) | \(\Theta(1)\) |
| delete(length - 1) | \(\Theta(1)\) | \(\Theta(1)\) | \(\Theta(1)\) | \(\Theta(1)\) |
| Resize | \(\Theta(n)\) | \(\Theta(n)\) | — | \(\Theta(n)\) |

## Grenzen gegenüber Python list

Die Eigenimplementierung zeigt die Kernmechanik, bildet aber nicht alle
Produktionsdetails nach:

- CPython verwendet keinen exakten Verdopplungsfaktor.
- Python list besitzt mehr Methoden, optimierten C-Code und ausgefeilte
  Fehlerbehandlung.
- Die genaue Speicherverwaltung und Freigabestrategie ist komplexer.
- Diese Klasse ist nicht threadsicher und nicht auf maximale Geschwindigkeit
  optimiert.
- Die Diagnosehistorie ist Lerninstrument, kein Bestandteil eines minimalen
  dynamischen Arrays.

Das Lernziel ist nicht, list praktisch zu ersetzen, sondern deren
Kapazitätsmodell, Verschiebungskosten und amortisierte Analyse aus einer
funktionierenden Eigenimplementierung herzuleiten.
