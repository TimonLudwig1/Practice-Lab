# Lösung und Auswertung

## Datenmodell

`Node` verwendet `eq=False`, damit zwei Knoten mit demselben Wert verschiedene
Identitäten bleiben. Eine strukturelle automatische Dataclass-Gleichheit würde
außerdem rekursiv `next` vergleichen und wäre bei Zyklen gefährlich.

Die Liste speichert genau drei Felder:

- `_head`: erster Knoten oder `None`,
- `_tail`: letzter Knoten oder `None`,
- `_size`: Zahl erreichbarer Knoten.

Nutzwerte liegen ausschließlich in Knoten; es existiert keine parallele
Python-Liste als Abkürzung.

## Append und Prepend

Append ist durch Tail konstant:

```text
leer:       head = node, tail = node
nichtleer:  tail.next = node, tail = node
```

Prepend ist symmetrisch am Head:

```text
node.next = head
head = node
falls vorher leer: tail = node
```

Die Reihenfolge verhindert, dass der alte Head verloren geht. Beide Operationen
erhöhen `_size` genau einmal.

## Insert

Die beiden Randpositionen delegieren an getestete Grundoperationen. Für eine
mittlere Position wird der Vorgänger gesucht:

```text
before ──▶ after

node.next = after
before.next = node
```

Der neue Knoten übernimmt zuerst die alte Nachfolgerreferenz. Erst danach wird
der Link des Vorgängers geändert. Die Suche kostet O(index), das Umhängen O(1).

## Delete

Negative Indizes werden durch `index += size` in einen normalen Zugriffsindex
überführt. Danach gibt es zwei strukturelle Fälle:

- Index null: Head auf den Nachfolger setzen; bei Größe eins Tail leeren.
- Sonst: Vorgänger suchen und dessen `next` am Ziel vorbeisetzen; war das Ziel
  Tail, wird der Vorgänger neuer Tail.

Erst nach dem Umhängen wird `target.next = None` gesetzt. Der Rest der Liste ist
dann bereits über einen anderen Link erreichbar. `_size` sinkt genau einmal.

## Remove

Die Traversierung hält gleichzeitig `previous` und `current`. Beim ersten
Treffer ist deshalb die für O(1)-Deletion nötige Vorgängerreferenz schon bekannt.
Die Gesamtlaufzeit bleibt O(n), weil die Suche dominiert.

Der Fall `previous is None` bedeutet, dass der Head getroffen wurde. Der Fall
`current is tail` aktualisiert den Tail. Beim einzigen Element treffen beide
Fälle gleichzeitig zu und beide Endpunkte werden `None`.

## Invariantenprüfung

`check_invariants` ist absichtlich unabhängig von `_size` traversierend. Sie
merkt besuchte Objektidentitäten und erkennt daher Zyklen, bevor eine Endlosschleife
entsteht. Anschließend vergleicht sie:

- gezählte Knoten mit `_size`,
- letzten erreichbaren Knoten mit `_tail`,
- Terminal-Link des Tails mit `None`.

Diese Prüfung kostet O(n) und gehört nicht automatisch in jede Produktionsoperation.
In Tests und beim Lernen ist sie jedoch wertvoll, weil der Fehler unmittelbar
nach der verursachenden Mutation sichtbar wird.

## Warum Delete Tail weiterhin O(n) ist

Der Tail macht Append O(1), enthält aber keine Referenz auf seinen Vorgänger. Um
den vorletzten Knoten zum neuen Tail zu machen, muss vom Head traversiert werden.
Eine Doubly Linked List löst das mit `tail.prev`, bezahlt dafür aber eine weitere
Referenz und komplexere Mutationen pro Knoten.

## Kantenfallmatrix

| Operation | leer | ein Element | Kopf | Mitte | Tail |
|---|---|---|---|---|---|
| Append | setzt beide | aktualisiert Tail | — | — | O(1) |
| Prepend | setzt beide | aktualisiert Head | O(1) | — | — |
| Insert | nur Index 0 | Index 0 oder 1 | Prepend | Vorgänger suchen | Append |
| Delete | Fehler | leert beide | Head ändern | Vorgänger umhängen | Tail ändern |
| Remove | Fehler | leert beide bei Treffer | Head ändern | Vorgänger umhängen | Tail ändern |

Die pytest-Suite prüft nicht nur Ergebniswerte, sondern Objektidentitäten,
abgetrennte Links und die vollständigen Invarianten nach jeder Mutation. Dadurch
würde auch eine oberflächlich korrekt aussehende Wertfolge mit falschem Tail oder
Size-Zähler auffallen.
