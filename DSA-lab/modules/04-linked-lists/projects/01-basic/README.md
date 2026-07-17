# 01-basic — Singly Linked List von Grund auf

## Ziel

Du implementierst eine vollständige Singly Linked List ohne `list`, `deque` oder
andere Container als internen Datenspeicher. Die Klasse hält nur Referenzen auf
Head und Tail sowie eine Größenvariable. Jede Mutation muss diese drei
Strukturinvarianten korrekt aktualisieren.

Das Projekt nutzt Python-Skripte und pytest. Eine Linked List besteht aus vielen
kleinen Randfällen; parametrisierte Tests machen die Spezifikation präziser und
leichter wiederholbar als ein Notebook.

## Öffentliche Schnittstelle

```python
linked = SinglyLinkedList(values=())

linked.append(value)       # O(1)
linked.prepend(value)      # O(1)
linked.insert(index, value)
linked.delete(index)
linked.remove(value)
linked.find(value)
linked.node_at(index)
linked.clear()

len(linked)
linked[index]
value in linked
list(linked)
repr(linked)
```

`append`, `prepend` und `insert` geben den erzeugten `Node` zurück. `delete` und
`remove` geben den entfernten Wert zurück. Dadurch können Tests nicht nur Werte,
sondern auch Knotenidentität und Abtrennung prüfen.

## Strukturinvarianten

Nach jeder erfolgreichen Mutation müssen gelten:

```text
size == 0:
    head is None
    tail is None

size > 0:
    head und tail sind Knoten
    tail.next is None
    vom head sind exakt size Knoten erreichbar
    der letzte erreichbare Knoten ist tail
    es gibt keinen Zyklus
```

`check_invariants()` prüft diese Bedingungen ausdrücklich. Rufe die Methode in
Tests nach jeder Mutation auf, nicht nur am Ende einer langen Operationsfolge.

## Aufgaben

### 1. Konstruktion und Iteration

- Erzeuge aus jedem Iterable dieselbe Reihenfolge.
- Leere Konstruktion setzt beide Endpunkte auf `None`.
- `__iter__` folgt ausschließlich `next`-Referenzen.
- `__len__` liest die gespeicherte Größe in O(1).
- `__repr__` liefert beispielsweise `SinglyLinkedList([1, 2])`.

### 2. Append und Prepend

Append verwendet den gespeicherten Tail und darf nicht vom Head aus traversieren.
Beim ersten Knoten werden Head und Tail dieselbe Objektidentität. Prepend setzt
den alten Head als Nachfolger; bei leerer Liste muss zusätzlich Tail gesetzt
werden.

### 3. Insert

`insert(index, value)` akzeptiert nur `0 <= index <= len(linked)`:

- `0` delegiert an Prepend,
- `len` delegiert an Append,
- dazwischen wird der Vorgänger an `index - 1` gesucht und lokal umgehängt.

Negative Insert-Indizes werden bewusst nicht normalisiert. So bleibt die
Zeigerposition eindeutig und die API lehrt die Grenzen explizit.

### 4. Delete

`delete(index)` unterstützt Python-artige negative Zugriffsindizes. Löschen am
Head und Löschen des einzigen Knotens benötigen besondere Endpunktupdates. Beim
Löschen des Tails wird dessen Vorgänger zum neuen Tail.

Der entfernte Knoten soll danach `next is None` erfüllen. So ist er wirklich aus
der Struktur abgetrennt, auch wenn ein Test noch eine externe Referenz hält.

### 5. Suche und Löschen nach Wert

`find(value)` liefert den ersten Index oder `None`. `remove(value)` entfernt nur
den ersten Treffer und löst bei fehlendem Wert `ValueError` aus. Wertgleichheit
ist hier erwünscht; Knotenidentität bleibt davon getrennt.

### 6. Zugriff und Lebenszyklus

`node_at` und `__getitem__` normalisieren negative Indizes. `clear` trennt alle
Knoten nacheinander ab und stellt den leeren Zustand wieder her.

## Ausführen

Abhängigkeit installieren:

```bash
python3 -m pip install -r requirements.txt
```

Tests:

```bash
python3 -m pytest -v
```

Vollständige Operationsdemo:

```bash
python3 demo.py
```

## Komplexität

| Operation | Zeit | Begründung |
|---|---:|---|
| `len` | O(1) | gespeicherte Größe |
| Append | O(1) | gespeicherter Tail |
| Prepend | O(1) | nur Head-Links |
| Zugriff/Suche | O(n) | Vorwärtstraversierung |
| Insert | O(n), an Endpunkten O(1) | Vorgänger gegebenenfalls suchen |
| Delete | O(n), am Head O(1) | Vorgänger gegebenenfalls suchen |
| Remove | O(n) | ersten Wert suchen |
| Clear | O(n) | jeden Link explizit trennen |

## Fertig, wenn …

- kein Python-Container die Nutzwerte intern speichert,
- Head, Tail und Size nach jeder Operation konsistent sind,
- Append und Prepend auf leerer sowie nichtleerer Liste funktionieren,
- Insert und Delete Kopf, Mitte, Tail und Einzelelement korrekt behandeln,
- negative Zugriffsindizes und alle ungültigen Indizes definiert reagieren,
- entfernte Knoten abgetrennt werden,
- Suche, Iteration, Länge, Membership und Repr korrekt sind,
- alle pytest-Tests und die vollständige Demo fehlerfrei laufen,
- du jede Mutation vor dem Code als Zeigerdiagramm darstellen kannst.
