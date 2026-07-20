# Projekt 01-basic: Binary Search Tree von Grund auf

## Ziel

In diesem Projekt baust du einen vollständigen **Binary Search Tree (BST)** ohne
fertige Baum- oder Queue-Implementierung. Dabei wird die zentrale Invariante zum
Arbeitswerkzeug: Für jeden Knoten liegen links nur kleinere und rechts nur
größere Werte. Suche, Minimum, Maximum und sortierte Ausgabe ergeben sich direkt
aus dieser Ordnung.

Das Projekt ist als Python-Skript mit separater pytest-Suite angelegt. Dieses
Format macht Zustandsänderungen und Kantenfälle transparenter als ein Notebook:
Die Datenstruktur bleibt importierbar, die Demo ist wiederholbar und jede
Operation lässt sich isoliert testen.

## Dateien

- `bst.py`: vollständige Implementierung mit `Node` und `BinarySearchTree`
- `demo.py`: nachvollziehbare Insert-, Such-, Traversierungs- und Delete-Demo
- `test_bst.py`: Unit- und Property-Tests gegen ein Python-`set` als Referenz

## Design-Entscheidungen

Der Baum speichert jeden Wert höchstens einmal. `insert` liefert `False`, wenn
der Wert bereits vorhanden ist; damit bleibt die Invariante strikt und der
Umgang mit Duplikaten ist nicht implizit. `delete` meldet ebenfalls über einen
Boolean, ob sich der Baum geändert hat.

Die Tiefensuchen sind bewusst rekursiv implementiert. Level-Order verwendet
eine selbst gebaute Queue aus Liste plus Leseposition. Dadurch wird weder
`collections.deque` als Abkürzung genutzt noch das langsame `pop(0)` benötigt.

## Aufgabenstellung

1. Lies die öffentliche API in `bst.py` und formuliere für jede Operation, an
   welcher Stelle die BST-Invariante die nächste Entscheidung bestimmt.
2. Führe `demo.py` aus. Prüfe nach jedem Insert, ob die Inorder-Ausgabe sortiert
   ist, und gleiche die vier Traversierungen mit der ASCII-Struktur ab.
3. Verfolge in `delete` die drei strukturellen Fälle:
   - Blatt: Der Verweis des Elternknotens wird zu `None`.
   - Ein Kind: Das Kind ersetzt den gelöschten Knoten.
   - Zwei Kinder: Der Inorder Successor ersetzt den Wert; anschließend wird
     dieser Successor im rechten Teilbaum entfernt.
4. Ergänze eigene Bäume: einen vollständig nach rechts degenerierten Baum,
   einen Baum mit nur linken Kindern und einen möglichst ausgeglichenen Baum.
5. Führe die Tests aus und ergänze mindestens einen eigenen Kantenfall.

## Ausführen

Wechsle in diesen Projektordner und starte:

```bash
python3 demo.py
python3 -m pytest -q
```

## Erwartete Komplexität

Mit Baumhöhe `h` benötigen Insert, Suche, Delete, Minimum und Maximum `O(h)`
Zeit. In einem ausgeglichenen Baum ist `h = O(log n)`, in einem degenerierten
Baum dagegen `h = O(n)`. Die Traversierungen und die ASCII-Ausgabe besuchen
jeden Knoten einmal und benötigen daher `O(n)` Zeit. Rekursive Traversierungen
belegen `O(h)` Stack-Speicher; die Level-Order-Queue benötigt im Worst Case
`O(n)` Speicher.

## Fertig, wenn …

- Insert und Suche für vorhandene, fehlende und doppelte Werte korrekt sind,
- Delete ein Blatt, einen Knoten mit einem Kind, einen Knoten mit zwei Kindern
  sowie die Wurzel korrekt behandelt,
- Minimum und Maximum für gefüllte Bäume stimmen und bei Leere klar scheitern,
- Preorder, Inorder, Postorder und Level-Order erwartete Folgen liefern,
- die ASCII-Darstellung die Links-/Rechtsstruktur eindeutig sichtbar macht,
- nach jeder getesteten Änderung Inorder-Ausgabe und Größe zur Referenz passen,
- `python3 demo.py` und `python3 -m pytest -q` fehlerfrei durchlaufen.
