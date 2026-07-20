# Projekt 02-medium: Baum-Aufgabenkatalog

## Ziel

Dieses Projekt trainiert fünf typische Interview- und Praxisaufgaben auf
Binärbäumen: Höhe, Balance, BST-Validierung, Lowest Common Ancestor (LCA) und
Serialisierung. Der Schwerpunkt liegt nicht auf einer weiteren Baumklasse,
sondern auf der Frage: **Welche Information muss ein rekursiver Aufruf an seinen
Elternknoten zurückgeben?**

Das Projekt nutzt eigenständige Python-Skripte und pytest. So lassen sich die
Algorithmen einzeln importieren und Kantenfälle präziser untersuchen als in einem
linearen Notebook. Die Datenstruktur ist absichtlich klein; die Rekursion steht
im Mittelpunkt.

## Dateien

- `tree_algorithms.py`: Knotenmodell und alle fünf Aufgaben mit Hilfsfunktionen
- `demo.py`: gemeinsamer Durchlauf auf einem konkreten Beispielbaum
- `test_tree_algorithms.py`: Tests für reguläre Fälle, Grenzfälle und Fehlerdaten

## Gemeinsamer Beispielbaum

Die Einfügefolge `8, 3, 10, 1, 6, 14, 4, 7, 13` erzeugt:

```text
              8
            /   \
           3     10
          / \      \
         1   6      14
            / \     /
           4   7   13
```

Die Höhe wird in **Kanten** gemessen. Deshalb hat ein Blatt Höhe `0`, der leere
Baum Höhe `-1` und der Beispielbaum Höhe `3`.

## 1. Höhe: Information von unten nach oben

Ein Knoten kennt seine Höhe erst, wenn beide Kinder geantwortet haben. Der
Rekursionsbaum für den Teilbaum mit Wurzel `6` sieht so aus:

```text
height(6)
+-- height(4)
|   +-- height(None) = -1
|   +-- height(None) = -1
|   `-- return 0
+-- height(7)
|   +-- height(None) = -1
|   +-- height(None) = -1
|   `-- return 0
`-- return 1 + max(0, 0) = 1
```

Damit gilt rekursiv `height(node) = 1 + max(height(left), height(right))`.
Jeder Knoten wird genau einmal besucht: Zeit `O(n)`, Call Stack `O(h)`.

## 2. Balance: zwei Antworten in einem Rückweg

Die langsame Idee berechnet an jedem Knoten die Höhen beider Teilbäume neu und
kann dadurch `O(n²)` erreichen. `balance_result` bündelt stattdessen Höhe und
Balancezustand in einer Antwort:

```text
balance(8)
+-- balance(3)  -> height=2, balanced=True
+-- balance(10) -> height=2, balanced=False
`-- combine     -> height=3, balanced=False
```

Ein Knoten ist nur dann ausgeglichen, wenn beide Kinder ausgeglichen sind und
ihre Höhen höchstens um eins differieren. Das ist eine Postorder-Auswertung in
`O(n)` Zeit.

## 3. BST validieren: Grenzen vererben

Nur Eltern und Kinder zu vergleichen reicht nicht. Im folgenden Baum ist `12`
zwar größer als sein Elternknoten `5`, liegt aber im linken Teilbaum von `10`:

```text
       10                    validate(10, -inf, +inf)
      /  \                   +-- validate(5, -inf, 10)
     5   15                  |   `-- validate(12, 5, 10) -> False
      \
      12
```

Jeder rekursive Aufruf erhält ein offenes Intervall. Links wird die aktuelle
Knotenzahl zur neuen Obergrenze, rechts zur neuen Untergrenze. Die offenen
Grenzen verbieten zugleich Duplikate. Aufwand: `O(n)` Zeit und `O(h)` Stack.

## 4. Lowest Common Ancestor: Funde zusammenführen

Der allgemeine LCA-Algorithmus nutzt die BST-Ordnung absichtlich nicht und
funktioniert deshalb auf jedem Binärbaum mit eindeutigen Werten:

```text
LCA(4, 7) unter 3
+-- subtree(1) -> found neither
+-- subtree(6)
|   +-- subtree(4) -> found 4
|   +-- subtree(7) -> found 7
|   `-- both found -> candidate 6
`-- propagate candidate 6
```

Zusätzlich zum Kandidaten liefert jeder Aufruf zwei Wahrheitswerte zurück. Daher
kann die Funktion `None` liefern, wenn eines der Ziele gar nicht im Baum liegt.
Auch hier wird jeder Knoten nur einmal besucht.

## 5. Serialisierung: Struktur durch Nullmarker erhalten

Nur die Werte zu speichern reicht nicht, weil verschiedene Strukturen dieselbe
Wertfolge besitzen können. Preorder plus explizite Nullmarker ist eindeutig:

```text
    2                 serialize(2)
   / \                +-- value 2
  1   3               +-- serialize(1) -> 1, null, null
                      `-- serialize(3) -> 3, null, null

tokens: [2, 1, null, null, 3, null, null]
```

`deserialize` liest die Folge mit einer gemeinsamen Position. Jeder Wert baut
einen Knoten und fordert danach genau zwei Teilbäume an; `null` beendet einen
Ast. Die Tokenfolge liegt als JSON vor, sodass auch Strings mit Leerzeichen oder
Kommas sicher erhalten bleiben. `None` ist als Knotenwert ausgeschlossen, weil
es den Nullmarker bezeichnet.

## Aufgabenstellung

1. Zeichne für jede Funktion den Rekursionsbaum auf einem Baum mit höchstens
   sieben Knoten und notiere die Rückgabe jedes Aufrufs.
2. Begründe, warum `balance_result` Höhe und Balance gemeinsam berechnet.
3. Konstruiere einen Baum, der alle lokalen Eltern-Kind-Vergleiche besteht, aber
   kein BST ist. Prüfe ihn mit `is_valid_bst`.
4. Teste LCA einmal mit Zielen in verschiedenen Teilbäumen, einmal mit einem
   Vorfahren als Ziel und einmal mit einem fehlenden Wert.
5. Serialisiere einen eigenen Baum von Hand. Zähle für `n` Knoten die Werte und
   Nullmarker und vergleiche das Ergebnis mit dem Test.
6. Ergänze mindestens einen eigenen Test pro Algorithmus.

## Ausführen

Im Projektordner:

```bash
python3 demo.py
python3 -m pytest -q
```

## Hinweise

- Formuliere immer zuerst den Base Case für `None`.
- Entscheide dann, welche Information der Elternaufruf benötigt.
- Verändere bei der BST-Validierung nicht den Baum; verändere die Grenzen.
- Prüfe bei Round Trips nicht nur die Werte, sondern die gesamte Struktur.
- Ein Algorithmus, der Höhe mehrfach je Teilbaum berechnet, ist nicht linear.

## Fertig, wenn …

- die Höhenkonvention für leeren Baum, Blatt und inneren Knoten klar ist,
- der Balance-Check jeden Knoten nur einmal besucht,
- die BST-Validierung auch tiefe und doppelte Verstöße erkennt,
- LCA auf allgemeinen Binärbäumen arbeitet und fehlende Ziele behandelt,
- Serialisierung und Deserialisierung Struktur und JSON-Werte erhalten,
- unvollständige oder überschüssige Tokenfolgen verständlich abgelehnt werden,
- Demo, Syntaxprüfung und alle Tests fehlerfrei durchlaufen.
