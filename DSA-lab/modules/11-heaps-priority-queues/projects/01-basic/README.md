# Projekt 01-basic: Binary Min-Heap selbst bauen

## Ziel

In diesem Projekt implementierst du einen Binary Min-Heap vollständig selbst.
Die Daten liegen in einem normalen Python-Array; Eltern- und Kindbeziehungen
werden ausschließlich aus Indizes berechnet. `heapq` wird nicht in der Lösung
verwendet, sondern nur in den Tests als unabhängige Referenz.

Das Projekt ist als importierbares Python-Modul mit separater Demo und
pytest-Suite aufgebaut. So bleiben die Datenstruktur, die sichtbare Simulation
und die automatische Korrektheitsprüfung klar getrennt.

## Dateien

- `binary_heap.py`: `MinHeap` und Heap Sort auf Basis des Eigenbaus
- `demo.py`: sichtbare Zustände bei Heapify, Push, Pop und Sortierung
- `test_binary_heap.py`: Unit- und Property-Tests gegen `heapq` und `sorted`

## Implementierte Schnittstelle

```text
MinHeap(values)       Eingabe kopieren und bottom-up heapifizieren
push(value)           am Ende einfügen, dann Sift-Up
peek()                Minimum an Index 0 lesen
pop()                 Minimum entfernen, dann Sift-Down
heapify(values)       Inhalt ersetzen und in O(n) aufbauen
to_list()             defensive Kopie des internen Heap-Arrays
is_valid()            jede Eltern-Kind-Kante prüfen
heap_sort(values)      Eigenheap bauen und wiederholt Minimum poppen
```

Duplikate sind erlaubt. Die Eingabe wird kopiert, damit Konstruktion und Heap
Sort die Liste des Aufrufers nicht verändern. Direktes Mutieren gespeicherter
Objekte ist nicht erlaubt, weil dadurch die Heap-Invariante unbemerkt ungültig
werden kann.

## Aufgabenstellung

1. Zeichne für mindestens zehn Arrayindizes Parent, linkes und rechtes Kind ein.
2. Verfolge `push(3)` auf `[4, 7, 9, 10, 12, 15]`. Notiere nach jedem Tausch,
   warum nur noch die nächste Elternkante verletzt sein kann.
3. Verfolge `pop()` auf `[2, 5, 4, 12, 9, 8, 7]`. Begründe, warum Sift-Down mit
   dem kleineren Kind tauschen muss.
4. Führe Bottom-up-Heapify auf einer eigenen unsortierten Liste von Hand aus.
   Starte beim letzten inneren Knoten `(n - 2) // 2`.
5. Vergleiche das interne Array des Eigenheaps nicht direkt mit `heapq`: Zwei
   unterschiedliche Arrays können dieselbe gültige Heapordnung repräsentieren.
   Vergleiche stattdessen Invariante, Minimum, Multimenge und Pop-Reihenfolge.
6. Erkläre, warum die hier verwendete Heap-Sort-Variante `O(n)` Zusatzspeicher
   belegt, obwohl eine In-place-Max-Heap-Variante mit `O(1)` möglich ist.
7. Ergänze mindestens je einen eigenen Kantenfall für Push, Pop und Heapify.

## Invarianten

Nach jeder öffentlichen Operation müssen zwei Aussagen gelten:

1. Das Array repräsentiert einen vollständigen Binärbaum, weil nur am Ende
   angehängt oder entfernt wird.
2. Für jeden Kindindex `i > 0` gilt
   `heap[(i - 1) // 2] <= heap[i]`.

`is_valid()` prüft die zweite Aussage explizit. Die Tests kontrollieren zusätzlich,
dass keine Werte verloren gehen oder dupliziert werden.

## Komplexität

| Operation | Zeit | Zusatzspeicher |
|---|---:|---:|
| `peek` | `O(1)` | `O(1)` |
| `push` | `O(log n)` | `O(1)` |
| `pop` | `O(log n)` | `O(1)` |
| Bottom-up-`heapify` | `O(n)` | `O(n)` wegen Eingabekopie |
| `is_valid` | `O(n)` | `O(1)` |
| `heap_sort` | `O(n log n)` | `O(n)` |

Die Heapify-Schleife selbst arbeitet in-place im kopierten Array mit `O(1)`
weiterem Speicher. Die Kopie ist eine bewusste API-Entscheidung, nicht eine
Eigenschaft des Heapify-Algorithmus.

## Ausführen

Im Projektordner:

```bash
python3 demo.py
python3 -m pytest -q
```

## Fertig, wenn …

- Parent- und Kindindizes für nullbasierte Arrays korrekt sind,
- Push die Invariante per Sift-Up wiederherstellt,
- Pop leere, einelementige und mehrstufige Heaps korrekt behandelt,
- Heapify beliebige Iterables bottom-up verarbeitet,
- Duplikate, negative Werte, Strings und vergleichbare Tupel funktionieren,
- gemischte Zufallsoperationen dieselben Minima wie `heapq` liefern,
- Heap Sort für Kantenfälle und Zufallslisten mit `sorted` übereinstimmt,
- Demo, Syntaxprüfung und alle Tests fehlerfrei durchlaufen.
