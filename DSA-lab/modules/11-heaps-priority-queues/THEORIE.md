# Modul 11: Heaps & Priority Queues

Ein Notaufnahme-System behandelt nicht einfach die Person, die am längsten
wartet. Ein Betriebssystem startet nicht zwingend den ältesten Prozess. Und wenn
eine Datenpipeline aus Millionen Messwerten nur die 20 größten Ausreißer sucht,
muss sie nicht alle Werte vollständig sortieren. In allen drei Situationen lautet
die zentrale Frage:

> Welches Element ist **jetzt** am wichtigsten?

Eine **Priority Queue** formuliert diese Frage als abstrakte Schnittstelle. Ein
**Binary Heap** ist die typische Datenstruktur, die diese Schnittstelle effizient
implementiert. Dieses Modul entwickelt den Heap konsequent in drei Schritten:
zuerst die Intuition, dann konkrete Zustandsübergänge, zuletzt Invarianten,
Korrektheit und Komplexität.

## Lernziele

Nach diesem Modul kannst du:

- Priority Queue und Binary Heap begrifflich trennen,
- einen Min- oder Max-Heap in einem Array darstellen,
- `push`, `peek`, `pop`, Sift-Up und Sift-Down begründen,
- Bottom-up-Heapify implementieren und dessen Laufzeit `O(n)` erklären,
- Heap Sort herleiten und seine Eigenschaften einordnen,
- Pythons `heapq` sicher mit Zahlen und Prioritätstupeln einsetzen,
- für Top-K den richtigen Heap-Typ und die richtige Heap-Größe wählen,
- k sortierte Folgen effizient zusammenführen,
- einen laufenden Median mit zwei Heaps pflegen,
- Priority Queues in Graphalgorithmen und Scheduling-Systemen wiedererkennen.

---

# Teil I — Intuition

## 1. Warum eine normale Queue nicht reicht

Eine FIFO-Queue beantwortet die Frage: „Wer kam zuerst?“ Das ist richtig für eine
Kassenschlange, aber falsch für zeitkritische Jobs. Eine Priority Queue ordnet
nicht nach Einfügezeit, sondern nach einem **Priority Key**.

Angenommen, die kleinere Zahl bedeutet höhere Dringlichkeit:

| Ankunft | Aufgabe | Priorität |
|---:|---|---:|
| 1 | Backup erstellen | 8 |
| 2 | Sicherheitsalarm prüfen | 1 |
| 3 | Bericht exportieren | 5 |

FIFO würde `Backup` zuerst liefern. Eine Min-Priority-Queue liefert
`Sicherheitsalarm`, weil `1` der kleinste Priority Key ist.

Die abstrakten Operationen lauten:

- `push(item, priority)`: ein Element aufnehmen,
- `peek()`: das wichtigste Element ansehen,
- `pop()`: das wichtigste Element entfernen,
- optional `change_priority`: eine Priorität ändern.

Die Priority Queue schreibt nicht vor, **wie** diese Operationen intern umgesetzt
werden. Sie ist ein Abstract Data Type. Ein Binary Heap ist eine konkrete
Implementierung.

## 2. Warum nicht einfach sortieren?

Zwei einfache Implementierungen zeigen den Zielkonflikt:

| Repräsentation | `push` | `peek` | `pop` des Minimums |
|---|---:|---:|---:|
| unsortierte Liste | `O(1)` | `O(n)` | `O(n)` |
| sortiertes Array | `O(n)` | `O(1)` | `O(1)` am Rand |
| Binary Min-Heap | `O(log n)` | `O(1)` | `O(log n)` |

Eine unsortierte Liste verschiebt die Kosten zur Entnahme. Ein sortiertes Array
verschiebt sie zum Einfügen, weil Platz geschaffen werden muss. Der Heap verlangt
weniger Ordnung: Er garantiert nur, dass das wichtigste Element an der Wurzel
liegt. Dadurch bleiben Einfügen und Entnehmen logarithmisch.

```python
def pop_min_unsorted(values: list[int]) -> int:
    """Remove a minimum after a linear scan."""
    if not values:
        raise IndexError("empty collection")
    minimum_index = min(range(len(values)), key=values.__getitem__)
    return values.pop(minimum_index)


data = [9, 2, 7, 4]
assert pop_min_unsorted(data) == 2
assert data == [9, 7, 4]
```

## 3. Die entscheidende Sparsamkeit des Heaps

Ein sortiertes Array kennt die vollständige Reihenfolge aller Elemente. Ein
Min-Heap kennt nur lokale Beziehungen:

> Jeder Elternknoten ist kleiner oder gleich seinen Kindern.

Aus dieser lokalen Regel folgt global nur: Die Wurzel ist ein Minimum. Zwischen
Geschwistern und zwischen verschiedenen Teilbäumen besteht keine
Sortiergarantie. Genau diese unvollständige Ordnung spart Arbeit.

```text
Gültiger Min-Heap:          Kein sortiertes Array:

          2                 [2, 5, 4, 12, 9, 8, 7]
       /     \
      5       4             5 steht vor 4,
    /  \     / \            trotzdem ist der Heap gültig.
   12   9   8   7
```

---

# Teil II — Simulation: Der Binary Heap als Array

## 4. Zwei Invarianten statt Zeiger

Ein Binary Heap erfüllt zwei unabhängige Invarianten.

### Forminvariante

Der Baum ist **vollständig** (complete): Alle Ebenen außer eventuell der letzten
sind gefüllt; die letzte Ebene wird lückenlos von links nach rechts belegt.

### Ordnungsinvariante

Beim Min-Heap gilt für jede Eltern-Kind-Kante `parent <= child`. Beim Max-Heap
gilt entsprechend `parent >= child`.

Die Forminvariante erlaubt eine kompakte Arraydarstellung ohne Node-Objekte und
ohne Zeiger. Bei nullbasierter Indizierung gelten:

```text
parent(i) = (i - 1) // 2       für i > 0
left(i)   = 2 * i + 1
right(i)  = 2 * i + 2
```

Für das Array `[2, 5, 4, 12, 9, 8, 7]` ergibt sich:

| Index `i` | Wert | Parent-Index | Left-Index | Right-Index |
|---:|---:|---:|---:|---:|
| 0 | 2 | — | 1 | 2 |
| 1 | 5 | 0 | 3 | 4 |
| 2 | 4 | 0 | 5 | 6 |
| 3 | 12 | 1 | 7 | 8 |

Indizes ab `n // 2` besitzen keine Kinder und sind Blätter.

```python
def relatives(index: int) -> tuple[int | None, int, int]:
    """Return parent, left-child, and right-child indices."""
    parent = None if index == 0 else (index - 1) // 2
    return parent, 2 * index + 1, 2 * index + 2


assert relatives(0) == (None, 1, 2)
assert relatives(5) == (2, 11, 12)
```

## 5. Die Heap-Eigenschaft prüfen

Die Arrayreihenfolge muss nicht sortiert sein. Geprüft werden ausschließlich
existierende Eltern-Kind-Kanten:

```python
def is_min_heap(values: list[int]) -> bool:
    """Check the local min-heap invariant."""
    for child in range(1, len(values)):
        parent = (child - 1) // 2
        if values[parent] > values[child]:
            return False
    return True


assert is_min_heap([2, 5, 4, 12, 9, 8, 7])
assert not is_min_heap([2, 5, 1])
assert is_min_heap([])
```

Lokale Vergleiche genügen: Folge von einem beliebigen Knoten den Elternkanten
bis zur Wurzel. Jeder Schritt kann den Wert im Min-Heap nur verkleinern oder
gleich lassen. Daher ist die Wurzel nicht größer als dieser Knoten. Das gilt für
jeden Knoten, also ist die Wurzel global minimal.

## 6. Push und Sift-Up von Hand

Wir starten mit:

```text
Array: [4, 7, 9, 10, 12, 15]

          4
       /     \
      7       9
    /  \     /
   10  12   15
```

### Schritt 1: Form erhalten

Der neue Wert `3` wird an das Arrayende angehängt. Dadurch bleibt der Baum
vollständig:

```text
[4, 7, 9, 10, 12, 15, 3]
                         ^ index 6
```

Die einzige mögliche Ordnungsverletzung liegt auf dem Pfad vom neuen Blatt zur
Wurzel.

### Schritt 2: Mit dem Elternknoten tauschen

| Zustand | Index | Parent | Vergleich | Aktion |
|---|---:|---:|---|---|
| `[4,7,9,10,12,15,3]` | 6 | 2 | `3 < 9` | tauschen |
| `[4,7,3,10,12,15,9]` | 2 | 0 | `3 < 4` | tauschen |
| `[3,7,4,10,12,15,9]` | 0 | — | Wurzel erreicht | stoppen |

Sift-Up berührt höchstens einen Knoten pro Ebene. Ein vollständiger Baum mit
`n` Elementen hat Höhe `floor(log2(n))`; damit kostet `push` `O(log n)`.

```python
def heappush(heap: list[int], value: int) -> None:
    """Push one value into a min-heap."""
    heap.append(value)
    child = len(heap) - 1
    while child > 0:
        parent = (child - 1) // 2
        if heap[parent] <= heap[child]:
            break
        heap[parent], heap[child] = heap[child], heap[parent]
        child = parent


heap = [4, 7, 9, 10, 12, 15]
heappush(heap, 3)
assert heap == [3, 7, 4, 10, 12, 15, 9]
assert is_min_heap(heap)
```

### Warum Sift-Up korrekt ist

Vor dem Anhängen war der alte Heap gültig. Das neue Blatt besitzt noch keine
Kinder; nur die Kante zu seinem Parent kann verletzt sein. Nach einem Tausch ist
die Kante nach unten korrekt: Der größere Elternwert nimmt die alte Kindposition
ein, unter der bereits nur noch größere Werte liegen. Eine neue Verletzung kann
nur zur nächsten Elternkante entstehen. Beim Stopp sind alle Kanten wieder
gültig.

## 7. Peek: billig, aber nicht entfernen

Die Wurzel steht an Index `0`. `peek` ist deshalb `O(1)`:

```python
def heappeek(heap: list[int]) -> int:
    """Return a heap minimum without removing it."""
    if not heap:
        raise IndexError("peek from empty heap")
    return heap[0]


assert heappeek([3, 7, 4, 10]) == 3
```

Das Entfernen von Index `0` mit normalem Listen-Shift würde `O(n)` kosten und
eine Lücke an der Wurzel hinterlassen. Der Heap nutzt stattdessen sein letztes
Element.

## 8. Pop und Sift-Down von Hand

Ausgangsheap:

```text
[2, 5, 4, 12, 9, 8, 7]
```

### Schritt 1: Minimum sichern, Form erhalten

Wir sichern `2`, entfernen das letzte Element `7` und setzen es an die Wurzel:

```text
[7, 5, 4, 12, 9, 8]
```

Der Baum bleibt vollständig. Nur die Ordnungsinvariante kann auf dem Wurzelpfad
nach unten verletzt sein.

### Schritt 2: Mit dem kleineren Kind tauschen

| Zustand | Index | Kinderwerte | Wahl | Aktion |
|---|---:|---|---:|---|
| `[7,5,4,12,9,8]` | 0 | `5`, `4` | `4` bei Index 2 | tauschen |
| `[4,5,7,12,9,8]` | 2 | `8`, keines | `8` | `7 <= 8`, stoppen |

Warum muss das **kleinere** Kind gewählt werden? Ein Tausch mit `5` ließe `4`
als Kind der neuen Wurzel `5` zurück und verletzte die Invariante sofort.

```python
def sift_down(heap: list[int], start: int, size: int | None = None) -> None:
    """Repair a min-heap downward within heap[:size]."""
    if size is None:
        size = len(heap)
    parent = start
    while True:
        left = 2 * parent + 1
        if left >= size:
            return
        right = left + 1
        smaller = left
        if right < size and heap[right] < heap[left]:
            smaller = right
        if heap[parent] <= heap[smaller]:
            return
        heap[parent], heap[smaller] = heap[smaller], heap[parent]
        parent = smaller


heap = [7, 5, 4, 12, 9, 8]
sift_down(heap, 0)
assert heap == [4, 5, 7, 12, 9, 8]
```

```python
def heappop(heap: list[int]) -> int:
    """Remove and return the minimum of a min-heap."""
    if not heap:
        raise IndexError("pop from empty heap")
    minimum = heap[0]
    last = heap.pop()
    if heap:
        heap[0] = last
        sift_down(heap, 0)
    return minimum


heap = [2, 5, 4, 12, 9, 8, 7]
assert heappop(heap) == 2
assert heap == [4, 5, 7, 12, 9, 8]
assert is_min_heap(heap)
```

Der Sonderfall mit genau einem Element ist wichtig: Nach `pop()` ist die Liste
leer, also darf nicht mehr auf Index `0` geschrieben werden.

## 9. Bottom-up-Heapify: ein Array zum Heap machen

Eine beliebige Liste ist selten ein Heap:

```text
[9, 4, 7, 1, -2, 6, 5]
```

Alle Blätter sind für sich bereits gültige Heaps. Bei `n = 7` ist der letzte
innere Knoten `(n - 2) // 2 = 2`. Wir bearbeiten innere Knoten rückwärts:

| Startindex | Zustand nach Sift-Down | Erklärung |
|---:|---|---|
| 2 | `[9,4,5,1,-2,6,7]` | `7` tauscht mit Kind `5` |
| 1 | `[9,-2,5,1,4,6,7]` | `4` tauscht mit `-2` |
| 0 | `[-2,1,5,9,4,6,7]` | `9` sinkt über `1` |

Wenn Index `i` bearbeitet wird, sind seine Kinderwurzeln bereits gültige Heaps.
Sift-Down macht daraus zusammen mit `i` einen größeren gültigen Heap.

```python
def heapify(values: list[int]) -> None:
    """Transform values into a min-heap in place."""
    for parent in range((len(values) - 2) // 2, -1, -1):
        sift_down(values, parent)


values = [9, 4, 7, 1, -2, 6, 5]
heapify(values)
assert values == [-2, 1, 5, 9, 4, 6, 7]
assert is_min_heap(values)
```

### Warum Heapify nicht `O(n log n)` kostet

Die grobe obere Schranke „`n` Knoten mal `log n`“ ist korrekt, aber nicht scharf.
Die meisten Knoten liegen weit unten:

- ungefähr `n/2` Knoten sind Blätter und kosten `0` Schritte,
- ungefähr `n/4` Knoten können höchstens `1` Ebene sinken,
- ungefähr `n/8` Knoten können höchstens `2` Ebenen sinken,
- nur sehr wenige Knoten liegen nahe der Wurzel und können weit sinken.

Die Gesamtarbeit wird durch folgende Reihe begrenzt:

```text
n/4 * 1 + n/8 * 2 + n/16 * 3 + ...
= n * (1/4 + 2/8 + 3/16 + ...)
< n
```

Die gewichtete geometrische Reihe konvergiert zu einer Konstanten. Deshalb ist
Bottom-up-Heapify `O(n)`. Das sukzessive Einfügen von `n` Elementen ist dagegen
im Worst Case `O(n log n)`.

```python
def heapify_by_push(values: list[int]) -> list[int]:
    """Build a heap by repeated pushes; correct but asymptotically slower."""
    result: list[int] = []
    for value in values:
        heappush(result, value)
    return result


source = [9, 4, 7, 1, -2, 6, 5]
pushed = heapify_by_push(source)
assert is_min_heap(pushed)
assert sorted(pushed) == sorted(source)
```

Wichtig: Verschiedene gültige Heap-Arrays können dieselbe Wertemenge
repräsentieren. Ein Test sollte deshalb Invariante und Multimenge prüfen, nicht
blind ein einziges Layout erwarten.

---

# Teil III — Formalisierung

## 10. Definition und Komplexitäten

Ein **Binary Min-Heap** ist ein vollständiger Binärbaum, für dessen jeden Knoten
`v` und jedes direkte Kind `c` gilt:

```text
key(v) <= key(c)
```

Beim Max-Heap wird die Relation umgedreht. Aus Vollständigkeit folgt Höhe
`h = floor(log2 n)`. Die Standardoperationen sind:

| Operation | Zeit | Zusatzspeicher | Kernidee |
|---|---:|---:|---|
| `peek` | `O(1)` | `O(1)` | Wurzel lesen |
| `push` | `O(log n)` | `O(1)` | Ende + Sift-Up |
| `pop` | `O(log n)` | `O(1)` | Ende zur Wurzel + Sift-Down |
| Bottom-up-Heapify | `O(n)` | `O(1)` | innere Knoten rückwärts |
| beliebigen Wert suchen | `O(n)` | `O(1)` | keine globale Ordnung |
| Minimum im Max-Heap | `O(n)` | `O(1)` | liegt irgendwo in Blättern |

### Schleifeninvariante von Sift-Up

Vor jeder Iteration sind alle Heap-Kanten korrekt, außer eventuell der Kante
zwischen `child` und seinem Parent. Ein Tausch repariert diese Kante nach unten;
die einzige mögliche Verletzung wandert eine Ebene nach oben. Beim Abbruch gibt
es keine Verletzung mehr.

### Schleifeninvariante von Sift-Down

Vor jeder Iteration sind beide Kindteilbäume gültige Heaps. Nur die Beziehungen
zwischen `parent` und seinen Kindern können falsch sein. Der Tausch mit dem
kleineren Kind setzt an der alten Parentposition das kleinste der drei Elemente
ein. Die einzige mögliche Verletzung wandert in den gewählten Kindteilbaum.

## 11. Heap Sort

Ein Max-Heap legt das größte Element an die Wurzel. Für aufsteigende Sortierung
wird diese Wurzel mit dem letzten Element des aktiven Bereichs getauscht. Danach
ist das Maximum an seiner endgültigen Position. Sift-Down repariert nur den
verkleinerten Präfix.

```text
Max-Heap:       [9, 7, 6, 4, 2, 5]
Swap 0, 5:      [5, 7, 6, 4, 2 | 9]
Sift-Down:      [7, 5, 6, 4, 2 | 9]
Swap 0, 4:      [2, 5, 6, 4 | 7, 9]
Sift-Down:      [6, 5, 2, 4 | 7, 9]
...
Ergebnis:       [2, 4, 5, 6, 7, 9]
```

```python
def max_sift_down(values: list[int], start: int, size: int) -> None:
    """Repair a max-heap inside values[:size]."""
    parent = start
    while True:
        left = 2 * parent + 1
        if left >= size:
            return
        right = left + 1
        larger = left
        if right < size and values[right] > values[left]:
            larger = right
        if values[parent] >= values[larger]:
            return
        values[parent], values[larger] = values[larger], values[parent]
        parent = larger


def heap_sort(values: list[int]) -> None:
    """Sort values ascending in place with a max-heap."""
    for parent in range((len(values) - 2) // 2, -1, -1):
        max_sift_down(values, parent, len(values))
    for end in range(len(values) - 1, 0, -1):
        values[0], values[end] = values[end], values[0]
        max_sift_down(values, 0, end)


values = [7, 2, 9, 4, 6, 5]
heap_sort(values)
assert values == [2, 4, 5, 6, 7, 9]
```

Heap Sort benötigt `O(n)` für Heapify und danach `n - 1` Entnahmen zu jeweils
`O(log n)`: insgesamt `O(n log n)` in Best, Average und Worst Case. Die
arraybasierte Variante arbeitet mit `O(1)` Zusatzspeicher. Sie ist aber nicht
stabil: Gleiche Schlüssel können beim Tauschen ihre relative Reihenfolge verlieren.

## 12. Python: `heapq`

Pythons Standardmodul `heapq` arbeitet direkt auf normalen Listen und stellt
einen Min-Heap bereit. Die wichtigsten Funktionen sind:

- `heapq.heapify(list)` verändert die Liste in `O(n)`,
- `heapq.heappush(heap, item)` fügt ein,
- `heapq.heappop(heap)` entfernt das Minimum,
- `heapq.heappushpop(heap, item)` kombiniert Push und Pop effizient,
- `heapq.heapreplace(heap, item)` poppt zuerst und pusht dann; der Heap muss
  nicht leer sein,
- `heapq.nsmallest` und `heapq.nlargest` sind Komfortfunktionen für kleine `k`.

```python
import heapq


values = [9, 4, 7, 1, -2, 6, 5]
heapq.heapify(values)
assert values[0] == -2
heapq.heappush(values, -5)
assert heapq.heappop(values) == -5
assert [heapq.heappop(values) for _ in range(len(values))] == [-2, 1, 4, 5, 6, 7, 9]
```

### Der Max-Heap-Trick

Für Zahlen wird meist das Vorzeichen negiert. Das kleinste negative Element
entspricht dem größten Originalwert:

```python
import heapq


max_heap: list[int] = []
for value in [4, 9, 2, 7]:
    heapq.heappush(max_heap, -value)
descending = [-heapq.heappop(max_heap) for _ in range(len(max_heap))]
assert descending == [9, 7, 4, 2]
```

Negation eignet sich nur für numerische Prioritäten. Bei komplexeren Objekten
wird ein vergleichbarer Key in ein Tupel gelegt oder ein passender Wrapper
definiert.

### Tupel und stabile Tie-Breaker

Tupel werden lexikografisch verglichen: zuerst Priorität, dann das zweite Feld
und so weiter. Wenn zwei Prioritäten gleich sind und die Nutzobjekte nicht
vergleichbar sind, verursacht `(priority, task)` einen `TypeError`. Ein monotoner
Zähler löst das Problem und erhält FIFO-Reihenfolge unter gleichen Prioritäten.

```python
import heapq
import itertools


counter = itertools.count()
queue: list[tuple[int, int, dict[str, str]]] = []
for priority, name in [(2, "report"), (1, "alarm"), (1, "hotfix")]:
    heapq.heappush(queue, (priority, next(counter), {"name": name}))

order = [heapq.heappop(queue)[2]["name"] for _ in range(len(queue))]
assert order == ["alarm", "hotfix", "report"]
```

### Prioritäten nachträglich ändern

`heapq` kennt keine direkte Decrease-Key-Operation. In vielen Algorithmen wird
stattdessen ein neuer Eintrag mit besserer Priorität gepusht. Beim Pop werden
veraltete Einträge übersprungen. Dieses **lazy deletion**-Muster ist oft einfacher
als die Position eines Elements im Heap aktuell zu halten.

---

# Teil IV — Kernmuster

## 13. Top-K: Den Heap klein halten

Die wichtigste Designfrage lautet nicht nur „Min- oder Max-Heap?“, sondern auch:

> Welche Elemente sollen im Heap bleiben, und welches davon soll leicht
> hinausgeworfen werden können?

### K größte Elemente

Halte einen **Min-Heap der Größe k**. Die Wurzel ist das kleinste Element unter
den bisher besten k und damit der aktuelle Grenzwert. Ein neuer Wert ersetzt die
Wurzel nur, wenn er größer ist.

Beispiel für `k = 3` und Stream `5, 1, 9, 3, 14, 8`:

| Wert | Heap der bisher 3 größten | Entscheidung |
|---:|---|---|
| 5 | `[5]` | aufnehmen |
| 1 | `[1,5]` | aufnehmen |
| 9 | `[1,5,9]` | aufnehmen |
| 3 | `[3,5,9]` | `3 > 1`, Grenze ersetzen |
| 14 | `[5,14,9]` | `14 > 3`, Grenze ersetzen |
| 8 | `[8,14,9]` | `8 > 5`, Grenze ersetzen |

```python
import heapq


def k_largest(values: list[int], k: int) -> list[int]:
    """Return the k largest values in descending order."""
    if k < 0:
        raise ValueError("k must be non-negative")
    heap: list[int] = []
    for value in values:
        if len(heap) < k:
            heapq.heappush(heap, value)
        elif k and value > heap[0]:
            heapq.heapreplace(heap, value)
    return sorted(heap, reverse=True)


assert k_largest([5, 1, 9, 3, 14, 8], 3) == [14, 9, 8]
assert k_largest([2, 2, 1], 2) == [2, 2]
assert k_largest([1, 2], 0) == []
```

Zeit: `O(n log k)`, Speicher: `O(k)`. Wenn `k` viel kleiner als `n` ist, ist das
attraktiver als vollständiges Sortieren in `O(n log n)`.

Für die **k kleinsten** Elemente ist die Logik spiegelbildlich: Ein Max-Heap der
Größe `k` hält das aktuell größte der kleinen Elemente an der Wurzel.

### Top-K häufigste Werte

Zuerst erzeugt eine Hash Map die Häufigkeiten. Danach hält ein Min-Heap nur die
`k` besten `(frequency, value)`-Paare. Das Muster kombiniert Hashing und Heap.

```python
import heapq
from collections import Counter


def top_k_frequent(values: list[str], k: int) -> list[tuple[str, int]]:
    """Return up to k values ordered by frequency descending, then name."""
    if k < 0:
        raise ValueError("k must be non-negative")
    counts = Counter(values)
    best = heapq.nlargest(k, counts.items(), key=lambda item: (item[1], item[0]))
    return [(value, frequency) for value, frequency in best]


result = top_k_frequent(["a", "b", "a", "c", "b", "a"], 2)
assert result == [("a", 3), ("b", 2)]
```

## 14. k-Way-Merge: Immer den kleinsten Kopf wählen

Gegeben sind `k` bereits sortierte Folgen mit insgesamt `n` Elementen. Ein
naiver Durchlauf könnte bei jeder Ausgabe alle `k` aktuellen Köpfe durchsuchen:
`O(nk)`. Ein Min-Heap speichert nur den aktuellen Kopf jeder nicht leeren Folge.

```text
A: [1, 7, 10]
B: [2, 3, 11]
C: [4, 8]

Heap startet mit (1,A), (2,B), (4,C).
Pop (1,A), danach Push (7,A).
Pop (2,B), danach Push (3,B).
Pop (3,B), danach Push (11,B).
...
```

Der Heap enthält höchstens `k` Einträge. Jede der `n` Ausgaben verursacht einen
Pop und meist einen Push: Zeit `O(n log k)`, Zusatzspeicher `O(k)` ohne Ausgabe.

```python
import heapq


def merge_sorted(sequences: list[list[int]]) -> list[int]:
    """Merge sorted sequences with a heap of current heads."""
    heap: list[tuple[int, int, int]] = []
    for sequence_index, sequence in enumerate(sequences):
        if sequence:
            heapq.heappush(heap, (sequence[0], sequence_index, 0))

    merged: list[int] = []
    while heap:
        value, sequence_index, position = heapq.heappop(heap)
        merged.append(value)
        next_position = position + 1
        sequence = sequences[sequence_index]
        if next_position < len(sequence):
            heapq.heappush(
                heap,
                (sequence[next_position], sequence_index, next_position),
            )
    return merged


assert merge_sorted([[1, 7, 10], [2, 3, 11], [4, 8]]) == [1, 2, 3, 4, 7, 8, 10, 11]
assert merge_sorted([[], [1], []]) == [1]
```

Der `sequence_index` ist zugleich Tie-Breaker, wenn zwei Kopfwerte gleich sind.
Das Muster ist die Grundlage externer Sortierung, Merge-Phasen in Datenbanken
und des Zusammenführens zeitlich sortierter Event-Streams.

## 15. Laufender Median mit zwei Heaps

Der Median eines Streams soll nach jeder neuen Zahl verfügbar sein. Vollständiges
Sortieren nach jedem Insert wäre teuer. Die Daten werden in zwei Hälften geteilt:

- `lower`: Max-Heap der kleineren Hälfte, in Python durch negierte Werte,
- `upper`: Min-Heap der größeren Hälfte.

Zwei Invarianten müssen gelten:

1. **Ordnung:** Jedes Element in `lower` ist `<=` jedem Element in `upper`.
2. **Balance:** Die Größen unterscheiden sich höchstens um eins; hier darf
   `lower` genau ein Element mehr besitzen.

Simulation für `5, 2, 10, 4`:

| Neu | Lower (inhaltlich) | Upper | Median |
|---:|---|---|---:|
| 5 | `[5]` | `[]` | 5 |
| 2 | `[2]` | `[5]` | 3,5 |
| 10 | `[5,2]` | `[10]` | 5 |
| 4 | `[4,2]` | `[5,10]` | 4,5 |

```python
import heapq


class RunningMedian:
    """Maintain a stream median with two heaps."""

    def __init__(self) -> None:
        self.lower: list[int] = []  # negatives form a max-heap
        self.upper: list[int] = []

    def add(self, value: int) -> None:
        if not self.lower or value <= -self.lower[0]:
            heapq.heappush(self.lower, -value)
        else:
            heapq.heappush(self.upper, value)

        if len(self.lower) > len(self.upper) + 1:
            heapq.heappush(self.upper, -heapq.heappop(self.lower))
        elif len(self.upper) > len(self.lower):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))

    def median(self) -> float:
        if not self.lower:
            raise ValueError("median of empty stream")
        if len(self.lower) == len(self.upper):
            return (-self.lower[0] + self.upper[0]) / 2
        return float(-self.lower[0])


tracker = RunningMedian()
medians = []
for value in [5, 2, 10, 4]:
    tracker.add(value)
    medians.append(tracker.median())
assert medians == [5.0, 3.5, 5.0, 4.5]
```

Jedes Insert benötigt `O(log n)`, der Median selbst `O(1)`, der Speicher `O(n)`.
Das Prinzip verallgemeinert sich auf Online-Statistiken, Quantile benötigen aber
andere oder erweiterte Datenstrukturen.

---

# Teil V — Priority Queues in Algorithmen und Systemen

## 16. Dijkstra-Vorgriff

Bei der Kürzeste-Wege-Suche nach Dijkstra ist die Priorität eines Knotens seine
bisher beste bekannte Distanz vom Start. Die nächste Verarbeitung nimmt immer
den Knoten mit der kleinsten vorläufigen Distanz.

```text
Priority Queue: (0, Start), (7, A), (12, B), ...
Pop liefert stets den aktuell nächsten unerledigten Knoten.
```

Findet der Algorithmus eine bessere Distanz, wird häufig ein neuer
`(distance, node)`-Eintrag gepusht. Ein später gepoppter Eintrag wird verworfen,
wenn seine Distanz nicht mehr der gespeicherten Bestdistanz entspricht. Das ist
dasselbe Lazy-Deletion-Muster wie bei veränderlichen Prioritäten.

Mit Adjazenzliste und Binary Heap ergibt sich typischerweise
`O((V + E) log V)`, oft als `O(E log V)` geschrieben. Modul 13 entwickelt den
vollständigen Algorithmus.

## 17. Scheduler: Priorität ist eine Produktentscheidung

Ein Scheduler kann Jobs nach Dringlichkeit, Deadline, Restlaufzeit oder einer
Kombination auswählen. Der Heap macht die Auswahl effizient, entscheidet aber
nicht, ob die Priority Policy fair ist.

Ein reiner Priority Scheduler kann **Starvation** erzeugen: Wenn fortlaufend
hochpriorisierte Jobs eintreffen, wartet ein niedriger Job unbegrenzt. Mögliche
Gegenmaßnahmen sind:

- **Aging:** Priorität steigt mit der Wartezeit,
- Quoten pro Prioritätsklasse,
- Round-Robin innerhalb gleicher Priorität,
- Deadline-basierte Keys,
- begrenzte Bursts hochpriorisierter Arbeit.

Der Tie-Breaker im Tupel ist deshalb nicht bloß technisch. `(priority,
arrival_counter, job)` definiert FIFO-Verhalten innerhalb einer Klasse und macht
die Simulation deterministisch.

## 18. Data-Science-Bezüge

Heaps erscheinen in Data-Science-Pipelines oft indirekt:

- Top-K Features, Anomalien, Suchergebnisse oder häufigste Kategorien,
- Zusammenführen vorsortierter Shards oder Log-Streams,
- Beam Search und Best-First Search,
- Nearest-Neighbor-Verfahren mit begrenzter Kandidatenmenge,
- Streaming-Median und laufende Rangstatistiken,
- Priorisierung teurer Experimente oder Datenqualitätsalarme.

Die gemeinsame Struktur lautet: Es existiert eine große oder wachsende Menge,
aber pro Schritt interessiert nur das aktuell beste Element oder eine kleine
Grenzmenge.

---

# Teil VI — Fallstricke und Designentscheidungen

## 19. Häufige Denkfehler

### „Ein Heap ist sortiert“

Falsch. Nur Eltern-Kind-Beziehungen sind geordnet. Inorder-Traversierung oder
Arrayreihenfolge liefern keine sortierte Folge. Wiederholtes Poppen liefert
dagegen Elemente in Prioritätsreihenfolge.

### „Ich kann einen beliebigen Wert binär suchen“

Falsch. Aus `target > root` folgt nicht, in welchem Teilbaum das Ziel liegt.
Beliebige Suche ist im Worst Case `O(n)`.

### „Für k größte Werte brauche ich einen Max-Heap“

Nicht, wenn der Heap nur `k` Elemente halten soll. Dann muss das **schlechteste
der aktuell besten** Elemente schnell entfernbar sein: ein Min-Heap.

### „Heapify ist wiederholtes Pushen“

Beide erzeugen einen Heap, aber nur Bottom-up-Heapify garantiert die lineare
Konstruktion. Das Ergebnislayout darf verschieden sein.

### „Gleiche Priorität bedeutet automatisch stabile Reihenfolge“

Ein Heap ist nicht stabil. Für Stabilität muss die Ankunftsreihenfolge explizit
Teil des Vergleichsschlüssels sein.

## 20. Mutationen, Duplikate und spezielle Zahlen

- Duplikate verletzen die Heap-Invariante nicht; `<=` erlaubt Gleichheit.
- Ein Element im Array direkt zu verändern kann die Invariante brechen. Nach
  Prioritätsverbesserung ist je nach Heap Sift-Up, nach Verschlechterung
  Sift-Down erforderlich.
- Mutable Nutzobjekte sollten nicht selbst den Vergleich bestimmen. Ein
  unveränderlicher Prioritätstupel ist sicherer.
- `float('nan')` besitzt keine normale totale Ordnung. Vergleiche mit NaN sind
  fast immer `False` und können Heap-Annahmen zerstören. Daten sollten vor dem
  Einfügen validiert werden.
- Bei negierten Max-Heap-Keys muss beim Lesen und Schreiben konsequent das
  Vorzeichen zurücktransformiert werden.

## 21. Welche Struktur ist die richtige?

| Problem | Geeignete Struktur | Begründung |
|---|---|---|
| Immer kleinstes Element entnehmen | Min-Heap | Wurzel ist Minimum |
| Immer größtes Element entnehmen | Max-Heap | Wurzel ist Maximum |
| Beliebige Mitgliedschaft schnell testen | Hash Set | Heap-Suche ist linear |
| Alle Werte sortiert iterieren | sortiertes Array / Baum | Heap ist nur partiell geordnet |
| K größte aus großem Stream | Min-Heap der Größe `k` | Grenzwert liegt an Wurzel |
| Median eines Streams | Max-Heap + Min-Heap | beide mittleren Grenzen `O(1)` |
| FIFO-Verarbeitung | Queue | Prioritätsordnung unnötig |

---

# Teil VII — Transfer und Selbstkontrolle

## 22. Entscheidungsrezept für Heap-Probleme

Stelle in dieser Reihenfolge Fragen:

1. Muss wiederholt ein Extremwert geliefert werden?
2. Ist das gesuchte Extremum Minimum oder Maximum?
3. Soll der Heap alle Elemente oder nur `k` Kandidaten halten?
4. Welches Element muss an der Wurzel liegen, damit eine schlechte Kandidatin
   leicht entfernt werden kann?
5. Gibt es gleiche Prioritäten, nicht vergleichbare Payloads oder Stabilität?
6. Können Prioritäten nachträglich veralten?
7. Reicht Lazy Deletion oder wird eine Index Map für echtes Update benötigt?

## 23. Kontrollfragen

1. Warum ist `[1, 4, 2, 9, 7, 3]` ein gültiger Min-Heap, obwohl `4 > 2`?
2. Welche Kanten können direkt nach einem Push verletzt sein?
3. Warum tauscht Sift-Down beim Min-Heap mit dem kleineren Kind?
4. Welcher Index ist bei `n` Elementen der letzte innere Knoten?
5. Warum ist Heapify trotz einzelner `O(log n)`-Sift-Downs insgesamt `O(n)`?
6. Welchen Heap nutzt du für die 100 größten Werte eines Milliarden-Streams?
7. Was muss ein k-Way-Merge-Eintrag außer dem Wert speichern?
8. Welche zwei Invarianten hält der Running-Median-Algorithmus?
9. Warum kann `(priority, task)` in `heapq` bei gleicher Priorität scheitern?
10. Warum löst ein Heap das Fairnessproblem eines Schedulers nicht?

### Kurzantworten

1. `4` und `2` sind Geschwister; nur Eltern-Kind-Kanten sind geordnet.
2. Nur der Pfad vom neuen Blatt zur Wurzel.
3. Sonst könnte das kleinere Geschwister unter einem größeren Parent bleiben.
4. `(n - 2) // 2` für `n >= 2`.
5. Fast alle Knoten liegen unten und können nur sehr kurz sinken.
6. Einen Min-Heap der Größe 100.
7. Herkunftsfolge und Position, damit der Nachfolger nachgeschoben werden kann.
8. Ordnung zwischen den Hälften und Größenbalance.
9. Python versucht danach die möglicherweise nicht vergleichbaren Tasks zu
   vergleichen; ein Counter dient als Tie-Breaker.
10. Der Heap führt nur die vorgegebene Priority Policy effizient aus.

## 24. Kompakte Gesamtübersicht

```text
Priority Queue (ADT)
    |
    +-- Binary Heap (Implementierung)
            |
            +-- complete-tree invariant -> kompaktes Array
            +-- heap-order invariant    -> Extremum an Wurzel
            +-- push                    -> append + sift-up
            +-- pop                     -> last-to-root + sift-down
            +-- heapify                 -> innere Knoten bottom-up, O(n)
            |
            +-- Muster
                    +-- Top-K            -> begrenzter Gegen-Heap
                    +-- k-Way-Merge      -> Heap aktueller Köpfe
                    +-- Running Median   -> Max-Heap + Min-Heap
                    +-- Dijkstra         -> kleinste bekannte Distanz
                    `-- Scheduler        -> bester Priority Key
```

## 25. Ausblick auf die Projekte

Im Projekt **01-basic** wird ein Min-Heap vollständig selbst gebaut, gegen
`heapq` property-getestet und zu Heap Sort erweitert. **02-medium** isoliert die
drei wichtigsten Muster Top-K, k-Way-Merge und Running Median. **03-final** nutzt
eine Priority Queue in einem reproduzierbaren Job-Scheduler und vergleicht dessen
Wartezeiten pro Prioritätsklasse mit FIFO.

Die zentrale Einsicht für alle drei Projekte lautet:

> Ein Heap sortiert nicht alles. Er investiert genau so viel Ordnung, dass der
> nächste relevante Extremwert effizient verfügbar ist.
