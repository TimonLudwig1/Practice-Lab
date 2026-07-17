# Modul 05: Stacks & Queues

Stacks und Queues sind weniger durch ihren Speicher als durch ihre **Regel für
den nächsten Zugriff** definiert. Beide verwalten eine Folge von Elementen, aber
sie beantworten eine andere Frage:

- Ein **Stack** fragt: Welches Element kam zuletzt hinzu?
- Eine **Queue** fragt: Welches Element wartet schon am längsten?

Diese Reihenfolgeregel ist häufig wichtiger als der Datentyp der Elemente. Wer
in einem Problem Wörter wie „zuletzt geöffnet“, „rückgängig“, „verschachtelt“,
„in Ankunftsreihenfolge“ oder „Ebene für Ebene“ erkennt, hat oft bereits die
passende Datenstruktur gefunden.

---

## 1. Abstrakter Datentyp und Implementierung trennen

Ein abstrakter Datentyp beschreibt **welche Operationen und Garantien** gelten.
Er legt nicht fest, wie die Daten intern gespeichert werden.

Ein Stack kann beispielsweise auf einem dynamischen Array oder einer Linked
List beruhen. Eine Queue kann als Linked List, Ringpuffer oder sogar mit zwei
Stacks implementiert werden. Für den aufrufenden Code bleibt trotzdem dieselbe
LIFO- beziehungsweise FIFO-Semantik sichtbar.

> „Stack“ bedeutet nicht automatisch „Python-Liste“ und „Queue“ bedeutet nicht
> automatisch „Linked List“. Das sind mögliche Implementierungen, nicht die
> Definition der Datenstruktur.

---

## 2. Stack: Last In, First Out

Ein Stack arbeitet nach **LIFO**: *Last In, First Out*. Das zuletzt abgelegte
Element wird zuerst entnommen. Eine passende Alltagsanalogie ist ein Stapel
Teller: Man legt oben ab und nimmt oben weg.

### 2.1 Grundoperationen

| Operation | Bedeutung | Typische Laufzeit |
|---|---|---:|
| `push(x)` | `x` oben ablegen | O(1) amortisiert |
| `pop()` | oberstes Element entfernen und liefern | O(1) |
| `peek()` / `top()` | oberstes Element ansehen | O(1) |
| `is_empty()` | auf Leerzustand prüfen | O(1) |
| `len(stack)` | Elementzahl liefern | O(1) |

„Amortisiert“ ist bei einem Stack auf dynamischem Array wichtig: Ein einzelnes
`push` kann beim Vergrößern des Arrays O(n) kosten. Über viele Einfügungen
verteilen sich diese seltenen Kopierkosten jedoch zu O(1) pro Operation.

### 2.2 Operationen simulieren

```text
Start       top
             v
            [ ]

push(A)     [A] <- top
push(B)     [B] <- top
            [A]
push(C)     [C] <- top
            [B]
            [A]

pop() -> C  [B] <- top
            [A]
```

Nur die Oberseite ist öffentlich zugänglich. Ein Stack, bei dem man beliebige
mittlere Elemente direkt entnimmt, hält seine eigene Abstraktion nicht mehr ein.

### 2.3 Arraybasierter Stack

In Python bildet das Ende einer `list` die effiziente Stack-Oberseite. `append`
und `pop()` am Ende verschieben keine übrigen Elemente.

```python
from typing import Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")


class StackUnderflowError(IndexError):
    """Raised when a stack operation requires an existing element."""


class ArrayStack(Generic[T]):
    """A LIFO stack backed by a dynamic array."""

    def __init__(self, values: Iterable[T] = ()) -> None:
        self._items = list(values)

    def push(self, value: T) -> None:
        self._items.append(value)

    def pop(self) -> T:
        if not self._items:
            raise StackUnderflowError("cannot pop from an empty stack")
        return self._items.pop()

    def peek(self) -> T:
        if not self._items:
            raise StackUnderflowError("cannot peek into an empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return reversed(self._items)


stack = ArrayStack([10, 20])
stack.push(30)
assert stack.peek() == 30
assert stack.pop() == 30
assert list(stack) == [20, 10]
```

`list.pop(0)` wäre dagegen O(n), weil alle verbleibenden Referenzen nach links
verschoben werden müssten. Deshalb wird beim Stack immer dasselbe Listenende für
`push` und `pop` verwendet.

### 2.4 Stack auf Linked List

Alternativ kann der Listenkopf als Stack-Oberseite dienen:

```text
top -> [C | next] -> [B | next] -> [A | None]
```

Einfügen und Entfernen am Kopf kosten jeweils O(1). Gegenüber dem Array entstehen
aber ein zusätzlicher Zeiger pro Knoten, schlechtere Cache-Lokalität und mehr
Objekte. Dafür ist kein gelegentliches Resize nötig. In Python ist eine `list`
für einen gewöhnlichen Stack meistens die pragmatischere Wahl.

---

## 3. Queue: First In, First Out

Eine Queue arbeitet nach **FIFO**: *First In, First Out*. Das zuerst eingetroffene
Element wird zuerst bedient. Die Alltagsanalogie ist eine Warteschlange an einer
Kasse.

| Operation | Alternative Namen | Bedeutung |
|---|---|---|
| `enqueue(x)` | `put`, `offer` | hinten anstellen |
| `dequeue()` | `get`, `poll` | vorn entnehmen |
| `front()` | `peek` | vorderstes Element ansehen |
| `is_empty()` | — | auf Leerzustand prüfen |

### 3.1 Operationen simulieren

```text
enqueue(A)
front -> [A] <- rear

enqueue(B), enqueue(C)
front -> [A] [B] [C] <- rear

dequeue() -> A
front -> [B] [C] <- rear
```

Ein neues Element betritt die Queue am `rear`; das nächste Element verlässt sie
am `front`. Die beiden Enden haben unterschiedliche Aufgaben.

### 3.2 Queue auf Singly Linked List

Mit Referenzen auf Kopf **und** Ende sind beide Kernoperationen O(1):

```text
front                                      rear
  |                                          |
  v                                          v
[A | next] -> [B | next] -> [C | None]
```

- `enqueue`: neuen Knoten an `rear.next` anhängen und `rear` aktualisieren.
- `dequeue`: `front` auf `front.next` setzen.
- Wird das letzte Element entnommen, müssen `front` und `rear` beide `None`
  werden.

Ohne `rear` müsste `enqueue` die gesamte Liste in O(n) durchlaufen. Ohne saubere
Behandlung des letzten Elements könnte `rear` auf einen bereits entfernten
Knoten zeigen.

### 3.3 Warum eine Python-Liste meist keine Queue sein sollte

Diese Variante sieht naheliegend aus, ist aber teuer:

```text
items.append(value)  # enqueue at the right end: amortized O(1)
items.pop(0)         # dequeue at the left end: O(n)
```

Nach `pop(0)` müssen alle verbleibenden Referenzen um einen Platz verschoben
werden. Bei n Dequeue-Operationen aus einer gefüllten Liste können dadurch
insgesamt O(n²) Verschiebungen entstehen.

Ein bloßer wachsender Front-Index vermeidet das Verschieben zunächst, hält aber
bereits entnommene Plätze im Speicher. Ein Ringpuffer löst beide Probleme.

---

## 4. Ringpuffer: ein Array logisch im Kreis

Ein Ringpuffer nutzt ein Array fester Kapazität wiederholt. Nach dem letzten
physischen Index folgt logisch wieder Index 0.

Er benötigt typischerweise:

- `front`: Index des nächsten zu entnehmenden Elements,
- `size`: aktuelle Elementzahl,
- `capacity`: Länge des zugrunde liegenden Arrays.

Der nächste freie Index ergibt sich aus:

```text
rear = (front + size) mod capacity
```

Nach einer Entnahme wandert die Front mit:

```text
front = (front + 1) mod capacity
```

### 4.1 Wrap-around simulieren

Kapazität 5, zunächst drei Elemente:

```text
Index:    0    1    2    3    4
Array:   [A]  [B]  [C]  [ ]  [ ]
          ^         ^
        front      last
```

Nach zwei `dequeue` und drei weiteren `enqueue`:

```text
Index:    0    1    2    3    4
Array:   [F]  [ ]  [C]  [D]  [E]
          ^         ^
        last      front

Logische Reihenfolge: C, D, E, F
```

Die logische Ordnung stimmt nicht mehr mit der Reihenfolge der physischen
Indizes überein. Das Modulo verbindet beide Enden des Arrays.

### 4.2 Voll und leer unterscheiden

Wenn nur `front` und `rear` gespeichert werden, kann `front == rear` sowohl
„leer“ als auch „voll“ bedeuten. Übliche Lösungen sind:

1. zusätzlich `size` speichern,
2. einen Arrayplatz absichtlich unbenutzt lassen,
3. ein separates Voll-Flag führen.

Eine explizite Größe ist didaktisch klar und ermöglicht `len(queue)` in O(1).

### 4.3 Eigenschaften

| Eigenschaft | Ringpuffer |
|---|---|
| `enqueue` | O(1) |
| `dequeue` | O(1) |
| Speicher | O(K) für feste Kapazität K |
| Resize | je nach Vertrag verboten oder O(n) |
| Lokalität | gut, da zusammenhängendes Array |
| Fehlerfälle | Underflow bei leer, Overflow bei voll |

Ein Ringpuffer passt besonders gut zu begrenzten Puffern, Streaming-Daten,
Producer-Consumer-Systemen und „letzte K Werte“-Fenstern.

---

## 5. Queue aus zwei Stacks

FIFO lässt sich ausschließlich mit zwei LIFO-Strukturen erzeugen:

- `incoming` nimmt neue Elemente auf,
- `outgoing` liefert das nächste Element aus.

Bei `dequeue` gilt:

1. Ist `outgoing` nicht leer, dort oben entnehmen.
2. Sonst alle Elemente aus `incoming` nach `outgoing` umstapeln.
3. Danach oben aus `outgoing` entnehmen.

```text
incoming       transfer        outgoing
 top                            top
 [C]           C ->            [A]
 [B]           B ->            [B]
 [A]           A ->            [C]

Danach verlässt A als ältestes Element zuerst die Queue.
```

```python
class TwoStackQueue(Generic[T]):
    """A FIFO queue implemented with two LIFO stacks."""

    def __init__(self) -> None:
        self._incoming: list[T] = []
        self._outgoing: list[T] = []

    def enqueue(self, value: T) -> None:
        self._incoming.append(value)

    def dequeue(self) -> T:
        self._prepare_outgoing()
        return self._outgoing.pop()

    def front(self) -> T:
        self._prepare_outgoing()
        return self._outgoing[-1]

    def _prepare_outgoing(self) -> None:
        if not self._outgoing:
            while self._incoming:
                self._outgoing.append(self._incoming.pop())
        if not self._outgoing:
            raise IndexError("cannot read from an empty queue")

    def __len__(self) -> int:
        return len(self._incoming) + len(self._outgoing)


queue = TwoStackQueue[int]()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
assert queue.dequeue() == 1
queue.enqueue(4)
assert [queue.dequeue(), queue.dequeue(), queue.dequeue()] == [2, 3, 4]
```

Ein einzelnes `dequeue` kann beim Transfer O(n) kosten. Jedes Element wird aber
höchstens einmal nach `incoming`, einmal nach `outgoing` und einmal aus
`outgoing` bewegt. Über eine lange Operationsfolge sind `enqueue` und `dequeue`
daher **amortisiert O(1)**.

Würde man bei jeder Entnahme alles hin- und wieder zurückstapeln, ginge diese
Garantie verloren. Entscheidend ist, `outgoing` erst vollständig zu leeren.

---

## 6. Deque: beide Enden freigeben

Eine **Double-Ended Queue** erlaubt Einfügen und Entfernen an beiden Enden:

```text
appendleft <- [ A | B | C ] -> append
popleft    <- [ A | B | C ] -> pop
```

Damit kann eine Deque sowohl Stack als auch Queue darstellen:

| Verhalten | Einfügen | Entnehmen |
|---|---|---|
| Stack | `append` | `pop` |
| Queue | `append` | `popleft` |

Pythons `collections.deque` ist für O(1)-Operationen an beiden Enden ausgelegt.
Sie ist die Standardwahl für allgemeine Queues.

```python
from collections import deque

work_queue = deque(["job-a", "job-b"])
work_queue.append("job-c")
assert work_queue.popleft() == "job-a"

undo_stack = deque(["state-1", "state-2"])
undo_stack.append("state-3")
assert undo_stack.pop() == "state-3"

recent = deque(maxlen=3)
recent.extend([10, 20, 30, 40])
assert list(recent) == [20, 30, 40]
```

Eine Deque ist kein Ersatz für zufälligen Zugriff: Zugriffe nahe der Mitte sind
O(n). Wer häufig `items[i]` benötigt, braucht eher ein Array beziehungsweise
eine Liste.

---

## 7. Anwendungsmuster des Stacks

### 7.1 Klammerprüfung

Verschachtelte Klammern werden in umgekehrter Reihenfolge geschlossen. Das ist
LIFO:

```text
Eingabe:   ([{}])
Öffnen:    ( -> [ -> {
Schließen: } passt zu {, ] passt zu [, ) passt zu (
```

```python
def brackets_are_balanced(text: str) -> bool:
    """Return whether all brackets are correctly nested and paired."""

    closing_to_opening = {")": "(", "]": "[", "}": "{"}
    opening = set(closing_to_opening.values())
    stack: list[str] = []

    for character in text:
        if character in opening:
            stack.append(character)
        elif character in closing_to_opening:
            if not stack or stack.pop() != closing_to_opening[character]:
                return False
    return not stack


assert brackets_are_balanced("total[(row + 1)]")
assert not brackets_are_balanced("([)]")
assert not brackets_are_balanced("(()")
```

Die Laufzeit ist O(n), weil jedes Zeichen einmal betrachtet und jede Klammer
höchstens einmal abgelegt und entfernt wird. Der Zusatzspeicher ist im Worst Case
O(n).

### 7.2 Undo und Redo

Eine Undo-Historie ist ein Stack abgeschlossener Aktionen. Für Redo werden
häufig zwei Stacks verwendet:

```text
execute(action): action ausführen, auf undo legen, redo leeren
undo():          von undo nehmen, rückgängig machen, auf redo legen
redo():          von redo nehmen, erneut ausführen, auf undo legen
```

Nach einem Undo und einer abweichenden neuen Aktion entsteht eine neue
Historienverzweigung. Die alte Zukunft passt nicht mehr zum aktuellen Zustand,
weshalb die neue Aktion den Redo-Stack leert.

In Produktionscode speichert man entweder vollständige Zustände oder Commands
mit `execute`- und `undo`-Operation. Zustandskopien sind einfacher, Commands oft
speichereffizienter.

### 7.3 Ausdrucksauswertung

Stacks passen zu verschachtelten Ausdrücken und Operatorprioritäten. Ein
klassischer zweistufiger Ansatz ist:

1. Infix-Ausdruck mit dem **Shunting-Yard-Algorithmus** in Postfix umwandeln.
2. Postfix-Ausdruck mit einem Operanden-Stack auswerten.

```text
Infix:    3 + 4 * 2
Postfix:  3 4 2 * +

Stack-Auswertung:
3      -> [3]
4      -> [3, 4]
2      -> [3, 4, 2]
*      -> [3, 8]
+      -> [11]
```

Bei einem binären Operator ist die Entnahmereihenfolge wichtig:

```python
import operator


def evaluate_postfix(tokens: list[str]) -> float:
    """Evaluate a valid postfix expression containing binary operators."""

    operations = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }
    operands: list[float] = []
    for token in tokens:
        if token not in operations:
            operands.append(float(token))
            continue
        if len(operands) < 2:
            raise ValueError("operator has too few operands")
        right = operands.pop()
        left = operands.pop()
        operands.append(operations[token](left, right))

    if len(operands) != 1:
        raise ValueError("expression does not reduce to one value")
    return operands[0]


assert evaluate_postfix(["3", "4", "2", "*", "+"]) == 11
assert evaluate_postfix(["10", "3", "-"]) == 7
```

`left - right` ist nicht dasselbe wie `right - left`. Dieser kleine Fehler ist
bei Stack-Evaluatoren besonders häufig.

### 7.4 Monotonic Stack

Ein monotonic Stack hält seine Elemente stets monoton steigend oder fallend. Er
wird genutzt, um für jedes Element den nächsten größeren oder kleineren Wert zu
finden, ohne für jedes Element den Rest des Arrays zu durchsuchen.

```python
def next_greater_values(values: list[int]) -> list[int | None]:
    """Return the next greater value to the right for every position."""

    result: list[int | None] = [None] * len(values)
    unresolved: list[int] = []

    for index, value in enumerate(values):
        while unresolved and values[unresolved[-1]] < value:
            previous_index = unresolved.pop()
            result[previous_index] = value
        unresolved.append(index)
    return result


assert next_greater_values([2, 1, 2, 4, 3]) == [4, 2, 4, None, None]
```

Der Stack enthält Indizes, deren Antwort noch fehlt. Seine Werte sind von unten
nach oben monoton nicht steigend. Sobald ein größerer Wert erscheint, löst er
alle kleineren Werte oben auf.

Obwohl eine innere `while`-Schleife vorkommt, beträgt die Gesamtlaufzeit O(n):
Jeder Index wird genau einmal gepusht und höchstens einmal gepoppt. Diese
amortisierte Argumentation ist das Herz des Musters.

Typische Anwendungen sind:

- nächstes größeres oder kleineres Element,
- tägliche Temperaturen,
- Stock Span,
- größte Rechteckfläche im Histogramm,
- Entfernen von Ziffern für eine lexikografisch minimale Zahl.

Vor dem Codieren sollte man drei Fragen beantworten:

1. Speichert der Stack Werte oder Indizes?
2. Soll er steigend oder fallend bleiben?
3. Werden gleiche Werte beim Vergleich entfernt (`<` oder `<=`)?

---

## 8. Anwendungsmuster der Queue

### 8.1 Task Queues

Eine Task Queue entkoppelt Produzenten von Arbeitern:

```text
producer(s) -> [job A, job B, job C] -> worker(s)
```

FIFO bietet eine leicht verständliche Fairness: ältere Jobs werden zuerst
angeboten. Reale Systeme müssen zusätzliche Fragen lösen:

- Was geschieht, wenn Produzenten schneller als Worker sind?
- Wie werden fehlgeschlagene Jobs wiederholt?
- Wann gilt ein Job als bestätigt?
- Darf ein Job doppelt verarbeitet werden?
- Braucht es Prioritäten, Deadlines oder mehrere Queues?

Eine begrenzte Queue kann **Backpressure** erzeugen: Ist sie voll, muss der
Produzent warten oder Arbeit ablehnen. Eine unbegrenzte Queue kann bei
dauerhafter Überlast den Speicher erschöpfen.

### 8.2 Message Queues

Message Broker transportieren Nachrichten zwischen unabhängigen Diensten. Die
logische Queue-Idee bleibt erhalten, aber verteilte Systeme ergänzen Persistenz,
Bestätigungen, Sichtbarkeits-Timeouts und Zustellgarantien.

„FIFO“ bedeutet dort nicht automatisch eine perfekte globale Reihenfolge. Mit
mehreren Partitionen oder parallelen Konsumenten kann Reihenfolge nur innerhalb
eines Teilstroms garantiert sein. Die Datenstruktur liefert ein Denkmodell; das
Systemdesign bestimmt die tatsächlichen Garantien.

### 8.3 Breadth-First Search als Vorgriff

Breadth-First Search (BFS) besucht einen Graphen Ebene für Ebene. Neu entdeckte
Knoten werden hinten angefügt, der älteste noch nicht bearbeitete Knoten wird
vorn entnommen:

```python
def breadth_first_order(graph: dict[str, list[str]], start: str) -> list[str]:
    """Return vertices in breadth-first discovery order."""

    frontier = deque([start])
    visited = {start}
    order: list[str] = []

    while frontier:
        vertex = frontier.popleft()
        order.append(vertex)
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return order


sample_graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": [],
    "F": [],
}
assert breadth_first_order(sample_graph, "A") == ["A", "B", "C", "D", "E", "F"]
```

Das Markieren erfolgt bereits beim Einreihen und nicht erst beim Entnehmen.
Sonst kann derselbe Knoten über mehrere Kanten mehrfach in der Queue landen.

---

## 9. Datenstrukturen vergleichen

| Struktur | Ordnungsregel | effiziente Enden | typischer Python-Typ |
|---|---|---|---|
| Stack | LIFO | ein Ende | `list` oder `deque` |
| Queue | FIFO | hinten rein, vorn raus | `collections.deque` |
| Deque | frei an beiden Enden | beide Enden | `collections.deque` |
| Ringpuffer | FIFO mit fester Kapazität | logisch beide Enden | eigene Array-Struktur |
| Priority Queue | höchste/niedrigste Priorität zuerst | nach Priorität | `heapq` |

Eine Priority Queue ist trotz ihres Namens keine FIFO-Queue. Die Entnahmeregel
lautet nicht „ältester Eintrag“, sondern „höchste Priorität“. Bei gleicher
Priorität kann ein zusätzlicher Sequenzzähler FIFO-Stabilität herstellen.

### Entscheidungshilfe

1. Muss das zuletzt hinzugefügte Element zuerst verarbeitet werden? **Stack.**
2. Muss das älteste Element zuerst verarbeitet werden? **Queue.**
3. Werden beide Enden benötigt? **Deque.**
4. Ist die Kapazität fest und sollen Speicherplätze wiederverwendet werden?
   **Ringpuffer.**
5. Bestimmt eine Priorität statt des Alters die Reihenfolge? **Priority Queue.**
6. Wird beliebiger Indexzugriff benötigt? Wahrscheinlich **Array/Liste**, nicht
   Stack oder Queue.

---

## 10. Laufzeiten sauber begründen

### 10.1 Worst Case und amortisierte Kosten

- `push` auf Array-Stack: O(1) **amortisiert**, O(n) im einzelnen Resize-Fall.
- `pop` am Array-Ende: O(1), abgesehen von möglichen Implementierungsdetails
  beim Schrumpfen.
- `popleft` auf `deque`: O(1).
- `pop(0)` auf `list`: O(n).

### 10.2 Speicherkomplexität

Für n gespeicherte Elemente benötigen Stack und wachsende Queue O(n) Speicher.
Ein Ringpuffer mit Kapazität K reserviert O(K), auch wenn gerade weniger Elemente
enthalten sind. Anwendungsalgorithmen können zusätzliche Grenzen besitzen:

- Klammerprüfung: O(n) im Worst Case.
- BFS: O(V) für Queue plus Besuchsmenge.
- Monotonic Stack: O(n) im Worst Case.
- Postfix-Auswertung: O(n) im Worst Case.

---

## 11. Häufige Fehler

### Fehler 1: Queue mit `list.pop(0)`

Die Semantik stimmt, aber die Laufzeit verschlechtert sich durch Verschiebungen
auf O(n) pro Entnahme. `deque.popleft()` oder ein Ringpuffer sind passender.

### Fehler 2: Underflow still verschlucken

`None` als Fehlerwert ist mehrdeutig, wenn `None` ein gültiges Element sein darf.
Eine Exception oder ein explizites Ergebnisobjekt macht den Vertrag eindeutig.

### Fehler 3: Ringpuffer nur über `front == rear` auswerten

Ohne Größe, freien Slot oder Voll-Flag sind „leer“ und „voll“ nicht
unterscheidbar.

### Fehler 4: Zwei-Stack-Queue bei jeder Entnahme komplett umstapeln

Der amortisierte Vorteil entsteht nur, wenn `outgoing` bis zum Leerzustand
weiterverwendet wird.

### Fehler 5: Ungültige Klammerreihenfolge übersehen

Gleiche Anzahlen reichen nicht. `([)]` enthält von jeder Klammer gleich viele,
ist aber falsch verschachtelt. Der Stack prüft Typ **und** Reihenfolge.

### Fehler 6: Operanden vertauschen

Beim Postfix-Stack ist der erste Pop der rechte und der zweite Pop der linke
Operand.

### Fehler 7: Monotonic Stack vorschnell als O(n²) bewerten

Die verschachtelte Schleife allein beweist keine quadratische Laufzeit. Zähle,
wie oft jeder Index insgesamt gepusht und gepoppt werden kann.

### Fehler 8: Nebenläufige Queue mit einer Datenstruktur verwechseln

`collections.deque` bietet effiziente Endoperationen, ersetzt aber nicht jeden
Synchronisationsvertrag. Threads, Prozesse und verteilte Dienste benötigen je
nach Fall Sperren, blockierende Operationen oder einen Message Broker.

---

## 12. Vorgehensmuster beim Lösen neuer Aufgaben

1. **Entnahmeregel formulieren:** zuletzt, ältestes oder höchste Priorität?
2. **Öffentliche Operationen festlegen:** Welche Fehlerfälle gehören zum
   Vertrag?
3. **Invariante notieren:** Was muss nach jeder Operation wahr bleiben?
4. **Implementierung wählen:** Array, Linked List, Ringpuffer, Deque oder
   Kombination?
5. **Zeiger beziehungsweise Indizes simulieren:** leer, ein Element, voll,
   Wrap-around.
6. **Laufzeit begründen:** Worst Case und amortisierte Kosten unterscheiden.
7. **Randfälle testen:** Underflow, Overflow, Duplikate, falsy Werte und lange
   Operationsfolgen.

Beispielinvarianten:

- Stack: `len(_items)` ist die Elementzahl; das letzte Arrayelement ist `top`.
- Linked Queue: leer genau dann, wenn `front is None and rear is None`.
- Ringpuffer: `0 <= size <= capacity`; alle logischen Elemente liegen auf den
  nächsten `size` Kreispositionen ab `front`.
- Two-Stack-Queue: Die logische Reihenfolge ist `reversed(outgoing)` gefolgt von
  `incoming` in Einfügereihenfolge.
- Monotonic Stack: Die Werte an gespeicherten Indizes erfüllen nach jeder
  Iteration die gewählte Monotonie.

---

## 13. Selbstkontrolle

Nach diesem Modul solltest du ohne Vorlage erklären und implementieren können:

- warum LIFO zu verschachtelten Strukturen und Undo passt,
- warum FIFO faire Ankunftsreihenfolge und BFS ermöglicht,
- warum `list.pop(0)` keine gute allgemeine Queue-Operation ist,
- wie `front`, `size` und Modulo einen Ringpuffer steuern,
- weshalb eine Queue aus zwei Stacks amortisiert O(1) erreicht,
- wann `deque` statt `list` verwendet werden sollte,
- warum ein monotonic Stack trotz innerer Schleife O(n) sein kann,
- welche Garantien eine In-Memory-Queue nicht automatisch für verteilte Systeme
  liefert.

### Übungsfragen

1. Simuliere einen Ringpuffer der Kapazität 4 für die Folge `enqueue(A)`,
   `enqueue(B)`, `dequeue()`, `enqueue(C)`, `enqueue(D)`, `enqueue(E)`.
2. Begründe mit einer Buchhaltung pro Element die amortisierte Laufzeit einer
   Two-Stack-Queue.
3. Erweitere die Klammerprüfung so, dass Position und Art des ersten Fehlers
   zurückgegeben werden.
4. Formuliere die Monotonie für „nächstes kleineres Element links“.
5. Entwirf einen Undo-/Redo-Vertrag und beschreibe, was nach einer neuen Aktion
   mit dem Redo-Stack passiert.
6. Vergleiche einen festen Ringpuffer mit einer unbegrenzten Task Queue unter
   dauerhafter Überlast.

Wer diese Fragen mit Invarianten, Randfällen und Laufzeiten beantworten kann,
hat nicht nur die APIs gelernt, sondern die zugrunde liegenden
Verarbeitungsmuster verstanden.
