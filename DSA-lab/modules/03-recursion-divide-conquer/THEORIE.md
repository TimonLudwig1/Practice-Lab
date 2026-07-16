# Modul 03 — Rekursion & Divide and Conquer

## Lernziele

Nach diesem Modul kannst du rekursive Funktionen nicht nur schreiben, sondern
systematisch beurteilen. Du kannst für einen konkreten Aufruf die aktiven
Stack-Frames notieren, aus einem Rekursionsbaum die Laufzeit ablesen und erklären,
warum eine Funktion sicher terminiert. Außerdem erkennst du, wann Rekursion die
Struktur eines Problems natürlich ausdrückt und wann ein expliziter Stack oder
eine Schleife die robustere Wahl ist.

Rekursion ist kein eigener „magischer“ Rechenmechanismus. Sie ist eine Form der
Funktionsausführung, bei der jeder noch nicht beendete Aufruf im Call Stack
gespeichert bleibt. Divide and Conquer nutzt Rekursion häufig, ist aber ein
separates Entwurfsparadigma: Ein Problem wird in kleinere Teilprobleme zerlegt,
diese werden gelöst und ihre Ergebnisse kombiniert.

---

## 1. Rekursion als Selbstähnlichkeit

### 1.1 Intuition: Eine Aufgabe derselben Art delegieren

Stell dir eine Reihe verschachtelter Kisten vor. Um herauszufinden, wie viele
Kisten enthalten sind, öffnest du die äußere Kiste und stellst für die innere
Kiste exakt dieselbe Frage. Irgendwann erreichst du eine Kiste ohne weitere
Kiste. Dort ist keine Delegation mehr nötig.

Eine rekursive Lösung besteht immer aus drei Verträgen:

1. **Basisfall:** Welche kleinste Eingabe kann unmittelbar beantwortet werden?
2. **Rekursionsfall:** Wie wird die Antwort mithilfe einer kleineren Instanz
   desselben Problems zusammengesetzt?
3. **Fortschrittsgarantie:** Warum nähert sich jeder rekursive Aufruf nachweisbar
   einem Basisfall?

Fehlt der Basisfall, läuft die Delegation endlos. Ist der Basisfall vorhanden,
aber die Eingabe wird nicht kleiner, ist er möglicherweise unerreichbar. Deshalb
ist „es gibt einen Basisfall“ schwächer als eine echte Terminierungsbegründung.

### 1.2 Simulation: Fakultät Schritt für Schritt

Die Fakultät ist definiert als

\[
n! = n \cdot (n-1) \cdot \ldots \cdot 1, \qquad 0! = 1.
\]

Die Definition enthält bereits dieselbe Aufgabe für `n - 1`:

```python
def factorial(n: int) -> int:
    """Return n! for a non-negative integer."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1
    return n * factorial(n - 1)


assert factorial(0) == 1
assert factorial(5) == 120
```

Für `factorial(4)` entsteht zunächst eine Kette offener Rechnungen:

```text
factorial(4) braucht 4 * factorial(3)
factorial(3) braucht 3 * factorial(2)
factorial(2) braucht 2 * factorial(1)
factorial(1) braucht 1 * factorial(0)
factorial(0) liefert 1
```

Erst danach werden Ergebnisse in umgekehrter Reihenfolge zurückgegeben:

```text
factorial(0) = 1
factorial(1) = 1 * 1  = 1
factorial(2) = 2 * 1  = 2
factorial(3) = 3 * 2  = 6
factorial(4) = 4 * 6  = 24
```

Die Eingabe `n` ist eine natürliche Fortschrittsgröße. Sie sinkt bei jedem
Aufruf strikt um eins und kann nicht unendlich oft positiv bleiben. Damit wird
der Basisfall `n == 0` sicher erreicht.

### 1.3 Formalisierung: Der Rekursionsvertrag

Für eine rekursive Funktion auf einem Problem der Größe `n` sollten diese
Behauptungen belegbar sein:

- Der Basisfall ist für die kleinste gültige Größe korrekt.
- Jeder Rekursionsfall ruft die Funktion nur mit strikt kleinerer Größe auf.
- Unter der Annahme, dass die kleineren Aufrufe korrekt sind, kombiniert der
  aktuelle Frame ihre Antworten zur korrekten Antwort für `n`.

Das ist die Struktur eines Induktionsbeweises. Die rekursive Implementierung und
der Korrektheitsbeweis spiegeln einander:

```text
Basisfall der Funktion     ↔ Induktionsanfang
rekursiver Aufruf          ↔ Induktionsannahme
Kombination im Frame       ↔ Induktionsschritt
```

---

## 2. Der Call Stack

### 2.1 Intuition: Pausierte Arbeitsblätter

Ein Funktionsaufruf besitzt lokale Variablen, Parameter und eine Rücksprungstelle.
Ruft er eine andere Funktion auf, ist seine Arbeit noch nicht fertig. Python legt
diesen pausierten Zustand als **Stack Frame** ab. Der zuletzt gestartete Aufruf
muss zuerst beendet werden: Last In, First Out.

Ein Frame enthält konzeptionell:

- die Argumente des Aufrufs,
- seine lokalen Variablen,
- die Stelle, an der nach dem Unteraufruf fortgesetzt wird,
- den späteren Rückgabewert.

### 2.2 Simulation: Frames von `factorial(3)`

Nach jedem Abstieg sieht der Call Stack so aus; der aktive Frame steht oben:

```text
Schritt 1                  Schritt 2                  Schritt 3
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ factorial(3)    │       │ factorial(2)    │       │ factorial(1)    │
│ wartet auf f(2) │       │ wartet auf f(1) │       │ wartet auf f(0) │
└─────────────────┘       ├─────────────────┤       ├─────────────────┤
                          │ factorial(3)    │       │ factorial(2)    │
                          │ wartet auf f(2) │       │ wartet auf f(1) │
                          └─────────────────┘       ├─────────────────┤
                                                    │ factorial(3)    │
                                                    │ wartet auf f(2) │
                                                    └─────────────────┘
```

Am tiefsten Punkt kommt `factorial(0)` hinzu und liefert sofort `1`. Danach
werden die Frames nacheinander entfernt:

| Ereignis | Aktiver Frame | Berechnung | Rückgabe |
|---|---|---:|---:|
| Basisfall | `factorial(0)` | direkt | 1 |
| Rückkehr | `factorial(1)` | `1 * 1` | 1 |
| Rückkehr | `factorial(2)` | `2 * 1` | 2 |
| Rückkehr | `factorial(3)` | `3 * 2` | 6 |

Wichtig ist die Trennung von **Abstieg** und **Aufstieg**. Code vor dem rekursiven
Aufruf läuft beim Abstieg. Code danach läuft erst, wenn ein tieferer Frame
zurückkehrt.

### 2.3 Ein verzweigter Call Stack: Fibonacci

Die naive Fibonacci-Funktion ruft sich pro innerem Knoten zweimal auf:

```python
def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number using the direct recurrence."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(10) == 55
```

Obwohl der Rekursionsbaum verzweigt, wird immer nur ein Pfad gleichzeitig im
Call Stack gehalten. Python wertet zuerst `fibonacci(n - 1)` vollständig aus,
kehrt zurück und beginnt danach `fibonacci(n - 2)`. Baumgröße und maximale
Stack-Tiefe sind deshalb verschiedene Größen.

---

## 3. Rekursionsbäume und Kosten

### 3.1 Intuition: Arbeit pro Ebene statt pro Zeile Code

Eine Rekurrenz beschreibt die Kosten eines Problems über kleinere Probleme. Ein
Rekursionsbaum macht sichtbar:

- wie viele Teilprobleme auf jeder Ebene existieren,
- wie groß diese Teilprobleme sind,
- wie viel nichtrekursive Arbeit jeder Knoten erledigt,
- wie viele Ebenen bis zum Basisfall entstehen.

Die Gesamtkosten sind die Summe der Arbeit aller Knoten. Häufig genügt es, die
Arbeit **pro Ebene** und die Anzahl der Ebenen zu bestimmen.

### 3.2 Lineare Rekursion

Für Fakultät fällt neben dem Unteraufruf nur eine Multiplikation an:

\[
T(n) = T(n-1) + \Theta(1).
\]

Der „Baum“ ist eine Kette mit `n + 1` Knoten:

```text
T(n)       Θ(1)
 └─T(n-1)  Θ(1)
    └─T(n-2) Θ(1)
       ...
          └─T(0) Θ(1)
```

Es gibt Θ(n) Ebenen mit Θ(1) Arbeit: insgesamt Θ(n) Zeit. Gleichzeitig liegen
bis zu Θ(n) Frames auf dem Stack: Θ(n) zusätzlicher Speicher.

### 3.3 Verzweigte Rekursion ohne Wiederverwendung

Für naive Fibonacci gilt näherungsweise:

\[
T(n) = T(n-1) + T(n-2) + \Theta(1).
\]

Ein Ausschnitt für `fibonacci(5)`:

```text
f(5)
├─ f(4)
│  ├─ f(3)
│  │  ├─ f(2)
│  │  └─ f(1)
│  └─ f(2)
└─ f(3)
   ├─ f(2)
   └─ f(1)
```

`f(3)` und `f(2)` werden mehrfach berechnet. Die Knotenzahl wächst exponentiell,
genauer Θ(φⁿ) mit dem goldenen Schnitt φ, und wird oft grob als O(2ⁿ)
angegeben. Die Tiefe bleibt dagegen Θ(n). Daraus folgen exponentielle Zeit, aber
nur linearer Stack-Speicher.

### 3.4 Halbierung mit linearer Ebenenarbeit

Merge Sort zerlegt ein Array in zwei Hälften und führt die sortierten Ergebnisse
linear zusammen:

\[
T(n) = 2T(n/2) + \Theta(n).
\]

```text
Ebene 0:       1 Problem  der Größe n       → Gesamtarbeit Θ(n)
Ebene 1:       2 Probleme der Größe n/2     → Gesamtarbeit Θ(n)
Ebene 2:       4 Probleme der Größe n/4     → Gesamtarbeit Θ(n)
...
Ebene log n:   n Probleme der Größe 1       → Gesamtarbeit Θ(n)
```

Es gibt Θ(log n) Ebenen, jede kostet Θ(n): insgesamt Θ(n log n).

---

## 4. Rekursion und Iteration

### 4.1 Intuition: Impliziter oder expliziter Zustand

Rekursion speichert den noch offenen Zustand implizit im Call Stack. Eine
iterative Lösung speichert ihn in Schleifenvariablen oder in einer eigenen
Datenstruktur. Beide Varianten müssen dieselbe Information erhalten; sie legen
sie nur an unterschiedlichen Orten ab.

Rekursion ist oft natürlich bei:

- hierarchischen Strukturen wie Bäumen und Ordnern,
- Divide-and-Conquer-Verfahren,
- Problemen, deren Definition selbst rekursiv ist,
- Backtracking mit klaren Entscheidungszuständen.

Iteration ist oft günstiger bei:

- sehr tiefen linearen Ketten,
- einfachem Akkumulieren über Sequenzen,
- produktivem Code mit unbekannter oder unkontrollierter Tiefe,
- Situationen, in denen Stack-Zustand explizit inspiziert oder begrenzt werden
  soll.

### 4.2 Dieselbe Summe in zwei Formen

```python
def recursive_sum(values: list[int], index: int = 0) -> int:
    """Return the sum from index to the end recursively."""
    if index == len(values):
        return 0
    return values[index] + recursive_sum(values, index + 1)


def iterative_sum(values: list[int]) -> int:
    """Return the sum using constant auxiliary stack space."""
    total = 0
    for value in values:
        total += value
    return total


assert recursive_sum([2, 4, 6]) == 12
assert iterative_sum([2, 4, 6]) == 12
```

Beide brauchen Θ(n) Zeit. Die rekursive Variante hält Θ(n) Frames, die iterative
nur konstanten Zusatzspeicher. Für eine flache Liste drückt die Schleife das
Problem daher direkter und robuster aus.

### 4.3 Python und das Rekursionslimit

CPython schützt den nativen Prozessstack durch ein Rekursionslimit. Es liegt
typischerweise ungefähr bei tausend Frames, ist aber eine Laufzeitkonfiguration
und kein Sprachvertrag. Es kann abgefragt werden:

```python
import sys


recursion_limit = sys.getrecursionlimit()
assert recursion_limit > 0
```

Ein Überschreiten führt zu `RecursionError`. Das Limit pauschal stark zu erhöhen
ist keine algorithmische Lösung und kann den Prozessstack gefährden. Bei
unbekannter Tiefe sollte der Algorithmus iterativ mit einem expliziten Stack
formuliert werden.

Python führt außerdem keine Tail-Call-Optimierung durch. Auch wenn der rekursive
Aufruf die letzte Operation ist, bleibt für jeden Aufruf ein eigener Frame
erhalten. Eine tail-rekursive lineare Funktion spart daher in Python keinen
Stack-Speicher.

---

## 5. Divide and Conquer

### 5.1 Intuition: Teilen, Erobern, Kombinieren

Divide and Conquer besteht aus drei Phasen:

1. **Divide:** Zerlege ein Problem in kleinere, möglichst ausgewogene Teile.
2. **Conquer:** Löse die Teilprobleme rekursiv.
3. **Combine:** Setze die Teilergebnisse zur Gesamtlösung zusammen.

Der Basisfall löst ausreichend kleine Teilprobleme direkt. Die Balance ist
wichtig: Halbierung erzeugt logarithmische Tiefe; eine Zerlegung in Größen `1`
und `n - 1` kann linear tief werden.

### 5.2 Simulation des Merge-Prinzips

Zwei bereits sortierte Folgen lassen sich mit zwei Zeigern linear verbinden:

```text
links  = [2, 5, 8]
rechts = [1, 4, 9]

vergleiche 2 und 1 → [1]
vergleiche 2 und 4 → [1, 2]
vergleiche 5 und 4 → [1, 2, 4]
vergleiche 5 und 9 → [1, 2, 4, 5]
vergleiche 8 und 9 → [1, 2, 4, 5, 8]
Rest rechts        → [1, 2, 4, 5, 8, 9]
```

```python
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def merge(left: Sequence[T], right: Sequence[T]) -> list[T]:
    """Merge two non-decreasing sequences."""
    merged: list[T] = []
    left_index = 0
    right_index = 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1

    merged.extend(left[left_index:])
    merged.extend(right[right_index:])
    return merged


def merge_sort(values: Sequence[T]) -> list[T]:
    """Return a sorted copy using divide and conquer."""
    if len(values) < 2:
        return list(values)
    middle = len(values) // 2
    left = merge_sort(values[:middle])
    right = merge_sort(values[middle:])
    return merge(left, right)


assert merge([2, 5, 8], [1, 4, 9]) == [1, 2, 4, 5, 8, 9]
assert merge_sort([7, 2, 5, 2, 9, 1]) == [1, 2, 2, 5, 7, 9]
```

Die Korrektheitsidee der Combine-Phase ist eine Invariante: Vor jedem Vergleich
enthält `merged` genau die kleinsten bereits verarbeiteten Elemente in sortierter
Reihenfolge. Das kleinere der beiden nächsten Kandidaten ist zwangsläufig das
nächste globale Element.

---

## 6. Das Master-Theorem intuitiv

### 6.1 Die Form

Viele ausgewogene Divide-and-Conquer-Rekurrenzen haben die Form

\[
T(n) = aT(n/b) + f(n).
\]

- `a`: Zahl der rekursiven Teilprobleme,
- `n/b`: Größe jedes Teilproblems,
- `f(n)`: Arbeit für Teilen und Kombinieren im aktuellen Knoten.

Die Blätterzahl des Rekursionsbaums wächst wie
\(n^{\log_b a}\). Diese Größe beschreibt grob die gesamte Arbeit an der
Blattebene. Das Master-Theorem vergleicht sie mit `f(n)`.

### 6.2 Fall 1: Die Blätter dominieren

Ist die Arbeit pro Knoten deutlich kleiner als das Wachstum der Teilprobleme,
dominiert die große Zahl der Blätter.

```text
T(n) = 4T(n/2) + Θ(n)
Blattmaß: n^(log_2 4) = n²
Ergebnis: Θ(n²)
```

Pro Ebene vervierfacht sich die Knotenzahl, während die Knotengröße halbiert
wird. Die Ebenenarbeit wächst geometrisch und die letzte Ebene dominiert.

### 6.3 Fall 2: Alle Ebenen sind gleich teuer

Sind `f(n)` und das Blattmaß gleich groß, trägt jede der logarithmisch vielen
Ebenen dieselbe Größenordnung bei.

```text
T(n) = 2T(n/2) + Θ(n)
Blattmaß: n^(log_2 2) = n
Ebenen: Θ(log n)
Ergebnis: Θ(n log n)
```

Das ist der Merge-Sort-Fall.

### 6.4 Fall 3: Die Wurzelarbeit dominiert

Ist `f(n)` polynomial größer als das Blattmaß und nimmt die Arbeit beim Abstieg
regelmäßig ab, dominiert die obere Ebene.

```text
T(n) = 2T(n/2) + Θ(n²)
Blattmaß: n
Ebenenarbeit: n², n²/2, n²/4, ...
Ergebnis: Θ(n²)
```

Die geometrische Summe bleibt in der Größenordnung ihres ersten Terms.

### 6.5 Grenzen

Das Master-Theorem passt nicht unverändert auf jede Rekurrenz. Beispiele:

- `T(n) = T(n - 1) + 1` teilt nicht durch einen konstanten Faktor.
- Naive Fibonacci erzeugt Teilprobleme unterschiedlicher Größen.
- Stark unregelmäßige Zusatzarbeit kann die Standardfälle verletzen.

Dann helfen ein direkter Rekursionsbaum, Substitution oder andere Verfahren.
Für dieses Modul ist entscheidend, die drei Fälle als Ebenenvergleich zu
verstehen, nicht eine formale Beweisschablone auswendig zu lernen.

---

## 7. Memoization als Ausblick

### 7.1 Intuition: Antworten auf wiederholte Fragen merken

Naive Fibonacci ist langsam, weil identische Teilprobleme wiederholt auftreten.
Memoization speichert die Antwort beim ersten Auftreten und liefert sie später in
O(1) aus einem Cache zurück.

```python
def memoized_fibonacci(n: int, cache: dict[int, int] | None = None) -> int:
    """Return Fibonacci(n) while caching overlapping subproblems."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if cache is None:
        cache = {}
    if n < 2:
        return n
    if n not in cache:
        cache[n] = memoized_fibonacci(n - 1, cache) + memoized_fibonacci(
            n - 2, cache
        )
    return cache[n]


assert memoized_fibonacci(10) == 55
assert memoized_fibonacci(100) == 354224848179261915075
```

Nun wird jedes `n` nur einmal vollständig berechnet. Die Zeit sinkt von
exponentiell auf Θ(n), der Cache benötigt Θ(n) Speicher und die Stack-Tiefe bleibt
Θ(n). Das ist ein Vorgriff auf Dynamic Programming: Überlappende Teilprobleme
werden nicht erneut gelöst.

Memoization verbessert nur Wiederholungen. Bei Merge Sort sind die Teilarrays
disjunkt; dort gibt es keine identischen Teilprobleme, die ein Cache einsparen
könnte.

---

## 8. Systematisches Tracen

Für einen manuellen Trace eignet sich eine Tabelle mit einem Eintrag pro
Aufrufereignis:

| Schritt | Tiefe | Aufruf | Basisfall? | Wartet auf | Rückgabe |
|---:|---:|---|---|---|---|
| 1 | 0 | `factorial(3)` | nein | `factorial(2)` | offen |
| 2 | 1 | `factorial(2)` | nein | `factorial(1)` | offen |
| 3 | 2 | `factorial(1)` | nein | `factorial(0)` | offen |
| 4 | 3 | `factorial(0)` | ja | — | 1 |
| 5 | 2 | Fortsetzung `factorial(1)` | — | — | 1 |
| 6 | 1 | Fortsetzung `factorial(2)` | — | — | 2 |
| 7 | 0 | Fortsetzung `factorial(3)` | — | — | 6 |

Bei verzweigten Aufrufen ergänzt du die Reihenfolge der Kinder. Ein
Depth-First-Ablauf bedeutet: Das erste Kind wird vollständig abgearbeitet, bevor
das zweite beginnt.

Ein guter Trace beantwortet vier Fragen:

1. Welche Argumente besitzt jeder Frame?
2. Welche lokale Arbeit ist schon erledigt?
3. Auf welchen Unteraufruf wartet der Frame?
4. Welcher Wert wird beim Aufstieg zurückgegeben?

---

## 9. Typische Fehler

### Unerreichbarer Basisfall

`n` wird erhöht, obwohl der Basisfall bei `0` liegt. Die Existenz des Basisfalls
hilft nicht; die Fortschrittsrichtung ist falsch.

### Verlorener Rückgabewert

Ein rekursiver Aufruf wird ausgeführt, aber sein Ergebnis nicht mit `return`
weitergegeben. Dann liefert der äußere Frame implizit `None`.

### Verdeckte Zusatzkosten

Slicing wie `values[1:]` erzeugt bei jedem linearen Rekursionsschritt eine neue
Liste. Eine scheinbare Rekurrenz `T(n)=T(n-1)+O(1)` wird dadurch zu
`T(n)=T(n-1)+O(n)` und insgesamt O(n²). Ein Indexparameter vermeidet die Kopien.

### Exponentielle Wiederholung

Mehrere rekursive Aufrufe sind nicht automatisch falsch. Sie werden problematisch,
wenn dieselben Zustände wiederholt berechnet werden. Ein Rekursionsbaum macht
diese Überlappung sichtbar.

### Baumgröße mit Stack-Tiefe verwechseln

Die Laufzeit zählt alle besuchten Knoten. Der Speicher zählt den längsten
gleichzeitig aktiven Pfad. Ein exponentieller Baum kann lineare Tiefe haben.

### Rekursionslimit ignorieren

Eine mathematisch korrekte lineare Rekursion kann in Python für große Eingaben
trotzdem praktisch ungeeignet sein. Die Laufzeitkomplexität allein beantwortet
nicht die Frage nach der robusten Implementierung.

---

## 10. Entscheidungsleitfaden

Bevor du Rekursion verwendest, beantworte:

1. Ist das Problem natürlich in gleichartige kleinere Probleme zerlegbar?
2. Gibt es einen klaren, direkt lösbaren Basisfall?
3. Welche messbare Größe sinkt bei jedem Aufruf?
4. Wie tief kann der längste Pfad realistisch werden?
5. Werden Teilprobleme mehrfach berechnet?
6. Welche Arbeit entsteht pro Knoten und pro Ebene?
7. Muss nach dem Unteraufruf noch Zustand erhalten bleiben?
8. Wäre eine Schleife oder ein expliziter Stack klarer oder robuster?

Für die Laufzeitanalyse:

1. Formuliere eine Rekurrenz.
2. Zeichne die ersten zwei bis drei Ebenen.
3. Bestimme Knotenzahl und Arbeit pro Knoten je Ebene.
4. Multipliziere zur Ebenenarbeit.
5. Bestimme die Tiefe.
6. Summiere die Ebenen oder ordne den passenden Master-Fall zu.

---

## 11. Zusammenfassung

- Rekursion braucht Basisfall, Rekursionsfall und Fortschrittsgarantie.
- Jeder offene Aufruf belegt einen Stack Frame; Rückgaben erfolgen in umgekehrter
  Aufrufreihenfolge.
- Rekursionsbäume trennen Gesamtknotenzahl, Ebenenarbeit und maximale Tiefe.
- Rekursion und Iteration speichern denselben logischen Zustand implizit oder
  explizit.
- Python begrenzt Rekursion und optimiert Tail Calls nicht.
- Divide and Conquer bedeutet Teilen, rekursiv Lösen und Kombinieren.
- Das Master-Theorem vergleicht Blattarbeit mit der nichtrekursiven Arbeit pro
  Knoten.
- Memoization beseitigt wiederholte Berechnung überlappender Teilprobleme, ist
  aber kein allgemeiner Beschleuniger für jede Rekursion.

Wer eine rekursive Funktion Frame für Frame simulieren und ihren Baum Ebene für
Ebene bewerten kann, muss Rekursion nicht mehr als Sprung ins Ungewisse behandeln.
