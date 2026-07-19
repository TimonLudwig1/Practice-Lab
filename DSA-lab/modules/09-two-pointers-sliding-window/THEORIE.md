# Modul 09 — Two Pointers & Sliding Window

Two Pointers und Sliding Window sind Muster zur Vermeidung wiederholter Arbeit.
Eine Brute-Force-Lösung betrachtet häufig alle Paare oder berechnet überlappende
Teilbereiche immer wieder von vorn. Die Muster nutzen Ordnung, Monotonie oder
einen inkrementell gepflegten Zustand, sodass jeder Index nur wenige Male
berührt wird.

Dieses Skript folgt drei Ebenen:

1. **Intuition:** Welche wiederholte Arbeit wird vermieden und welche Struktur
   macht das möglich?
2. **Simulation:** Wie bewegen sich Zeiger und Fenstergrenzen auf konkreten
   Eingaben?
3. **Formalisierung:** Welche Invarianten tragen die Algorithmen, warum ist die
   Laufzeit linear und wann darf das Muster nicht eingesetzt werden?

---

# Ebene I — Intuition

## 1. Von allen Paaren zu gezielten Bewegungen

Wer in einem Array zwei Werte mit einer bestimmten Summe sucht, kann jedes Paar
prüfen. Bei `n` Werten entstehen ungefähr `n² / 2` Paare. Ist das Array jedoch
sortiert, liefert die Summe der beiden Außenwerte eine Richtung:

- Ist die Summe zu klein, muss der kleinere Wert wachsen: linker Zeiger nach
  rechts.
- Ist die Summe zu groß, muss der größere Wert sinken: rechter Zeiger nach links.
- Ist sie gleich dem Ziel, ist das Paar gefunden.

Eine Bewegung verwirft nicht nur ein Paar, sondern eine ganze Klasse unmöglicher
Paare. Beide Zeiger bewegen sich höchstens `n - 1` Positionen. Aus `O(n²)` wird
`O(n)` — allerdings erst, nachdem die Sortierung als Vorbedingung erfüllt ist.

```python
def pair_sum_brute(values, target):
    """Return one index pair with the target sum by checking every pair."""
    checks = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            checks += 1
            if values[left] + values[right] == target:
                return (left, right), checks
    return None, checks


def pair_sum_two_pointers(values, target):
    """Return one target-sum pair from a sorted sequence."""
    left, right = 0, len(values) - 1
    checks = 0
    while left < right:
        checks += 1
        current = values[left] + values[right]
        if current == target:
            return (left, right), checks
        if current < target:
            left += 1
        else:
            right -= 1
    return None, checks


ordered = list(range(2_000))
pair, linear_checks = pair_sum_two_pointers(ordered, 3_997)
assert pair == (1_998, 1_999)
assert linear_checks <= len(ordered)
```

## 2. Zwei Zeiger sind mehr als „links und rechts“

Es gibt zwei wichtige Bewegungsformen:

### Gegenläufig

Ein Zeiger startet links, einer rechts. Beide bewegen sich aufeinander zu.
Typische Signale sind ein sortiertes Array, Paarsummen, Palindrome oder ein
Flächen-/Abstandsproblem, bei dem eine Seite gezielt verworfen werden kann.

### Gleichläufig

Beide Zeiger laufen von links nach rechts, aber mit verschiedener Bedeutung. Ein
Lesezeiger untersucht jedes Element; ein Schreibzeiger markiert die nächste
Position im gültigen Präfix. Das passt zu In-Place-Filterung, Duplikatentfernung
oder stabiler Komprimierung.

Der entscheidende Gedanke ist nicht die Zahl zwei. Jeder Zeiger besitzt eine
semantische Rolle, und seine Bewegung muss durch eine Invariante begründet sein.

## 3. Sliding Window als wiederverwendeter Teilbereich

Bei einer Zeitreihe soll etwa die Summe jeder Gruppe aus 100 aufeinanderfolgenden
Messungen berechnet werden. Die naive Lösung addiert für jedes Fenster alle 100
Werte neu. Benachbarte Fenster teilen aber 99 Werte.

```text
altes Fenster: [a b c d]
neues Fenster:   [b c d e]
```

Die neue Summe entsteht durch

```text
neue_summe = alte_summe - a + e
```

Das ist ein **festes Fenster**: Seine Breite bleibt konstant. Bei einem
**variablen Fenster** wächst die rechte Grenze und die linke Grenze schrumpft so
lange, bis eine Bedingung wieder gilt. Beispiele sind „längster Substring ohne
Wiederholung“ oder „kleinstes Teilarray mit Summe mindestens Ziel“.

## 4. Die gemeinsame Idee

Two Pointers und Sliding Window überlappen konzeptionell. Ein variables Fenster
besitzt ebenfalls zwei Zeiger. Die Bezeichnung Sliding Window betont zusätzlich:

- Der Bereich zwischen den Zeigern ist zusammenhängend.
- Über diesen Bereich wird ein Zustand gepflegt, etwa Summe, Häufigkeiten oder
  Zahl ungültiger Elemente.
- Beim Eintritt und Austritt eines Elements wird der Zustand inkrementell
  aktualisiert.

Two Pointers kann dagegen auch Paare an den Außenrändern vergleichen, ohne einen
aggregierten Zustand des gesamten Zwischenbereichs zu pflegen.

## 5. Wann lohnt sich das Muster?

Typische Erkennungssignale sind:

- „sortiertes Array“ plus Paar, Differenz oder Zielsumme,
- „in-place“, „Duplikate entfernen“ oder „gültige Elemente nach vorn“,
- „zusammenhängendes Teilarray“ oder „Substring“,
- „längstes“, „kürzestes“, „höchstens k“ oder „mindestens k“,
- wiederholte Statistiken über benachbarte Zeitintervalle,
- eine Bedingung, die beim Wachsen und Schrumpfen des Fensters vorhersagbar
  reagiert.

Nicht jedes Teilarrayproblem ist ein Sliding-Window-Problem. Negative Zahlen
können beispielsweise die Monotonie einer Summenbedingung zerstören: Entfernen
eines linken negativen Werts erhöht die Summe, statt sie zu senken.

---

# Ebene II — Simulation

## 6. Gegenläufige Zeiger: Paarsumme

Gesucht ist Summe `13` in `[1, 2, 4, 6, 8, 11]`.

| Schritt | `left` | `right` | Werte | Summe | Entscheidung |
|---:|---:|---:|---|---:|---|
| 1 | 0 | 5 | `1 + 11` | 12 | zu klein, `left += 1` |
| 2 | 1 | 5 | `2 + 11` | 13 | gefunden |

Warum darf `1` verworfen werden? Zusammen mit dem größten verfügbaren Wert `11`
ist die Summe bereits zu klein. Mit jedem kleineren rechten Wert wäre sie erst
recht zu klein.

```python
values = [1, 2, 4, 6, 8, 11]
pair, checks = pair_sum_two_pointers(values, 13)
assert pair == (1, 5)
assert checks == 2
assert pair_sum_two_pointers(values, 100)[0] is None
```

### 6.1 Sortieren verändert Indizes

Ist die Eingabe unsortiert, kann sie zunächst sortiert werden. Das kostet
`O(n log n)` und verändert ohne Dekoration die ursprünglichen Indizes. Wenn die
Indizes Teil der Ausgabe sind, müssen Paare `(value, original_index)` sortiert
oder alternativ eine Hash Map verwendet werden.

```python
def pair_sum_unsorted(values, target):
    """Return original indices after sorting decorated values."""
    decorated = sorted((value, index) for index, value in enumerate(values))
    left, right = 0, len(decorated) - 1
    while left < right:
        current = decorated[left][0] + decorated[right][0]
        if current == target:
            return decorated[left][1], decorated[right][1]
        if current < target:
            left += 1
        else:
            right -= 1
    return None


unsorted_values = [8, 1, 11, 4, 2, 6]
original_pair = pair_sum_unsorted(unsorted_values, 13)
assert original_pair is not None
assert sum(unsorted_values[index] for index in original_pair) == 13
```

## 7. Gegenläufige Zeiger: größter Wasserbehälter

Zwei Höhen bilden mit ihrem Abstand eine Fläche:

```text
area = min(height[left], height[right]) * (right - left)
```

Nach jeder Prüfung wird die kleinere Höhe verworfen. Wird stattdessen die höhere
Seite bewegt, sinkt der Abstand, während die begrenzende kleinere Höhe bleibt;
die Fläche kann nicht wachsen.

```python
def max_container_area(heights):
    """Return the maximum area formed by two vertical lines."""
    left, right = 0, len(heights) - 1
    best = 0
    while left < right:
        best = max(best, min(heights[left], heights[right]) * (right - left))
        if heights[left] <= heights[right]:
            left += 1
        else:
            right -= 1
    return best


assert max_container_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
assert max_container_area([]) == 0
assert max_container_area([5]) == 0
```

## 8. Gleichläufige Zeiger: stabil filtern

Aus `[3, -1, 4, -2, 0, 5]` sollen negative Werte in-place entfernt werden. Der
Lesezeiger besucht jedes Element. Der Schreibzeiger zeigt stets auf den Beginn
des noch unbestimmten Bereichs.

| Lesen | Wert | Aktion | gültiges Präfix | `write` danach |
|---:|---:|---|---|---:|
| 0 | 3 | schreiben | `[3]` | 1 |
| 1 | -1 | überspringen | `[3]` | 1 |
| 2 | 4 | schreiben | `[3, 4]` | 2 |
| 3 | -2 | überspringen | `[3, 4]` | 2 |
| 4 | 0 | schreiben | `[3, 4, 0]` | 3 |
| 5 | 5 | schreiben | `[3, 4, 0, 5]` | 4 |

```python
def filter_in_place(values, keep):
    """Stably keep matching values and return the valid prefix length."""
    write = 0
    for read, value in enumerate(values):
        if keep(value):
            values[write] = value
            write += 1
    del values[write:]
    return write


filtered = [3, -1, 4, -2, 0, 5]
length = filter_in_place(filtered, lambda value: value >= 0)
assert length == 4
assert filtered == [3, 4, 0, 5]
```

Vor jeder Iteration enthält `values[:write]` genau die bisher gelesenen gültigen
Werte in ursprünglicher Reihenfolge. Diese Invariante erklärt gleichzeitig
Korrektheit und Stabilität.

## 9. Gleichläufige Zeiger: Duplikate entfernen

In einem sortierten Array stehen gleiche Werte nebeneinander. Der Schreibzeiger
markiert das letzte eindeutige Element; der Lesezeiger sucht den nächsten anderen
Wert.

```python
def deduplicate_sorted(values):
    """Remove adjacent duplicates in place and return the unique length."""
    if not values:
        return 0
    write = 1
    for read in range(1, len(values)):
        if values[read] != values[write - 1]:
            values[write] = values[read]
            write += 1
    del values[write:]
    return write


duplicates = [1, 1, 2, 2, 2, 4, 7, 7]
assert deduplicate_sorted(duplicates) == 4
assert duplicates == [1, 2, 4, 7]
```

Ohne Sortierung reichen benachbarte Vergleiche nicht aus. Dann wird zusätzlicher
Zustand wie ein Set benötigt, oder die Eingabe muss zuerst sortiert werden.

## 10. Festes Fenster: gleitende Summen

Für Werte `[2, 1, 5, 1, 3, 2]` und Fensterbreite `3`:

| Fenster | Update | Summe |
|---|---|---:|
| `[2, 1, 5]` | initial | 8 |
| `[1, 5, 1]` | `8 - 2 + 1` | 7 |
| `[5, 1, 3]` | `7 - 1 + 3` | 9 |
| `[1, 3, 2]` | `9 - 5 + 2` | 6 |

```python
def rolling_sums(values, width):
    """Return sums of all contiguous windows with a fixed positive width."""
    if width < 1:
        raise ValueError("width must be positive")
    if width > len(values):
        return []
    current = sum(values[:width])
    result = [current]
    for right in range(width, len(values)):
        current += values[right] - values[right - width]
        result.append(current)
    return result


assert rolling_sums([2, 1, 5, 1, 3, 2], 3) == [8, 7, 9, 6]
assert rolling_sums([1, 2], 3) == []
```

Das erste Fenster kostet `O(k)`, danach jedes Update `O(1)`. Insgesamt entstehen
`O(n)` Zeit und `O(1)` Arbeitszustand zusätzlich zur Ergebnisliste.

## 11. Variables Fenster: kleinste Länge mit Zielsumme

Gesucht ist das kürzeste zusammenhängende Teilarray positiver Zahlen mit Summe
mindestens `7` in `[2, 3, 1, 2, 4, 3]`.

Die rechte Grenze wächst immer. Sobald die Summe ausreicht, wird links so lange
geschrumpft, wie die Bedingung noch erfüllt ist.

| `right` | Wert hinzu | Fenster vor Schrumpfen | Summe | gefundene Längen |
|---:|---:|---|---:|---|
| 0 | 2 | `[2]` | 2 | — |
| 1 | 3 | `[2, 3]` | 5 | — |
| 2 | 1 | `[2, 3, 1]` | 6 | — |
| 3 | 2 | `[2, 3, 1, 2]` | 8 | 4 |
| 4 | 4 | `[3, 1, 2, 4]` | 10 | 4, 3 |
| 5 | 3 | `[2, 4, 3]` | 9 | 3, 2 |

```python
def minimum_length_at_least(values, target):
    """Return the shortest positive-value window whose sum reaches target."""
    if target <= 0:
        return 0
    left = 0
    current = 0
    best = len(values) + 1
    for right, value in enumerate(values):
        if value <= 0:
            raise ValueError("values must be positive")
        current += value
        while current >= target:
            best = min(best, right - left + 1)
            current -= values[left]
            left += 1
    return 0 if best == len(values) + 1 else best


assert minimum_length_at_least([2, 3, 1, 2, 4, 3], 7) == 2
assert minimum_length_at_least([1, 1], 5) == 0
```

Positive Werte sind hier eine notwendige Vorbedingung. Beim Entfernen des linken
Werts sinkt die Summe garantiert. Mit negativen Zahlen ist die erforderliche
Monotonie nicht gegeben; Präfixsummen und andere Datenstrukturen sind dann häufig
passender.

## 12. Variables Fenster mit Hash Map: längster eindeutiger Substring

Für jedes Zeichen speichert eine Hash Map dessen zuletzt gesehenen Index. Tritt
ein Duplikat innerhalb des aktuellen Fensters auf, springt `left` direkt hinter
das frühere Vorkommen.

Für `abcaeb`:

| `right` | Zeichen | vorheriger Index | `left` danach | Fenster | beste Länge |
|---:|---|---:|---:|---|---:|
| 0 | a | — | 0 | `a` | 1 |
| 1 | b | — | 0 | `ab` | 2 |
| 2 | c | — | 0 | `abc` | 3 |
| 3 | a | 0 | 1 | `bca` | 3 |
| 4 | e | — | 1 | `bcae` | 4 |
| 5 | b | 1 | 2 | `caeb` | 4 |

```python
def longest_unique_substring(text):
    """Return the longest substring without repeated characters."""
    last_seen = {}
    left = 0
    best_start = 0
    best_length = 0
    for right, character in enumerate(text):
        previous = last_seen.get(character)
        if previous is not None and previous >= left:
            left = previous + 1
        last_seen[character] = right
        length = right - left + 1
        if length > best_length:
            best_start = left
            best_length = length
    return text[best_start : best_start + best_length]


assert longest_unique_substring("abcaeb") == "bcae"
assert longest_unique_substring("bbbbb") == "b"
assert longest_unique_substring("") == ""
```

Das `max` beziehungsweise die Prüfung `previous >= left` ist wesentlich. Ein
älteres Duplikat links außerhalb des Fensters darf `left` niemals rückwärts
bewegen.

## 13. Frequenzzustand: Anagrammfenster

Bei einer festen Fensterbreite kann eine Hash Map die Zeichenhäufigkeiten
pflegen. Ein Fenster ist ein Anagramm des Musters, wenn seine Häufigkeiten gleich
sind.

```python
from collections import Counter


def anagram_starts(text, pattern):
    """Return start indices of windows that are anagrams of pattern."""
    width = len(pattern)
    if width == 0 or width > len(text):
        return []
    required = Counter(pattern)
    window = Counter(text[:width])
    starts = [0] if window == required else []
    for right in range(width, len(text)):
        entering = text[right]
        leaving = text[right - width]
        window[entering] += 1
        window[leaving] -= 1
        if window[leaving] == 0:
            del window[leaving]
        if window == required:
            starts.append(right - width + 1)
    return starts


assert anagram_starts("cbaebabacd", "abc") == [0, 6]
assert anagram_starts("abab", "ab") == [0, 1, 2]
```

Für ein festes kleines Alphabet kann ein Array schneller und speichersparender
als eine Hash Map sein. Das Zustandsprinzip bleibt identisch.

## 14. Data Science: Rolling Mean und Varianz

Zeitreihenanalysen verwenden Rolling Windows für Glättung, Volatilität,
Anomalieerkennung und lokale Features. Mittelwert und Populationsvarianz lassen
sich über Summe und Quadratsumme pflegen:

```text
mean = sum / k
variance = sum_of_squares / k - mean²
```

```python
def rolling_mean_variance(values, width):
    """Return population mean and variance for every fixed-width window."""
    if width < 1:
        raise ValueError("width must be positive")
    if width > len(values):
        return []
    current_sum = sum(values[:width])
    square_sum = sum(value * value for value in values[:width])
    result = []
    for start in range(len(values) - width + 1):
        mean = current_sum / width
        variance = max(0.0, square_sum / width - mean * mean)
        result.append((mean, variance))
        next_index = start + width
        if next_index < len(values):
            leaving = values[start]
            entering = values[next_index]
            current_sum += entering - leaving
            square_sum += entering * entering - leaving * leaving
    return result


statistics = rolling_mean_variance([1.0, 2.0, 3.0, 6.0], 3)
assert statistics[0][0] == 2.0
assert abs(statistics[0][1] - 2 / 3) < 1e-12
assert statistics[1][0] == 11 / 3
```

Die Formel über Quadratsummen kann bei sehr großen, fast gleichen
Gleitkommazahlen numerisch instabil werden. Projekt `03-final` behandelt den
Trade-off zwischen `O(1)`-Update, Referenzberechnung und numerischer Toleranz.

## 15. Ein Operationsexperiment

Die lineare Laufzeit eines variablen Fensters ist nicht offensichtlich, weil
eine `while`-Schleife in einer `for`-Schleife steht. Entscheidend ist: `left`
wird über den gesamten Lauf höchstens `n`-mal erhöht.

```python
def window_pointer_moves(values, target):
    """Count right expansions and left contractions for positive values."""
    left = 0
    current = 0
    expansions = 0
    contractions = 0
    for value in values:
        current += value
        expansions += 1
        while current >= target:
            current -= values[left]
            left += 1
            contractions += 1
    return expansions, contractions


expansions, contractions = window_pointer_moves([1] * 10_000, 100)
assert expansions == 10_000
assert contractions <= 10_000
```

---

# Ebene III — Formalisierung

## 16. Gegenläufige Paarsuche: Invariante und Korrektheit

Für ein sortiertes Array lautet die Invariante:

> Falls ein noch nicht ausgeschlossenes Zielpaar existiert, gibt es eines mit
> beiden Indizes im aktuellen Bereich `[left, right]`.

Ist `values[left] + values[right] < target`, kann `values[left]` mit keinem Index
zwischen `left + 1` und `right` das Ziel erreichen, denn `values[right]` ist
bereits der größte Partner. `left` darf daher erhöht werden. Das symmetrische
Argument gilt bei einer zu großen Summe für `right`.

Die Distanz `right - left` sinkt in jeder Runde strikt. Die Schleife terminiert
nach höchstens `n - 1` Bewegungen. Zeit: `O(n)`, Zusatzspeicher: `O(1)`.

## 17. Gleichläufige Filterung: Präfixinvariante

Vor jeder Leseiteration gilt:

1. `values[:write]` enthält genau die gültigen Werte aus dem bereits gelesenen
   Eingabepräfix.
2. Ihre relative Reihenfolge entspricht der Eingabe.
3. `write <= read` beziehungsweise nach dem Lesen `write <= read + 1`.

Ein ungültiger Wert verändert das Präfix nicht. Ein gültiger Wert wird genau an
`write` angehängt. Induktiv bleibt die Invariante erhalten. Nach dem letzten
Element ist das Präfix die vollständige stabile Filterung.

## 18. Festes Fenster: Zustandsinvariante

Vor der Ausgabe jedes Fensters `[left, right]` repräsentiert der gepflegte Zustand
genau die Elemente dieses Fensters. Beim Verschieben

1. verlässt das Element an `left` den Zustand,
2. betritt das neue Element an `right + 1` den Zustand,
3. beide Grenzen steigen um eins.

Jedes Element wird einmal hinzugefügt und höchstens einmal entfernt. Ist ein
Update `O(1)`, beträgt die Gesamtlaufzeit `O(n)`.

Nicht jeder Zustand kann günstig entfernt werden. Minimum und Maximum lassen
sich nicht durch einfache Subtraktion pflegen; dafür wird häufig eine monotone
Deque verwendet. Median benötigt typischerweise zwei Heaps oder eine geordnete
Multimenge.

## 19. Variables Fenster: Gültigkeit und Amortisierung

Ein typisches Maximierungsfenster hat diese Form:

```text
for right in range(n):
    add(values[right])
    while window_is_invalid():
        remove(values[left])
        left += 1
    update_best(left, right)
```

Nach der `while`-Schleife ist das Fenster gültig. Für „längstes gültiges Fenster“
wird erst dann das Optimum aktualisiert. Bei „kürzestes Fenster, das eine
Mindestbedingung erfüllt“ wird dagegen während des Schrumpfens aktualisiert,
solange das Fenster noch ausreichend ist.

Obwohl die Schleifen verschachtelt sind, betritt jedes Element das Fenster
einmal über `right` und verlässt es höchstens einmal über `left`. Damit gibt es
höchstens `2n` Grenzbewegungen: amortisiert `O(n)`.

## 20. Wann die Bedingung monotone Bewegungen erlaubt

Sliding Window benötigt eine gerichtete Reaktion:

- Wenn ein Fenster ungültig wird, muss Schrumpfen es wieder gültig machen können.
- Wenn ein ausreichendes Fenster geschrumpft wird, muss erkennbar sein, wann es
  nicht mehr ausreicht.
- `left` darf nie rückwärts laufen.

Beispiele:

| Problem | Wirkung des Wachsens | Wirkung des Schrumpfens | geeignet? |
|---|---|---|---|
| Summe positiver Werte | Summe steigt | Summe sinkt | ja |
| höchstens `k` Nullen | Nullenzahl steigt/bleibt | sinkt/bleibt | ja |
| keine Duplikate | Konflikte können entstehen | Konflikte verschwinden | ja |
| Summe mit negativen Werten | kann steigen oder sinken | kann steigen oder sinken | meist nein |
| beliebige nicht zusammenhängende Auswahl | kein einzelnes Fenster | — | nein |

## 21. Hash-Map-Zustand korrekt pflegen

Zwei verbreitete Repräsentationen sind:

1. **Häufigkeiten:** Beim Eintritt erhöhen, beim Austritt verringern und Nullwerte
   entfernen. Geeignet für Anagramme, Sollhäufigkeiten und „höchstens k
   verschiedene Werte“.
2. **Letzter Index:** Bei einem Duplikat springt `left` direkt hinter dessen
   letztes Vorkommen. Geeignet für eindeutige Substrings.

Die Invariante muss sagen, ob die Map das aktuelle Fenster oder die gesamte
bisherige Historie beschreibt. `last_seen` enthält Historie; deshalb ist die
Prüfung gegen `left` erforderlich. Eine Frequenz-Map soll dagegen exakt das
Fenster spiegeln und muss austretende Werte aktualisieren.

## 22. Mustervergleich

| Muster | Vorbedingung | Zeigerbewegung | Zustand | typische Ausgabe |
|---|---|---|---|---|
| Gegenläufige Two Pointers | meist sortiert oder Verwerfungsargument | außen nach innen | oft `O(1)` | Paar, Fläche, Wahrheitswert |
| Gleichläufige Two Pointers | lokales Keep/Drop-Kriterium | beide vorwärts | gültiges Präfix | neue Länge, In-Place-Ergebnis |
| Festes Sliding Window | zusammenhängende feste Breite | synchron vorwärts | Summe/Counter/Statistik | Wert pro Fenster |
| Variables Sliding Window | monotone Gültigkeitsreaktion | rechts wächst, links schrumpft | Summe/Counter/Map | längstes/kürzestes Fenster |

## 23. Abgrenzung zu verwandten Techniken

### Präfixsummen

Präfixsummen beantworten viele beliebige Bereichssummen nach `O(n)`
Vorverarbeitung in `O(1)`. Sliding Window ist direkter, wenn alle benachbarten
Fenster einmal in Reihenfolge verarbeitet oder Streamdaten online ausgewertet
werden.

### Hash Map

Für Two Sum auf unsortierten Daten liefert eine Hash Map eine `O(n)`-Lösung ohne
Sortierung. Two Pointers ist attraktiv, wenn die Daten bereits sortiert sind,
Bereichsstruktur genutzt wird oder `O(1)` Zusatzspeicher wichtig ist.

### Binary Search

Binary Search halbiert einen geordneten Kandidatenraum. Two Pointers verwirft
durch gerichtete Grenzbewegungen Kandidaten. Manche Probleme kombinieren beide
Ideen, etwa eine sortierte Paarsuche innerhalb einer äußeren Schleife.

### Dynamic Programming

Wenn optimale Teilstrukturen über viele nicht zusammenhängende Entscheidungen
entstehen und keine monotone Fensterbewegung möglich ist, ist Dynamic Programming
oft die passendere Technik.

## 24. Laufzeitvergleich

| Aufgabe | Brute Force | Musterlösung | Zusatzspeicher |
|---|---:|---:|---:|
| Paarsumme, sortiert | `O(n²)` | `O(n)` | `O(1)` |
| stabiler In-Place-Filter | Kopie oft `O(n)` | `O(n)` | `O(1)` zusätzlich |
| alle Summen der Breite `k` | `O(nk)` | `O(n)` | `O(1)` |
| kürzestes positives Summenfenster | `O(n²)` | `O(n)` | `O(1)` |
| längster eindeutiger Substring | `O(n²)` oder `O(n³)` | `O(n)` erwartet | `O(a)` Alphabet |
| Rolling Mean/Varianz | `O(nk)` | `O(n)` | `O(1)` |

Die Hash-Map-Angabe `O(n)` verwendet die erwartete konstante Laufzeit von
Hash-Operationen. `a` bezeichnet die Zahl verschiedener Werte im Fenster.

## 25. Klassische Fehler

### Unbegründete Zeigerbewegung

„Summe zu klein, also links erhöhen“ ist nur mit sortierter Eingabe korrekt.
Jede Bewegung braucht ein Verwerfungsargument.

### Fenstergrenzen vermischen

Bei inklusivem Fenster `[left, right]` ist die Länge `right - left + 1`. Ein
fehlendes `+1` erzeugt typische Off-by-One-Fehler.

### Zustand in falscher Reihenfolge aktualisieren

Wird `left` erhöht, bevor das austretende Element aus Summe oder Counter entfernt
wurde, beschreibt der Zustand nicht mehr das dokumentierte Fenster.

### Nur einmal statt solange schrumpfen

Ein variables Fenster benötigt häufig `while`, nicht `if`. Nach einem einzigen
Entfernen kann die Bedingung weiterhin verletzt oder weiterhin erfüllt sein.

### Linke Grenze zurücksetzen

Beim letzten-Index-Muster muss `left = max(left, previous + 1)` gelten. Ein altes
Duplikat außerhalb des Fensters darf die Grenze nicht rückwärts bewegen.

### Leere und unmögliche Fenster

Breite null, Breite größer als die Eingabe, leere Strings und nicht erreichbare
Zielbedingungen brauchen explizite Verträge.

## 26. Teststrategie

Gute Tests prüfen:

- leere Eingabe, ein Element und Fensterbreite eins,
- Treffer an beiden Außenrändern,
- kein gültiges Paar oder Fenster,
- nur gleiche Werte und nur verschiedene Werte,
- Ziel bereits im ersten Fenster und erst im letzten Fenster,
- mehrfaches Schrumpfen in einer rechten Iteration,
- Unicode-Zeichen bei Stringfenstern,
- Zustandsinvarianten nach jedem Eintritt und Austritt,
- Ergebnisgleichheit mit einer einfachen Brute-Force-Referenz,
- reproduzierbare Zufallsfälle mit festem Seed.

Ein kompakter Property-Test für feste Fenstersummen:

```python
import random


rng = random.Random(90901)  # Fixed seed makes failures reproducible.
for _ in range(200):
    random_values = [rng.randrange(-20, 21) for _ in range(rng.randrange(30))]
    width = rng.randrange(1, 35)
    expected = [
        sum(random_values[start : start + width])
        for start in range(len(random_values) - width + 1)
    ]
    assert rolling_sums(random_values, width) == expected
```

## 27. Entscheidungsleitfaden

Vor dem Coden helfen diese Fragen:

1. Ist die gesuchte Auswahl zusammenhängend? Falls ja, ist Sliding Window ein
   Kandidat.
2. Ist die Fensterbreite fest oder hängt sie von einer Bedingung ab?
3. Welche Information muss beim Eintritt und Austritt eines Elements geändert
   werden?
4. Reagiert die Bedingung monoton genug, damit `left` nur vorwärts laufen darf?
5. Ist die Eingabe sortiert, sodass eine zu kleine oder zu große Paargröße eine
   Zeigerbewegung begründet?
6. Wird ein gültiges Präfix in-place aufgebaut, sodass Lese- und Schreibzeiger
   passen?
7. Kann jedes Element nachweislich nur konstant oft betreten oder verlassen
   werden?
8. Welche Brute-Force-Lösung dient als Referenztest?

## 28. Zusammenfassung

- Gegenläufige Two Pointers nutzen meist Sortierung oder ein geometrisches
  Verwerfungsargument.
- Gleichläufige Two Pointers bauen ein gültiges Präfix stabil und in-place auf.
- Feste Fenster ersetzen vollständige Neuberechnung durch Eintritts- und
  Austrittsupdates.
- Variable Fenster wachsen rechts und schrumpfen links, solange eine monotone
  Bedingungslogik dies erlaubt.
- Hash Maps repräsentieren Häufigkeiten oder letzte Positionen im
  Fensterzustand.
- Die verschachtelte Schleifenform bleibt linear, weil jede Grenze höchstens `n`
  Schritte vorwärts macht.
- Negative Zahlen, unsortierte Daten oder nicht zusammenhängende Auswahlen
  können die Voraussetzungen zerstören.
- Rolling Windows übertragen das Muster direkt auf Zeitreihenfeatures und
  Streaming-Anomalieerkennung.

Wer die Invariante des Bereichs und die Bedeutung jeder Zeigerbewegung benennen
kann, erkennt die Muster zuverlässig und vermeidet zufällige, fehleranfällige
Schablonenanwendung.
