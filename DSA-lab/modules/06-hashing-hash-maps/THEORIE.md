# Modul 06: Hashing & Hash Maps

Hash Maps gehören zu den wichtigsten Werkzeugen in Algorithmen und
Datenpipelines. Sie beantworten Fragen wie „Habe ich diesen Schlüssel schon
gesehen?“, „Welcher Wert gehört zu dieser ID?“ oder „Wie oft kam dieses Ereignis
vor?“ im Average Case in O(1). Dieses Versprechen ist jedoch nicht kostenlos:
Eine Hash-Funktion muss Schlüssel auf einen begrenzten Speicher abbilden,
Kollisionen sind unvermeidbar, und bei ungünstiger Verteilung bleibt als Worst
Case O(n).

Dieses Skript entwickelt das Konzept in drei Ebenen:

1. **Intuition:** Warum Direktzugriff schneller als lineare Suche sein kann.
2. **Simulation:** Wie Schlüssel in Buckets landen, wie Kollisionen behandelt
   werden und warum der Load Factor entscheidend ist.
3. **Formalisierung:** Invarianten, Komplexitäten, Rehashing, Python-Interna und
   wiederverwendbare Lösungsmuster.

Nach dem Modul sollst du eine eigene Hash Map mit Kollisionsbehandlung bauen,
den Einfluss der Füllung messen und erkennen können, wann zusätzlicher Speicher
eine lineare Suche vermeidet.

---

## Ebene 1: Intuition

### 1. Vom Durchsuchen zum Adressieren

Stell dir ein Lager mit 10.000 Paketen vor. Liegen alle Pakete unsortiert in
einem Raum, musst du im Worst Case jedes Etikett lesen. Die Suche nach Paket
`DSA-4711` kostet O(n).

Ein Regal mit nummerierten Fächern verändert die Frage. Statt „Wo liegt das
Paket?“ fragst du „Welche Fachnummer ergibt sich aus dem Etikett?“. Kannst du die
Fachnummer direkt berechnen, springst du ohne Durchsuchen zum Ziel. Genau diese
Übersetzung übernimmt eine **Hash-Funktion**.

```text
Schlüssel --Hash-Funktion--> Hashwert --Kompression--> Bucket-Index
"DSA-4711"                    8347219                   3
```

Die Hash Map speichert ein Paar aus Schlüssel und Wert im berechneten Bucket.
Bei einer späteren Suche berechnet sie denselben Index erneut. Der Schlüssel ist
also nicht selbst die Adresse; er wird deterministisch in eine Kandidatenadresse
übersetzt.

### 2. Warum O(1) fast zu gut klingt

Ein Arrayzugriff an einem bekannten Index kostet O(1). Wenn das Berechnen des
Hashwerts ebenfalls als konstant gilt, scheint auch die gesamte Suche konstant
zu sein. Drei Einschränkungen gehören aber immer zum Versprechen:

- Mehrere Schlüssel können denselben Bucket erhalten. Diese **Kollisionen** muss
  die Struktur zusätzlich auflösen.
- Die Tabelle braucht freie Kapazität. Mit wachsendem **Load Factor** werden
  Kollisionen beziehungsweise lange Suchpfade wahrscheinlicher.
- O(1) ist der erwartete beziehungsweise amortisierte Average Case. Erzwingen
  viele Schlüssel denselben Hashwert, degeneriert die Suche zu O(n).

Die korrekte Kurzform lautet deshalb:

> Hash Maps bieten Put, Get und Delete im erwarteten Average Case in O(1),
> benötigen dafür O(n) Speicher und behalten O(n) als Worst Case.

### 3. Hash Map und Hash Set

Eine Hash Map speichert `key -> value`. Ein Hash Set benötigt nur Schlüssel und
beantwortet die Frage nach Mitgliedschaft.

| Frage | Geeignete Struktur |
|---|---|
| Welcher Name gehört zu Kundennummer 42? | Hash Map |
| Ist Ereignis-ID `evt-17` schon aufgetreten? | Hash Set |
| Wie oft kam Status 500 vor? | Hash Map mit Zählerwerten |
| Welche eindeutigen IP-Adressen existieren? | Hash Set |

Ein Set kann konzeptionell als Hash Map betrachtet werden, deren Werte nicht
benötigt werden.

---

## Ebene 2: Simulation

### 4. Von einem String zum Bucket

Pythons eingebautes `hash()` ist innerhalb eines Prozesses geeignet, aber die
Hashwerte von Strings werden aus Sicherheitsgründen zwischen Prozessen
typischerweise randomisiert. Für eine nachvollziehbare Simulation verwenden wir
daher eine bewusst einfache, stabile Funktion.

```python
def stable_text_hash(text: str) -> int:
    """Return a small deterministic polynomial hash for teaching purposes."""
    value = 0
    for character in text:
        value = value * 31 + ord(character)
    return value


def bucket_index(key: str, capacity: int) -> int:
    """Map a text key into the valid bucket range."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return stable_text_hash(key) % capacity


example_keys = ["Ada", "Linus", "Grace", "Edsger", "Barbara"]
example_capacity = 8
example_distribution = {
    key: bucket_index(key, example_capacity) for key in example_keys
}
print(example_distribution)
assert all(0 <= index < example_capacity for index in example_distribution.values())
assert bucket_index("Ada", 8) == bucket_index("Ada", 8)
```

Die Multiplikation mit 31 sorgt dafür, dass die Reihenfolge der Zeichen wirkt:
`"ab"` und `"ba"` erhalten gewöhnlich verschiedene Hashwerte. Das Modulo
komprimiert einen beliebig großen Hashwert auf den Bereich `0 .. capacity - 1`.

Eine brauchbare Hash-Funktion für eine Hash Map sollte:

- für denselben unveränderten Schlüssel denselben Hashwert liefern,
- gleiche Schlüssel gleich hashen,
- kleine Unterschiede möglichst über viele Hashwerte verteilen,
- schnell zu berechnen sein und
- die erwarteten Eingaben möglichst gleichmäßig auf Buckets verteilen.

Sie muss nicht kollisionsfrei sein. Bei mehr möglichen Schlüsseln als Buckets
ist Kollisionsfreiheit nach dem Schubfachprinzip unmöglich. Eine kryptografische
Hash-Funktion verfolgt zusätzliche Sicherheitsziele und ist für eine gewöhnliche
In-Memory-Hash-Map meist unnötig teuer.

### 5. Verteilung sichtbar machen

Eine Verteilung lässt sich als Bucket-Belegung zählen. Perfekte Gleichverteilung
ist bei kleinen Stichproben nicht zu erwarten; problematisch ist eine systematische
Häufung.

```python
def bucket_histogram(keys: list[str], capacity: int) -> list[int]:
    """Count how many keys map to every bucket."""
    counts = [0] * capacity
    for key in keys:
        counts[bucket_index(key, capacity)] += 1
    return counts


simulation_keys = [f"sensor-{number:03d}" for number in range(100)]
simulation_counts = bucket_histogram(simulation_keys, capacity=16)
print("Bucket occupancy:", simulation_counts)
assert sum(simulation_counts) == len(simulation_keys)
```

Die Histogrammwerte messen noch keine Suchzeit. Sie zeigen aber, wo eine
Kollisionsstrategie zusätzliche Arbeit leisten muss.

### 6. Kollisionen mit Chaining auflösen

Beim **Separate Chaining** ist jeder Bucket ein kleiner Container von
Schlüssel-Wert-Paaren. Kollidierende Schlüssel werden im selben Bucket
gespeichert. Innerhalb des Buckets wird der echte Schlüssel verglichen, denn ein
gleicher Bucket-Index bedeutet nicht, dass die Schlüssel gleich sind.

Nehmen wir Kapazität 5 und diese bereits berechneten Indizes:

| Operation | Index | Zustand des Buckets |
|---|---:|---|
| `put("Ada", 10)` | 2 | `[('Ada', 10)]` |
| `put("Grace", 20)` | 2 | `[('Ada', 10), ('Grace', 20)]` |
| `put("Linus", 30)` | 4 | `[('Linus', 30)]` |
| `put("Ada", 99)` | 2 | `[('Ada', 99), ('Grace', 20)]` |

Die letzte Operation fügt keinen zweiten `Ada`-Eintrag ein. Sie findet im Bucket
den gleichen Schlüssel und aktualisiert dessen Wert.

```python
class TinyChainedMap:
    """A minimal fixed-capacity hash map for collision tracing."""

    def __init__(self, capacity: int = 5) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buckets: list[list[tuple[str, int]]] = [
            [] for _ in range(capacity)
        ]

    def put(self, key: str, value: int) -> None:
        bucket = self._buckets[bucket_index(key, len(self._buckets))]
        for position, (stored_key, _) in enumerate(bucket):
            if stored_key == key:
                bucket[position] = (key, value)
                return
        bucket.append((key, value))

    def get(self, key: str) -> int:
        bucket = self._buckets[bucket_index(key, len(self._buckets))]
        for stored_key, value in bucket:
            if stored_key == key:
                return value
        raise KeyError(key)

    def delete(self, key: str) -> None:
        bucket = self._buckets[bucket_index(key, len(self._buckets))]
        for position, (stored_key, _) in enumerate(bucket):
            if stored_key == key:
                bucket.pop(position)
                return
        raise KeyError(key)


chained = TinyChainedMap(capacity=3)
chained.put("alpha", 1)
chained.put("beta", 2)
chained.put("alpha", 3)
assert chained.get("alpha") == 3
assert chained.get("beta") == 2
chained.delete("beta")
try:
    chained.get("beta")
except KeyError:
    pass
else:
    raise AssertionError("deleted key must be absent")
```

Chaining bleibt korrekt, selbst wenn alle Schlüssel im selben Bucket landen.
Die Laufzeit wird dann allerdings linear in der Länge dieser Chain.

### 7. Kollisionen mit Linear Probing auflösen

**Open Addressing** speichert alle Einträge direkt im Bucket-Array. Ist der
berechnete Bucket belegt, untersucht Linear Probing nacheinander weitere Plätze:

```text
index_i = (start_index + i) mod capacity
```

Bei Kapazität 7 starten drei Schlüssel am Index 4:

| Schritt | Bucket 4 | Bucket 5 | Bucket 6 |
|---|---|---|---|
| `put(A)` | A | leer | leer |
| `put(B)` | A | B | leer |
| `put(C)` | A | B | C |

Eine Suche nach C darf nicht bei Bucket 4 oder 5 abbrechen. Sie folgt derselben
Probe Sequence bis C oder ein wirklich leerer Bucket gefunden wird.

```python
def linear_probe_insert(
    table: list[str | None], start_index: int, key: str
) -> int:
    """Insert a key into the first free slot and return its index."""
    for offset in range(len(table)):
        index = (start_index + offset) % len(table)
        if table[index] is None:
            table[index] = key
            return index
    raise OverflowError("hash table is full")


probe_table: list[str | None] = [None] * 7
assert linear_probe_insert(probe_table, 4, "A") == 4
assert linear_probe_insert(probe_table, 4, "B") == 5
assert linear_probe_insert(probe_table, 4, "C") == 6
assert linear_probe_insert(probe_table, 6, "D") == 0
print("Linear probing:", probe_table)
```

Die vierte Einfügung demonstriert Wrap-around. Nach Index 6 geht die Probe
Sequence bei Index 0 weiter.

#### Warum Löschen einen Tombstone braucht

Würde B einfach durch einen leeren Bucket ersetzt, könnte eine spätere Suche
nach C fälschlich bei diesem Loch abbrechen. Open-Addressing-Tabellen verwenden
daher meist drei Zustände:

1. **never used:** Hier darf eine erfolglose Suche abbrechen.
2. **occupied:** Schlüssel und Wert sind gespeichert.
3. **deleted/Tombstone:** Die Suche muss weiterlaufen; eine Einfügung darf den
   Platz später wiederverwenden.

Linear Probing ist cache-freundlich, kann aber **Primary Clustering** erzeugen:
Zusammenhängende belegte Bereiche wachsen und verlängern weitere Probe Sequences.

### 8. Load Factor und Rehashing simulieren

Der Load Factor beschreibt die Füllung:

```text
alpha = number_of_entries / number_of_buckets
```

Bei Chaining kann `alpha` größer als 1 werden. Bei Open Addressing muss stets
mindestens ein nutzbarer freier Platz existieren. In beiden Fällen verschlechtert
ein hoher Load Factor typischerweise die Performance.

Angenommen, eine Tabelle startet mit Kapazität 4 und rehasht vor einer Einfügung,
die `alpha > 0.75` verursachen würde:

| Einträge | Kapazität | Load Factor | Aktion |
|---:|---:|---:|---|
| 1 | 4 | 0,25 | keine |
| 2 | 4 | 0,50 | keine |
| 3 | 4 | 0,75 | keine |
| 4 geplant | 4 | 1,00 | auf 8 vergrößern, alle Einträge neu hashen |
| 4 | 8 | 0,50 | Einfügung abschließen |

Beim Rehashing reicht es nicht, die alten Buckets an denselben Index zu kopieren.
Der Index hängt von der Kapazität ab:

```text
old_index = hash(key) mod 4
new_index = hash(key) mod 8
```

Jeder gespeicherte Schlüssel muss deshalb erneut verteilt werden. Ein einzelnes
Rehashing kostet O(n), tritt bei geometrischem Wachstum aber nur gelegentlich
auf. Dadurch bleibt Put amortisiert O(1).

### 9. Den Worst Case empirisch provozieren

Python erlaubt eigene Objekte mit kontrolliertem Hashwert. Die folgende Klasse
erzwingt für alle Instanzen denselben Hash und macht aus dem schnellen Lookup
eine lineare Suche durch viele Kollisionen.

```python
from time import perf_counter


class BadHashKey:
    """A key that intentionally collides with every sibling key."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BadHashKey) and self.value == other.value


def measure_repeated_lookup(mapping: dict[object, int], key: object) -> float:
    """Measure many lookups and return elapsed seconds."""
    start = perf_counter()
    checksum = 0
    for _ in range(2_000):
        checksum += mapping[key]
    elapsed = perf_counter() - start
    assert checksum >= 0
    return elapsed


for measurement_size in (100, 400, 800):
    normal_map = {number: number for number in range(measurement_size)}
    colliding_keys = [BadHashKey(number) for number in range(measurement_size)]
    colliding_map = {key: key.value for key in colliding_keys}
    normal_time = measure_repeated_lookup(normal_map, measurement_size - 1)
    collision_time = measure_repeated_lookup(
        colliding_map, colliding_keys[-1]
    )
    print(
        f"n={measurement_size:4d}: normal={normal_time:.6f}s, "
        f"colliding={collision_time:.6f}s"
    )
```

Absolute Zeiten hängen vom Rechner ab. Relevant ist das Wachstum: Beim normalen
Dictionary bleibt die Lookup-Zeit pro Zugriff ungefähr stabil; beim absichtlich
kollidierenden Dictionary wächst die Anzahl notwendiger Gleichheitsvergleiche
mit n. Das Experiment zeigt, warum O(1) kein Worst-Case-Versprechen ist.

---

## Ebene 3: Formalisierung

### 10. Abstrakte Definition

Eine Map repräsentiert eine endliche Menge von Paaren

```text
M = {(k_1, v_1), (k_2, v_2), ..., (k_n, v_n)}
```

mit eindeutigen Schlüsseln. Die Kernoperationen sind:

- `put(k, v)`: Einfügen oder den Wert eines vorhandenen Schlüssels ersetzen.
- `get(k)`: Den zu k gehörenden Wert liefern oder Abwesenheit melden.
- `delete(k)`: Das Paar mit Schlüssel k entfernen oder Abwesenheit melden.
- `contains(k)`: Prüfen, ob k vorhanden ist.

Eine Hash Table implementiert diese abstrakte Map mit:

1. einer Hash-Funktion `h(k)`,
2. einer Kompression auf einen Bucket-Index und
3. einer Strategie zur Auflösung von Kollisionen.

Die zentrale Korrektheitsbedingung lautet:

> Wenn zwei Schlüssel gemäß `==` gleich sind, müssen ihre Hashwerte gleich sein.

Die Umkehrung gilt nicht: Gleiche Hashwerte dürfen zu ungleichen Schlüsseln
gehören. Deshalb muss die Struktur nach dem Hashvergleich immer noch die
Schlüsselgleichheit prüfen.

### 11. Chaining formal

Für eine Tabelle mit `m` Buckets berechnet sich der Index häufig als

```text
i = h(k) mod m
```

Bucket `T[i]` enthält eine Folge von Einträgen. Die Map-Invariante fordert:

- Jeder gespeicherte Schlüssel kommt genau einmal vor.
- Jeder Eintrag `(k, v)` liegt in dem Bucket, den die aktuelle Kapazität für k
  bestimmt.
- Der gespeicherte Größenwert entspricht der Gesamtzahl aller Einträge.

Pseudocode für Get:

```text
GET(key):
    index <- HASH(key) mod capacity
    for (stored_key, value) in buckets[index]:
        if stored_key == key:
            return value
    raise KeyError
```

Put durchsucht dieselbe Chain. Findet es den Schlüssel, ersetzt es den Wert;
andernfalls hängt es ein neues Paar an und erhöht die Größe. Delete entfernt nur
bei erfolgreichem Fund und vermindert die Größe genau einmal.

Unter der Annahme gleichmäßiger Verteilung beträgt die erwartete Chain-Länge
ungefähr `alpha = n / m`. Hält Rehashing alpha durch eine Konstante beschränkt,
bleiben die Operationen erwartet O(1).

### 12. Open Addressing formal

Open Addressing definiert für jeden Schlüssel eine Folge möglicher Indizes:

```text
p(k, 0), p(k, 1), ..., p(k, m - 1)
```

Linear Probing verwendet:

```text
p(k, i) = (h(k) + i) mod m
```

Die Probe Sequence muss im erlaubten Bereich jeden relevanten Platz erreichen.
Get, Put und Delete müssen dieselbe Folge verwenden. Eine Suche endet nur bei:

- dem gesuchten Schlüssel,
- einem `never used`-Slot oder
- nach höchstens m untersuchten Buckets.

Tombstones erhalten die Suchpfade, erhöhen aber langfristig die effektive
Füllung. Ein Rehashing entfernt sie, weil nur aktive Einträge in ein frisches
Array übertragen werden.

### 13. Komplexitäten einordnen

| Operation | Erwartet/amortisiert | Worst Case | Ursache des Worst Case |
|---|---:|---:|---|
| Put | O(1) | O(n) | lange Kollisionen oder Rehashing |
| Get | O(1) | O(n) | alle relevanten Schlüssel kollidieren |
| Delete | O(1) | O(n) | lange Chain/Probe Sequence |
| Contains | O(1) | O(n) | wie Get |
| Iteration | O(n) | O(n + m) intern möglich | Einträge beziehungsweise Buckets besuchen |
| Speicher | O(n + m) | O(n + m) | Einträge plus Bucket-Array |

Bei einer gut dimensionierten Tabelle ist `m = Theta(n)`, daher wird der Speicher
gewöhnlich als O(n) angegeben. Die O(1)-Angabe behandelt außerdem das Hashen
eines Schlüssels als konstant. Für einen String der Länge L kostet seine erste
Hashberechnung konzeptionell O(L); manche Laufzeitumgebungen cachen Hashwerte
unveränderlicher Objekte.

### 14. Chaining und Open Addressing vergleichen

| Aspekt | Separate Chaining | Open Addressing |
|---|---|---|
| Speicherort | Bucket plus externe Einträge | Einträge direkt im Array |
| Load Factor | darf größer als 1 sein | muss unter 1 bleiben |
| Löschen | direkt aus Chain | Tombstone oder aufwendige Reparatur |
| Cache-Lokalität | oft schwächer | oft stärker |
| Kollisionskosten | Länge der Chain | Länge der Probe Sequence |
| Zusatzspeicher | Container/Referenzen je Bucket | Status pro Array-Slot |
| Typisches Problem | einzelne lange Chains | Clustering und Tombstones |

Keine Strategie ist universell überlegen. Entscheidend sind Speicherlayout,
Schlüsselverteilung, erwartete Füllung und das Verhältnis von Lese-, Schreib- und
Löschoperationen.

### 15. Python: `dict`, `set` und Hashability

Python stellt Hash Maps als `dict` und Hash Sets als `set` bereit. Ihre konkrete
CPython-Implementierung ist versionsabhängig, konzeptionell verwendet ein
Dictionary eine hochoptimierte Open-Addressing-Struktur. Seit Python 3.7 ist die
Einfügereihenfolge eines `dict` Teil der Sprachgarantie. Diese Reihenfolge macht
das Dictionary aber weder sortiert noch zu einer Priority Queue.

Ein Objekt ist **hashable**, wenn:

- es einen Hashwert besitzt, der sich während seiner Lebenszeit nicht ändert,
- es mit anderen Objekten verglichen werden kann und
- Gleichheit die Regel `a == b -> hash(a) == hash(b)` einhält.

Typische unveränderliche Built-ins sind hashable: Zahlen, Strings, Bytes und
`frozenset`. Ein Tuple ist nur dann hashable, wenn alle seine Elemente hashable
sind. Listen, Dictionaries und Sets sind veränderlich und daher keine Keys.

```python
valid_python_map = {
    "sensor-17": "online",
    (52.5, 13.4): "Berlin",
    frozenset({"read", "write"}): "editor",
}
assert valid_python_map[(52.5, 13.4)] == "Berlin"

hashability_checks = {
    "string": isinstance(hash("key"), int),
    "tuple": isinstance(hash((1, 2)), int),
}
try:
    hash([1, 2])
except TypeError:
    hashability_checks["list_rejected"] = True

assert all(hashability_checks.values())
```

Warum wäre eine Liste als Key gefährlich? Würde `[1, 2]` zunächst in Bucket 5
liegen und später zu `[1, 2, 3]` mutieren, könnte sich ihr Hashwert ändern. Die
Map würde beim Lookup einen anderen Bucket untersuchen und den gespeicherten Key
nicht mehr zuverlässig finden.

Auch eigene Klassen brauchen Sorgfalt. Wer `__eq__` fachlich definiert, muss dazu
ein konsistentes `__hash__` liefern oder die Klasse bewusst unhashable lassen.
Unveränderliche `dataclass`-Objekte können beispielsweise mit `frozen=True` als
Keys modelliert werden, sofern ihre Felder selbst hashable sind.

#### Praktische Hinweise zu Python-Dictionaries

- `key in mapping` prüft Keys, nicht Values.
- `mapping.get(key)` unterscheidet ohne Sentinel nicht zwischen „fehlt“ und
  „vorhandener Wert ist None“.
- `set` entfernt Duplikate, garantiert aber keine sortierte Ausgabe.
- `hash(text)` sollte nicht als dauerhafte Datenbank-ID gespeichert werden; sein
  Wert ist nicht als prozess- und versionsübergreifendes Format gedacht.
- Während einer Iteration sollte die Größe eines Dictionarys oder Sets nicht
  verändert werden.

### 16. Anwendungsmuster

Hash-basierte Lösungen erkennt man häufig an einer der Fragen:

- Muss ich oft prüfen, ob etwas bereits gesehen wurde?
- Brauche ich zu einer ID schnell Metadaten?
- Muss ich Elemente nach einem berechenbaren Merkmal gruppieren?
- Kann ich frühere Ergebnisse speichern, statt erneut zu suchen?
- Darf ich O(n) zusätzlichen Speicher einsetzen, um O(n²) Zeit zu vermeiden?

#### 16.1 Frequency Counting

Ein Dictionary ordnet jedem Wert seinen bisherigen Zähler zu.

```python
def frequency_count(items: list[str]) -> dict[str, int]:
    """Count occurrences in one pass."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


status_counts = frequency_count(["ok", "error", "ok", "timeout", "ok"])
assert status_counts == {"ok": 3, "error": 1, "timeout": 1}
```

Zeit: erwartet O(n). Speicher: O(k) für k verschiedene Werte.

#### 16.2 Deduplizierung mit stabiler Reihenfolge

Ein Set übernimmt schnelle Mitgliedschaft, eine Liste bewahrt die erste
Auftretensreihenfolge.

```python
def unique_in_order(items: list[str]) -> list[str]:
    """Keep the first occurrence of every item."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


event_ids = ["e3", "e1", "e3", "e2", "e1"]
assert unique_in_order(event_ids) == ["e3", "e1", "e2"]
```

`list(set(items))` entfernt zwar ebenfalls Duplikate, drückt die gewünschte
Reihenfolge aber nicht aus und eignet sich nicht, wenn das erste Auftreten
erhalten bleiben soll.

#### 16.3 Lookup-Tabelle statt wiederholter Suche

Beim Verknüpfen kleiner Datensätze kann eine Seite einmal indexiert werden.

```python
def attach_names(
    measurements: list[tuple[int, float]],
    sensors: list[tuple[int, str]],
) -> list[tuple[str, float]]:
    """Join measurements with sensor names through an indexed lookup."""
    name_by_id = {sensor_id: name for sensor_id, name in sensors}
    return [
        (name_by_id[sensor_id], value)
        for sensor_id, value in measurements
    ]


sensor_rows = [(7, "north"), (9, "south")]
measurement_rows = [(9, 18.2), (7, 17.8), (9, 18.4)]
assert attach_names(measurement_rows, sensor_rows) == [
    ("south", 18.2),
    ("north", 17.8),
    ("south", 18.4),
]
```

Ohne Index könnte jede der n Messungen alle m Sensoren durchsuchen: O(nm). Mit
Dictionary kostet der Aufbau erwartet O(m), das anschließende Join O(n), zusammen
O(n + m) bei O(m) zusätzlichem Speicher.

#### 16.4 Two Sum als Komplement-Suche

Gesucht sind zwei Werte mit Summe `target`. Statt für jedes Paar O(n²) zu testen,
speichern wir bereits gesehene Werte und fragen nach dem Komplement.

```python
def two_sum(numbers: list[int], target: int) -> tuple[int, int] | None:
    """Return indices of two values adding up to target."""
    index_by_value: dict[int, int] = {}
    for right_index, value in enumerate(numbers):
        complement = target - value
        if complement in index_by_value:
            return index_by_value[complement], right_index
        index_by_value[value] = right_index
    return None


assert two_sum([2, 7, 11, 15], 9) == (0, 1)
assert two_sum([3, 3], 6) == (0, 1)
assert two_sum([1, 2, 3], 100) is None
```

Die Reihenfolge ist wichtig: Erst nach dem Komplement suchen, dann den aktuellen
Wert speichern. So wird dasselbe Element nicht zweimal verwendet. Das Muster
verallgemeinert sich auf Paarungen, Differenzen und bereits gesehene Zustände.

#### 16.5 Gruppierung nach einem kanonischen Schlüssel

Für Anagramme ist die sortierte Zeichenfolge ein möglicher Gruppenschlüssel.

```python
def group_anagrams(words: list[str]) -> list[list[str]]:
    """Group words that contain the same character multiset."""
    groups: dict[str, list[str]] = {}
    for word in words:
        signature = "".join(sorted(word))
        groups.setdefault(signature, []).append(word)
    return list(groups.values())


anagram_groups = group_anagrams(["eat", "tea", "tan", "ate", "nat"])
assert sorted(sorted(group) for group in anagram_groups) == [
    ["ate", "eat", "tea"],
    ["nat", "tan"],
]
```

Die Hash Map löst hier nicht die gesamte Aufgabe. Entscheidend ist zuerst eine
Signatur, die für fachlich gleiche Gruppen identisch und hashable ist.

### 17. Speicher gegen Zeit: eine Entscheidungsregel

Hashing ist besonders attraktiv, wenn viele spätere Lookups die einmaligen
Kosten eines Indexaufbaus amortisieren. Ein typischer Umbau lautet:

```text
Vorher: Für jedes Element erneut linear suchen -> O(n²), O(1) Zusatzspeicher
Nachher: Gesehene Elemente indexieren       -> O(n) erwartet, O(n) Speicher
```

Eine Hash Map ist nicht automatisch die beste Wahl:

- Für sehr kleine Daten kann eine lineare Liste einfacher und schneller sein.
- Benötigst du sortierte Reihenfolge oder Bereichsabfragen, hilft Hashing allein
  nicht; sortierte Arrays oder Bäume passen besser.
- Bei knappem Speicher kann der zusätzliche Tabellen-Overhead zu groß sein.
- Für persistente oder verteilte Schlüssel braucht es ein klar definiertes,
  stabiles Hashverfahren statt des laufzeitinternen `hash()`.
- Bei sicherheitskritischen, fremdgesteuerten Keys muss auch der Worst Case und
  Kollisionsmissbrauch berücksichtigt werden.

### 18. Häufige Denk- und Implementierungsfehler

#### Kollision mit Gleichheit verwechseln

`hash(a) == hash(b)` beweist nicht `a == b`. Ein Bucket-Fund ist nur ein Kandidat;
der echte Schlüsselvergleich bleibt notwendig.

#### Beim Update die Größe erhöhen

`put(existing_key, new_value)` ersetzt einen Wert. Die Anzahl der Einträge darf
sich nicht ändern.

#### Beim Rehashing alte Indizes kopieren

Nach einer Kapazitätsänderung muss jeder aktive Schlüssel mit der neuen
Kapazität verteilt werden.

#### Open-Addressing-Einträge einfach leeren

Ein echtes Loch kann eine Probe Sequence vorzeitig beenden. Delete benötigt
Tombstones oder eine korrekte Reparatur der folgenden Cluster.

#### Nur den Durchschnitt messen

Ein durchschnittlich kurzer Lookup kann einzelne lange Chains verbergen.
Hilfreich sind zusätzlich Maximum, Quantile, Bucket-Histogramme und Probe-Längen.

#### `get` mit einem fachlichen Wert als Sentinel verwenden

```python
sentinel = object()
mapping_with_none = {"known": None}
known_value = mapping_with_none.get("known", sentinel)
missing_value = mapping_with_none.get("missing", sentinel)
assert known_value is None
assert missing_value is sentinel
```

Nur ein eindeutiger Sentinel unterscheidet zuverlässig zwischen „Key fehlt“ und
„Key existiert mit Wert None“.

### 19. Selbstkontrolle

Du hast die Kernideen verstanden, wenn du diese Fragen ohne Auswendiglernen
begründen kannst:

1. Warum kann eine Hash Map O(1) im Average Case, aber O(n) im Worst Case haben?
2. Warum prüft Get nach dem Bucket-Index zusätzlich den Schlüssel?
3. Welche Invariante verletzt eine mutable Liste als Key?
4. Warum müssen Einträge beim Resize neu gehasht werden?
5. Weshalb braucht Linear Probing beim Löschen einen Tombstone?
6. Was verändert ein hoher Load Factor an Chain- beziehungsweise Probe-Längen?
7. Wann ist ein Set geeigneter als ein Dictionary?
8. Wie ersetzt ein Lookup-Index eine verschachtelte Suche?
9. Welche Speicher-Zeit-Abwägung macht Two Sum von O(n²) zu erwartet O(n)?
10. Warum sollte eine Performance-Messung absichtlich schlechte Hashwerte
    einschließen?

Eine praktische Abschlussübung ist, für die Schlüssel `A, B, C, D` eine Tabelle
mit Kapazität 5 auf Papier zu simulieren: einmal mit Chaining, einmal mit Linear
Probing. Verwende absichtlich dieselbe Startposition, lösche B und trace danach
eine Suche nach D. Der Unterschied macht die Tombstone-Regel unmittelbar
sichtbar.

---

## Zusammenfassung

Hashing ersetzt wiederholte Suche durch berechneten Direktzugriff. Eine
Hash-Funktion verteilt Keys auf ein begrenztes Bucket-Array; Kollisionen sind
unvermeidbar und werden durch Chaining oder Open Addressing aufgelöst. Der Load
Factor verbindet Speicherverbrauch mit Kollisionskosten. Rehashing ist einzeln
linear, hält Operationen bei geometrischem Wachstum aber amortisiert konstant.

Python stellt mit `dict` und `set` hochoptimierte Hash-Strukturen bereit. Ihre
Keys müssen hashable sein: Der Hash darf sich während der Lebenszeit nicht
ändern und muss zur Gleichheitsdefinition passen. Frequency Counting,
Deduplizierung, Lookup-Tabellen, Gruppierung und Two Sum sind Varianten derselben
Idee: O(n) zusätzlichen Speicher einsetzen, um wiederholte lineare Suchen zu
vermeiden.

Das O(1)-Versprechen ist damit weder Magie noch Garantie für jede Eingabe. Es ist
das Ergebnis aus guter Hash-Funktion, kontrolliertem Load Factor, korrekter
Kollisionsstrategie und einer Average-Case-Annahme, deren Worst Case bewusst
verstanden und gemessen werden muss.
