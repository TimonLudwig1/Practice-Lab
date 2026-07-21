# Module 06: Hashing & Hash Maps

Hash Maps are among the most important tools in algorithms and data pipelines. They
answer questions like "Have I seen this key before?", "What value belongs to this ID?"
or "How often did this event occur?" in the Average Case in O(1). However, this promise
is not free: a hash function must map keys to a limited memory, collisions are
unavoidable, and if the distribution is unfavourable, it remains as Worst Case O(n).

This script develops the concept in three levels:

1. **Intuition:** Why direct access can be faster than linear search.
2. **Simulation:** How keys land in buckets, how collisions are treated, and why the
   load factor is crucial.
3. **Formalization:** Invariants, complexities, rehashing, Python internals and reusable
   solution patterns.

After this module, you should build your own hash map with collision handling, measure
the influence of the load factor and see when additional memory avoids a linear search.

---

## Level 1: Intuition

### 1. From Browse to Address

Imagine a warehouse with 10,000 packages. If all packages are unsorted in one room, you
must read each label in the worst case. Searching for package `DSA-4711` costs O(n).

A shelf with numbered compartments changes the question. Instead of "Where is the
package?" you ask "Which technical number results from the label?". If you can calculate
the subject number directly, you jump to the target without searching through it. This
translation is done by a **hash function**.

```text
Key --hash function--> hash value --compression--> bucket index
"DSA-4711"                    8347219                   3
```

The hash map stores a pair of keys and values in the calculated bucket. In a later
search, it recalculates the same index. So the key is not the address itself; it is
deterministically translated into a candidate address.

### 2. Why O(1) sounds almost too good

An array access to a known index costs O(1). If the calculation of the hash value is
also considered constant, the whole search seems to be constant. But three limitations
are always part of the promise:

- Several keys can get the same bucket. This **collisions** must dissolve the structure
  additionally.
- The table needs free capacity. With growing **Load Factor** collisions or long search
  paths become more likely.
- O(1) is the expected or amortised average case. If many keys force the same hash
  value, the search will degenerate to O(n).

The correct short form is therefore:

> Hash Maps offer Put, Get and Delete in the expected Average Case in O(1),
> require O(n) memory and keep O(n), as a worst case.

### 3. Hash Map and Hash Set

A hash map stores `key -> value`. A hash set requires only keys and answers the question
of membership.

| Question | Suitable structure |
|---|---|
| Which name belongs to customer number 42? | Hash Map |
| Has Event ID `evt-17` already occurred? | Hash Set |
| How many times did Status 500 occur? | Hash Map with counter values |
| Which unique IP addresses exist? | Hash Set |

A set can be conceptually regarded as a hash map whose values are not required.

---

## Level 2: Simulation

### 4. From a string to a bucket

Pythons built-in `hash()` is suitable within a process, but string hash values are
typically randomized between processes for security reasons. For a comprehensible
simulation, we therefore use a consciously simple, stable function.

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

The multiplication with 31 ensures that the order of the characters works: `"ab"` and
`"ba"` usually receive different hash values. The modulo compresses an arbitrary hash
value to the area `0 .. capacity - 1`.

A useful hash function for a hash map should:

- provide the same hash value for the same unchanged key;
- same key equal to hashen,
- distribute small differences as much as possible over many hash values,
- be fast to calculate; and
- distribute the expected inputs to Buckets as evenly as possible.

It doesn't have to be collision-free. With more possible keys than Buckets, collision
freedom is impossible according to the drawer principle. A cryptographic hash function
pursues additional security targets and is usually unnecessarily expensive for an
ordinary in-memory hash map.

### 5. Make Distribution Visible

A distribution can be counted as a bucket occupancy. Perfect equal distribution is not
to be expected for small samples; a systematic accumulation is problematic.

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

The histogram values do not yet measure search time. They show, however, where a
collision strategy has to do additional work.

### 6. Dissolve Collision with Chaining

With **Separate Chaining**, each bucket is a small container of key-value pairs.
Colliding keys are stored in the same bucket. Within the bucket, the real key is
compared, because an identical bucket index does not mean that the keys are the same.

Take capacity 5 and these already calculated indices:

| Operation | Index | Bucket condition |
|---|---:|---|
| `put("Ada", 10)` | 2 | `[('Ada', 10)]` |
| `put("Grace", 20)` | 2 | `[('Ada', 10), ('Grace', 20)]` |
| `put("Linus", 30)` | 4 | `[('Linus', 30)]` |
| `put("Ada", 99)` | 2 | `[('Ada', 99), ('Grace', 20)]` |

The last operation does not add a second `Ada` entry. It finds the same key in the
bucket and updates its value.

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

Chaining remains correct even if all keys end up in the same bucket. The runtime then
becomes linear in the length of this chain.

### 7. Dissolve collisions with linear probing

**Open Addressing** stores all entries directly in the bucket array. If the calculated
bucket is used, linear probing examines further places one after the other:

```text
index_i = (start_index + i) mod capacity
```

At capacity 7, three keys start at index 4:

| Step | Bucket 4 | Bucket 5 | Bucket 6 |
|---|---|---|---|
| `put(A)` | A | empty | empty |
| `put(B)` | A | B | empty |
| `put(C)` | A | B | C |

A search for C must not stop at Bucket 4 or 5. It follows the same sample sequence until
C or a really empty bucket is found.

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

The fourth insertion demonstrates wrap-around. After index 6, the sample sequence
continues with index 0.

#### Why Delete Needs a Tombstone

If B were simply replaced by an empty bucket, a later search for C could falsely break
off in this hole. Therefore, open-addressing tables usually use three states:

1. **never used:** Here may abort an unsuccessful search.
2. **occupied:** Key and value are stored.
3. **deleted/Tombstone:** The search must continue; an insertion may reuse the space
   later.

Linear probing is cache-friendly, but can create **Primary clustering**: Related
occupied areas grow and extend further sample sequences.

### 8. Simulate Load Factor and Rehashing

The load factor describes the filling:

```text
alpha = number_of_entrys/number_of_buckets
```

In Chaining, `alpha` may be greater than 1. Open Addressing must always have at least
one usable free space. In both cases, a high load factor typically worsens performance.

Suppose a table starts with capacity 4 and rehases before an insertion that would cause
`alpha > 0.75`:

| Entries | Capacity | Load Factor | Action |
|---:|---:|---:|---|
| 1 | 4 | 0,25 | none |
| 2 | 4 | 0,50 | none |
| 3 | 4 | 0,75 | none |
| 4 planned | 4 | 1,00 | increase to 8, new hashen all entries |
| 4 | 8 | 0,50 | Complete insertion |

When rehashing, it is not enough to copy the old buckets to the same index. The index
depends on the capacity:

```text
old_index = hash(key) mod 4
new_index = hash(key) mod 8
```

Each stored key must therefore be distributed again. A single rehashing costs O(n), but
occurs only occasionally with geometric growth. Put remains amortized O(1).

### 9. Provoking the worst case empirically

Python allows own objects with a controlled hash value. The following class forces the
same hash for all instances and makes the quick lookup a linear search through many
collisions.

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

Absolute times depend on the computer. The growth is relevant: In the normal dictionary,
the lookup time per access remains approximately stable; in the deliberately colliding
dictionary, the number of necessary equality comparisons increases with n. The
experiment shows why O(1) is not a worst-case promise.

---

## Level 3: Formalisation

### 10. Abstract definition

A map represents a finite amount of pairsn

```text
M = {(k_1, v_1, (k_2, v_2),..., (k_n, v_n)}
```

with clear keys. The core operations are:

- `put(k, v)`: Insert or replace the value of an existing key.
- `get(k)`: Deliver the value belonging to k or report absence.
- `delete(k)`: Remove the pair with key k or report absence.
- `contains(k)`: Check if k is present.

A hash table implements this abstract map with:

1. a hash function `h(k)`,
2. a compression to a bucket index; and
3. a strategy to resolve collisions.

The central condition of correctness is:

> If two keys are equal according to `==`, their hash values must be equal.

The reversal does not apply: the same hash values may belong to unequal keys. Therefore,
after the hash comparison, the structure still has to check the key equality.

### 11. Chaining formal

For a table with `m` buckets, the index is often calculated as

```text
i = h(k) mod m
```

Bucket `T[i]` contains a sequence of entries. The map invariant requires:

- Each stored key occurs exactly once.
- Each entry `(k, v)` is located in the bucket that determines the current capacity for
  k.
- The stored size value corresponds to the total number of entries.

Pseudocode for Get:

```text
GET(key):
    index <- HASH(key) mod capacity
    for (stored_key, value) in buckets[index]:
        if stored_key.
            return value
    raise KeyError
```

Put searches the same chain. If it finds the key, it replaces the value; otherwise it
will attach a new pair and increase the size. Delete removes only if found successfully
and reduces the size exactly once.

Assuming uniform distribution, the expected chain length is approximately `alpha = n /
m`. If Rehashing alpha is limited by a constant, operations are expected to remain O(1).

### 12. Open Addressing formal

Open Addressing defines a sequence of possible indices for each key:

```text
(k, 0), (k, 1),..., (k, m - 1)
```

Linear probing used:

```text
p(k, i) = (h(k) + i) mod m
```

The Sequence sample must reach any relevant place in the permitted area. Get, put and
delete must use the same sequence. A search only ends with:

- the key sought,
- a `never used` slot, or
- after a maximum of m of Buckets.

Tombstones receive the search paths, but increase the effective filling in the long
term. A rehashing removes them because only active entries are transferred to a fresh
array.

### 13. Classify Complexities

| Operation | Expected/amortised | Worst Case | Cause of the Worst Case |
|---|---:|---:|---|
| Put | O(1) | O(n) | long collisions or rehashing |
| Get | O(1) | O(n) | all relevant keys collide |
| Delete | O(1) | O(n) | long chain/sample sequence |
| Contains | O(1) | O(n) | how to get |
| Iteration | O(n) | O(n + m) internally possible | Visit Entries or Buckets |
| Memory | O(n + m) | O(n + m) | Entries plus Bucket array |

A well-dimensioned table is `m = Theta(n)`, so the memory is usually given as O(n). The
O(1) specification also treats the hash of a key as constant. For a string of length L,
its first hash calculation costs conceptual O(L); some runtime environments cache hash
values of immutable objects.

### 14. Compare Chaining and Open Addressing

| Aspect | Separate Chaining | Open Addressing |
|---|---|---|
| Location | Bucket plus external entries | Entries directly in the array |
| Load Factor | may be greater than 1 | must remain below 1 |
| Delete | Direct from Chain | Tombstone or elaborate repair |
| Cache locality | often weaker | often stronger |
| Collision costs | Length of Chain | Length of sample Sequence |
| Add-on memory | Containers/references per bucket | Status per array slot |
| Typical problem | single long chains | Clustering and Tombstones |

No strategy is universally superior. The decisive factors are memory layout, key
distribution, expected filling and the ratio of read, write and delete operations.

### 15. Python: `dict`, `set` and hashability

Python provides hash maps as `dict` and hash sets as `set`. Their concrete CPython
implementation is version-dependent, conceptually a dictionary uses a highly optimized
open-addressing structure. Since Python 3.7, the insertion order of a `dict` is part of
the language guarantee. This order makes the dictionary neither sorted nor a priority
queue.

An object is **hashable** if:

- it has a hash value that does not change during its lifetime,
- it can be compared with other objects; and
- Equality complies with the rule `a == b -> hash(a) == hash(b)`.

Typical unchangeable build-ins are hashable: numbers, strings, bytes and `frozenset`. A
tuple is only hashable if all its elements are hashable. Lists, dictionaries and sets
are variable and therefore no keys.

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

Why would a list as a key be dangerous? If `[1, 2]` were to lie in Bucket 5 and later
mutate to `[1, 2, 3]`, its hash value could change. The map would examine another bucket
during the lookup and find the stored key no longer reliable.

Also own classes need care. If you define `__eq__` professionally, you must provide a
consistent `__hash__` or leave the class deliberately unhashable. Unchangeable
`dataclass` objects can, for example, be modeled as keys with `frozen=True` if their
fields themselves are hashable.

#### Practical information on python dictionaries

- `key in mapping` checks keys, not values.
- Without Sentinel, `mapping.get(key)` does not distinguish between "miss" and "existing
  value is None".
- `set` removes duplicates, but does not guarantee sorted output.
- `hash(text)` should not be stored as a permanent database ID; its value is not
  intended as a cross-process and cross-version format.
- During iteration, the size of a dictionary or set should not be changed.

### 16. Application patterns

Hash-based solutions are often identified by one of the questions:

- Do I often have to check if something has already been seen?
- Do I need metadata for an ID quickly?
- Do I have to group elements according to a predictable characteristic?
- Can I save previous results instead of looking again?
- May I use O(n) additional memory to avoid O(n2) time?

#### 16.1 Frequency Counting

A dictionary assigns its previous counter to each value.

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

Time: expected O(n). Memory: O(k) for k different values.

#### 16.2 Deduplication with stable order

A set takes over fast membership, a list retains the first sequence of occurrences.

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

`list(set(items))` also removes duplicates, but does not express the desired order and
is not suitable if the first occurrence is to be preserved.

#### 16.3 Lookup table instead of repeated search

When linking small records, a page can be indexed once.

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

Without index, each of the n measurements could search all m sensors: O(nm). With
Dictionary, the setup is expected to cost O(m), the subsequent Join O(n), together O(n +
m) at O(m) additional memory.

#### 16.4 Two Sum as complement search

Two values with a sum of `target` are searched. Instead of testing O(n2) for each pair,
we save already seen values and ask for the complement.

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

The order is important: First search for the complement, then save the current value. So
the same element is not used twice. The pattern generalizes to pairings, differences and
already seen states.

#### 16.5 Grouping by a canonical key

For anagrams, the sorted string is a possible group key.

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

The hash map does not solve the entire task here. First, a signature that is identical
and hashable for the same groups is crucial.

### 17. Memory against time: a rule of decision

Hashing is particularly attractive when many later lookups amortize the one-time cost of
an index setup. A typical conversion is:

```text
Before: Search linearly for each element again -> O(n2), O(1) Additional memory
After: Indexing Viewed Elements       -> O(s) expected, O(n) memory
```

A hash map is not automatically the best choice:

- For very small data, a linear list can be easier and faster.
- If you need sorted order or area queries, hashing alone does not help; sorted arrays
  or trees fit better.
- With scarce memory, the additional table overhead can be too large.
- For persistent or distributed keys, a clearly defined, stable hash process is needed
  instead of the internal `hash()`.
- The worst case and collision abuse must also be taken into account in security-
  critical, externally controlled keys.

### 18. Common thinking and implementation errors

#### Confusion with equality

`hash(a) == hash(b)` does not prove `a == b`. A Bucket Fund is only a candidate; the
real key comparison remains necessary.

#### Increase the size of the update

`put(existing_key, new_value)` replaces a value. The number of entries must not change.

#### Copy old indices on rehashing

After a capacity change, each active key must be distributed with the new capacity.

#### Just empty open-addressing entries

A real hole can finish a sample sequence prematurely. Delete requires Tombstones or a
correct repair of the following clusters.

#### Measure only the average

An average short lookup can hide individual long chains. Additionally, maximum,
quantile, bucket histograms and sample lengths are helpful.

#### Use `get` with a technical value as a sentinel

```python
sentinel = object()
mapping_with_none = {"known": None}
known_value = mapping_with_none.get("known", sentinel)
missing_value = mapping_with_none.get("missing", sentinel)
assert known_value is None
assert missing_value is sentinel
```

Only a unique Sentinel reliably distinguishes between "key missing" and "key exists with
value None".

### 19. Self-control

You have understood the core ideas if you can justify these questions without
memorizing:

1. Why can a Hash Map O(1) in the Average Case, but O(n) in the Worst Case?
2. Why does Get check the key after the bucket index?
3. What invariant violates a mutable list as a key?
4. Why do entries have to be re-hashed during resize?
5. Why does Linear Probing need a Tombstone when deleting it?
6. What changes a high load factor in chain or sample lengths?
7. When is a set more suitable than a dictionary?
8. How does a lookup index replace a nested search?
9. Which memory-time balance does Two Sum expect from O(n2) O(n)?
10. Why should a performance measurement intentionally include bad hash values?

A practical final exercise is to simulate a table with capacity 5 on paper for the keys
`A, B, C, D`: once with Chaining, once with Linear Probing. Use intentionally the same
starting position, delete B and trace a search for D. The difference makes the Tombstone
rule immediately visible.

---

## Executive summary

Hashing replaces repeated search with calculated direct access. A hash function
distributes keys to a limited bucket array; collisions are unavoidable and are resolved
by chaining or open addressing. The load factor combines storage consumption with
collision costs. Rehashing is individually linear, keeping operations at geometrical
growth but amortized constant.

Python provides highly optimized hash structures with `dict` and `set`. Your keys must
be hashable: The hash must not change during the lifetime and must fit the definition of
equality. Frequency counting, deduplication, lookup tables, grouping and two sum are
variants of the same idea: Use O(n) additional memory to avoid repeated linear searches.

The O(1) promise is thus neither magic nor guarantee for any input. It is the result of
good hash function, controlled load factor, correct collision strategy and an average
case assumption whose worst case must be consciously understood and measured.

---

# Deutsche Fassung

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
Key --hash function--> hash value --compression--> bucket index
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
alpha = number_of_entrys/number_of_buckets
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
M = {(k_1, v_1, (k_2, v_2),..., (k_n, v_n)}
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
        if stored_key.
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
(k, 0), (k, 1),..., (k, m - 1)
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
Before: Search linearly for each element again -> O(n2), O(1) Additional memory
After: Indexing Viewed Elements       -> O(s) expected, O(n) memory
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
