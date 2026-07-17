# Projekt 01-basic: Stack & Queue selbst bauen

## Ziel

In diesem Projekt entstehen zwei lineare Datenstrukturen ohne fertige Container
aus `collections`: ein arraybasierter `Stack` und eine `Queue` als Ringpuffer
mit fixer Kapazität. Im Mittelpunkt stehen LIFO und FIFO, konstante Laufzeiten
sowie die Zustände, in denen eine Operation nicht möglich ist.

Das Projekt ist als Python-Skript mit pytest-Suite angelegt. Dieses Format macht
die Zustandsübergänge im Terminal sichtbar und trennt Implementierung, Demo und
automatische Prüfung klar voneinander.

## Aufgabenstellung

1. Implementiere in `stack_queue.py` einen generischen `Stack` auf Basis einer
   Python-Liste. `push`, `pop` und `peek` arbeiten ausschließlich am Listenende.
2. Behandle `pop` und `peek` auf einem leeren Stack mit einer eigenen
   `StackUnderflowError`-Exception.
3. Implementiere eine `FixedCapacityQueue` auf einem einmalig angelegten Puffer.
   Elemente dürfen beim Entfernen nicht verschoben werden. Nutze stattdessen
   einen Head-Index, die aktuelle Größe und Modulo-Arithmetik.
4. Behandle das Lesen aus einer leeren Queue als Underflow und das Einfügen in
   eine volle Queue als Overflow. Ein fehlgeschlagener Aufruf darf den Zustand
   nicht verändern.
5. Weise mit Tests nach, dass LIFO, FIFO, Kapazitätsgrenzen und mehrere
   Wrap-around-Zyklen korrekt funktionieren.

Eine vollständige Referenzimplementierung liegt bei. Für den größten Lerneffekt
sollten zuerst nur README und Tests gelesen und die Implementierung anschließend
selbst rekonstruiert werden.

## Ringpuffer von Hand simulieren

Für Kapazität 4 gilt für die nächste freie Position:

```text
tail = (head + size) % capacity
```

Nach `enqueue(A), enqueue(B), enqueue(C), dequeue(), enqueue(D), enqueue(E)`
liegt der physische Puffer möglicherweise als `[E, B, C, D]` vor. Die logische
FIFO-Reihenfolge ist trotzdem `[B, C, D, E]`, weil `head` auf Index 1 zeigt.
`to_list()` zeigt immer diese logische Reihenfolge.

## Invarianten und Laufzeiten

Während jeder Queue-Operation gelten:

- `0 <= size <= capacity`
- `0 <= head < capacity`
- bei einer nicht vollen Queue zeigt `(head + size) % capacity` auf den nächsten
  freien Slot
- die logischen Elemente liegen, zyklisch gelesen, in FIFO-Reihenfolge ab `head`

| Struktur | Operation | Laufzeit | Begründung |
|---|---|---:|---|
| Stack | `push` | amortisiert O(1) | Append am Ende; seltenes Array-Resize |
| Stack | `pop`, `peek` | O(1) | Zugriff am Ende |
| Queue | `enqueue` | O(1) | Indexberechnung und eine Zuweisung |
| Queue | `dequeue`, `peek` | O(1) | Zugriff am Head, kein Verschieben |
| beide | `to_list` | O(n) | bewusste Kopie für Beobachtung/Debugging |

## Ausführen

Im Projektordner:

```bash
python demo.py
python -m pytest -q
```

Die Demo zeigt erst LIFO, danach FIFO, Fehlerfälle und die Wiederverwendung eines
bereits freigewordenen Pufferslots.

## Hinweise

- Eine Queue auf einer Python-Liste mit `pop(0)` ist kein Ringpuffer: Jedes
  Entfernen verschiebt alle übrigen Elemente und kostet O(n).
- Nutze nicht den Wert `None`, um einen freien Slot fachlich zu erkennen. `None`
  darf selbst ein reguläres Element der Datenstruktur sein.
- Setze einen ausgelesenen Pufferslot wieder auf einen internen Sentinel. So hält
  der Puffer keine unnötigen Referenzen auf entfernte Objekte.
- Prüfe Grenzen vor jeder Mutation. Dadurch bleiben die Strukturen auch nach
  einer Exception unverändert.

## Fertig, wenn …

- der Stack bei allen Eingaben nach LIFO arbeitet,
- `peek` bei beiden Strukturen den Zustand nicht verändert,
- die Queue bei voller Kapazität weder überschreibt noch wächst,
- Underflow und Overflow mit den vorgesehenen Exception-Typen gemeldet werden,
- die FIFO-Reihenfolge über mehrere Wrap-around-Zyklen erhalten bleibt,
- auch `None`, Kapazität 1 und wiederholtes Leeren korrekt funktionieren,
- `python demo.py` ohne Fehler läuft und
- alle Tests mit `python -m pytest -q` erfolgreich sind.
