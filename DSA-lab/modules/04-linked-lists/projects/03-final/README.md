# Projekt 03-final: LRU-Cache aus Hash Map und Doubly Linked List

Dieses Projekt implementiert einen **Least-Recently-Used-Cache ohne
`collections.OrderedDict`**. Eine Hash Map ermöglicht den direkten Zugriff auf
einen Eintrag. Eine selbst gebaute, zirkuläre Doubly Linked List speichert seine
Nutzungsreihenfolge. `get` und `put` benötigen dadurch im Mittel O(1).

Anschließend dient der Cache als Funktions-Cache für eine absichtlich teure,
deterministische Datenabfrage. Ein reproduzierbarer Workload mit häufigen und
seltenen Schlüsseln macht Hit-Rate, vermiedene Datenquellenaufrufe und Laufzeit
messbar.

## Architektur

```text
Hash Map
  "B" ───────────────┐
  "C" ────────┐      │
  "A" ──┐     │      │
         v     v      v
      [ A ] <-> [ C ] <-> [ B ]
        ^                     ^
       LRU                   MRU
        ^                     v
        └────── [ SENTINEL ] ─┘
```

Der Sentinel besitzt keinen Nutzereintrag. Bei einer leeren Liste zeigen sein
`next` und `previous` auf ihn selbst. Dadurch benötigen Einfügen und Entfernen
keine Sonderbehandlung für Kopf, Ende oder leere Liste. Ein echter Schlüssel
oder Wert darf deshalb sogar `None` sein: Der Sentinel wird über seine Identität
und nicht über seinen Inhalt erkannt.

## Zeigeroperationen

Ein Knoten `node` wird mit vier Zuweisungen aus seiner aktuellen Position
gelöst:

```text
before <-> node <-> after

before.next   = after
after.previous = before
node.previous = None
node.next     = None
```

Das Anhängen an die MRU-Seite erfolgt ebenfalls mit vier lokalen Änderungen:

```text
old_mru <-> SENTINEL

node.previous     = old_mru
node.next         = SENTINEL
old_mru.next      = node
SENTINEL.previous = node
```

- **Treffer:** Map-Lookup, Knoten lösen, an MRU anhängen.
- **Update:** Map-Lookup, Wert ersetzen, an MRU verschieben.
- **Insert:** Knoten an MRU anhängen und in der Map registrieren.
- **Eviction:** `sentinel.next` ist direkt der LRU-Knoten; lösen und aus der Map
  entfernen.

Jede Operation verändert unabhängig von der Cache-Größe nur konstant viele
Zeiger und Map-Einträge. Hash-Map-Zugriffe sind dabei im Mittel O(1).

## Dateien

- `lru_cache.py`: generischer LRU-Cache, Statistiken und Invariantenprüfung
- `query_simulation.py`: reproduzierbarer Workload, Datenquelle, Messung und CSV
- `test_lru_cache.py`: Unit-, Randfall-, Integrations- und Referenzmodelltests
- `LOESUNG.md`: Entwurfsentscheidungen und Auswertungshilfe

Das Format **Python-Skripte + pytest + CSV** passt zum Lernziel: Die
Zeigermanipulation bleibt direkt im Code sichtbar, Tests sichern strukturelle
Invarianten ab und die CSV macht den Anwendungseffekt weiterverarbeitbar.

## Ausführen

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -v
python3 query_simulation.py
```

Die Standardsimulation verwendet 1.000 Requests, 100 mögliche Schlüssel, zehn
Hot Keys, 80 Prozent Hot-Key-Wahrscheinlichkeit und eine Kapazität von 20. Die
Zufallsfolge ist durch den Seed reproduzierbar. Laufzeiten bleiben naturgemäß
maschinenabhängig.

Eigene Parameter sind über die Kommandozeile möglich:

```bash
python3 query_simulation.py \
  --requests 2000 --key-space 200 --hot-keys 20 \
  --hot-probability 0.85 --capacity 30 --rounds 5000
```

Die Ergebnisdatei entsteht standardmäßig unter
`results/cache_simulation.csv`. Generierte CSV-Dateien werden nicht committet.

## Fertig, wenn …

- `get` und `put` im Mittel O(1) benötigen,
- Zugriffe einen Eintrag korrekt auf die MRU-Seite verschieben,
- Überkapazität immer den LRU-Eintrag verdrängt,
- Map und Liste nach jeder Operation dieselben Knoten enthalten,
- die gecachte und ungecachte Simulation identische Werte liefert,
- `hits + misses == requests` und `source_calls == misses` gilt,
- alle Tests sowie die vollständige Standardsimulation erfolgreich laufen.
