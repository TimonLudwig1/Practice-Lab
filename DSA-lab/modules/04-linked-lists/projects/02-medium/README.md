# 02-medium — Zeiger-Klassiker

## Ziel

Dieses Projekt trainiert vier Algorithmen, bei denen Knotenidentität und
Zeigerreihenfolge wichtiger sind als Indexarithmetik:

1. eine Kette in-place umkehren,
2. die Mitte mit Runnern finden,
3. einen Zyklus mit Floyd erkennen und vermessen,
4. zwei sortierte Ketten stabil ohne neue Datenknoten mergen.

Jeder Algorithmus kann einen optionalen Trace füllen. Die Demo verbindet diese
konkreten Schritte mit den Zeigerdiagrammen dieser Dokumentation. Python-Skripte
plus pytest eignen sich dafür, weil Tests neben Werten auch Objektidentitäten und
unveränderte Links prüfen können.

## Gemeinsames Knotenmodell

`Node` verwendet Identitätsgleichheit. `build_chain`, `chain_nodes` und
`chain_values` sind nur Aufbau-/Prüfhilfen; die Algorithmen selbst arbeiten
direkt auf `next`-Referenzen. `chain_nodes` lehnt Zyklen ab, damit ein fehlerhafter
Test nicht endlos läuft.

## 1. In-place-Reversal

```python
new_head = reverse_in_place(head, trace=trace)
```

Vor einem Schritt:

```text
previous        current         next_node
   │               │                │
   ▼               ▼                ▼
 [A] ◀── ...      [B] ───────────▶ [C] ──▶ [D]
```

Zuweisungsreihenfolge:

```text
next_node = current.next    # Rest sichern
current.next = previous     # aktuellen Link umdrehen
previous = current          # umgedrehten Prefix erweitern
current = next_node         # Rest weiterbearbeiten
```

Danach:

```text
             previous        current
                 │              │
                 ▼              ▼
 [A] ◀── ... ◀── [B]           [C] ──▶ [D]
```

Invariante: `previous` ist der Kopf des umgedrehten Prefix, `current` der Kopf
des noch unveränderten Suffix. Zeit O(n), Extra-Speicher O(1).

## 2. Mitte mit Runner-Technik

```python
middle = middle_node(head, trace=trace)
```

`slow` läuft einen, `fast` zwei Knoten pro Runde:

```text
Start:       S/F
              ▼
             [1] → [2] → [3] → [4] → [5] → [6]

Runde 1:            S     F
                    ▼     ▼
             [1] → [2] → [3] → [4] → [5] → [6]

Runde 2:                  S           F
                          ▼           ▼
             [1] → [2] → [3] → [4] → [5] → [6]

Runde 3:                        S                 F=None
                                ▼
             [1] → [2] → [3] → [4] → [5] → [6]
```

Bei gerader Länge liefert diese Variante den zweiten mittleren Knoten. Die Links
werden nicht verändert. Zeit O(n), Extra-Speicher O(1).

## 3. Floyd-Zyklenerkennung

```python
info = detect_cycle(head, trace=trace)
```

Phase 1 lässt `slow` einen und `fast` zwei Schritte laufen. Treffen beide sich,
existiert ein Zyklus:

```text
[A] → [B] → [C] → [D] → [E]
            ▲                 │
            └─────────────────┘
```

Phase 2 setzt einen Zeiger auf Head und lässt ihn gemeinsam mit dem
Treffpunktzeiger jeweils einen Schritt laufen. Ihr nächstes Treffen ist der
Zykluseintritt. Eine zusätzliche Runde um den Zyklus bestimmt seine Länge.

`CycleInfo` enthält:

- `entry`: Knotenidentität des Zykluseintritts,
- `cycle_length`: Zahl der Knoten im Kreis,
- `prefix_length`: Zahl der Knoten vor dem Eintritt.

Werte sind irrelevant; verglichen wird ausschließlich mit `is`. Zeit O(n),
Extra-Speicher O(1). `has_cycle` bietet die einfache boolesche Schnittstelle.

## 4. Stabiler In-place-Merge

```python
merged = merge_sorted(left, right, trace=trace)
```

Ein lokaler Dummy hält das Ergebnisende. Der kleinere aktuelle Knoten wird dort
angehängt und sein Quellzeiger weitergeschoben:

```text
left:   [1] → [4] → [7]
right:  [2] → [3] → [8]

dummy → [1] → [2] → [3] → [4] → [7] → [8]
```

Bei gleichen Werten wird links gewählt; der Merge ist dadurch stabil. Der Dummy
ist kein Datenknoten des Ergebnisses. Alle ursprünglichen Knoten werden genau
einmal wiederverlinkt, neue Nutzknoten werden nicht erzeugt.

Vorbedingungen: Beide Eingaben sind aufsteigend sortiert, azyklisch und
knotendisjunkt. Zeit O(n + m), Extra-Speicher O(1).

## Ausführen

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -v
python3 pointer_demo.py
```

## Fertig, wenn …

- Reversal leere, einzelne und längere Ketten ohne neue Knoten verarbeitet,
- der alte Head nach Reversal korrekt terminal ist,
- die Runner-Methode für gerade Länge den zweiten mittleren Knoten liefert,
- Floyd keinen Zusatzcontainer verwendet und Eintritt sowie Längen korrekt sind,
- gleiche Werte nicht mit gleicher Knotenidentität verwechselt werden,
- Merge sortiert, stabil und vollständig in-place arbeitet,
- alle vier Traces die entscheidenden Zeigerbewegungen dokumentieren,
- pytest-Suite und vollständige Pointer-Demo fehlerfrei laufen,
- du jedes Verfahren anhand seines Diagramms ohne Versuch-und-Irrtum erklären
  kannst.
