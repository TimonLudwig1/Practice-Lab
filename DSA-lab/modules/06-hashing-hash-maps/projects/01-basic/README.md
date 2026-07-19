# Projekt 01-basic: Hash Map von Grund auf

## Ziel

Dieses Projekt implementiert eine generische Hash Map mit Separate Chaining,
ohne intern `dict` oder `set` zu verwenden. Put, Get und Delete werden um
automatisches Rehashing ergänzt: Würde eine neue Einfügung den Load Factor 0,75
überschreiten, verdoppelt sich das Bucket-Array und alle Einträge werden neu
verteilt.

Die Lösung besteht aus Python-Skripten, pytest und einer CSV-Messreihe. Dieses
Format trennt Datenstruktur, nachvollziehbare Demo und reproduzierbares
Performance-Experiment. Ein Notebook wäre für Einzelmessungen bequem, würde die
testbare API aber unnötig mit Darstellung vermischen.

## Aufgabenstellung

1. Lege ein Array aus Buckets an. Jeder Bucket ist eine Liste von Einträgen.
2. Berechne den Bucket mit `hash(key) % capacity`.
3. Implementiere `put`, `get` und `delete`. Ein Update ersetzt den Wert, ohne die
   Größe zu verändern.
4. Löse Kollisionen durch lineare Suche innerhalb der betroffenen Chain.
5. Verdopple die Kapazität, sobald eine neue Einfügung den Load Factor 0,75
   überschreiten würde. Verteile jeden aktiven Eintrag mit der neuen Kapazität.
6. Instrumentiere Anzahl Kollisionen, maximale Chain-Länge und Rehashes.
7. Vergleiche bei wachsendem `n` eine normal rehashende Map mit einer
   absichtlich auf 16 Buckets fixierten Variante.

## Zustands-Simulation

Bei Kapazität 4 und einer Hash-Funktion, die alle Keys nach Bucket 1 schickt:

| Operation | Größe | Kapazität | Load Factor | Chain 1 |
|---|---:|---:|---:|---|
| Start | 0 | 4 | 0,00 | `[]` |
| `put(Ada, 10)` | 1 | 4 | 0,25 | `[Ada]` |
| `put(Grace, 20)` | 2 | 4 | 0,50 | `[Ada, Grace]` |
| `put(Linus, 30)` | 3 | 4 | 0,75 | `[Ada, Grace, Linus]` |
| `put(Edsger, 40)` | 4 | 8 | 0,50 | alle vier Keys neu verteilt |

Die absichtlich schlechte Hash-Funktion erzeugt auch nach dem Resize dieselbe
Chain. Das Resize kontrolliert die durchschnittliche Füllung, kann eine schlechte
Hash-Verteilung aber nicht reparieren.

## Invarianten und Komplexität

Nach jeder Operation gelten:

- Jeder Key kommt höchstens einmal vor.
- Jeder Eintrag liegt im Bucket `hash(key) % capacity`.
- `size` entspricht der Summe aller Chain-Längen.
- Ein Update verändert `size` nicht.
- Bei aktiviertem Rehashing gilt nach einer abgeschlossenen Einfügung
  `load_factor <= 0.75`.

| Operation | Erwartet/amortisiert | Worst Case |
|---|---:|---:|
| Put | O(1) | O(n) |
| Get | O(1) | O(n) |
| Delete | O(1) | O(n) |
| Rehashing | selten O(n) | O(n) |
| Speicher | O(n + capacity) | O(n + capacity) |

Der Worst Case entsteht, wenn alle Keys in derselben Chain liegen. Geometrisches
Wachstum verteilt die Kosten seltener Rehashes über viele Einfügungen, sodass Put
amortisiert O(1) bleibt.

## Ausführen

Im Projektordner:

```bash
python demo.py
python benchmark.py
python -m pytest -q
```

`benchmark.py` schreibt `output/benchmark.csv`. Absolute Nanosekundenwerte hängen
vom Rechner ab. Verglichen werden deshalb vor allem Load Factor, Chain-Länge und
das Wachstum der Lookup-Kosten.

## Hinweise

- Suche beim Put zuerst nach einem vorhandenen Key. Sonst kann ein reines Update
  unnötig ein Resize auslösen.
- Rehashing bedeutet Neuverteilung, nicht Kopieren an denselben Bucket-Index.
- Python liefert auch negative Hashwerte; Modulo erzeugt trotzdem einen gültigen
  nichtnegativen Listenindex.
- Ein gespeicherter Wert `None` ist nicht dasselbe wie ein fehlender Key.
- Eine fixe Chaining-Tabelle darf einen Load Factor über 1 besitzen; ihre Chains
  werden dann entsprechend länger.

## Fertig, wenn …

- Put, Get, Update und Delete für kollidierende Keys korrekt funktionieren,
- fehlende Keys eine `KeyError` auslösen,
- die vierte Einfügung bei Kapazität 4 und Grenze 0,75 auf 8 Buckets rehasht,
- mehrere Resizes keinen Eintrag verlieren,
- Größen-, Bucket- und Eindeutigkeitsinvarianten geprüft werden können,
- die Messreihe beide Strategien für dieselben wachsenden Eingaben vergleicht,
- Demo und Benchmark vollständig laufen und
- alle Tests mit `python -m pytest -q` erfolgreich sind.
