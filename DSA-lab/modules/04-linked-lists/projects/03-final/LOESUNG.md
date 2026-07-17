# Lösung und Entwurfsbegründung

## Warum zwei Datenstrukturen nötig sind

Eine Hash Map allein findet einen Schlüssel in durchschnittlich O(1), kennt aber
keine Nutzungsreihenfolge. Eine Linked List hält die Reihenfolge, müsste einen
Schlüssel allein jedoch in O(n) suchen. Der Cache kombiniert beide Stärken:

```text
key --Hash Map--> node --previous/next--> recency order
```

Die Map speichert nicht nur Werte, sondern Referenzen auf die tatsächlichen
Listenknoten. Ein Treffer kann den gefundenen Knoten daher ohne Suche lösen und
an das MRU-Ende hängen.

## Repräsentationsinvarianten

`check_invariants()` prüft die Regeln, auf denen alle Operationen beruhen:

1. Der Sentinel hat immer einen Vorgänger und Nachfolger.
2. Vorwärts- und Rückwärtszeiger sind paarweise konsistent.
3. Die Liste schließt sich ausschließlich am Sentinel zum Kreis.
4. Jeder Listenknoten steht unter seinem Schlüssel exakt einmal in der Map.
5. Jeder Map-Knoten gehört zur Liste.
6. Listenlänge, Map-Größe und Kapazitätsgrenze stimmen überein.

Diese Prüfung selbst ist O(n) und deshalb bewusst kein Bestandteil von `get`
oder `put`. Sie ist ein Diagnosewerkzeug für Tests und Entwicklung.

## Ablauf der öffentlichen Operationen

### `get(key)`

1. Schlüssel in der Map suchen.
2. Bei Fehlen: Miss zählen und `KeyError` auslösen.
3. Bei Treffer: Hit zählen, Knoten lösen, als MRU anhängen, Wert liefern.

### `put(key, value)`

- Existiert der Schlüssel, wird sein Wert überschrieben und der Knoten MRU.
- Ist er neu, entsteht genau ein neuer Knoten an der MRU-Seite.
- Überschreitet die Größe die Kapazität, liefert `sentinel.next` sofort den
  LRU-Knoten. Er wird aus Liste und Map entfernt.

### `get_or_compute(key, loader)`

Diese Methode bildet das Cache-Aside-Muster ab. Ein Treffer liefert den
gespeicherten Wert. Bei einem Miss wird `loader` genau einmal aufgerufen und das
Ergebnis eingefügt. Schlägt der Loader fehl, bleibt kein halbfertiger Eintrag im
Cache. Auch falsy Werte wie `0`, `False` oder `None` sind reguläre Ergebnisse.

## Komplexität

| Operation | Zeit im Mittel | Zusatzspeicher |
|---|---:|---:|
| `get` | O(1) | O(1) |
| `peek` | O(1) | O(1) |
| `put` / Update | O(1) | O(1) pro neuem Eintrag |
| `delete` | O(1) | O(1) |
| Eviction | O(1) | O(1) |
| Reihenfolge als Tupel | O(n) | O(n) |
| Invariantenprüfung | O(n) | O(n) |

Die O(1)-Aussage für die Map gilt wie üblich amortisiert beziehungsweise im
Durchschnitt bei brauchbarer Hash-Verteilung.

## Messaufbau der Simulation

`generate_workload` erzeugt eine feste Sequenz aus Hot und Cold Keys. Beide
Messpfade verarbeiten exakt diese Sequenz und verwenden dieselbe deterministische
Datenquellenfunktion:

- Der ungecachte Pfad ruft die Quelle für jeden Request auf.
- Der gecachte Pfad ruft sie ausschließlich bei einem Miss auf.
- Vor der Auswertung wird die vollständige Ergebnisliste beider Pfade verglichen.

Die zentrale Konsistenzgleichung lautet:

```text
hits + misses = requests
source_calls  = misses
hit_rate      = hits / requests
```

Eine hohe Hit-Rate bedeutet, dass der Cache gut zu Zugriffsmuster und Kapazität
passt. Der Speedup hängt zusätzlich davon ab, wie teuer die Quelle gegenüber
Cache-Verwaltung und Python-Overhead ist. Deshalb prüfen Tests die
Ergebnisgleichheit und Metrikkonsistenz, aber keinen festen Mindest-Speedup.

## Grenzen und sinnvolle Erweiterungen

- Die Implementierung ist nicht threadsicher; parallele Zugriffe bräuchten eine
  Sperre um Map und Liste als gemeinsame atomare Struktur.
- Es gibt keine zeitbasierte Invalidierung (TTL).
- Größe bedeutet Anzahl der Einträge, nicht Speicherverbrauch in Bytes.
- Die CPU-Simulation modelliert keine Netzwerkvarianz oder asynchrone Abfragen.
- Produktionssysteme benötigen gegebenenfalls Metrikexport, Persistenz und eine
  Strategie gegen einen Cache Stampede.

Gerade diese Grenzen zeigen den Kern des Projekts klar: Die Kombination aus
Hash Map und Doubly Linked List löst die LRU-Ordnungsfrage ohne lineare Suche.
