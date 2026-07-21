# 03-final — Routing on a Synthetic Road Network

This final project builds a reproducible weighted grid road network, computes
shortest routes with a custom Dijkstra implementation, measures the impact of
road closures, and visualizes the resulting detours.

Matplotlib is used only for presentation. The graph representation, closure
filtering, priority-queue search, distance updates, and path reconstruction are
implemented in this project without NetworkX routing functions.

## Network model

- Grid: 12 rows × 16 columns
- Nodes: 192 intersections
- Roads: 356 undirected orthogonal connections
- Seed: `1313`
- Edge weights: synthetic positive travel times from 1.0 to 6.0
- Start: south-west corner `r11c0`
- Target: south-east corner `r11c15`

Every horizontal and vertical neighboring pair is connected. The seed controls
traffic-like travel-time variation while preserving the same topology.

## Routing algorithm

`shortest_route` is a from-scratch Dijkstra implementation:

1. initialize the start distance to zero and all others to infinity,
2. repeatedly settle the node with the smallest current distance,
3. skip roads and nodes marked as closed,
4. relax every available outgoing road,
5. stop when the target is settled,
6. reconstruct the route through predecessor pointers.

The function returns the path, total travel time, hop count, and number of
settled nodes. Unreachable targets are represented explicitly by `path=None` and
infinite travel time.

## Closure scenarios

### 1. Baseline

All roads are open. This establishes the optimal travel time used for every
delay comparison.

### 2. Single route road

The middle edge of the baseline shortest path is closed. This measures local
resilience: the new route usually leaves the old path briefly and rejoins it
nearby.

### 3. North-gap barrier

All east-west crossings at the center column are closed except the crossing in
the northern row. The road network remains connected, but a south-side trip must
make a structural detour to the only remaining gap.

## Visualization

`route_closures.png` uses three aligned panels with a shared spatial scale:

- thin gray lines: open roads,
- thick blue line: shortest route,
- dashed red lines with crosses: closures,
- green circle: start,
- purple square: target.

Each panel directly labels travel time and percentage delay. This makes the
local single-road detour and the network-wide barrier effect visually comparable.

## Outputs

- `results/road_network.csv`: every generated road and its travel time
- `results/route_scenarios.csv`: route, cost, delay, closures, and search effort
- `results/REPORT.md`: concise English analysis
- `results/route_closures.png`: three-panel route visualization

## Complexity

With an adjacency list and binary heap, routing takes `O((V + E) log V)` time
and `O(V + E)` total storage. Closed-road membership uses a hash set, so each
availability check is expected `O(1)`.

## Run

From this project directory:

```bash
python3 run_routing.py
python3 -m pytest -q
```

---

# Deutsch — 03-final: Routing auf einem synthetischen Straßennetz

Dieses Abschlussprojekt erzeugt ein reproduzierbares gewichtetes
Gitterstraßennetz, berechnet kürzeste Routen mit einer eigenen
Dijkstra-Implementierung, misst die Auswirkungen von Straßensperrungen und
visualisiert die entstehenden Umwege.

Matplotlib dient ausschließlich der Darstellung. Graphrepräsentation,
Sperrungsfilterung, Priority-Queue-Suche, Distanzaktualisierung und
Pfadrekonstruktion werden in diesem Projekt ohne Routingfunktionen von NetworkX
implementiert.

## Netzmodell

- Gitter: 12 Zeilen × 16 Spalten
- Knoten: 192 Kreuzungen
- Straßen: 356 ungerichtete orthogonale Verbindungen
- Seed: `1313`
- Kantengewichte: synthetische positive Fahrzeiten von 1,0 bis 6,0
- Start: südwestliche Ecke `r11c0`
- Ziel: südöstliche Ecke `r11c15`

Jedes horizontal oder vertikal benachbarte Knotenpaar ist verbunden. Der Seed
steuert verkehrsähnliche Schwankungen der Fahrzeiten bei unveränderter Topologie.

## Routingalgorithmus

`shortest_route` ist eine selbst implementierte Dijkstra-Variante:

1. Startdistanz auf null und alle anderen Distanzen auf unendlich setzen,
2. wiederholt den Knoten mit der kleinsten aktuellen Distanz fixieren,
3. als gesperrt markierte Straßen und Knoten überspringen,
4. jede verfügbare ausgehende Straße relaxieren,
5. stoppen, sobald das Ziel fixiert ist,
6. die Route über Vorgängerzeiger rekonstruieren.

Die Funktion liefert Pfad, Gesamtfahrzeit, Kantenzahl und Anzahl fixierter
Knoten. Unerreichbare Ziele werden explizit durch `path=None` und unendliche
Fahrzeit dargestellt.

## Sperrungsszenarien

### 1. Basisfall

Alle Straßen sind geöffnet. Dieser Fall bestimmt die optimale Fahrzeit als
Referenz für alle Verzögerungsvergleiche.

### 2. Einzelne Routenstraße

Die mittlere Kante der ursprünglichen kürzesten Route wird gesperrt. Das misst
lokale Robustheit: Die neue Route verlässt den alten Pfad meist kurz und trifft
in der Nähe wieder auf ihn.

### 3. Barriere mit nördlicher Lücke

Alle Ost-West-Übergänge an der mittleren Spalte werden bis auf den Übergang in
der nördlichen Zeile gesperrt. Das Straßennetz bleibt zusammenhängend, eine Fahrt
auf der Südseite muss aber einen strukturellen Umweg zur einzigen Lücke nehmen.

## Visualisierung

`route_closures.png` verwendet drei ausgerichtete Teilabbildungen mit gemeinsamem
räumlichem Maßstab:

- dünne graue Linien: offene Straßen,
- dicke blaue Linie: kürzeste Route,
- gestrichelte rote Linien mit Kreuzen: Sperrungen,
- grüner Kreis: Start,
- violettes Quadrat: Ziel.

Jede Teilabbildung beschriftet Fahrzeit und prozentuale Verzögerung direkt. So
lassen sich der lokale Umweg durch eine einzelne Sperrung und der netzweite
Barriereeffekt visuell vergleichen.

## Ausgaben

- `results/road_network.csv`: jede erzeugte Straße und ihre Fahrzeit
- `results/route_scenarios.csv`: Route, Kosten, Verzögerung, Sperrungen und Suchaufwand
- `results/REPORT.md`: kompakte englische Analyse
- `results/route_closures.png`: dreiteilige Routenvisualisierung

## Komplexität

Mit Adjazenzliste und binärem Heap benötigt das Routing `O((V + E) log V)` Zeit
und insgesamt `O(V + E)` Speicher. Die Zugehörigkeitsprüfung gesperrter Straßen
verwendet eine Hash-Menge und kostet daher erwartet `O(1)`.

## Ausführen

In diesem Projektordner:

```bash
python3 run_routing.py
python3 -m pytest -q
```
