# Project 01 (basic) — Pathfinding in a maze: from BFS to A\*

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format:** Jupyter notebook (`astar_pathfinding.ipynb`).
**Why this format?** Search algorithms are understood best when you can *see* which cells the algorithm touches — the notebook interleaves code, explanation and the visualisation of the expanded nodes directly.

**Data:** no external data needed — the maze is defined as a small, hand-built ASCII grid right inside the notebook (deliberately chosen: you can rebuild it yourself and immediately see what happens).

## Goal

You implement breadth-first search (BFS) and A\* with the Manhattan heuristic for a grid maze and compare the two: the same (optimal) path, but noticeably fewer expanded nodes for A\*. At the end you see experimentally that an *overestimating* heuristic is faster still, but destroys optimality.

## Prior knowledge

- Section 1.3 of the module script (search) read
- Python basics: functions, lists, dictionaries, loops
- New and explained in the notebook: `collections.deque` (queue) and `heapq` (priority queue)

## Tasks (step by step)

1. Open the notebook: `jupyter lab`, then navigate to this folder (for setup see `SETUP.md` in the repository root).
2. **Cell "neighbouring cells"**: implement `neighbours(pos)` — the 4 walkable neighbouring positions. The mini tests below it must print `True`.
3. **Cell "BFS"**: complete the search loop (take an element from the queue, goal test, enter unvisited neighbours).
4. Run the visualisation and look at *how* BFS spreads out.
5. **Cell "A\*"**: implement the Manhattan heuristic and the neighbour update with `f = g + h`.
6. Run the heuristic comparison at the end and explain the three rows of the result table to yourself.

## What should work in the end

- Both mini tests in step 2 print `True`.
- BFS finds a path with **29 steps** and expands **54 nodes**.
- A\* also finds 29 steps, but expands only **35 nodes**.
- The greedy variant (5x Manhattan) expands 32 nodes but finds a **31**-step path — it is no longer optimal.

## Solution

Complete, executed reference solution: [`solution/solution.ipynb`](solution/solution.ipynb) — please look at it only after your own attempt. The TODO spots are solved there in a few lines each; if your solution looks different but the numbers match, that is entirely fine.

---
---

# Projekt 01 (basic) — Wegsuche im Labyrinth: von BFS zu A\* (deutsche Fassung)

**Format:** Jupyter Notebook (`astar_pathfinding.ipynb`).
**Warum dieses Format?** Suchalgorithmen versteht man am besten, wenn man *sieht*, welche Felder der Algorithmus anfasst — das Notebook verzahnt Code, Erklärung und die Visualisierung der expandierten Knoten direkt miteinander.

**Daten:** Keine externen Daten nötig — das Labyrinth ist als kleines, handgebautes ASCII-Gitter direkt im Notebook definiert (bewusst so gewählt: du kannst es selbst umbauen und sofort sehen, was passiert).

## Ziel

Du implementierst Breitensuche (BFS) und A\* mit Manhattan-Heuristik für ein Gitter-Labyrinth und vergleichst beide: gleicher (optimaler) Pfad, aber deutlich weniger expandierte Knoten bei A\*. Zum Schluss siehst du experimentell, dass eine *überschätzende* Heuristik zwar noch schneller ist, aber die Optimalität zerstört.

## Vorwissen

- Modul-Skript Abschnitt 1.3 (Suche) gelesen
- Python-Basics: Funktionen, Listen, Dictionaries, Schleifen
- Neu und im Notebook erklärt: `collections.deque` (Warteschlange) und `heapq` (Prioritätswarteschlange)

## Aufgaben (Schritt für Schritt)

1. Öffne das Notebook: `jupyter lab`, dann zu diesem Ordner navigieren (Setup siehe `SETUP.md` im Repo-Wurzelverzeichnis).
2. **Zelle „Nachbarfelder"**: Implementiere `nachbarn(pos)` — die 4 begehbaren Nachbarpositionen. Die Mini-Tests darunter müssen `True` ausgeben.
3. **Zelle „BFS"**: Vervollständige die Suchschleife (Element aus der Queue nehmen, Zieltest, unbesuchte Nachbarn eintragen).
4. Führe die Visualisierung aus und schau dir an, *wie* BFS sich ausbreitet.
5. **Zelle „A\*"**: Implementiere die Manhattan-Heuristik und die Nachbar-Aktualisierung mit `f = g + h`.
6. Führe den Heuristik-Vergleich am Ende aus und erkläre dir selbst die drei Zeilen der Ergebnistabelle.

## Was am Ende funktionieren soll

- Beide Mini-Tests in Schritt 2 geben `True` aus.
- BFS findet einen Pfad mit **29 Schritten** und expandiert **54 Knoten**.
- A\* findet ebenfalls 29 Schritte, expandiert aber nur **35 Knoten**.
- Die gierige Variante (5×Manhattan) expandiert 32 Knoten, findet aber einen **31**-Schritte-Pfad — sie ist nicht mehr optimal.

## Lösung

Komplette, ausgeführte Musterlösung: [`solution/solution.ipynb`](solution/solution.ipynb) — bitte erst nach eigenem Versuch anschauen. Die TODO-Stellen sind dort jeweils mit wenigen Zeilen gelöst; wenn deine Lösung anders aussieht, aber die Zahlen stimmen, ist das völlig in Ordnung.
