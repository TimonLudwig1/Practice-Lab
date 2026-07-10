# Projekt 01 (basic) — Wegsuche im Labyrinth: von BFS zu A\*

**Format:** Jupyter Notebook (`astar_wegsuche.ipynb`).
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

Komplette, ausgeführte Musterlösung: [`loesung/loesung.ipynb`](loesung/loesung.ipynb) — bitte erst nach eigenem Versuch anschauen. Die TODO-Stellen sind dort jeweils mit wenigen Zeilen gelöst; wenn deine Lösung anders aussieht, aber die Zahlen stimmen, ist das völlig in Ordnung.
