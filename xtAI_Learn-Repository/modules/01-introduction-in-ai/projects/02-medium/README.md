# Project 02 (medium) — Tic-tac-toe with an unbeatable AI (minimax + alpha-beta)

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The identifiers named here are the ones currently in `tictactoe.py`; the Python sources are still German and get translated in a later pass.

**Format:** Python script project (`tictactoe.py` + `test_tictactoe.py`).
**Why this format?** This is about a recursive algorithm and an interactive game — both belong in proper `.py` files with tests, not in a notebook. Along the way you learn the normal workflow "write code, run the test script", which you will need in all later modules.

**Data:** none — tic-tac-toe is fully defined by its rules.

## Goal

You implement the **minimax** algorithm with **alpha-beta pruning** and use it to build an AI that never loses. The built-in node counter shows you that pruning does not change the result but visits only about 6 % of the nodes.

## Prior knowledge

- Section 2.1 of the module script (games: minimax, alpha-beta)
- Project 01-basic (where you have already seen the "search in a state space" pattern)
- **Recursion** — minimax calls itself. If that still feels shaky: write a small recursion example yourself first (factorial, Fibonacci).

## Tasks

Deliberately less is given than in the basic project — the game mechanics (display, game loop) are in place, the AI is missing. Three functions in `tictactoe.py` are marked with `TODO`:

1. **`gewinner(brett)`** — determine the outcome of the game (`"X"`, `"O"`, `"unentschieden"` or `None`). Start here; the first five tests check only this function.
2. **`minimax(brett, am_zug, max_spieler, ...)`** — the core. The procedure is given as a four-point plan in the docstring. Important details:
   - Set moves **in place** and undo them afterwards (`brett[feld] = am_zug` ... `brett[feld] = LEER`) — this saves copying the board.
   - Scoring with depth: `10 - tiefe` for a win, `tiefe - 10` for a loss. This makes the AI win *quickly* and lose *slowly* instead of playing indifferently.
   - The alpha-beta cut-off is in the end exactly **one** `if` line with `break`.
3. **`beste_aktion(brett, spieler)`** — tries all moves and takes the one with the highest minimax value.

To check (in the project folder, venv active):

```bash
python test_tictactoe.py
```

If everything is green, play against your AI:

```bash
python tictactoe.py
```

## What should work in the end

- All tests in `test_tictactoe.py` pass, in particular:
  - AI against AI **always ends in a draw** (that is the mathematically correct outcome with perfect play),
  - alpha-beta visits **under 25 %** of the nodes of plain minimax (reference solution: about 6 %, 34,202 instead of 549,945 nodes for the first move).
- You can play against the AI and **cannot win** (a draw at best).

## Questions to think about (after the tests pass)

1. Why does `beste_aktion` try all 9 cells on the very first move, although by symmetry only 3 are really different (corner, edge, centre)? How could this be exploited?
2. The node savings from pruning depend on the **move ordering**. Experiment: sort `freie_felder` so that cell 4 (the centre) is tried first — does the node counter get smaller?
3. Why is this approach not directly usable for chess, and which two ways out does the script name?

## Solution

Complete reference solution: [`solution/tictactoe_solution.py`](solution/tictactoe_solution.py) — only after your own attempt! You can also run the tests against the solution:

```bash
TTT_MODUL=tictactoe_solution python test_tictactoe.py
```

---
---

# Projekt 02 (medium) — Tic-Tac-Toe mit unschlagbarer KI (Minimax + Alpha-Beta) (deutsche Fassung)

**Format:** Python-Skript-Projekt (`tictactoe.py` + `test_tictactoe.py`).
**Warum dieses Format?** Hier geht es um einen rekursiven Algorithmus und ein interaktives Spiel — beides gehört in richtige `.py`-Dateien mit Tests, nicht in ein Notebook. Nebenbei lernst du den normalen Workflow „Code schreiben → Testskript laufen lassen", den du in allen späteren Modulen brauchst.

**Daten:** Keine — Tic-Tac-Toe ist vollständig durch seine Regeln definiert.

## Ziel

Du implementierst den **Minimax**-Algorithmus mit **Alpha-Beta-Pruning** und baust damit eine KI, die nie verliert. Der eingebaute Knotenzähler zeigt dir, dass Pruning das Ergebnis nicht verändert, aber nur ~6 % der Knoten besucht.

## Vorwissen

- Modul-Skript Abschnitt 2.1 (Spiele: Minimax, Alpha-Beta)
- Projekt 01-basic (dort hast du das Muster „Suche im Zustandsraum" schon gesehen)
- **Rekursion** — Minimax ruft sich selbst auf. Wenn dir das noch wackelig vorkommt: erst ein kleines Rekursionsbeispiel (Fakultät, Fibonacci) selbst schreiben.

## Aufgaben

Es ist bewusst weniger vorgegeben als im Basic-Projekt — die Spielmechanik (Anzeige, Spielschleife) steht, die KI fehlt. In `tictactoe.py` sind drei Funktionen mit `TODO` markiert:

1. **`gewinner(brett)`** — Spielausgang bestimmen (`"X"`, `"O"`, `"unentschieden"` oder `None`). Fang hiermit an; die ersten fünf Tests prüfen nur diese Funktion.
2. **`minimax(brett, am_zug, max_spieler, ...)`** — der Kern. Das Vorgehen steht als 4-Punkte-Plan im Docstring. Wichtige Details:
   - Züge **in-place** setzen und danach zurücknehmen (`brett[feld] = am_zug` … `brett[feld] = LEER`) — das spart das Kopieren des Bretts.
   - Bewertung mit Tiefe: `10 - tiefe` für Sieg, `tiefe - 10` für Niederlage. So gewinnt die KI *schnell* und verliert *langsam* statt gleichgültig zu spielen.
   - Der Alpha-Beta-Schnitt ist am Ende genau **eine** `if`-Zeile mit `break`.
3. **`beste_aktion(brett, spieler)`** — probiert alle Züge und nimmt den mit dem höchsten Minimax-Wert.

Prüfen (im Projektordner, venv aktiv):

```bash
python test_tictactoe.py
```

Wenn alles grün ist, spiel gegen deine KI:

```bash
python tictactoe.py
```

## Was am Ende funktionieren soll

- Alle Tests in `test_tictactoe.py` bestanden, insbesondere:
  - KI gegen KI endet **immer unentschieden** (das ist der mathematisch korrekte Ausgang bei perfektem Spiel),
  - Alpha-Beta besucht **unter 25 %** der Knoten von reinem Minimax (Referenzlösung: ~6 %, 34.202 statt 549.945 Knoten für den ersten Zug).
- Du kannst gegen die KI spielen und **nicht gewinnen** (bestenfalls Remis).

## Denkfragen (nach dem Bestehen der Tests)

1. Warum probiert `beste_aktion` beim allerersten Zug 9 Felder durch, obwohl aus Symmetriegründen nur 3 wirklich verschieden sind (Ecke, Kante, Mitte)? Wie könnte man das ausnutzen?
2. Die Knotenersparnis durch Pruning hängt von der **Zugreihenfolge** ab. Experimentiere: sortiere `freie_felder` so, dass Feld 4 (Mitte) zuerst probiert wird — wird der Knotenzähler kleiner?
3. Warum ist dieser Ansatz für Schach nicht direkt verwendbar, und welche zwei Auswege nennt das Skript?

## Lösung

Vollständige Musterlösung: [`solution/tictactoe_solution.py`](solution/tictactoe_solution.py) — erst nach eigenem Versuch! Du kannst die Tests auch gegen die Lösung laufen lassen:

```bash
TTT_MODUL=tictactoe_solution python test_tictactoe.py
```
