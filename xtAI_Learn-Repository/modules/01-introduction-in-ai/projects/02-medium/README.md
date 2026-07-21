# Projekt 02 (medium) — Tic-Tac-Toe mit unschlagbarer KI (Minimax + Alpha-Beta)

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
