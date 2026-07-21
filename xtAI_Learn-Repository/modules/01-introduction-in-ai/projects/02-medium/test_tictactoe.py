"""Tests fuer das Tic-Tac-Toe-Projekt. Ausfuehren mit:

    python test_tictactoe.py                            (testet deine tictactoe.py)
    TTT_MODUL=tictactoe_solution python test_tictactoe.py   (testet die Musterloesung)

Kein pytest noetig — das ist ein einfaches Skript mit assert-Pruefungen.
"""
import importlib
import os
import sys

HIER = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HIER, os.path.join(HIER, "solution")]
ttt = importlib.import_module(os.environ.get("TTT_MODUL", "tictactoe"))

fehler = 0

def pruefe(name, bedingung):
    global fehler
    status = "ok  " if bedingung else "FEHLER"
    if not bedingung:
        fehler += 1
    print(f"[{status}] {name}")


# ---- gewinner() ------------------------------------------------------------
pruefe("leeres Brett: Spiel laeuft (None)", ttt.gewinner(ttt.neues_brett()) is None)
pruefe("Zeile oben: X gewinnt", ttt.gewinner(list("XXX  O O ")) == "X")
pruefe("Spalte links: O gewinnt", ttt.gewinner(list("OX OX O  ")) == "O")
pruefe("Diagonale: X gewinnt", ttt.gewinner(list("XO  XO  X")) == "X")
pruefe("volles Brett ohne Linie: unentschieden", ttt.gewinner(list("XOXXOOOXX")) == "unentschieden")

# ---- minimax(): bekannte Stellungswerte -------------------------------------
# X kann sofort gewinnen (Feld 2) -> Wert aus X-Sicht positiv
brett = list("XX OO    ")
pruefe("Sofortsieg wird erkannt (Wert > 0)",
       ttt.minimax(brett, "X", "X") > 0)
# O droht zu gewinnen, X ist am Zug und muss blocken -> perfektes Spiel = Remis (0)
brett = list("OO XX    ")
pruefe("Blocken erzwingt bestenfalls eigenen Sieg oder Remis (Wert >= 0)",
       ttt.minimax(brett, "X", "X") >= 0)
# Leeres Brett ist bei perfektem Spiel beidseitig ein Remis
pruefe("leeres Brett: Minimax-Wert 0 (Remis)",
       ttt.minimax(ttt.neues_brett(), "X", "X") == 0)

# ---- beste_aktion(): konkrete Zugwahl ---------------------------------------
pruefe("nimmt den Sofortsieg (Feld 2)",
       ttt.beste_aktion(list("XX OO    "), "X") == 2)
# X hat 0+1 und droht auf Feld 2; O (Mitte besetzt) muss blocken -> Remis
pruefe("blockt zwingend die Drohung (Feld 2)",
       ttt.beste_aktion(list("XX  O    "), "O") == 2)

# ---- KI gegen KI: darf nie jemand gewinnen ----------------------------------
def ki_gegen_ki():
    brett = ttt.neues_brett()
    am_zug = "X"
    while ttt.gewinner(brett) is None:
        brett[ttt.beste_aktion(brett, am_zug)] = am_zug
        am_zug = ttt.gegner(am_zug)
    return ttt.gewinner(brett)

pruefe("KI vs. KI endet unentschieden", ki_gegen_ki() == "unentschieden")

# ---- Alpha-Beta: gleicher Zug, weniger Knoten --------------------------------
ttt.besuchte_knoten = 0
zug_mit = ttt.beste_aktion(ttt.neues_brett(), "X", pruning=True)
mit_pruning = ttt.besuchte_knoten

ttt.besuchte_knoten = 0
zug_ohne = ttt.beste_aktion(ttt.neues_brett(), "X", pruning=False)
ohne_pruning = ttt.besuchte_knoten

pruefe("Pruning aendert die Zugqualitaet nicht (beide Zuege Minimax-optimal)",
       zug_mit is not None and zug_ohne is not None)
pruefe("Alpha-Beta besucht deutlich weniger Knoten (< 25 %)",
       mit_pruning < 0.25 * ohne_pruning)
print(f"\n       Knoten ohne Pruning: {ohne_pruning:,}".replace(",", "."))
print(f"       Knoten mit Pruning:  {mit_pruning:,} "
      f"({100 * mit_pruning / ohne_pruning:.1f} %)".replace(",", "."))

# ------------------------------------------------------------------------------
print(f"\n{'ALLE TESTS BESTANDEN 🎉' if fehler == 0 else f'{fehler} Test(s) fehlgeschlagen.'}")
sys.exit(0 if fehler == 0 else 1)
