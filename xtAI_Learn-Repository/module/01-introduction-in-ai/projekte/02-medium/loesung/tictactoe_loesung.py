"""Tic-Tac-Toe mit unschlagbarer KI — MUSTERLOESUNG.

Identisch zu tictactoe.py, aber mit ausgefuellten TODO-Stellen.
Spielen:   python loesung/tictactoe_loesung.py
Testen:    TTT_MODUL=tictactoe_loesung python test_tictactoe.py
"""

# Das Brett ist eine Liste mit 9 Feldern (Index 0-8), Inhalt: "X", "O" oder " ".
#
#   0 | 1 | 2
#  ---+---+---
#   3 | 4 | 5
#  ---+---+---
#   6 | 7 | 8
#
# "X" zieht immer zuerst. Die KI kann beide Seiten spielen.

LEER = " "

GEWINNLINIEN = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # Zeilen
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # Spalten
    (0, 4, 8), (2, 4, 6),              # Diagonalen
]

# Zaehler fuer besuchte Knoten — damit vergleichen wir Minimax mit/ohne Pruning.
besuchte_knoten = 0


def neues_brett():
    return [LEER] * 9


def freie_felder(brett):
    """Liste der Indizes aller freien Felder (das sind die moeglichen Aktionen)."""
    return [i for i, feld in enumerate(brett) if feld == LEER]


def gewinner(brett):
    """"X" oder "O", falls jemand gewonnen hat; "unentschieden", falls das Brett
    voll ist; None, falls das Spiel noch laeuft."""
    for a, b, c in GEWINNLINIEN:
        if brett[a] != LEER and brett[a] == brett[b] == brett[c]:
            return brett[a]
    if LEER not in brett:
        return "unentschieden"
    return None


def gegner(spieler):
    return "O" if spieler == "X" else "X"


def minimax(brett, am_zug, max_spieler, tiefe=0, alpha=-float("inf"), beta=float("inf"), pruning=True):
    """Minimax-Wert des Bretts aus Sicht von max_spieler.

    Bewertung: +10 - tiefe fuer Sieg (frueher Sieg ist besser),
               tiefe - 10 fuer Niederlage (spaete Niederlage ist besser),
               0 fuer Unentschieden.
    alpha = bester bereits garantierter Wert fuer MAX auf dem Pfad,
    beta  = bester bereits garantierter Wert fuer MIN auf dem Pfad.
    Mit pruning=False wird nie abgeschnitten (fuer den Vergleich).
    """
    global besuchte_knoten
    besuchte_knoten += 1

    ende = gewinner(brett)
    if ende is not None:
        if ende == max_spieler:
            return 10 - tiefe
        if ende == gegner(max_spieler):
            return tiefe - 10
        return 0

    ist_max = (am_zug == max_spieler)
    bester = -float("inf") if ist_max else float("inf")

    for feld in freie_felder(brett):
        brett[feld] = am_zug
        wert = minimax(brett, gegner(am_zug), max_spieler, tiefe + 1, alpha, beta, pruning)
        brett[feld] = LEER

        if ist_max:
            bester = max(bester, wert)
            alpha = max(alpha, bester)
        else:
            bester = min(bester, wert)
            beta = min(beta, bester)
        if pruning and beta <= alpha:
            break  # Alpha-Beta-Schnitt: Rest kann das Ergebnis nicht mehr aendern

    return bester


def beste_aktion(brett, spieler, pruning=True):
    """Waehlt das Feld mit dem hoechsten Minimax-Wert fuer `spieler`."""
    bester_wert, bestes_feld = -float("inf"), None
    for feld in freie_felder(brett):
        brett[feld] = spieler
        wert = minimax(brett, gegner(spieler), spieler, tiefe=1, pruning=pruning)
        brett[feld] = LEER
        if wert > bester_wert:
            bester_wert, bestes_feld = wert, feld
    return bestes_feld


# ---------------------------------------------------------------- Anzeige & Spiel

def zeige(brett):
    z = [feld if feld != LEER else str(i) for i, feld in enumerate(brett)]
    print(f"\n {z[0]} | {z[1]} | {z[2]}\n---+---+---\n {z[3]} | {z[4]} | {z[5]}\n---+---+---\n {z[6]} | {z[7]} | {z[8]}\n")


def spiele_mensch_gegen_ki():
    brett = neues_brett()
    mensch = input("Willst du X (faengt an) oder O sein? [X/O]: ").strip().upper()
    mensch = mensch if mensch in ("X", "O") else "X"
    ki = gegner(mensch)
    am_zug = "X"
    while gewinner(brett) is None:
        zeige(brett)
        if am_zug == mensch:
            try:
                feld = int(input(f"Dein Zug ({mensch}), Feld 0-8: "))
            except ValueError:
                continue
            if feld not in freie_felder(brett):
                print("Feld ist nicht frei.")
                continue
        else:
            feld = beste_aktion(brett, ki)
            print(f"KI ({ki}) setzt auf Feld {feld}.")
        brett[feld] = am_zug
        am_zug = gegner(am_zug)
    zeige(brett)
    ergebnis = gewinner(brett)
    print("Unentschieden!" if ergebnis == "unentschieden" else f"{ergebnis} gewinnt!")


if __name__ == "__main__":
    spiele_mensch_gegen_ki()
