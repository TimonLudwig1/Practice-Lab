"""Tic-Tac-Toe mit unschlagbarer KI — DEINE AUFGABE.

Fuelle die drei mit TODO markierten Funktionen (gewinner, minimax, beste_aktion).
Pruefen:  python test_tictactoe.py
Spielen:  python tictactoe.py   (erst sinnvoll, wenn die Tests gruen sind)
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
    # TODO 1: Gehe alle GEWINNLINIEN durch. Stehen auf allen drei Feldern einer
    #         Linie dieselben (nicht-leeren) Zeichen, hat dieser Spieler gewonnen.
    # TODO 2: Kein Gewinner, aber kein LEER mehr im Brett -> "unentschieden".
    # TODO 3: Sonst -> None.
    raise NotImplementedError


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

    Vorgehen (siehe Skript, Abschnitt 2.1):
      1. Terminaltest: gewinner(brett) -> Wert zurueckgeben (Bewertung siehe oben).
      2. Sonst: alle freien Felder durchprobieren —
         Zug setzen (brett[feld] = am_zug), rekursiv bewerten, Zug zuruecknehmen!
      3. Ist max_spieler am Zug: Maximum bilden und alpha aktualisieren;
         sonst Minimum bilden und beta aktualisieren.
      4. Falls pruning und beta <= alpha: Schleife abbrechen (Schnitt!).
    """
    global besuchte_knoten
    besuchte_knoten += 1

    # TODO: Schritte 1-4 aus dem Docstring implementieren.
    raise NotImplementedError


def beste_aktion(brett, spieler, pruning=True):
    """Waehlt das Feld mit dem hoechsten Minimax-Wert fuer `spieler`."""
    # TODO: Fuer jedes freie Feld — Zug setzen, mit minimax(...) bewerten
    #       (naechster am Zug: gegner(spieler), tiefe=1), Zug zuruecknehmen,
    #       und das Feld mit dem hoechsten Wert zurueckgeben.
    raise NotImplementedError


# ---------------------------------------------------------------- Anzeige & Spiel
# Ab hier ist alles fertig vorgegeben — nichts zu tun.

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
