"""Sudoku auf zwei Wegen loesen: als CSP und als SAT-Instanz (via DPLL).

Ein Problem -> zwei Formalismen. Der CSP-Weg nutzt dein Backtracking+MRV+AC-3
(csp.py), der SAT-Weg kodiert Sudoku in Klauseln und laesst dein DPLL (dpll.py)
entscheiden.

DEINE AUFGABE hier: `at_most_one` und `encode_sudoku` (die SAT-Kodierung).
Der Rest (Parsing, CSP-Modell, Ausgabe, Demo) ist vorgegeben.
"""
from csp import CSP, backtracking_search
from dpll import dpll_satisfiable

N = 9
BOX = 3

# ---------------------------------------------------------------- Ein-/Ausgabe
def parse_grid(s):
    s = "".join(ch for ch in s if ch.isdigit() or ch == ".")
    assert len(s) == 81, f"erwarte 81 Felder, bekam {len(s)}"
    return [[0 if c in ".0" else int(c) for c in s[r*9:r*9+9]] for r in range(9)]


def show(grid):
    lines = []
    for r in range(9):
        if r % 3 == 0 and r:
            lines.append("------+-------+------")
        row = " ".join((str(grid[r][c]) if grid[r][c] else ".")
                        + (" |" if c % 3 == 2 and c < 8 else "")
                        for c in range(9))
        lines.append(row)
    return "\n".join(lines)


def is_valid_solution(grid):
    if grid is None:
        return False
    full = range(1, 10)
    def ok(vals): return sorted(vals) == list(full)
    for r in range(9):
        if not ok(grid[r]):                 return False
    for c in range(9):
        if not ok([grid[r][c] for r in range(9)]): return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = [grid[br+i][bc+j] for i in range(3) for j in range(3)]
            if not ok(box):                 return False
    return True


# ---------------------------------------------------------------- CSP-Modell (vorgegeben)
def sudoku_neighbors():
    nb = {}
    for r in range(9):
        for c in range(9):
            peers = set()
            for k in range(9):
                peers.add((r, k)); peers.add((k, c))
            br, bc = (r // 3) * 3, (c // 3) * 3
            for i in range(3):
                for j in range(3):
                    peers.add((br + i, bc + j))
            peers.discard((r, c))
            nb[(r, c)] = peers
    return nb


def build_csp(grid):
    variables = [(r, c) for r in range(9) for c in range(9)]
    domains = {(r, c): ({grid[r][c]} if grid[r][c] else set(range(1, 10)))
               for (r, c) in variables}
    return CSP(variables, domains, sudoku_neighbors(),
               lambda A, a, B, b: a != b)          # AllDifferent, binaer zerlegt


def solve_csp(grid):
    sol = backtracking_search(build_csp(grid))
    if sol is None:
        return None
    return [[sol[(r, c)] for c in range(9)] for r in range(9)]


# ---------------------------------------------------------------- SAT-Modell
def var(r, c, d):
    """Boolesche Variable v(r,c,d): 'Zelle (r,c) traegt Ziffer d'.
    Eindeutige positive ID in 1..729. (vorgegeben)"""
    return r * 81 + c * 9 + (d - 1) + 1


def at_most_one(lits):
    """Klauseln, die 'hoechstens eines dieser Literale ist wahr' erzwingen.
    Inspiration: fuer jedes Paar (li, lj) mit i<j reicht die Klausel (-li v -lj).
    Gib eine Liste solcher Klauseln (Mengen) zurueck.
    """
    # TODO
    raise NotImplementedError


def encode_sudoku(grid):
    """Sudoku -> KNF-Klauselmenge (Liste von Mengen von Literalen).

    Baue diese Klauselgruppen (nutze var(...) und at_most_one(...)):
      (1) jede Zelle traegt MINDESTENS eine Ziffer:  { var(r,c,d) fuer alle d }
      (2) jede Zelle traegt HOECHSTENS eine Ziffer:  at_most_one ueber die d
      (3) jede Ziffer d hoechstens einmal pro ZEILE
      (4) jede Ziffer d hoechstens einmal pro SPALTE
      (5) jede Ziffer d hoechstens einmal pro 3x3-BLOCK
      (6) Vorgaben (grid[r][c] != 0) als Unit-Klausel { var(r,c,grid[r][c]) }
    """
    clauses = []
    digits = range(1, 10)
    # TODO: Gruppen (1)-(6) aufbauen und an `clauses` anhaengen
    raise NotImplementedError


def solve_sat(grid):
    clauses = encode_sudoku(grid)
    model = dpll_satisfiable(clauses)
    if model is None:
        return None, len(clauses)
    out = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            for d in range(1, 10):
                if model.get(var(r, c, d), False):
                    out[r][c] = d
    return out, len(clauses)


# ---------------------------------------------------------------- Demo
PUZZLE = ("53..7...."
          "6..195..."
          ".98....6."
          "8...6...3"
          "4..8.3..1"
          "7...2...6"
          ".6....28."
          "...419..5"
          "....8..79")

if __name__ == "__main__":
    import time
    grid = parse_grid(PUZZLE)
    print("Ausgangsraetsel:\n" + show(grid) + "\n")

    t = time.perf_counter()
    csp_sol = solve_csp(grid)
    print(f"[CSP]  geloest in {(time.perf_counter()-t)*1000:.1f} ms, "
          f"gueltig={is_valid_solution(csp_sol)}")
    print(show(csp_sol) + "\n")

    t = time.perf_counter()
    sat_sol, ncl = solve_sat(grid)
    print(f"[SAT]  {ncl} Klauseln, DPLL in {(time.perf_counter()-t)*1000:.1f} ms, "
          f"gueltig={is_valid_solution(sat_sol)}")
    print(show(sat_sol) + "\n")

    assert csp_sol == sat_sol, "Beide Wege muessen dieselbe Loesung liefern!"
    print("OK — CSP- und SAT-Loesung sind identisch und gueltig.")
