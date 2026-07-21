"""Sudoku auf zwei Wegen loesen: als CSP und als SAT-Instanz (via DPLL).

Zeigt den Kern des Skripts praktisch: Ein Problem -> zwei Formalismen.
Der CSP-Weg nutzt Backtracking+MRV+AC-3, der SAT-Weg kodiert Sudoku in
aussagenlogische Klauseln und laesst DPLL entscheiden.
"""
from csp import CSP, backtracking_search
from dpll import dpll_satisfiable

N = 9
BOX = 3

# ---------------------------------------------------------------- Ein-/Ausgabe
def parse_grid(s):
    """81-Zeichen-String ('.' oder '0' = leer) -> 9x9-Liste von ints."""
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


# ---------------------------------------------------------------- CSP-Modell
def sudoku_neighbors():
    """Fuer jede Zelle (r,c) die Menge der Zellen in derselben Zeile, Spalte
    oder demselben 3x3-Block (ohne die Zelle selbst)."""
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
    domains = {}
    for (r, c) in variables:
        domains[(r, c)] = {grid[r][c]} if grid[r][c] else set(range(1, 10))
    neighbors = sudoku_neighbors()
    constraints = lambda A, a, B, b: a != b        # AllDifferent, binaer zerlegt
    return CSP(variables, domains, neighbors, constraints)


def solve_csp(grid):
    csp = build_csp(grid)
    sol = backtracking_search(csp)
    if sol is None:
        return None
    return [[sol[(r, c)] for c in range(9)] for r in range(9)]


# ---------------------------------------------------------------- SAT-Modell
def var(r, c, d):
    """Boolesche Variable v(r,c,d): 'Zelle (r,c) traegt Ziffer d'.
    Eindeutige positive ID in 1..729."""
    return r * 81 + c * 9 + (d - 1) + 1


def at_most_one(lits):
    """Klauseln, die erzwingen, dass hoechstens ein Literal der Liste wahr ist:
    fuer jedes Paar (li, lj) die Klausel (-li v -lj)."""
    clauses = []
    for i in range(len(lits)):
        for j in range(i + 1, len(lits)):
            clauses.append({-lits[i], -lits[j]})
    return clauses


def encode_sudoku(grid):
    """Sudoku -> KNF-Klauselmenge (Liste von Mengen von Literalen)."""
    clauses = []
    digits = range(1, 10)
    # (1) Jede Zelle traegt mindestens eine Ziffer.
    for r in range(9):
        for c in range(9):
            clauses.append({var(r, c, d) for d in digits})
            # (2) ... und hoechstens eine.
            clauses += at_most_one([var(r, c, d) for d in digits])
    # (3) Jede Ziffer hoechstens einmal pro Zeile.
    for r in range(9):
        for d in digits:
            clauses += at_most_one([var(r, c, d) for c in range(9)])
    # (4) ... pro Spalte.
    for c in range(9):
        for d in digits:
            clauses += at_most_one([var(r, c, d) for r in range(9)])
    # (5) ... pro 3x3-Block.
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            for d in digits:
                cells = [var(br + i, bc + j, d) for i in range(3) for j in range(3)]
                clauses += at_most_one(cells)
    # (6) Vorgaben als Unit-Klauseln.
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                clauses.append({var(r, c, grid[r][c])})
    return clauses


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
    t_csp = time.perf_counter() - t
    print(f"[CSP]  geloest in {t_csp*1000:.1f} ms, gueltig={is_valid_solution(csp_sol)}")
    print(show(csp_sol) + "\n")

    t = time.perf_counter()
    sat_sol, ncl = solve_sat(grid)
    t_sat = time.perf_counter() - t
    print(f"[SAT]  {ncl} Klauseln, DPLL in {t_sat*1000:.1f} ms, "
          f"gueltig={is_valid_solution(sat_sol)}")
    print(show(sat_sol) + "\n")

    assert csp_sol == sat_sol, "Beide Wege muessen dieselbe Loesung liefern!"
    print("OK — CSP- und SAT-Loesung sind identisch und gueltig.")
