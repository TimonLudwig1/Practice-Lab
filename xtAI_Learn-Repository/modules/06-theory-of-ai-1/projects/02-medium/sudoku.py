"""Solving Sudoku by two routes: as a CSP and as a SAT instance (via DPLL).

One problem -> two formalisms. The CSP route uses your backtracking+MRV+AC-3
(csp.py), the SAT route encodes Sudoku into clauses and lets your DPLL
(dpll.py) decide.

YOUR TASK here: `at_most_one` and `encode_sudoku` (the SAT encoding).
The rest (parsing, the CSP model, the output, the demo) is given.
"""
from csp import CSP, backtracking_search
from dpll import dpll_satisfiable

N = 9
BOX = 3

# ---------------------------------------------------------------- Input/output
def parse_grid(s):
    s = "".join(ch for ch in s if ch.isdigit() or ch == ".")
    assert len(s) == 81, f"expected 81 cells, got {len(s)}"
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


# ---------------------------------------------------------------- The CSP model (given)
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
               lambda A, a, B, b: a != b)          # AllDifferent, decomposed into binary form


def solve_csp(grid):
    sol = backtracking_search(build_csp(grid))
    if sol is None:
        return None
    return [[sol[(r, c)] for c in range(9)] for r in range(9)]


# ---------------------------------------------------------------- The SAT model
def var(r, c, d):
    """The boolean variable v(r,c,d): 'cell (r,c) carries the digit d'.
    A unique positive ID in 1..729. (given)"""
    return r * 81 + c * 9 + (d - 1) + 1


def at_most_one(lits):
    """Clauses that enforce 'at most one of these literals is true'.
    Inspiration: for every pair (li, lj) with i<j the clause (-li v -lj)
    suffices. Return a list of such clauses (sets).
    """
    # TODO
    raise NotImplementedError


def encode_sudoku(grid):
    """Sudoku -> a set of CNF clauses (a list of sets of literals).

    Build these groups of clauses (use var(...) and at_most_one(...)):
      (1) every cell carries AT LEAST one digit:  { var(r,c,d) for all d }
      (2) every cell carries AT MOST one digit:   at_most_one over the d
      (3) every digit d at most once per ROW
      (4) every digit d at most once per COLUMN
      (5) every digit d at most once per 3x3 BLOCK
      (6) the givens (grid[r][c] != 0) as a unit clause { var(r,c,grid[r][c]) }
    """
    clauses = []
    digits = range(1, 10)
    # TODO: build the groups (1)-(6) and append them to `clauses`
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
    print("The initial puzzle:\n" + show(grid) + "\n")

    t = time.perf_counter()
    csp_sol = solve_csp(grid)
    print(f"[CSP]  solved in {(time.perf_counter()-t)*1000:.1f} ms, "
          f"valid={is_valid_solution(csp_sol)}")
    print(show(csp_sol) + "\n")

    t = time.perf_counter()
    sat_sol, ncl = solve_sat(grid)
    print(f"[SAT]  {ncl} clauses, DPLL in {(time.perf_counter()-t)*1000:.1f} ms, "
          f"valid={is_valid_solution(sat_sol)}")
    print(show(sat_sol) + "\n")

    assert csp_sol == sat_sol, "both routes must deliver the same solution!"
    print("OK — the CSP and SAT solutions are identical and valid.")
