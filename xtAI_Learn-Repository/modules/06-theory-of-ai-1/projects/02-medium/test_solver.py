"""A small test suite. Run it while you fill in csp.py / dpll.py / sudoku.py —
that way you check every building block individually.

    python test_solver.py

All tests have to end with 'OK'. Before you implement anything they fail (as
expected) with NotImplementedError.
"""
from csp import CSP, backtracking_search
from dpll import dpll_satisfiable, clause_value
import sudoku


def test_dpll_sat():
    # (A v B) & (-A) -> must force B=True
    model = dpll_satisfiable([{1, 2}, {-1}])
    assert model is not None, "the formula is satisfiable"
    assert model[2] is True and model[1] is False
    print("  DPLL SAT ................ OK")


def test_dpll_unsat():
    # A & -A -> unsatisfiable
    assert dpll_satisfiable([{1}, {-1}]) is None
    # (A v B) & (A v -B) & (-A v B) & (-A v -B) -> unsatisfiable
    assert dpll_satisfiable([{1, 2}, {1, -2}, {-1, 2}, {-1, -2}]) is None
    print("  DPLL UNSAT .............. OK")


def test_clause_value():
    assert clause_value({1, 2}, {1: True}) is True
    assert clause_value({-1, 2}, {1: True, 2: False}) is False
    assert clause_value({1, 2}, {1: False}) is None
    print("  clause_value ............ OK")


def test_csp_map_coloring():
    # Colouring the map of Australia (3 colours) — the CSP classic.
    WA, NT, SA, Q, NSW, V, T = "WA NT SA Q NSW V T".split()
    variables = [WA, NT, SA, Q, NSW, V, T]
    domains = {v: {"red", "green", "blue"} for v in variables}
    edges = [(WA, NT), (WA, SA), (NT, SA), (NT, Q), (SA, Q),
             (SA, NSW), (SA, V), (Q, NSW), (NSW, V)]
    neighbors = {v: set() for v in variables}
    for a, b in edges:
        neighbors[a].add(b); neighbors[b].add(a)
    csp = CSP(variables, domains, neighbors, lambda A, a, B, b: a != b)
    sol = backtracking_search(csp)
    assert sol is not None
    for a, b in edges:
        assert sol[a] != sol[b], f"{a} and {b} are coloured the same!"
    print("  CSP map colouring ....... OK")


def test_sudoku_both_ways():
    grid = sudoku.parse_grid(sudoku.PUZZLE)
    csp_sol = sudoku.solve_csp(grid)
    sat_sol, _ = sudoku.solve_sat(grid)
    assert sudoku.is_valid_solution(csp_sol), "the CSP solution is invalid"
    assert sudoku.is_valid_solution(sat_sol), "the SAT solution is invalid"
    assert csp_sol == sat_sol, "the CSP and SAT routes return different solutions"
    print("  Sudoku CSP == SAT ....... OK")


if __name__ == "__main__":
    print("Tests:")
    for t in (test_clause_value, test_dpll_sat, test_dpll_unsat,
              test_csp_map_coloring, test_sudoku_both_ways):
        t()
    print("\nAll tests passed.")
