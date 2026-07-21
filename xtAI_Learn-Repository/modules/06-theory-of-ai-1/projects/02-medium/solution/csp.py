"""Generisches CSP-Framework: Backtracking + MRV + AC-3 (Kantenkonsistenz).

Ein CSP ist (Variablen, Domaenen, Nachbarn, Constraint-Praedikat). Die
Constraints hier sind binaer: constraints(A, a, B, b) -> bool sagt, ob die
Belegung A=a, B=b erlaubt ist. Fuer Sudoku ist das schlicht "a != b".
"""
from collections import deque


class CSP:
    def __init__(self, variables, domains, neighbors, constraints):
        self.variables = list(variables)
        self.domains = {v: set(domains[v]) for v in variables}  # volle Domaenen
        self.neighbors = neighbors                              # var -> iterable(var)
        self.constraints = constraints                         # (A,a,B,b) -> bool


# ---------------------------------------------------------------- AC-3
def revise(csp, domains, Xi, Xj):
    """Streicht aus domains[Xi] jeden Wert ohne zulaessigen Partner in
    domains[Xj]. Gibt True zurueck, wenn etwas entfernt wurde."""
    revised = False
    for x in set(domains[Xi]):
        if not any(csp.constraints(Xi, x, Xj, y) for y in domains[Xj]):
            domains[Xi].discard(x)
            revised = True
    return revised


def ac3(csp, domains, queue=None):
    """Stellt Kantenkonsistenz auf domains (mutierend) her.
    Gibt False zurueck, sobald eine Domaene leer wird (Inkonsistenz)."""
    if queue is None:
        queue = deque((Xi, Xj) for Xi in csp.variables for Xj in csp.neighbors[Xi])
    else:
        queue = deque(queue)
    while queue:
        Xi, Xj = queue.popleft()
        if revise(csp, domains, Xi, Xj):
            if not domains[Xi]:
                return False
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.append((Xk, Xi))
    return True


# ---------------------------------------------------------------- Backtracking
def select_unassigned_variable(csp, domains, assignment):
    """MRV: waehle die unbelegte Variable mit den wenigsten Restwerten."""
    unassigned = [v for v in csp.variables if v not in assignment]
    return min(unassigned, key=lambda v: len(domains[v]))


def consistent(csp, var, value, assignment):
    """Vertraeglich mit allen bereits belegten Nachbarn?"""
    return all(csp.constraints(var, value, B, assignment[B])
               for B in csp.neighbors[var] if B in assignment)


def backtracking_search(csp):
    """Backtracking mit MRV und MAC (Maintaining Arc Consistency via AC-3).
    Gibt eine vollstaendige Belegung (dict) oder None zurueck."""
    domains = {v: set(csp.domains[v]) for v in csp.variables}
    if not ac3(csp, domains):           # Vorverarbeitung
        return None

    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return dict(assignment)
        var = select_unassigned_variable(csp, domains, assignment)
        for value in list(domains[var]):
            if consistent(csp, var, value, assignment):
                saved = {v: set(domains[v]) for v in domains}   # Snapshot
                assignment[var] = value
                domains[var] = {value}
                if ac3(csp, domains, [(Xk, var) for Xk in csp.neighbors[var]]):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                domains.update(saved)                           # Rollback
                del assignment[var]
        return None

    return backtrack({})
