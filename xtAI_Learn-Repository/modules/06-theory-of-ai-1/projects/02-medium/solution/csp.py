"""A generic CSP framework: backtracking + MRV + AC-3 (arc consistency).

A CSP is (variables, domains, neighbours, a constraint predicate). The
constraints here are binary: constraints(A, a, B, b) -> bool says whether the
assignment A=a, B=b is permitted. For Sudoku that is simply "a != b".
"""
from collections import deque


class CSP:
    def __init__(self, variables, domains, neighbors, constraints):
        self.variables = list(variables)
        self.domains = {v: set(domains[v]) for v in variables}  # the full domains
        self.neighbors = neighbors                              # var -> iterable(var)
        self.constraints = constraints                          # (A,a,B,b) -> bool


# ---------------------------------------------------------------- AC-3
def revise(csp, domains, Xi, Xj):
    """Deletes from domains[Xi] every value without a permitted partner in
    domains[Xj]. Returns True if something was removed."""
    revised = False
    for x in set(domains[Xi]):
        if not any(csp.constraints(Xi, x, Xj, y) for y in domains[Xj]):
            domains[Xi].discard(x)
            revised = True
    return revised


def ac3(csp, domains, queue=None):
    """Establishes arc consistency on domains (mutating it).
    Returns False as soon as a domain becomes empty (an inconsistency)."""
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
    """MRV: pick the unassigned variable with the fewest remaining values."""
    unassigned = [v for v in csp.variables if v not in assignment]
    return min(unassigned, key=lambda v: len(domains[v]))


def consistent(csp, var, value, assignment):
    """Compatible with all already assigned neighbours?"""
    return all(csp.constraints(var, value, B, assignment[B])
               for B in csp.neighbors[var] if B in assignment)


def backtracking_search(csp):
    """Backtracking with MRV and MAC (maintaining arc consistency via AC-3).
    Returns a complete assignment (a dict) or None."""
    domains = {v: set(csp.domains[v]) for v in csp.variables}
    if not ac3(csp, domains):           # preprocessing
        return None

    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return dict(assignment)
        var = select_unassigned_variable(csp, domains, assignment)
        for value in list(domains[var]):
            if consistent(csp, var, value, assignment):
                saved = {v: set(domains[v]) for v in domains}   # snapshot
                assignment[var] = value
                domains[var] = {value}
                if ac3(csp, domains, [(Xk, var) for Xk in csp.neighbors[var]]):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                domains.update(saved)                           # rollback
                del assignment[var]
        return None

    return backtrack({})
