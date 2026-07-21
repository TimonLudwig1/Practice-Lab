"""A generic CSP framework: backtracking + MRV + AC-3 (arc consistency).

YOUR TASK: fill in the functions marked with TODO. The data structure (the CSP
class) and the MRV selection are given. The core of the project are `revise`,
`ac3` and `backtracking_search`.

A CSP is (variables, domains, neighbours, a constraint predicate). The
constraints are binary: constraints(A, a, B, b) -> bool says whether the
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
    """Delete from domains[Xi] every value x for which there is NO partner
    value y in domains[Xj] with constraints(Xi, x, Xj, y).
    Return True if at least one value was removed.

    Hint: iterate over set(domains[Xi]) (a copy!), otherwise you are modifying
    what you are iterating over.
    """
    # TODO
    raise NotImplementedError


def ac3(csp, domains, queue=None):
    """Establish arc consistency on `domains` (mutating it).
    Return False as soon as a domain becomes empty (an inconsistency), else True.

    The scaffold:
      - queue = all directed arcs (Xi, Xj), if None was passed.
      - while the queue is not empty:
            (Xi, Xj) = queue.popleft()
            if revise(...):  # Xi has changed
                if domains[Xi] is empty: return False
                add all (Xk, Xi) back for the neighbours Xk != Xj
      - return True
    """
    if queue is None:
        queue = deque((Xi, Xj) for Xi in csp.variables for Xj in csp.neighbors[Xi])
    else:
        queue = deque(queue)
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Backtracking
def select_unassigned_variable(csp, domains, assignment):
    """MRV (given): the unassigned variable with the fewest remaining values."""
    unassigned = [v for v in csp.variables if v not in assignment]
    return min(unassigned, key=lambda v: len(domains[v]))


def consistent(csp, var, value, assignment):
    """Compatible with all already assigned neighbours? (given)"""
    return all(csp.constraints(var, value, B, assignment[B])
               for B in csp.neighbors[var] if B in assignment)


def backtracking_search(csp):
    """Backtracking with MRV and MAC (maintaining arc consistency via AC-3).
    Return a complete assignment (a dict) or None.

    The scaffold:
      domains = a fresh copy of the domains
      if not ac3(csp, domains): return None      # preprocessing

      def backtrack(assignment):
          if all variables are assigned: return dict(assignment)
          var = select_unassigned_variable(...)
          for every value in domains[var]:
              if consistent(csp, var, value, assignment):
                  saved = a snapshot of all domains
                  assignment[var] = value ; domains[var] = {value}
                  if ac3(csp, domains, [(Xk, var) for Xk in neighbors[var]]):
                      result = backtrack(assignment)
                      if result: return result
                  restore the domains (saved) ; del assignment[var]
          return None
      return backtrack({})
    """
    # TODO
    raise NotImplementedError
