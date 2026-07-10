"""Generisches CSP-Framework: Backtracking + MRV + AC-3 (Kantenkonsistenz).

DEINE AUFGABE: Fuelle die mit TODO markierten Funktionen. Die Datenstruktur
(CSP-Klasse) und die MRV-Auswahl sind vorgegeben. Kern des Projekts sind
`revise`, `ac3` und `backtracking_search`.

Ein CSP ist (Variablen, Domaenen, Nachbarn, Constraint-Praedikat). Die
Constraints sind binaer: constraints(A, a, B, b) -> bool sagt, ob die
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
    """Streiche aus domains[Xi] jeden Wert x, fuer den es in domains[Xj]
    KEINEN Partnerwert y mit constraints(Xi, x, Xj, y) gibt.
    Gib True zurueck, wenn mindestens ein Wert entfernt wurde.

    Tipp: iteriere ueber set(domains[Xi]) (Kopie!), sonst aenderst du,
    worueber du gerade laeufst.
    """
    # TODO
    raise NotImplementedError


def ac3(csp, domains, queue=None):
    """Stelle Kantenkonsistenz auf `domains` (mutierend) her.
    Gib False zurueck, sobald eine Domaene leer wird (Inkonsistenz), sonst True.

    Geruest:
      - queue = alle gerichteten Kanten (Xi, Xj), falls None uebergeben.
      - solange queue nicht leer:
            (Xi, Xj) = queue.popleft()
            wenn revise(...):  # Xi hat sich geaendert
                wenn domains[Xi] leer: return False
                fuege alle (Xk, Xi) fuer Nachbarn Xk != Xj wieder hinzu
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
    """MRV (vorgegeben): unbelegte Variable mit den wenigsten Restwerten."""
    unassigned = [v for v in csp.variables if v not in assignment]
    return min(unassigned, key=lambda v: len(domains[v]))


def consistent(csp, var, value, assignment):
    """Vertraeglich mit allen bereits belegten Nachbarn? (vorgegeben)"""
    return all(csp.constraints(var, value, B, assignment[B])
               for B in csp.neighbors[var] if B in assignment)


def backtracking_search(csp):
    """Backtracking mit MRV und MAC (Maintaining Arc Consistency via AC-3).
    Gib eine vollstaendige Belegung (dict) oder None zurueck.

    Geruest:
      domains = frische Kopie der Domaenen
      wenn nicht ac3(csp, domains): return None      # Vorverarbeitung

      def backtrack(assignment):
          wenn alle Variablen belegt: return dict(assignment)
          var = select_unassigned_variable(...)
          fuer jeden value in domains[var]:
              wenn consistent(csp, var, value, assignment):
                  saved = Snapshot aller domains
                  assignment[var] = value ; domains[var] = {value}
                  wenn ac3(csp, domains, [(Xk, var) fuer Xk in neighbors[var]]):
                      result = backtrack(assignment)
                      wenn result: return result
                  domains zuruecksetzen (saved) ; del assignment[var]
          return None
      return backtrack({})
    """
    # TODO
    raise NotImplementedError
