"""DPLL — Erfuellbarkeits-Solver fuer Aussagenlogik in KNF.

Kodierung: Eine Variable ist eine positive ganze Zahl. Ein Literal ist +v
(positiv) oder -v (negiert). Eine Klausel ist eine Menge von Literalen
(Disjunktion), eine Formel eine Liste von Klauseln (Konjunktion).

Rueckgabe: ein Modell {var: bool} (SAT) oder None (UNSAT).
"""
import sys
sys.setrecursionlimit(100000)


def clause_value(clause, model):
    """Wahrheitswert einer Klausel unter (partiellem) model:
    True (erfuellt), False (verletzt) oder None (noch offen)."""
    unknown = False
    for lit in clause:
        v = abs(lit)
        if v in model:
            if model[v] == (lit > 0):
                return True                # ein wahres Literal genuegt
        else:
            unknown = True
    return None if unknown else False


def find_unit_clause(clauses, model):
    """Eine noch offene Klausel mit genau EINEM unbelegten Literal (alle
    anderen falsch). Gibt (var, wert) zurueck, den das Literal erzwingt."""
    for clause in clauses:
        if clause_value(clause, model) is not None:
            continue
        unbound = [lit for lit in clause if abs(lit) not in model]
        if len(unbound) == 1:
            lit = unbound[0]
            return abs(lit), (lit > 0)
    return None, None


def find_pure_symbol(symbols, clauses, model):
    """Ein Symbol, das in allen noch offenen Klauseln nur mit EINER Polaritaet
    auftritt. Gibt (var, wert) zurueck."""
    lits = set()
    for clause in clauses:
        if clause_value(clause, model) is None:
            lits.update(clause)
    for s in symbols:
        pos, neg = (s in lits), (-s in lits)
        if pos and not neg:
            return s, True
        if neg and not pos:
            return s, False
    return None, None


def dpll_satisfiable(clauses):
    clauses = [frozenset(c) for c in clauses]
    symbols = sorted({abs(l) for c in clauses for l in c})
    return _dpll(clauses, symbols, {})


def _dpll(clauses, symbols, model):
    # 1) Fruehterminierung: verletzte Klausel? -> UNSAT auf diesem Zweig.
    for clause in clauses:
        if clause_value(clause, model) is False:
            return None
    # alle erfuellt?
    if all(clause_value(c, model) is True for c in clauses):
        return dict(model)
    # 2) Unit Propagation (der wirkungsvollste Schritt).
    P, val = find_unit_clause(clauses, model)
    if P is not None:
        m = dict(model); m[P] = val
        return _dpll(clauses, [s for s in symbols if s != P], m)
    # 3) Pure-Literal-Regel.
    P, val = find_pure_symbol(symbols, clauses, model)
    if P is not None:
        m = dict(model); m[P] = val
        return _dpll(clauses, [s for s in symbols if s != P], m)
    # 4) Verzweigen ueber ein freies Symbol.
    P, rest = symbols[0], symbols[1:]
    for val in (True, False):
        m = dict(model); m[P] = val
        r = _dpll(clauses, rest, m)
        if r is not None:
            return r
    return None
