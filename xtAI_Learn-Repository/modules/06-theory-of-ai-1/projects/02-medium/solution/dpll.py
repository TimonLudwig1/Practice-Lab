"""DPLL — a satisfiability solver for propositional logic in CNF.

The encoding: a variable is a positive integer. A literal is +v (positive) or
-v (negated). A clause is a set of literals (a disjunction), a formula a list
of clauses (a conjunction).

Returns: a model {var: bool} (SAT) or None (UNSAT).
"""
import sys
sys.setrecursionlimit(100000)


def clause_value(clause, model):
    """The truth value of a clause under the (partial) model:
    True (satisfied), False (violated) or None (still open)."""
    unknown = False
    for lit in clause:
        v = abs(lit)
        if v in model:
            if model[v] == (lit > 0):
                return True                # one true literal is enough
        else:
            unknown = True
    return None if unknown else False


def find_unit_clause(clauses, model):
    """A still open clause with exactly ONE unassigned literal (all others
    false). Returns (var, value) that the literal forces."""
    for clause in clauses:
        if clause_value(clause, model) is not None:
            continue
        unbound = [lit for lit in clause if abs(lit) not in model]
        if len(unbound) == 1:
            lit = unbound[0]
            return abs(lit), (lit > 0)
    return None, None


def find_pure_symbol(symbols, clauses, model):
    """A symbol that occurs in all still open clauses with only ONE polarity.
    Returns (var, value)."""
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
    # 1) Early termination: a violated clause? -> UNSAT on this branch.
    for clause in clauses:
        if clause_value(clause, model) is False:
            return None
    # all satisfied?
    if all(clause_value(c, model) is True for c in clauses):
        return dict(model)
    # 2) Unit propagation (the most effective step).
    P, val = find_unit_clause(clauses, model)
    if P is not None:
        m = dict(model); m[P] = val
        return _dpll(clauses, [s for s in symbols if s != P], m)
    # 3) The pure literal rule.
    P, val = find_pure_symbol(symbols, clauses, model)
    if P is not None:
        m = dict(model); m[P] = val
        return _dpll(clauses, [s for s in symbols if s != P], m)
    # 4) Branch on a free symbol.
    P, rest = symbols[0], symbols[1:]
    for val in (True, False):
        m = dict(model); m[P] = val
        r = _dpll(clauses, rest, m)
        if r is not None:
            return r
    return None
