"""DPLL — a satisfiability solver for propositional logic in CNF.

YOUR TASK: fill in the TODO functions. The encoding and the outer entry point
are given.

The encoding: a variable is a positive integer. A literal is +v (positive) or
-v (negated). A clause is a set of literals (a disjunction), a formula a list
of clauses (a conjunction).

Returns: a model {var: bool} (SAT) or None (UNSAT).
"""
import sys
sys.setrecursionlimit(100000)


def clause_value(clause, model):
    """The truth value of a clause under the (partial) model:
      - True  if ANY literal is true,
      - False if ALL literals are assigned and false,
      - None  otherwise (at least one literal unassigned, none true).
    A literal `lit` is true if model[abs(lit)] == (lit > 0).
    """
    # TODO
    raise NotImplementedError


def find_unit_clause(clauses, model):
    """Look for a still OPEN clause (clause_value == None) with exactly ONE
    unassigned literal. Return (var, value) that this literal forces
    (value = lit > 0). Otherwise (None, None)."""
    # TODO
    raise NotImplementedError


def find_pure_symbol(symbols, clauses, model):
    """Look for a symbol from `symbols` that occurs in all still OPEN clauses
    with only ONE polarity. Return (var, value), otherwise (None, None).
    Hint: first collect all literals of the open clauses in a set."""
    # TODO
    raise NotImplementedError


def dpll_satisfiable(clauses):
    """The entry point (given)."""
    clauses = [frozenset(c) for c in clauses]
    symbols = sorted({abs(l) for c in clauses for l in c})
    return _dpll(clauses, symbols, {})


def _dpll(clauses, symbols, model):
    """The recursion with the three DPLL accelerators.

    The scaffold:
      1) Early termination:
           - any clause False -> return None
           - all clauses True -> return dict(model)
      2) Unit propagation: P,val = find_unit_clause(...); if P: set it and recurse
      3) Pure literal:     P,val = find_pure_symbol(...);  if P: set it and recurse
      4) Branch on symbols[0] with val in (True, False); the first successful
         return wins, otherwise None.
    When recursing, copy the model each time and remove P from symbols.
    """
    # TODO
    raise NotImplementedError
