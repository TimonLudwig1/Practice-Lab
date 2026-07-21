"""The automatic acceptance test of the prover (the reference solution).

    python test_prover.py    ->  "All tests passed."
"""
from logic import (Atom, Not, And, Or, Implies, ForAll, Exists, Var, Fn, Const,
                   unify, unify_atoms, occurs, to_clauses, prove)
import scenarios


def test_unify_basics():
    x, y = Var("x"), Var("y")
    John, Mary = Const("John"), Const("Mary")
    # Knows(John,x) vs Knows(y,Mary)
    s = unify_atoms(Atom("Knows", (John, x)), Atom("Knows", (y, Mary)), {})
    assert s is not None and s["x"] == Mary and s["y"] == John
    # the occurs check: x must NOT unify with f(x)
    assert unify(x, Fn("f", (x,)), {}) is None
    # different functors fail
    assert unify(Fn("a"), Fn("b"), {}) is None
    print("  Unification & occurs check ... OK")


def test_cnf_skolem():
    # Exists without a universal -> a Skolem constant
    cls = to_clauses(Exists("y", Atom("P", (Var("y"),))))
    assert len(cls) == 1 and len(list(cls[0])) == 1
    # ForAll Exists -> a Skolem function (the argument = the universal)
    cls = to_clauses(ForAll("x", Exists("y", Atom("R", (Var("y"), Var("x"))))))
    lit = next(iter(cls[0]))
    arg0 = lit.atom.args[0]
    assert isinstance(arg0, Fn) and len(arg0.args) == 1, "a Skolem function of x was expected"
    print("  CNF & Skolemization .......... OK")


def test_scenarios():
    assert scenarios.scenario_west(verbose=False)
    assert scenarios.scenario_ancestor(verbose=False)
    assert scenarios.scenario_propositional(verbose=False)
    assert scenarios.scenario_nonentailment()
    assert scenarios.demo_occurs_check()
    print("  All scenarios ................ OK")


if __name__ == "__main__":
    print("Tests:")
    test_unify_basics()
    test_cnf_skolem()
    test_scenarios()
    print("\nAll tests passed.")
