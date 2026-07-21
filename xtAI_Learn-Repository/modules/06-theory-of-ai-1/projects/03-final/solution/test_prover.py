"""Automatische Abnahme des Beweisers (Musterlösung).

    python test_prover.py    ->  "Alle Tests bestanden."
"""
from logic import (Atom, Not, And, Or, Implies, ForAll, Exists, Var, Fn, Const,
                   unify, unify_atoms, occurs, to_clauses, prove)
import scenarios


def test_unify_basics():
    x, y = Var("x"), Var("y")
    Hans, Maria = Const("Hans"), Const("Maria")
    # Kennt(Hans,x) vs Kennt(y,Maria)
    s = unify_atoms(Atom("Kennt", (Hans, x)), Atom("Kennt", (y, Maria)), {})
    assert s is not None and s["x"] == Maria and s["y"] == Hans
    # occurs-check: x mit f(x) darf NICHT unifizieren
    assert unify(x, Fn("f", (x,)), {}) is None
    # verschiedene Funktoren scheitern
    assert unify(Fn("a"), Fn("b"), {}) is None
    print("  Unifikation & Occurs-Check ... OK")


def test_cnf_skolem():
    # Exists ohne Universelle -> Skolem-Konstante
    cls = to_clauses(Exists("y", Atom("P", (Var("y"),))))
    assert len(cls) == 1 and len(list(cls[0])) == 1
    # ForAll Exists -> Skolem-Funktion (Argument = die Universelle)
    cls = to_clauses(ForAll("x", Exists("y", Atom("R", (Var("y"), Var("x"))))))
    lit = next(iter(cls[0]))
    arg0 = lit.atom.args[0]
    assert isinstance(arg0, Fn) and len(arg0.args) == 1, "Skolem-Funktion von x erwartet"
    print("  KNF & Skolemisierung ......... OK")


def test_scenarios():
    assert scenarios.scenario_west(verbose=False)
    assert scenarios.scenario_ancestor(verbose=False)
    assert scenarios.scenario_propositional(verbose=False)
    assert scenarios.scenario_nonentailment()
    assert scenarios.demo_occurs_check()
    print("  Alle Szenarien ............... OK")


if __name__ == "__main__":
    print("Tests:")
    test_unify_basics()
    test_cnf_skolem()
    test_scenarios()
    print("\nAlle Tests bestanden.")
