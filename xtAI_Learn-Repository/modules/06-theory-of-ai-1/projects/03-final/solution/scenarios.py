"""Application scenarios for the resolution prover (logic.py)."""
from logic import (Atom, Not, And, Or, Implies, Iff, ForAll, Exists,
                   Var, Fn, Const, unify, unify_atoms, occurs, prove, to_clauses,
                   clause_str)

# Short forms
def V(n): return Var(n)
def C(n): return Const(n)

# ======================================================================
#  Scenario A — "Colonel West is a criminal" (the AIMA classic, Horn)
# ======================================================================
def scenario_west(verbose=True):
    x, y, z = V("x"), V("y"), V("z")
    West, Nono, America, M1 = C("West"), C("Nono"), C("America"), C("M1")

    def A(p, *a): return Atom(p, a)

    kb = [
        # It is a crime for an American to sell weapons to hostile nations.
        ForAll("x", ForAll("y", ForAll("z",
            Implies(And(A("American", x), And(A("Weapon", y),
                        And(A("Sells", x, y, z), A("Hostile", z)))),
                    A("Criminal", x))))),
        A("Owns", Nono, M1),                                   # Nono owns M1
        A("Missile", M1),                                      # M1 is a missile
        # All missiles that Nono owns were sold by West.
        ForAll("x", Implies(And(A("Missile", x), A("Owns", Nono, x)),
                            A("Sells", West, x, Nono))),
        ForAll("x", Implies(A("Missile", x), A("Weapon", x))), # missiles are weapons
        ForAll("x", Implies(A("Enemy", x, America), A("Hostile", x))),  # enemies are hostile
        A("American", West),                                   # West is an American
        A("Enemy", Nono, America),                             # Nono is an enemy of America
    ]
    goal = A("Criminal", West)
    ok, steps = prove(kb, goal, verbose=verbose)
    print(f"\n[A] Is West a criminal?  ->  {'YES, proved' if ok else 'not provable'} "
          f"({steps} resolution steps)")
    return ok

# ======================================================================
#  Scenario B — a Skolem function: "everyone has an ancestor"
#  KB: everyone has a parent;  parent => ancestor.   Goal: everyone has an ancestor.
#  It exercises the Skolem FUNCTION (the parent depends on x) and unification
#  into function terms.
# ======================================================================
def scenario_ancestor(verbose=True):
    def A(p, *a): return Atom(p, a)
    x, y = V("x"), V("y")
    kb = [
        ForAll("x", Exists("y", A("Parent", y, x))),                       # everyone has a parent
        ForAll("x", ForAll("y", Implies(A("Parent", y, x), A("Ancestor", y, x)))),
    ]
    goal = ForAll("x", Exists("y", A("Ancestor", y, x)))                   # everyone has an ancestor
    ok, steps = prove(kb, goal, verbose=verbose)
    print(f"\n[B] Does everyone have an ancestor?  ->  {'YES, proved' if ok else 'not provable'} "
          f"({steps} resolution steps)")
    return ok

# ======================================================================
#  Scenario C — pure propositional logic (the special case without variables)
#  Modus tollens: (P=>Q), ¬Q  |=  ¬P
# ======================================================================
def scenario_propositional(verbose=True):
    P, Q = Atom("P"), Atom("Q")
    kb = [Implies(P, Q), Not(Q)]
    goal = Not(P)
    ok, steps = prove(kb, goal, verbose=verbose)
    print(f"\n[C] Propositional logic, modus tollens (P⇒Q, ¬Q ⊢ ¬P)  ->  "
          f"{'YES' if ok else 'no'} ({steps} steps)")
    return ok

# ======================================================================
#  Scenario D — the counter-check: something that does NOT follow is not proved
# ======================================================================
def scenario_nonentailment():
    def A(p, *a): return Atom(p, a)
    kb = [A("Sunny"), Implies(A("Sunny"), A("Warm"))]
    goal = A("Raining")                    # does not follow
    ok, steps = prove(kb, goal, max_steps=500, verbose=False)
    print(f"\n[D] Does 'Raining' follow from (Sunny, Sunny⇒Warm)?  ->  "
          f"{'YES (AN ERROR!)' if ok else 'no, correctly not provable'} ({steps} steps)")
    return not ok

# ======================================================================
#  A demonstration of the occurs check
# ======================================================================
def demo_occurs_check():
    x = Var("x")
    fx = Fn("f", (x,))
    s = unify(x, fx, {})
    print("\n[Occurs check] unify(x, f(x)) =", s,
          "->", "correctly rejected (an infinite term)" if s is None else "AN ERROR")
    return s is None

# ======================================================================
if __name__ == "__main__":
    print("=" * 68)
    print(" Resolution theorem prover — a demonstration")
    print("=" * 68)
    r = []
    r.append(scenario_west(verbose=True))
    r.append(scenario_ancestor(verbose=True))
    r.append(scenario_propositional(verbose=True))
    r.append(scenario_nonentailment())
    r.append(demo_occurs_check())
    print("\n" + "=" * 68)
    print(" Result:", "ALL scenarios as expected." if all(r) else "AN ERROR in one scenario!")
    print("=" * 68)
