"""Anwendungsszenarien fuer den Resolutions-Beweiser (logic.py)."""
from logic import (Atom, Not, And, Or, Implies, Iff, ForAll, Exists,
                   Var, Fn, Const, unify, unify_atoms, occurs, prove, to_clauses,
                   clause_str)

# Kurzschreibweisen
def V(n): return Var(n)
def C(n): return Const(n)

# ======================================================================
#  Szenario A — "Colonel West ist ein Krimineller" (AIMA-Klassiker, Horn)
# ======================================================================
def scenario_west(verbose=True):
    x, y, z = V("x"), V("y"), V("z")
    West, Nono, America, M1 = C("West"), C("Nono"), C("America"), C("M1")

    def A(p, *a): return Atom(p, a)

    kb = [
        # Es ist ein Verbrechen fuer einen Amerikaner, Waffen an feindliche Nationen zu verkaufen.
        ForAll("x", ForAll("y", ForAll("z",
            Implies(And(A("American", x), And(A("Weapon", y),
                        And(A("Sells", x, y, z), A("Hostile", z)))),
                    A("Criminal", x))))),
        A("Owns", Nono, M1),                                   # Nono besitzt M1
        A("Missile", M1),                                      # M1 ist eine Rakete
        # Alle Raketen, die Nono besitzt, wurden von West verkauft.
        ForAll("x", Implies(And(A("Missile", x), A("Owns", Nono, x)),
                            A("Sells", West, x, Nono))),
        ForAll("x", Implies(A("Missile", x), A("Weapon", x))), # Raketen sind Waffen
        ForAll("x", Implies(A("Enemy", x, America), A("Hostile", x))),  # Feinde sind feindlich
        A("American", West),                                   # West ist Amerikaner
        A("Enemy", Nono, America),                             # Nono ist Feind Amerikas
    ]
    goal = A("Criminal", West)
    ok, steps = prove(kb, goal, verbose=verbose)
    print(f"\n[A] Ist West ein Krimineller?  ->  {'JA, bewiesen' if ok else 'nicht beweisbar'} "
          f"({steps} Resolutionsschritte)")
    return ok

# ======================================================================
#  Szenario B — Skolem-Funktion: "jeder hat einen Vorfahren"
#  KB: jeder hat einen Elter;  Elter => Vorfahr.   Ziel: jeder hat einen Vorfahren.
#  Uebt Skolem-FUNKTION (Elter haengt von x ab) und Unifikation in Funktionsterme.
# ======================================================================
def scenario_ancestor(verbose=True):
    def A(p, *a): return Atom(p, a)
    x, y = V("x"), V("y")
    kb = [
        ForAll("x", Exists("y", A("Parent", y, x))),                       # jeder hat einen Elter
        ForAll("x", ForAll("y", Implies(A("Parent", y, x), A("Ancestor", y, x)))),
    ]
    goal = ForAll("x", Exists("y", A("Ancestor", y, x)))                   # jeder hat einen Vorfahren
    ok, steps = prove(kb, goal, verbose=verbose)
    print(f"\n[B] Hat jeder einen Vorfahren?  ->  {'JA, bewiesen' if ok else 'nicht beweisbar'} "
          f"({steps} Resolutionsschritte)")
    return ok

# ======================================================================
#  Szenario C — reine Aussagenlogik (Sonderfall ohne Variablen)
#  Modus Tollens: (P=>Q), ¬Q  |=  ¬P
# ======================================================================
def scenario_propositional(verbose=True):
    P, Q = Atom("P"), Atom("Q")
    kb = [Implies(P, Q), Not(Q)]
    goal = Not(P)
    ok, steps = prove(kb, goal, verbose=verbose)
    print(f"\n[C] Aussagenlogik, Modus Tollens (P⇒Q, ¬Q ⊢ ¬P)  ->  "
          f"{'JA' if ok else 'nein'} ({steps} Schritte)")
    return ok

# ======================================================================
#  Szenario D — Gegenprobe: etwas, das NICHT folgt, wird nicht bewiesen
# ======================================================================
def scenario_nonentailment():
    def A(p, *a): return Atom(p, a)
    kb = [A("Sunny"), Implies(A("Sunny"), A("Warm"))]
    goal = A("Raining")                    # folgt nicht
    ok, steps = prove(kb, goal, max_steps=500, verbose=False)
    print(f"\n[D] Folgt 'Raining' aus (Sunny, Sunny⇒Warm)?  ->  "
          f"{'JA (FEHLER!)' if ok else 'nein, korrekt nicht beweisbar'} ({steps} Schritte)")
    return not ok

# ======================================================================
#  Occurs-Check-Demonstration
# ======================================================================
def demo_occurs_check():
    x = Var("x")
    fx = Fn("f", (x,))
    s = unify(x, fx, {})
    print("\n[Occurs-Check] unify(x, f(x)) =", s,
          "->", "korrekt abgelehnt (unendlicher Term)" if s is None else "FEHLER")
    return s is None

# ======================================================================
if __name__ == "__main__":
    print("=" * 68)
    print(" Resolutions-Theorembeweiser — Demonstration")
    print("=" * 68)
    r = []
    r.append(scenario_west(verbose=True))
    r.append(scenario_ancestor(verbose=True))
    r.append(scenario_propositional(verbose=True))
    r.append(scenario_nonentailment())
    r.append(demo_occurs_check())
    print("\n" + "=" * 68)
    print(" Ergebnis:", "ALLE Szenarien wie erwartet." if all(r) else "FEHLER in einem Szenario!")
    print("=" * 68)
