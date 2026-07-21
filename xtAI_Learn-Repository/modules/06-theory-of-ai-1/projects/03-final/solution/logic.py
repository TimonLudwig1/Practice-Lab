"""A resolution theorem prover for first-order predicate logic.

The pipeline (parts 3 and 4 of the script):
    formula --eliminate implications--> --NNF--> --standardize apart-->
            --skolemize--> --drop universals--> --distribute-->
            a set of clauses --resolution with unification--> a refutation.

The proof strategy: refutation. To show KB |= alpha, one adds ¬alpha to the KB
and derives the empty clause.
"""
from dataclasses import dataclass
from itertools import count
from typing import Tuple

# ======================================================================
#  Terms:  Var (a variable)  and  Fn (a function/constant symbol)
# ======================================================================
@dataclass(frozen=True)
class Var:
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class Fn:
    name: str
    args: Tuple = ()
    def __repr__(self):
        return self.name if not self.args else f"{self.name}({','.join(map(repr,self.args))})"

def Const(name):            # a constant = a 0-ary function
    return Fn(name, ())

# ======================================================================
#  Formulas (AST)
# ======================================================================
@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple = ()
    def __repr__(self):
        return self.pred if not self.args else f"{self.pred}({','.join(map(repr,self.args))})"

@dataclass(frozen=True)
class Not:   f: object
@dataclass(frozen=True)
class And:   a: object; b: object
@dataclass(frozen=True)
class Or:    a: object; b: object
@dataclass(frozen=True)
class Implies: a: object; b: object
@dataclass(frozen=True)
class Iff:   a: object; b: object
@dataclass(frozen=True)
class ForAll: x: str; f: object
@dataclass(frozen=True)
class Exists: x: str; f: object

# ======================================================================
#  Literals and clauses
# ======================================================================
@dataclass(frozen=True)
class Literal:
    neg: bool
    atom: Atom
    def __repr__(self):
        return ("¬" if self.neg else "") + repr(self.atom)

# A clause = frozenset[Literal];  the empty clause = frozenset()

# ======================================================================
#  CNF conversion
# ======================================================================
def eliminate_implications(f):
    if isinstance(f, Atom):    return f
    if isinstance(f, Not):     return Not(eliminate_implications(f.f))
    if isinstance(f, And):     return And(eliminate_implications(f.a), eliminate_implications(f.b))
    if isinstance(f, Or):      return Or(eliminate_implications(f.a), eliminate_implications(f.b))
    if isinstance(f, Implies): return Or(Not(eliminate_implications(f.a)), eliminate_implications(f.b))
    if isinstance(f, Iff):
        a, b = eliminate_implications(f.a), eliminate_implications(f.b)
        return And(Or(Not(a), b), Or(Not(b), a))
    if isinstance(f, ForAll):  return ForAll(f.x, eliminate_implications(f.f))
    if isinstance(f, Exists):  return Exists(f.x, eliminate_implications(f.f))
    raise TypeError(f)

def to_nnf(f):
    """Move negations inwards (assumes implications have been eliminated)."""
    if isinstance(f, Atom):   return f
    if isinstance(f, And):    return And(to_nnf(f.a), to_nnf(f.b))
    if isinstance(f, Or):     return Or(to_nnf(f.a), to_nnf(f.b))
    if isinstance(f, ForAll): return ForAll(f.x, to_nnf(f.f))
    if isinstance(f, Exists): return Exists(f.x, to_nnf(f.f))
    if isinstance(f, Not):
        g = f.f
        if isinstance(g, Atom):   return f
        if isinstance(g, Not):    return to_nnf(g.f)
        if isinstance(g, And):    return Or(to_nnf(Not(g.a)), to_nnf(Not(g.b)))
        if isinstance(g, Or):     return And(to_nnf(Not(g.a)), to_nnf(Not(g.b)))
        if isinstance(g, ForAll): return Exists(g.x, to_nnf(Not(g.f)))
        if isinstance(g, Exists): return ForAll(g.x, to_nnf(Not(g.f)))
    raise TypeError(f)

def _rename_term(t, env):
    if isinstance(t, Var): return Var(env.get(t.name, t.name))
    if isinstance(t, Fn):  return Fn(t.name, tuple(_rename_term(a, env) for a in t.args))
    raise TypeError(t)

def _rename_atom(a, env):
    return Atom(a.pred, tuple(_rename_term(t, env) for t in a.args))

_std_counter = count(1)
def standardize_apart(f, env=None):
    """Rename every quantified variable so that it is globally unique."""
    env = env or {}
    if isinstance(f, Atom): return _rename_atom(f, env)
    if isinstance(f, Not):  return Not(standardize_apart(f.f, env))
    if isinstance(f, And):  return And(standardize_apart(f.a, env), standardize_apart(f.b, env))
    if isinstance(f, Or):   return Or(standardize_apart(f.a, env), standardize_apart(f.b, env))
    if isinstance(f, (ForAll, Exists)):
        fresh = f"{f.x}_{next(_std_counter)}"
        env2 = dict(env); env2[f.x] = fresh
        body = standardize_apart(f.f, env2)
        return (ForAll if isinstance(f, ForAll) else Exists)(fresh, body)
    raise TypeError(f)

def _subst_term(t, sub):
    if isinstance(t, Var): return sub.get(t.name, t)
    if isinstance(t, Fn):  return Fn(t.name, tuple(_subst_term(a, sub) for a in t.args))
    raise TypeError(t)

_sk_counter = count(1)
def skolemize(f, univ=(), sub=None):
    """Replace existential quantifiers by Skolem constants/functions.
    `univ` = the surrounding universally quantified variables (as terms)."""
    sub = sub or {}
    if isinstance(f, ForAll):
        return ForAll(f.x, skolemize(f.f, univ + (Var(f.x),), sub))
    if isinstance(f, Exists):
        sk = Fn(f"Sk{next(_sk_counter)}", tuple(univ))   # the witness depends on univ
        sub2 = dict(sub); sub2[f.x] = sk
        return skolemize(f.f, univ, sub2)
    if isinstance(f, And): return And(skolemize(f.a, univ, sub), skolemize(f.b, univ, sub))
    if isinstance(f, Or):  return Or(skolemize(f.a, univ, sub), skolemize(f.b, univ, sub))
    if isinstance(f, Not): return Not(skolemize(f.f, univ, sub))   # only Not(Atom) in NNF
    if isinstance(f, Atom): return Atom(f.pred, tuple(_subst_term(t, sub) for t in f.args))
    raise TypeError(f)

def drop_universals(f):
    while isinstance(f, ForAll):
        f = f.f
    return f

def distribute(f):
    """Distribute Or over And -> the CNF structure."""
    if isinstance(f, And): return And(distribute(f.a), distribute(f.b))
    if isinstance(f, Or):
        a, b = distribute(f.a), distribute(f.b)
        if isinstance(a, And): return And(distribute(Or(a.a, b)), distribute(Or(a.b, b)))
        if isinstance(b, And): return And(distribute(Or(a, b.a)), distribute(Or(a, b.b)))
        return Or(a, b)
    return f

def _collect_clauses(f, acc):
    if isinstance(f, And):
        _collect_clauses(f.a, acc); _collect_clauses(f.b, acc)
    else:
        lits = set(); _collect_literals(f, lits)
        acc.append(frozenset(lits))

def _collect_literals(f, acc):
    if isinstance(f, Or):
        _collect_literals(f.a, acc); _collect_literals(f.b, acc)
    elif isinstance(f, Not):
        acc.add(Literal(True, f.f))
    elif isinstance(f, Atom):
        acc.add(Literal(False, f))
    else:
        raise TypeError(f"unexpected inside a clause: {f}")

def to_clauses(f):
    """The complete conversion of a (closed) formula into clauses."""
    f = eliminate_implications(f)
    f = to_nnf(f)
    f = standardize_apart(f)
    f = skolemize(f)
    f = drop_universals(f)
    f = distribute(f)
    acc = []
    _collect_clauses(f, acc)
    return acc

# ======================================================================
#  Unification (with the occurs check)
# ======================================================================
def occurs(vname, t, sub):
    if isinstance(t, Var):
        if t.name == vname: return True
        if t.name in sub:   return occurs(vname, sub[t.name], sub)
        return False
    if isinstance(t, Fn):
        return any(occurs(vname, a, sub) for a in t.args)
    return False

def unify(x, y, sub):
    if sub is None: return None
    if x == y:      return sub
    if isinstance(x, Var): return unify_var(x, y, sub)
    if isinstance(y, Var): return unify_var(y, x, sub)
    if isinstance(x, Fn) and isinstance(y, Fn):
        if x.name != y.name or len(x.args) != len(y.args): return None
        for a, b in zip(x.args, y.args):
            sub = unify(a, b, sub)
            if sub is None: return None
        return sub
    return None

def unify_var(var, x, sub):
    if var.name in sub:              return unify(sub[var.name], x, sub)
    if isinstance(x, Var) and x.name in sub: return unify(var, sub[x.name], sub)
    if occurs(var.name, x, sub):     return None          # the occurs check!
    s2 = dict(sub); s2[var.name] = x
    return s2

def unify_atoms(a1, a2, sub):
    if a1.pred != a2.pred or len(a1.args) != len(a2.args): return None
    for t1, t2 in zip(a1.args, a2.args):
        sub = unify(t1, t2, sub)
        if sub is None: return None
    return sub

# ======================================================================
#  Substitution on literals/clauses
# ======================================================================
def subst_literal(lit, sub):
    return Literal(lit.neg, Atom(lit.atom.pred,
                                 tuple(_subst_term(t, sub) for t in lit.atom.args)))

def subst_clause(clause, sub):
    return frozenset(subst_literal(l, sub) for l in clause)

_ren_counter = count(1)
def rename_clause(clause):
    """Rename all variables of the clause freshly (standardizing apart for resolution)."""
    names = {t.name for lit in clause for t in _vars_in_atom(lit.atom)}
    sub = {n: Var(f"{n}#{next(_ren_counter)}") for n in names}
    return subst_clause(clause, sub)

def _vars_in_term(t):
    if isinstance(t, Var): yield t
    elif isinstance(t, Fn):
        for a in t.args: yield from _vars_in_term(a)

def _vars_in_atom(a):
    for t in a.args: yield from _vars_in_term(t)

# ======================================================================
#  Resolution + factoring
# ======================================================================
def resolve(c1, c2):
    """All resolvents of two clauses (with the variables renamed freshly first)."""
    c1, c2 = rename_clause(c1), rename_clause(c2)
    resolvents = []
    for L1 in c1:
        for L2 in c2:
            if L1.neg == L2.neg:
                continue                       # not complementary
            sub = unify_atoms(L1.atom, L2.atom, {})
            if sub is None:
                continue
            new = (c1 - {L1}) | (c2 - {L2})
            resolvents.append(subst_clause(new, sub))
    return resolvents

def factors(clause):
    """Factors: unify two literals of the same name and polarity."""
    clause = rename_clause(clause)
    result = []
    lits = list(clause)
    for i in range(len(lits)):
        for j in range(i + 1, len(lits)):
            L1, L2 = lits[i], lits[j]
            if L1.neg != L2.neg:
                continue
            sub = unify_atoms(L1.atom, L2.atom, {})
            if sub is not None:
                result.append(subst_clause(clause, sub))
    return result

# ======================================================================
#  The main refutation loop
# ======================================================================
EMPTY = frozenset()

def prove(kb_formulas, goal, max_steps=100000, verbose=False):
    """Shows kb |= goal by refutation with the set-of-support strategy
    (a given-clause loop). Returns (proved: bool, steps).

    The idea: only clauses derived from the (negated) goal enter the 'set of
    support' (SOS), and they are used one after another as the 'given clause'
    to be resolved against ALL known clauses. That is refutation complete as
    long as the KB without the goal is satisfiable, and it avoids the many
    useless KB-with-KB resolutions of blind saturation."""
    kb_clauses = []
    for f in kb_formulas:
        kb_clauses.extend(to_clauses(f))
    goal_clauses = to_clauses(Not(goal))       # negate the goal
    kb_clauses = _dedup(kb_clauses)
    goal_clauses = _dedup(goal_clauses)

    parents = {}
    for c in kb_clauses + goal_clauses:
        parents.setdefault(c, None)

    known = set(kb_clauses) | set(goal_clauses)
    sos = list(goal_clauses)                    # the queue of given clauses
    step = 0

    def emit(res, p):
        nonlocal step
        res = frozenset(res)
        if res == EMPTY:
            parents[EMPTY] = p
            return True
        if not _is_tautology(res) and res not in known:
            known.add(res); parents[res] = p; sos.append(res)
        return False

    while sos and step < max_steps:
        given = sos.pop(0)
        for other in list(known):
            for res in resolve(given, other):
                step += 1
                if emit(res, (given, other)):
                    if verbose: _print_proof(EMPTY, parents)
                    return True, step
        for fc in factors(given):               # factoring of the given clause
            step += 1
            if emit(fc, (given, None)):
                if verbose: _print_proof(EMPTY, parents)
                return True, step
    return False, step

# ---- Helpers --------------------------------------------------------
def _normalize(clause):
    return frozenset(clause)

def _dedup(clauses):
    seen, out = set(), []
    for c in clauses:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def _is_tautology(clause):
    atoms_pos = {l.atom for l in clause if not l.neg}
    atoms_neg = {l.atom for l in clause if l.neg}
    return bool(atoms_pos & atoms_neg)

def clause_str(c):
    return "□ (empty)" if not c else "{" + ", ".join(sorted(map(repr, c))) + "}"

def _print_proof(target, parents):
    print("\n--- Refutation found (the derivation of the empty clause) ---")
    order, seen = [], set()
    def walk(c):
        if c in seen: return
        seen.add(c)
        p = parents.get(c)
        if p:
            if p[0] is not None: walk(p[0])
            if p[1] is not None: walk(p[1])
        order.append(c)
    walk(target)
    for c in order:
        p = parents.get(c)
        src = "" if not p else "   <= " + " , ".join(
            clause_str(x) for x in p if x is not None)
        print(" ", clause_str(c), src)
