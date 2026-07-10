"""Inferenz in Bayes-Netzen: exakt (Aufzaehlung, Variable Elimination) und
approximativ (Likelihood Weighting)."""
import random


def normalize(dist):
    """dict {True: a, False: b} -> normalisiert auf Summe 1."""
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}


# ====================================================================
#  1) Inferenz durch Aufzaehlung
# ====================================================================
def enumeration_ask(X, e, bn):
    """P(X | e) durch Aufzaehlung ueber alle versteckten Variablen."""
    Q = {}
    for xi in (True, False):
        e2 = dict(e); e2[X] = xi
        Q[xi] = enumerate_all(bn.variables, e2, bn)
    return normalize(Q)


def enumerate_all(variables, e, bn):
    if not variables:
        return 1.0
    Y, rest = variables[0], variables[1:]
    node = bn.lookup[Y]
    if Y in e:
        return node.p(e[Y], e) * enumerate_all(rest, e, bn)
    total = 0.0
    for y in (True, False):
        e2 = dict(e); e2[Y] = y
        total += node.p(y, e2) * enumerate_all(rest, e2, bn)
    return total


# ====================================================================
#  2) Variable Elimination
# ====================================================================
class Factor:
    def __init__(self, variables, cpt):
        self.variables = variables            # Liste von Variablennamen
        self.cpt = cpt                        # dict {Tupel der Werte in var-Reihenfolge: Zahl}

    def get(self, event):
        return self.cpt[tuple(event[v] for v in self.variables)]

    def pointwise_product(self, other):
        """Punktweises Produkt zweier Faktoren (ueber Vereinigung der Variablen)."""
        variables = self.variables + [v for v in other.variables if v not in self.variables]
        cpt = {}
        for event in bool_events(variables):
            cpt[tuple(event[v] for v in variables)] = self.get(event) * other.get(event)
        return Factor(variables, cpt)

    def sum_out(self, var):
        """Variable var ausmarginalisieren (summing out)."""
        variables = [v for v in self.variables if v != var]
        cpt = {}
        for event in bool_events(variables):
            s = 0.0
            for val in (True, False):
                e2 = dict(event); e2[var] = val
                s += self.get(e2)
            cpt[tuple(event[v] for v in variables)] = s
        return Factor(variables, cpt)


def bool_events(variables):
    """Alle booleschen Belegungen einer Variablenliste (als dicts)."""
    if not variables:
        yield {}
        return
    X, rest = variables[0], variables[1:]
    for e in bool_events(rest):
        for x in (True, False):
            yield {**e, X: x}


def make_factor(var, e, bn):
    """Faktor fuer Knoten var, mit fixierter Evidenz e (Evidenzvariablen fallen weg)."""
    node = bn.lookup[var]
    variables = [v for v in [var] + node.parents if v not in e]
    cpt = {}
    for event in bool_events(variables):
        full = {**e, **event}
        cpt[tuple(event[v] for v in variables)] = node.p(full[var], full)
    return Factor(variables, cpt)


def elimination_ask(X, e, bn):
    """P(X | e) mit Variable Elimination."""
    factors = []
    for var in reversed(bn.variables):
        factors.append(make_factor(var, e, bn))
        if var != X and var not in e:                    # versteckte Variable
            factors = sum_out_var(var, factors)
    # verbleibende Faktoren multiplizieren -> Faktor ueber {X}
    result = factors[0]
    for f in factors[1:]:
        result = result.pointwise_product(f)
    dist = {val: result.get({X: val}) for val in (True, False)}
    return normalize(dist)


def sum_out_var(var, factors):
    contains, rest = [], []
    for f in factors:
        (contains if var in f.variables else rest).append(f)
    if contains:
        prod = contains[0]
        for f in contains[1:]:
            prod = prod.pointwise_product(f)
        rest.append(prod.sum_out(var))
    return rest


# ====================================================================
#  3) Likelihood Weighting (approximativ)
# ====================================================================
def likelihood_weighting(X, e, bn, N=10000, seed=0):
    """Schaetzt P(X | e) aus N gewichteten Stichproben."""
    rng = random.Random(seed)
    W = {True: 0.0, False: 0.0}
    for _ in range(N):
        event, w = weighted_sample(bn, e, rng)
        W[event[X]] += w
    return normalize(W)


def weighted_sample(bn, e, rng):
    """Eine Stichprobe: Evidenz fixieren, Rest samplen, mit Evidenz-Likelihood gewichten."""
    w = 1.0
    event = dict(e)
    for var in bn.variables:
        node = bn.lookup[var]
        if var in e:
            w *= node.p(e[var], event)               # gewichten statt samplen
        else:
            event[var] = node.sample(event, rng)     # aus P(var | Eltern) ziehen
    return event, w
