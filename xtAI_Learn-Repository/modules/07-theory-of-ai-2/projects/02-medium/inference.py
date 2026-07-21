"""Inferenz in Bayes-Netzen: exakt (Aufzaehlung, Variable Elimination) und
approximativ (Likelihood Weighting).

DEINE AUFGABE: Fuelle die TODO-Funktionen. Vorgegeben ist die Faktor-Plumbing
(make_factor, bool_events, elimination_ask-Orchestrierung) — DU implementierst die
konzeptionellen Kerne: die Aufzaehlungs-Rekursion, die zwei Faktor-Operationen
(punktweises Produkt, summing out) und das Likelihood Weighting.
"""
import random


def normalize(dist):
    """dict {True: a, False: b} -> normalisiert auf Summe 1. (vorgegeben)"""
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}


# ====================================================================
#  AUFGABE 1 — Inferenz durch Aufzaehlung
# ====================================================================
def enumeration_ask(X, e, bn):
    """P(X | e): fuer xi in (True, False) enumerate_all(bn.variables, e∪{X:xi}, bn),
    dann normalisieren. (Geruest vorgegeben.)"""
    Q = {}
    for xi in (True, False):
        e2 = dict(e); e2[X] = xi
        Q[xi] = enumerate_all(bn.variables, e2, bn)
    return normalize(Q)


def enumerate_all(variables, e, bn):
    """Rekursive Aufzaehlung ueber die (topologisch geordneten) variables.
    - variables leer -> 1.0
    - erste Variable Y: node = bn.lookup[Y]
        * Y in e     -> node.p(e[Y], e) * enumerate_all(rest, e, bn)
        * Y versteckt -> sum ueber y in (True,False):
                          node.p(y, e∪{Y:y}) * enumerate_all(rest, e∪{Y:y}, bn)
    """
    # TODO
    raise NotImplementedError


# ====================================================================
#  AUFGABE 2 — Variable Elimination (Faktor-Operationen)
# ====================================================================
class Factor:
    def __init__(self, variables, cpt):
        self.variables = variables            # Liste von Variablennamen
        self.cpt = cpt                        # dict {Werte-Tupel in var-Reihenfolge: Zahl}

    def get(self, event):
        return self.cpt[tuple(event[v] for v in self.variables)]

    def pointwise_product(self, other):
        """Punktweises Produkt (Skript 2.5): Ergebnis-Variablen = Vereinigung
        beider Variablenlisten; fuer jede Belegung das Produkt der beiden Werte.
        Tipp: iteriere ueber bool_events(vereinigte_variablen) und nutze self.get / other.get.
        """
        # TODO
        raise NotImplementedError

    def sum_out(self, var):
        """Variable var ausmarginalisieren: neue Variablen = ohne var; fuer jede
        Belegung der restlichen Variablen die Summe ueber var in (True, False).
        """
        # TODO
        raise NotImplementedError


def bool_events(variables):
    """Alle booleschen Belegungen einer Variablenliste (als dicts). (vorgegeben)"""
    if not variables:
        yield {}
        return
    X, rest = variables[0], variables[1:]
    for e in bool_events(rest):
        for x in (True, False):
            yield {**e, X: x}


def make_factor(var, e, bn):
    """Faktor fuer Knoten var mit fixierter Evidenz e. (vorgegeben)"""
    node = bn.lookup[var]
    variables = [v for v in [var] + node.parents if v not in e]
    cpt = {}
    for event in bool_events(variables):
        full = {**e, **event}
        cpt[tuple(event[v] for v in variables)] = node.p(full[var], full)
    return Factor(variables, cpt)


def elimination_ask(X, e, bn):
    """P(X | e) mit Variable Elimination. (Orchestrierung vorgegeben — sie ruft
    deine pointwise_product / sum_out auf.)"""
    factors = []
    for var in reversed(bn.variables):
        factors.append(make_factor(var, e, bn))
        if var != X and var not in e:
            factors = sum_out_var(var, factors)
    result = factors[0]
    for f in factors[1:]:
        result = result.pointwise_product(f)
    dist = {val: result.get({X: val}) for val in (True, False)}
    return normalize(dist)


def sum_out_var(var, factors):
    """Alle Faktoren mit var multiplizieren und var heraussummieren. (vorgegeben)"""
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
#  AUFGABE 3 — Likelihood Weighting
# ====================================================================
def likelihood_weighting(X, e, bn, N=10000, seed=0):
    """Schaetzt P(X | e) aus N gewichteten Stichproben.
    Geruest: rng = random.Random(seed); W = {True:0.0, False:0.0};
    N-mal (event, w) = weighted_sample(...); W[event[X]] += w; normalize(W)."""
    # TODO
    raise NotImplementedError


def weighted_sample(bn, e, rng):
    """Eine gewichtete Stichprobe (Skript 2.6):
    - event = dict(e); w = 1.0
    - fuer var in bn.variables (topologisch): node = bn.lookup[var]
        * var in e   -> w *= node.p(e[var], event)     (gewichten, nicht samplen)
        * sonst      -> event[var] = node.sample(event, rng)
    - gib (event, w) zurueck.
    """
    # TODO
    raise NotImplementedError
