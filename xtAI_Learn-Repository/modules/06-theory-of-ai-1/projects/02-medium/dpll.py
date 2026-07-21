"""DPLL — Erfuellbarkeits-Solver fuer Aussagenlogik in KNF.

DEINE AUFGABE: Fuelle die TODO-Funktionen. Die Kodierung und der aeussere
Einstieg sind vorgegeben.

Kodierung: Eine Variable ist eine positive ganze Zahl. Ein Literal ist +v
(positiv) oder -v (negiert). Eine Klausel ist eine Menge von Literalen
(Disjunktion), eine Formel eine Liste von Klauseln (Konjunktion).

Rueckgabe: ein Modell {var: bool} (SAT) oder None (UNSAT).
"""
import sys
sys.setrecursionlimit(100000)


def clause_value(clause, model):
    """Wahrheitswert einer Klausel unter (partiellem) model:
      - True  wenn IRGENDEIN Literal wahr ist,
      - False wenn ALLE Literale belegt und falsch sind,
      - None  sonst (mind. ein Literal unbelegt, keins wahr).
    Ein Literal `lit` ist wahr, wenn model[abs(lit)] == (lit > 0).
    """
    # TODO
    raise NotImplementedError


def find_unit_clause(clauses, model):
    """Suche eine noch OFFENE Klausel (clause_value == None) mit genau EINEM
    unbelegten Literal. Gib (var, wert) zurueck, den dieses Literal erzwingt
    (wert = lit > 0). Sonst (None, None)."""
    # TODO
    raise NotImplementedError


def find_pure_symbol(symbols, clauses, model):
    """Suche ein Symbol aus `symbols`, das in allen noch OFFENEN Klauseln nur
    mit EINER Polaritaet vorkommt. Gib (var, wert) zurueck, sonst (None, None).
    Tipp: sammle zuerst alle Literale offener Klauseln in einer Menge."""
    # TODO
    raise NotImplementedError


def dpll_satisfiable(clauses):
    """Einstieg (vorgegeben)."""
    clauses = [frozenset(c) for c in clauses]
    symbols = sorted({abs(l) for c in clauses for l in c})
    return _dpll(clauses, symbols, {})


def _dpll(clauses, symbols, model):
    """Rekursion mit den drei DPLL-Beschleunigern.

    Geruest:
      1) Fruehterminierung:
           - irgendeine Klausel False -> return None
           - alle Klauseln True       -> return dict(model)
      2) Unit Propagation: P,val = find_unit_clause(...); wenn P: setze und rekursiere
      3) Pure Literal:     P,val = find_pure_symbol(...);  wenn P: setze und rekursiere
      4) Verzweige ueber symbols[0] mit val in (True, False); erste erfolgreiche
         Rueckgabe gewinnt, sonst None.
    Beim Rekursieren jeweils model kopieren und P aus symbols entfernen.
    """
    # TODO
    raise NotImplementedError
