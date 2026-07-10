"""Datenstruktur fuer diskrete (boolesche) Bayes-Netze + zwei Beispielnetze.

Konvention: Variablen sind boolesch (True/False). Ein 'event' ist ein dict
{Variablenname: bool}. Die CPT eines Knotens gibt P(Knoten=True | Eltern) an.
"""
from dataclasses import dataclass


class BayesNode:
    def __init__(self, name, parents, cpt):
        """parents: Liste von Elternnamen (Reihenfolge zaehlt!).
        cpt: - ohne Eltern: eine Zahl P(name=True)
             - mit Eltern:  dict {(elternwerte-Tupel): P(name=True)}"""
        if isinstance(parents, str):
            parents = parents.split()
        self.name = name
        self.parents = parents
        if isinstance(cpt, (int, float)):
            cpt = {(): cpt}
        elif cpt and isinstance(next(iter(cpt)), bool):
            cpt = {(k,): v for k, v in cpt.items()}
        self.cpt = cpt
        self.children = []

    def p(self, value, event):
        """P(name=value | Eltern wie in event)."""
        ptrue = self.cpt[tuple(event[p] for p in self.parents)]
        return ptrue if value else 1 - ptrue

    def sample(self, event, rng):
        """Ziehe einen Wert fuer name gegeben die Elternwerte in event."""
        return rng.random() < self.cpt[tuple(event[p] for p in self.parents)]

    def __repr__(self):
        return f"BayesNode({self.name})"


class BayesNet:
    def __init__(self, node_specs):
        self.variables = []                 # topologisch geordnet
        self.lookup = {}
        for spec in node_specs:
            self.add(BayesNode(*spec))

    def add(self, node):
        assert node.name not in self.lookup
        assert all(p in self.lookup for p in node.parents), \
            f"Eltern von {node.name} muessen vorher definiert sein (topologisch!)"
        self.lookup[node.name] = node
        self.variables.append(node.name)
        for p in node.parents:
            self.lookup[p].children.append(node)

    def variable_node(self, name):
        return self.lookup[name]


# --------------------------------------------------------------- Beispielnetze
def alarm_net():
    """Pearls Alarm-Netz.  B,E -> A -> J,M."""
    T, F = True, False
    return BayesNet([
        ("Burglary", "", 0.001),
        ("Earthquake", "", 0.002),
        ("Alarm", "Burglary Earthquake",
            {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
        ("JohnCalls", "Alarm", {T: 0.90, F: 0.05}),
        ("MaryCalls", "Alarm", {T: 0.70, F: 0.01}),
    ])


def diagnosis_net():
    """Kleines medizinisches Diagnosenetz.
    Rauchen/Umwelt -> Bronchitis/Krebs -> Symptome (Husten, Roentgen, Dyspnoe)."""
    T, F = True, False
    return BayesNet([
        ("Smoker", "", 0.30),
        ("Pollution", "", 0.10),
        ("Cancer", "Smoker Pollution",
            {(T, T): 0.05, (T, F): 0.03, (F, T): 0.02, (F, F): 0.001}),
        ("Bronchitis", "Smoker", {T: 0.25, F: 0.05}),
        ("XRay", "Cancer", {T: 0.90, F: 0.20}),           # positiver Befund
        ("Dyspnoea", "Cancer Bronchitis",
            {(T, T): 0.90, (T, F): 0.70, (F, T): 0.80, (F, F): 0.10}),
    ])
