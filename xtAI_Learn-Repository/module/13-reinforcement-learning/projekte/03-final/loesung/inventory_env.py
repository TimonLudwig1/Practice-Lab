"""Stochastisches Bestandsmanagement (stochastic inventory control) — ein MDP.

Ein Haendler haelt ein Lager der Kapazitaet M. Zu jeder Periode:
  1. Er beobachtet den Bestand i (Zustand, 0..M).
  2. Er bestellt a Einheiten (Aktion), die SOFORT eintreffen; Bestand -> x = i + a (<= M).
  3. Eine stochastische Nachfrage D ~ Poisson(lam) tritt ein.
  4. Verkauf = min(x, D). Nicht befriedigte Nachfrage geht VERLOREN (lost sales).
     Neuer Bestand i' = max(x - D, 0).
Kosten (negative Belohnung) je Periode:
     - Bestellfixkosten K, falls a > 0            (macht die optimale Politik zu (s,S))
     - variable Bestellkosten c pro Einheit
     - Lagerkosten h pro Einheit Endbestand
     - Strafkosten p pro Einheit verlorener Nachfrage (Fehlmenge)
  reward = -(K*[a>0] + c*a + h*max(x-D,0) + p*max(D-x,0))

Das ist ein klassisches Operations-Research-Problem. Weil das Modell (Poisson-Nachfrage +
Kosten) hier BEKANNT ist, kann man die exakte optimale Politik per Value Iteration berechnen
(siehe dp_reference.py) — die dient als Referenz fuer die modellfreien Lerner, die das Modell
NICHT benutzen duerfen.
"""
from __future__ import annotations
import numpy as np
from math import exp, lgamma


def poisson_pmf(lam: float, d_max: int) -> np.ndarray:
    """PMF von Poisson(lam) auf 0..d_max, wobei der Rest-Tail auf d_max gebuendelt wird
    (damit sich die Wahrscheinlichkeiten exakt zu 1 summieren)."""
    ks = np.arange(0, d_max + 1)
    logpmf = ks * np.log(lam) - lam - np.array([lgamma(k + 1) for k in ks])
    pmf = np.exp(logpmf)
    pmf[-1] += 1.0 - pmf.sum()      # Tail P(D > d_max) auf d_max legen
    return pmf


class InventoryEnv:
    def __init__(self, capacity=20, lam=8.0, c=2.0, K=10.0, h=1.0, p=6.0,
                 d_max=30, seed=0):
        self.M = capacity
        self.lam = lam
        self.c, self.K, self.h, self.p = c, K, h, p
        self.d_max = d_max
        self.n_states = capacity + 1
        self.n_actions = capacity + 1          # Bestellmengen 0..M (durch Kapazitaet gedeckelt)
        self.pmf = poisson_pmf(lam, d_max)
        self.rng = np.random.default_rng(seed)
        self._state = 0

    # ---- welche Aktionen sind in Zustand i zulaessig? (nicht ueber Kapazitaet bestellen) ----
    def valid_actions(self, i: int) -> np.ndarray:
        return np.arange(0, self.M - i + 1)

    def action_mask(self, i: int) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[: self.M - i + 1] = True
        return mask

    # ---- deterministische Kostenfunktion gegeben realisierte Nachfrage d ----
    def _reward(self, i: int, a: int, d: int) -> float:
        x = i + a
        ending = max(x - d, 0)
        lost = max(d - x, 0)
        cost = self.K * (a > 0) + self.c * a + self.h * ending + self.p * lost
        return -cost

    # ---- Gym-artige API (modellfrei nutzbar) ----
    def reset(self, state=None) -> int:
        self._state = self.rng.integers(self.n_states) if state is None else int(state)
        return self._state

    def step(self, a: int):
        i = self._state
        assert 0 <= a <= self.M - i, f"unzulaessige Aktion a={a} in Zustand i={i}"
        d = int(self.rng.choice(len(self.pmf), p=self.pmf))
        r = self._reward(i, a, d)
        self._state = max(i + a - d, 0)
        return self._state, r, False        # fortlaufende Aufgabe: nie 'done'

    # ---- das explizite Modell P, R (NUR fuer die DP-Referenz, nicht fuers Lernen!) ----
    def expected_reward(self, i: int, a: int) -> float:
        x = i + a
        d = np.arange(len(self.pmf))
        ending = np.maximum(x - d, 0)
        lost = np.maximum(d - x, 0)
        stage = self.h * ending + self.p * lost
        return -(self.K * (a > 0) + self.c * a + float(self.pmf @ stage))

    def transition_probs(self, i: int, a: int) -> np.ndarray:
        """P(i' | i, a) als Vektor der Laenge n_states."""
        x = i + a
        probs = np.zeros(self.n_states)
        d = np.arange(len(self.pmf))
        nxt = np.maximum(x - d, 0)           # naechster Zustand je Nachfrage d
        np.add.at(probs, nxt, self.pmf)
        return probs
