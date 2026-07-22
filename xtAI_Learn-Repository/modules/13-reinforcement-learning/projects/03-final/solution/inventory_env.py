"""Stochastic inventory control — an MDP.

A retailer holds a warehouse of capacity M. Each period:
  1. It observes the inventory i (the state, 0..M).
  2. It orders a units (the action), which arrive IMMEDIATELY; inventory -> x = i + a (<= M).
  3. A stochastic demand D ~ Poisson(lam) occurs.
  4. Sales = min(x, D). Unsatisfied demand is LOST (lost sales).
     New inventory i' = max(x - D, 0).
Cost (negative reward) per period:
     - fixed ordering cost K, if a > 0            (makes the optimal policy an (s,S) one)
     - variable ordering cost c per unit
     - holding cost h per unit of ending inventory
     - penalty cost p per unit of lost demand (shortage)
  reward = -(K*[a>0] + c*a + h*max(x-D,0) + p*max(D-x,0))

This is a classical operations-research problem. Because the model (Poisson demand + costs)
is KNOWN here, one can compute the exact optimal policy via value iteration (see
dp_reference.py) — that serves as the reference for the model-free learners, which must NOT
use the model.
"""
from __future__ import annotations
import numpy as np
from math import exp, lgamma


def poisson_pmf(lam: float, d_max: int) -> np.ndarray:
    """PMF of Poisson(lam) on 0..d_max, with the remaining tail bundled onto d_max (so that
    the probabilities sum exactly to 1)."""
    ks = np.arange(0, d_max + 1)
    logpmf = ks * np.log(lam) - lam - np.array([lgamma(k + 1) for k in ks])
    pmf = np.exp(logpmf)
    pmf[-1] += 1.0 - pmf.sum()      # put the tail P(D > d_max) onto d_max
    return pmf


class InventoryEnv:
    def __init__(self, capacity=20, lam=8.0, c=2.0, K=10.0, h=1.0, p=6.0,
                 d_max=30, seed=0):
        self.M = capacity
        self.lam = lam
        self.c, self.K, self.h, self.p = c, K, h, p
        self.d_max = d_max
        self.n_states = capacity + 1
        self.n_actions = capacity + 1          # order quantities 0..M (capped by the capacity)
        self.pmf = poisson_pmf(lam, d_max)
        self.rng = np.random.default_rng(seed)
        self._state = 0

    # ---- which actions are admissible in state i? (do not order above the capacity) ----
    def valid_actions(self, i: int) -> np.ndarray:
        return np.arange(0, self.M - i + 1)

    def action_mask(self, i: int) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[: self.M - i + 1] = True
        return mask

    # ---- deterministic cost function given a realized demand d ----
    def _reward(self, i: int, a: int, d: int) -> float:
        x = i + a
        ending = max(x - d, 0)
        lost = max(d - x, 0)
        cost = self.K * (a > 0) + self.c * a + self.h * ending + self.p * lost
        return -cost

    # ---- gym-like API (usable model-free) ----
    def reset(self, state=None) -> int:
        self._state = self.rng.integers(self.n_states) if state is None else int(state)
        return self._state

    def step(self, a: int):
        i = self._state
        assert 0 <= a <= self.M - i, f"inadmissible action a={a} in state i={i}"
        d = int(self.rng.choice(len(self.pmf), p=self.pmf))
        r = self._reward(i, a, d)
        self._state = max(i + a - d, 0)
        return self._state, r, False        # a continuing task: never 'done'

    # ---- the explicit model P, R (ONLY for the DP reference, not for learning!) ----
    def expected_reward(self, i: int, a: int) -> float:
        x = i + a
        d = np.arange(len(self.pmf))
        ending = np.maximum(x - d, 0)
        lost = np.maximum(d - x, 0)
        stage = self.h * ending + self.p * lost
        return -(self.K * (a > 0) + self.c * a + float(self.pmf @ stage))

    def transition_probs(self, i: int, a: int) -> np.ndarray:
        """P(i' | i, a) as a vector of length n_states."""
        x = i + a
        probs = np.zeros(self.n_states)
        d = np.arange(len(self.pmf))
        nxt = np.maximum(x - d, 0)           # the next state per demand d
        np.add.at(probs, nxt, self.pmf)
        return probs
