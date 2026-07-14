"""Modellfreie Kontrolle auf dem Bestandsmanagement-MDP.

Vier Verfahren, alle epsilon-greedy und mit Aktionsmasken (nicht ueber Kapazitaet bestellen):
  - MonteCarloControl : every-visit MC-Kontrolle (lernt aus ganzen Episoden)
  - SARSA             : on-policy TD-Kontrolle
  - QLearning         : off-policy TD-Kontrolle
  - ExpectedSarsa     : Erwartungswert ueber die epsilon-greedy-Policy im Ziel

KEINES benutzt env.expected_reward / env.transition_probs — sie lernen NUR aus (s,a,r,s')-
Erfahrung. Das Modell steckt allein in der DP-Referenz (dp_reference.py).
"""
from __future__ import annotations
import numpy as np


class TabularAgent:
    """Gemeinsame Basis: Q-Tabelle, maskierte epsilon-greedy-Auswahl, greedy-Politik."""
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1, seed=0):
        self.env = env
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.Q = np.zeros((env.n_states, env.n_actions))
        self.rng = np.random.default_rng(seed)
        # unzulaessige Aktionen dauerhaft ausblenden
        self._masks = np.array([env.action_mask(i) for i in range(env.n_states)])
        self.Q[~self._masks] = -np.inf

    def _masked_q(self, s):
        return self.Q[s]      # unzulaessige Eintraege sind bereits -inf

    def select_action(self, s):
        if self.rng.random() < self.epsilon:
            valid = np.flatnonzero(self._masks[s])
            return int(self.rng.choice(valid))
        q = self._masked_q(s)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def greedy_policy(self):
        pol = np.argmax(np.where(self._masks, self.Q, -np.inf), axis=1)
        return pol.astype(int)

    def greedy_prob(self, s):
        """Wahrscheinlichkeitsvektor der epsilon-greedy-Policy in s (fuer Expected SARSA)."""
        q = self._masked_q(s)
        valid = self._masks[s]
        n_valid = valid.sum()
        probs = np.zeros(self.env.n_actions)
        probs[valid] = self.epsilon / n_valid
        best = np.flatnonzero(q == q.max())
        probs[best] += (1.0 - self.epsilon) / len(best)   # ties teilen die greedy-Masse
        return probs


class SARSA(TabularAgent):
    def train(self, n_steps=200_000):
        env = self.env
        s = env.reset()
        a = self.select_action(s)
        hist = np.zeros(n_steps)
        for t in range(n_steps):
            s2, r, _ = env.step(a)
            a2 = self.select_action(s2)
            target = r + self.gamma * self.Q[s2, a2]
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
            s, a = s2, a2
            hist[t] = r
        return hist


class QLearning(TabularAgent):
    def train(self, n_steps=200_000):
        env = self.env
        s = env.reset()
        hist = np.zeros(n_steps)
        for t in range(n_steps):
            a = self.select_action(s)
            s2, r, _ = env.step(a)
            target = r + self.gamma * self.Q[s2].max()   # -inf-Eintraege stoeren max nicht
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
            s = s2
            hist[t] = r
        return hist


class ExpectedSarsa(TabularAgent):
    def train(self, n_steps=200_000):
        env = self.env
        s = env.reset()
        hist = np.zeros(n_steps)
        for t in range(n_steps):
            a = self.select_action(s)
            s2, r, _ = env.step(a)
            probs = self.greedy_prob(s2)
            exp_q = float(probs[self._masks[s2]] @ self.Q[s2, self._masks[s2]])
            target = r + self.gamma * exp_q
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
            s = s2
            hist[t] = r
        return hist


class MonteCarloControl(TabularAgent):
    """every-visit MC-Kontrolle mit inkrementellem Mittelwert. Die fortlaufende Aufgabe
    wird in Episoden fester Laenge geschnitten (episode_len Schritte)."""
    def __init__(self, env, gamma=0.95, epsilon=0.1, seed=0, episode_len=200):
        super().__init__(env, alpha=0.0, gamma=gamma, epsilon=epsilon, seed=seed)
        self.episode_len = episode_len
        self.N = np.zeros((env.n_states, env.n_actions))

    def train(self, n_steps=200_000):
        env = self.env
        n_episodes = n_steps // self.episode_len
        hist = []
        for _ in range(n_episodes):
            s = env.reset()
            traj = []
            for _ in range(self.episode_len):
                a = self.select_action(s)
                s2, r, _ = env.step(a)
                traj.append((s, a, r))
                hist.append(r)
                s = s2
            G = 0.0
            for (s, a, r) in reversed(traj):        # Returns rueckwaerts
                G = r + self.gamma * G
                self.N[s, a] += 1
                self.Q[s, a] += (G - self.Q[s, a]) / self.N[s, a]
        return np.array(hist)


AGENTS = {"MC": MonteCarloControl, "SARSA": SARSA,
          "Q-Learning": QLearning, "Expected SARSA": ExpectedSarsa}
