"""Tabellarische TD-Kontrolle: SARSA (on-policy) und Q-Learning (off-policy).

>>> DEINE AUFGABE <<<  Fuelle die drei mit TODO markierten Stellen aus:
   1) select_action  — epsilon-greedy Auswahl
   2) update         — das eine TD-Update, das SARSA von Q-Learning unterscheidet
   3) train          — die Trainingsschleife (Achtung SARSA: a' vor dem Update ziehen!)

Der EINZIGE Unterschied der Algorithmen steckt im Bootstrapping-Ziel:
  SARSA:      Ziel = r + gamma * Q[s', a']        (a' ~ aktuelle epsilon-greedy-Policy)
  Q-Learning: Ziel = r + gamma * max_a Q[s', a]   (greedy, unabhaengig von a')
Am terminalen Uebergang (done=True) gibt es keinen Folgewert: Ziel = r.

Die Musterloesung liegt in loesung/td_control.py — erst selbst versuchen!
"""
from __future__ import annotations
import numpy as np


class TDAgent:
    def __init__(self, n_states, n_actions, algo="sarsa",
                 alpha=0.5, gamma=1.0, epsilon=0.1, seed=0):
        assert algo in ("sarsa", "qlearning")
        self.n_states = n_states
        self.n_actions = n_actions
        self.algo = algo
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng(seed)

    def select_action(self, s: int) -> int:
        # TODO: mit Wkt. self.epsilon eine zufaellige Aktion
        #       (self.rng.integers(self.n_actions)), sonst argmax_a Q[s,a].
        #   Tipp fuer faires Tie-Breaking bei Gleichstand:
        #       best = np.flatnonzero(q == q.max()); self.rng.choice(best)
        raise NotImplementedError

    def update(self, s, a, r, s_next, a_next, done):
        # TODO: berechne das Ziel (siehe Modulkopf) und mache das Update
        #       Q[s,a] += alpha * (Ziel - Q[s,a]).
        #   - done:        Ziel = r
        #   - sarsa:       Ziel = r + gamma * Q[s_next, a_next]
        #   - qlearning:   Ziel = r + gamma * max(Q[s_next])
        raise NotImplementedError

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)


def train(env, agent, n_episodes=500, max_steps=1000):
    """Trainiere den Agenten. Rueckgabe: Array (Laenge n_episodes) der Belohnungssummen.

    Ablauf einer Episode (fuer SARSA korrekt, fuer Q-Learning ebenfalls gueltig):
      s = env.reset(); a = agent.select_action(s)
      wiederhole bis done oder max_steps:
        s', r, done = env.step(a)
        a' = agent.select_action(s')          # a' VOR dem Update ziehen
        agent.update(s, a, r, s', a', done)
        s, a = s', a'                          # a' wird naechster Schritt ausgefuehrt
        Belohnungssumme aktualisieren
    """
    returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        # TODO: eine Episode nach obigem Ablauf spielen und returns[ep] setzen.
        raise NotImplementedError
    return returns


def rollout_greedy(env, agent, max_steps=1000):
    """Fuehre die greedy-Policy einmal aus (ohne Exploration). Rueckgabe: Ertrag."""
    s = env.reset()
    total = 0.0
    for _ in range(max_steps):
        a = int(np.argmax(agent.Q[s]))
        s, r, done = env.step(a)
        total += r
        if done:
            break
    return total
