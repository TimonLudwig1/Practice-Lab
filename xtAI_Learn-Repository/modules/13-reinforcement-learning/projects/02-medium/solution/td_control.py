"""Tabellarische TD-Kontrolle: SARSA (on-policy) und Q-Learning (off-policy).

Beide teilen sich alles bis auf das Bootstrapping-Ziel im Update:
  SARSA:      Ziel = r + gamma * Q[s', a']        (a' ~ aktuelle epsilon-greedy-Policy)
  Q-Learning: Ziel = r + gamma * max_a Q[s', a]   (greedy, unabhaengig von a')

Das ist der ganze Unterschied zwischen "lerne den Wert der Policy, die ich AUSFUEHRE"
(on-policy) und "lerne die OPTIMALE Policy, waehrend ich explorierend handle" (off-policy).
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

    # ---- epsilon-greedy Auswahl (mit fairem Tie-Breaking) ----
    def select_action(self, s: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def greedy_action(self, s: int) -> int:
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    # ---- das eine TD-Update, das beide Algorithmen unterscheidet ----
    def update(self, s, a, r, s_next, a_next, done):
        if done:
            target = r
        elif self.algo == "sarsa":
            target = r + self.gamma * self.Q[s_next, a_next]
        else:  # qlearning
            target = r + self.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)


def train(env, agent, n_episodes=500, max_steps=1000):
    """Trainiere den Agenten. Rueckgabe: Array der Belohnungssummen je Episode.

    Wichtig fuer SARSA: die naechste Aktion a' muss VOR dem Update aus s' gezogen und
    dann im naechsten Schritt tatsaechlich ausgefuehrt werden (daher der (s,a)->(s',a')-
    Uebergang). Q-Learning ignoriert a' im Update, laeuft aber mit derselben Schleife.
    """
    returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s = env.reset()
        a = agent.select_action(s)
        total = 0.0
        for _ in range(max_steps):
            s_next, r, done = env.step(a)
            a_next = agent.select_action(s_next)
            agent.update(s, a, r, s_next, a_next, done)
            s, a = s_next, a_next
            total += r
            if done:
                break
        returns[ep] = total
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
