"""Tabular TD control: SARSA (on-policy) and Q-learning (off-policy).

Both share everything except the bootstrapping target in the update:
  SARSA:      target = r + gamma * Q[s', a']        (a' ~ the current epsilon-greedy policy)
  Q-learning: target = r + gamma * max_a Q[s', a]   (greedy, independent of a')

That is the whole difference between "learn the value of the policy I EXECUTE" (on-policy)
and "learn the OPTIMAL policy while acting with exploration" (off-policy).
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

    # ---- epsilon-greedy selection (with fair tie-breaking) ----
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

    # ---- the one TD update that distinguishes the two algorithms ----
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
    """Train the agent. Returns: an array of the reward sums per episode.

    Important for SARSA: the next action a' must be drawn from s' BEFORE the update and then
    actually executed in the next step (hence the (s,a)->(s',a') transition). Q-learning
    ignores a' in the update but runs with the same loop.
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
    """Execute the greedy policy once (without exploration). Returns: the return."""
    s = env.reset()
    total = 0.0
    for _ in range(max_steps):
        a = int(np.argmax(agent.Q[s]))
        s, r, done = env.step(a)
        total += r
        if done:
            break
    return total
