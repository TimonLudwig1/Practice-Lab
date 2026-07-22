"""Tabular TD control: SARSA (on-policy) and Q-learning (off-policy).

>>> YOUR TASK <<<  Fill in the three places marked with TODO:
   1) select_action  — epsilon-greedy selection
   2) update         — the one TD update that distinguishes SARSA from Q-learning
   3) train          — the training loop (attention SARSA: draw a' before the update!)

The ONLY difference of the algorithms is in the bootstrapping target:
  SARSA:      target = r + gamma * Q[s', a']        (a' ~ the current epsilon-greedy policy)
  Q-learning: target = r + gamma * max_a Q[s', a]   (greedy, independent of a')
At the terminal transition (done=True) there is no successor value: target = r.

The reference solution is in solution/td_control.py — try it yourself first!
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
        # TODO: with prob. self.epsilon a random action
        #       (self.rng.integers(self.n_actions)), otherwise argmax_a Q[s,a].
        #   Tip for fair tie-breaking on ties:
        #       best = np.flatnonzero(q == q.max()); self.rng.choice(best)
        raise NotImplementedError

    def update(self, s, a, r, s_next, a_next, done):
        # TODO: compute the target (see the module header) and do the update
        #       Q[s,a] += alpha * (target - Q[s,a]).
        #   - done:        target = r
        #   - sarsa:       target = r + gamma * Q[s_next, a_next]
        #   - qlearning:   target = r + gamma * max(Q[s_next])
        raise NotImplementedError

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)


def train(env, agent, n_episodes=500, max_steps=1000):
    """Train the agent. Returns: an array (length n_episodes) of the reward sums.

    The flow of an episode (correct for SARSA, also valid for Q-learning):
      s = env.reset(); a = agent.select_action(s)
      repeat until done or max_steps:
        s', r, done = env.step(a)
        a' = agent.select_action(s')          # draw a' BEFORE the update
        agent.update(s, a, r, s', a', done)
        s, a = s', a'                          # a' is executed in the next step
        update the reward sum
    """
    returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        # TODO: play one episode following the flow above and set returns[ep].
        raise NotImplementedError
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
