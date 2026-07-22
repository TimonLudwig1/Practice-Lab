"""Model-free control on the inventory-management MDP.

Four methods, all epsilon-greedy and with action masks (do not order above the capacity):
  - MonteCarloControl : every-visit MC control (learns from whole episodes)
  - SARSA             : on-policy TD control
  - QLearning         : off-policy TD control
  - ExpectedSarsa     : the expectation over the epsilon-greedy policy in the target

NONE uses env.expected_reward / env.transition_probs — they learn ONLY from (s,a,r,s')
experience. The model is solely in the DP reference (dp_reference.py).
"""
from __future__ import annotations
import numpy as np


class TabularAgent:
    """Shared base: a Q table, masked epsilon-greedy selection, the greedy policy."""
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1, seed=0):
        self.env = env
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.Q = np.zeros((env.n_states, env.n_actions))
        self.rng = np.random.default_rng(seed)
        # permanently hide inadmissible actions
        self._masks = np.array([env.action_mask(i) for i in range(env.n_states)])
        self.Q[~self._masks] = -np.inf

    def _masked_q(self, s):
        return self.Q[s]      # inadmissible entries are already -inf

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
        """The probability vector of the epsilon-greedy policy in s (for expected SARSA)."""
        q = self._masked_q(s)
        valid = self._masks[s]
        n_valid = valid.sum()
        probs = np.zeros(self.env.n_actions)
        probs[valid] = self.epsilon / n_valid
        best = np.flatnonzero(q == q.max())
        probs[best] += (1.0 - self.epsilon) / len(best)   # ties share the greedy mass
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
            target = r + self.gamma * self.Q[s2].max()   # -inf entries do not disturb the max
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
    """Every-visit MC control with the incremental mean. The continuing task is cut into
    episodes of fixed length (episode_len steps)."""
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
            for (s, a, r) in reversed(traj):        # returns backwards
                G = r + self.gamma * G
                self.N[s, a] += 1
                self.Q[s, a] += (G - self.Q[s, a]) / self.N[s, a]
        return np.array(hist)


AGENTS = {"MC": MonteCarloControl, "SARSA": SARSA,
          "Q-Learning": QLearning, "Expected SARSA": ExpectedSarsa}
