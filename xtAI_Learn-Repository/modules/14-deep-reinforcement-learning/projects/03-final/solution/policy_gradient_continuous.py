"""Model-free policy gradient for CONTINUOUS control inputs.

The trick of this project: LQR theory says the optimal policy is **linear** (u* = -K x). We
therefore parametrize the policy linearly too:

    pi_theta(u | x) = N( w^T x , sigma^2 )

This makes the learned weights `w` DIRECTLY comparable with the exact solution -K — the agent
does not have to discover the structure first, we can hold parameter against parameter.
(An MLP would work just as well but would no longer be interpretable.)

The agent uses ONLY (x, u, r, x') — it does not know A, B, Q, R.

Two variance-reduction ingredients, without which REINFORCE practically does not converge here:
  1. **A batch of episodes per update** (instead of a single one) — averages the gradient.
  2. **A per-time-step baseline**: center+scale at each t over the batch. Returns at t=0 are much
     larger in magnitude than at t=T-1; if one normalized globally, this time trend would
     dominate the learning signal.
Additionally log(sigma) is clamped from below — otherwise the exploration collapses (sigma -> 0),
the log probabilities explode and the training diverges.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -1.5      # sigma >= ~0.22 : exploration must not collapse
LOG_STD_MAX = 0.5


class LinearGaussianPolicy:
    def __init__(self, n_states=2, lr=0.05, init_log_std=-0.5, seed=0):
        torch.manual_seed(seed)
        self.mean = nn.Linear(n_states, 1, bias=False)
        nn.init.zeros_(self.mean.weight)                  # start: u = 0
        self.log_std = nn.Parameter(torch.tensor([float(init_log_std)]))
        self.opt = torch.optim.Adam(
            list(self.mean.parameters()) + [self.log_std], lr=lr)

    def std(self):
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def dist(self, x):
        xt = torch.as_tensor(x, dtype=torch.float32)
        return Normal(self.mean(xt), self.std())

    def select_action(self, x):
        d = self.dist(x)
        u = d.sample()
        return float(u.detach()), d.log_prob(u).sum()

    def weights(self):
        return self.mean.weight.detach().numpy().ravel()

    def update(self, log_probs, returns):
        """log_probs: (batch, T) tensor · returns: (batch, T) ndarray."""
        G = torch.as_tensor(np.asarray(returns), dtype=torch.float32)
        # per-time-step baseline (over the batch)
        adv = (G - G.mean(0, keepdim=True)) / (G.std(0, keepdim=True) + 1e-8)
        loss = -(log_probs * adv).sum(1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())


def discounted_returns(rewards, gamma=1.0):
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    return out


def train_policy_gradient(env, policy, n_updates=600, batch=32, gamma=1.0, verbose_every=None):
    """Trains `policy` model-free on `env`. Returns: a list of mean episode costs."""
    history = []
    for up in range(n_updates):
        LP, GG, costs = [], [], []
        for _ in range(batch):
            x = env.reset()
            logps, rews = [], []
            done = False
            while not done:
                u, lp = policy.select_action(x)
                x, r, done = env.step(u)
                logps.append(lp); rews.append(r)
            LP.append(torch.stack(logps))
            GG.append(discounted_returns(rews, gamma))
            costs.append(-sum(rews))
        policy.update(torch.stack(LP), np.array(GG))
        history.append(float(np.mean(costs)))
        if verbose_every and up % verbose_every == 0:
            print(f"  update {up:4d}  mean episode cost {history[-1]:8.3f}"
                  f"  w={np.round(policy.weights(),3)}  sigma={float(policy.std().detach()):.3f}")
    return history
