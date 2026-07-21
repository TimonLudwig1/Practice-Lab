"""Modellfreier Policy-Gradient fuer KONTINUIERLICHE Stellgroessen.

Der Clou dieses Projekts: die LQR-Theorie sagt, dass die optimale Politik **linear** ist
(u* = -K x). Deshalb parametrisieren wir die Politik ebenfalls linear:

    pi_theta(u | x) = N( w^T x , sigma^2 )

Damit sind die gelernten Gewichte `w` DIREKT mit der exakten Loesung -K vergleichbar — der
Agent muss die Struktur nicht erst entdecken, wir koennen Parameter gegen Parameter halten.
(Ein MLP wuerde genauso funktionieren, waere aber nicht mehr interpretierbar.)

Der Agent benutzt NUR (x, u, r, x') — er kennt A, B, Q, R nicht.

Zwei Varianzreduktions-Zutaten, ohne die REINFORCE hier praktisch nicht konvergiert:
  1. **Batch von Episoden je Update** (statt einer einzigen) — mittelt den Gradienten.
  2. **Per-Zeitschritt-Baseline**: an jedem t ueber den Batch zentrieren+skalieren. Returns bei
     t=0 sind betragsmaessig viel groesser als bei t=T-1; wuerde man global normieren, wuerde
     dieser Zeittrend das Lernsignal dominieren.
Zusaetzlich wird log(sigma) nach unten geklemmt — sonst kollabiert die Exploration
(sigma -> 0), die log-Wahrscheinlichkeiten explodieren und das Training divergiert.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -1.5      # sigma >= ~0.22 : Exploration darf nicht kollabieren
LOG_STD_MAX = 0.5


class LinearGaussianPolicy:
    def __init__(self, n_states=2, lr=0.05, init_log_std=-0.5, seed=0):
        torch.manual_seed(seed)
        self.mean = nn.Linear(n_states, 1, bias=False)
        nn.init.zeros_(self.mean.weight)                  # Start: u = 0
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
        """log_probs: (batch, T) Tensor · returns: (batch, T) ndarray."""
        G = torch.as_tensor(np.asarray(returns), dtype=torch.float32)
        # Per-Zeitschritt-Baseline (ueber den Batch)
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
    """Trainiert `policy` modellfrei auf `env`. Rueckgabe: Liste mittlerer Episodenkosten."""
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
            print(f"  update {up:4d}  mittlere Episodenkosten {history[-1]:8.3f}"
                  f"  w={np.round(policy.weights(),3)}  sigma={float(policy.std().detach()):.3f}")
    return history
