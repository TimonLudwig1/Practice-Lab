"""Policy-Gradient-Methoden: REINFORCE (mit/ohne Baseline) und Actor-Critic (A2C).

Der Gegensatz zu Projekt 01 (DQN, wertbasiert): hier wird die **Policy direkt** parametrisiert,
pi_theta(a|s), und per Gradientenaufstieg auf die erwartete Rendite optimiert.

Policy-Gradient-Theorem (Modul 14, Abschnitt 3.2):
    grad J = E[ sum_t  grad log pi_theta(a_t|s_t) * Psi_t ]
mit dem Kredit-Signal Psi_t:
    REINFORCE:            Psi_t = G_t                       (voller Return-to-go)
    REINFORCE + Baseline: Psi_t = G_t normalisiert          (Varianzreduktion)
    Actor-Critic:         Psi_t = A_t = G_t - V(s_t)        (Advantage; Critic schaetzt V)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def compute_returns(rewards, gamma):
    """Diskontierte Return-to-go: G_t = r_t + gamma*r_{t+1} + ... , rueckwaerts berechnet.
    Rueckgabe: Liste gleicher Laenge wie rewards."""
    out = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out


class PolicyNet(nn.Module):
    def __init__(self, n_states=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, s):
        return self.net(s)          # Logits (unnormierte log-Wahrscheinlichkeiten)


class ValueNet(nn.Module):
    def __init__(self, n_states=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s):
        return self.net(s).squeeze(-1)


class REINFORCE:
    def __init__(self, gamma=0.99, lr=1e-2, normalize=True, seed=0):
        torch.manual_seed(seed)
        self.gamma = gamma; self.normalize = normalize
        self.pi = PolicyNet()
        self.opt = torch.optim.Adam(self.pi.parameters(), lr=lr)

    def select_action(self, state):
        logits = self.pi(torch.as_tensor(state, dtype=torch.float32))
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a), dist.log_prob(a)

    def update(self, log_probs, rewards):
        G = torch.tensor(compute_returns(rewards, self.gamma), dtype=torch.float32)
        if self.normalize:                       # Baseline: zentrieren + skalieren
            G = (G - G.mean()) / (G.std() + 1e-8)
        loss = -(torch.stack(log_probs) * G).sum()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.detach())


class ActorCritic:
    def __init__(self, gamma=0.99, lr=1e-2, value_coef=0.5, seed=0):
        torch.manual_seed(seed)
        self.gamma = gamma; self.value_coef = value_coef
        self.actor = PolicyNet()
        self.critic = ValueNet()
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def select_action(self, state):
        st = torch.as_tensor(state, dtype=torch.float32)
        dist = Categorical(logits=self.actor(st))
        a = dist.sample()
        return int(a), dist.log_prob(a), self.critic(st)

    def update(self, log_probs, values, rewards):
        G = torch.tensor(compute_returns(rewards, self.gamma), dtype=torch.float32)
        V = torch.stack(values)
        advantage = (G - V).detach()             # Critic liefert die Baseline
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        actor_loss = -(torch.stack(log_probs) * advantage).sum()
        critic_loss = nn.functional.smooth_l1_loss(V, G)
        loss = actor_loss + self.value_coef * critic_loss
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(actor_loss.detach()), float(critic_loss.detach())
