"""Policy-Gradient-Methoden: REINFORCE (mit/ohne Baseline) und Actor-Critic (A2C).

>>> DEINE AUFGABE <<<  Hier steckt die ganze Arbeit dieses Projekts. Vorgegeben sind nur die
zwei Netz-Klassen (Standard-MLPs) und die Konstruktoren. Zu implementieren:
   1) compute_returns
   2) REINFORCE.select_action  /  REINFORCE.update
   3) ActorCritic.select_action /  ActorCritic.update

Der Gegensatz zu Projekt 01 (DQN, wertbasiert): hier wird die **Policy direkt**
parametrisiert, pi_theta(a|s), und per Gradientenaufstieg auf die erwartete Rendite optimiert.

Policy-Gradient-Theorem (Modul 14, Abschnitt 3.2):
    grad J = E[ sum_t  grad log pi_theta(a_t|s_t) * Psi_t ]
mit dem Kredit-Signal Psi_t:
    REINFORCE:            Psi_t = G_t                       (voller Return-to-go)
    REINFORCE + Baseline: Psi_t = G_t normalisiert          (Varianzreduktion)
    Actor-Critic:         Psi_t = A_t = G_t - V(s_t)        (Advantage; Critic schaetzt V)

WICHTIG (Vorzeichen): Optimierer *minimieren*. Der Policy-Gradient will J *maximieren*.
Der Verlust ist deshalb das NEGATIVE des Ziels:  loss = -(log_prob * Psi).sum()

Musterloesung: loesung/policy_gradient.py — erst selbst versuchen!
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def compute_returns(rewards, gamma):
    """Diskontierte Return-to-go: G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    Rueckgabe: Liste gleicher Laenge wie `rewards`.
    Tipp: rueckwaerts durch die Belohnungen laufen und G = r + gamma*G fortschreiben
          (dieselbe Rekursion wie bei Monte Carlo in Modul 13) — das ist O(T) statt O(T^2).
    """
    # TODO
    raise NotImplementedError


# ---- Netze: vorgegeben (Standard-MLPs, hier steckt nicht der Lernstoff) ----
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
        """Ziehe eine Aktion aus pi(.|s). Rueckgabe: (action:int, log_prob:Tensor).

        Kleine Inspiration (Standard-Muster in PyTorch):
            logits = self.pi(torch.as_tensor(state, dtype=torch.float32))
            dist   = Categorical(logits=logits)
            a      = dist.sample()
            return int(a), dist.log_prob(a)
        Wichtig: den log_prob als TENSOR zurueckgeben (er muss den Graphen behalten,
        damit spaeter backward() funktioniert) — nicht mit .item()/float() abschneiden!
        """
        # TODO
        raise NotImplementedError

    def update(self, log_probs, rewards):
        """Ein REINFORCE-Update aus einer vollstaendigen Episode.

        Schritte:
          1. G = compute_returns(rewards, self.gamma)  -> torch-Tensor (float32)
          2. falls self.normalize: G = (G - G.mean()) / (G.std() + 1e-8)   # Baseline
          3. loss = -(torch.stack(log_probs) * G).sum()
          4. zero_grad -> backward -> step
        Rueckgabe: float(loss.detach())
        """
        # TODO
        raise NotImplementedError


class ActorCritic:
    def __init__(self, gamma=0.99, lr=1e-2, value_coef=0.5, seed=0):
        torch.manual_seed(seed)
        self.gamma = gamma; self.value_coef = value_coef
        self.actor = PolicyNet()
        self.critic = ValueNet()
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def select_action(self, state):
        """Wie oben, aber zusaetzlich den Critic-Wert V(s) zurueckgeben.
        Rueckgabe: (action:int, log_prob:Tensor, value:Tensor)
        """
        # TODO
        raise NotImplementedError

    def update(self, log_probs, values, rewards):
        """Ein Actor-Critic-Update aus einer vollstaendigen Episode.

        Schritte:
          1. G = compute_returns(...) -> Tensor;  V = torch.stack(values)
          2. advantage = (G - V).detach()      # WARUM .detach()? Der Advantage ist fuer den
             advantage = normalisieren        #   Actor nur ein *Gewicht* — durch ihn soll
                                              #   KEIN Gradient in den Critic zurueckfliessen.
          3. actor_loss  = -(torch.stack(log_probs) * advantage).sum()
             critic_loss = nn.functional.smooth_l1_loss(V, G)   # Critic regressiert auf G
          4. loss = actor_loss + self.value_coef * critic_loss; zero_grad/backward/step
        Rueckgabe: (float(actor_loss.detach()), float(critic_loss.detach()))
        """
        # TODO
        raise NotImplementedError
