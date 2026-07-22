"""Policy-gradient methods: REINFORCE (with/without a baseline) and actor-critic (A2C).

>>> YOUR TASK <<<  This is where the whole work of this project sits. Given are only the two
network classes (standard MLPs) and the constructors. To implement:
   1) compute_returns
   2) REINFORCE.select_action  /  REINFORCE.update
   3) ActorCritic.select_action /  ActorCritic.update

The contrast to project 01 (DQN, value-based): here the **policy is parametrized directly**,
pi_theta(a|s), and optimized by gradient ascent on the expected return.

The policy-gradient theorem (module 14, section 3.2):
    grad J = E[ sum_t  grad log pi_theta(a_t|s_t) * Psi_t ]
with the credit signal Psi_t:
    REINFORCE:            Psi_t = G_t                       (the full return-to-go)
    REINFORCE + baseline: Psi_t = G_t normalized            (variance reduction)
    actor-critic:         Psi_t = A_t = G_t - V(s_t)        (advantage; the critic estimates V)

IMPORTANT (sign): optimizers *minimize*. The policy gradient wants to *maximize* J.
The loss is therefore the NEGATIVE of the objective:  loss = -(log_prob * Psi).sum()

The reference solution: solution/policy_gradient.py — try it yourself first!
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def compute_returns(rewards, gamma):
    """Discounted return-to-go: G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    Returns: a list of the same length as `rewards`.
    Tip: walk backwards through the rewards and carry G = r + gamma*G (the same recursion as
         with Monte Carlo in module 13) — that is O(T) instead of O(T^2).
    """
    # TODO
    raise NotImplementedError


# ---- networks: given (standard MLPs, the learning content is not here) ----
class PolicyNet(nn.Module):
    def __init__(self, n_states=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, s):
        return self.net(s)          # logits (unnormalized log probabilities)


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
        """Draw an action from pi(.|s). Returns: (action:int, log_prob:Tensor).

        A small inspiration (the standard pattern in PyTorch):
            logits = self.pi(torch.as_tensor(state, dtype=torch.float32))
            dist   = Categorical(logits=logits)
            a      = dist.sample()
            return int(a), dist.log_prob(a)
        Important: return the log_prob as a TENSOR (it has to keep the graph, so that backward()
        works later) — do not truncate it with .item()/float()!
        """
        # TODO
        raise NotImplementedError

    def update(self, log_probs, rewards):
        """One REINFORCE update from a complete episode.

        Steps:
          1. G = compute_returns(rewards, self.gamma)  -> a torch tensor (float32)
          2. if self.normalize: G = (G - G.mean()) / (G.std() + 1e-8)   # baseline
          3. loss = -(torch.stack(log_probs) * G).sum()
          4. zero_grad -> backward -> step
        Returns: float(loss.detach())
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
        """As above, but additionally return the critic value V(s).
        Returns: (action:int, log_prob:Tensor, value:Tensor)
        """
        # TODO
        raise NotImplementedError

    def update(self, log_probs, values, rewards):
        """One actor-critic update from a complete episode.

        Steps:
          1. G = compute_returns(...) -> Tensor;  V = torch.stack(values)
          2. advantage = (G - V).detach()      # WHY .detach()? The advantage is only a *weight*
             advantage = normalize             #   for the actor — NO gradient should flow
                                               #   through it back into the critic.
          3. actor_loss  = -(torch.stack(log_probs) * advantage).sum()
             critic_loss = nn.functional.smooth_l1_loss(V, G)   # the critic regresses onto G
          4. loss = actor_loss + self.value_coef * critic_loss; zero_grad/backward/step
        Returns: (float(actor_loss.detach()), float(critic_loss.detach()))
        """
        # TODO
        raise NotImplementedError
