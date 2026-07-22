"""Tests for the policy-gradient building blocks.  Call:  python test_pg.py"""
import numpy as np
import torch
from torch.distributions import Categorical
from cartpole import CartPole
from policy_gradient import (compute_returns, PolicyNet, ValueNet,
                             REINFORCE, ActorCritic)


# ---------------------------- returns ----------------------------
def test_compute_returns_undiscounted():
    # gamma=1: G_t = the sum of the remaining rewards
    assert compute_returns([1, 1, 1], 1.0) == [3, 2, 1]


def test_compute_returns_discounted():
    r = [0.0, 0.0, 1.0]
    G = compute_returns(r, 0.5)
    assert np.allclose(G, [0.25, 0.5, 1.0])


def test_compute_returns_length():
    r = [1.0] * 17
    assert len(compute_returns(r, 0.99)) == 17


# ---------------------------- networks / distribution ----------------------------
def test_policy_net_yields_valid_distribution():
    pi = PolicyNet()
    s = torch.zeros(4)
    dist = Categorical(logits=pi(s))
    p = dist.probs
    assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-5)
    assert p.shape == (2,) and torch.all(p >= 0)


def test_value_net_scalar():
    v = ValueNet()
    assert v(torch.zeros(4)).shape == torch.Size([])       # one scalar per state


def test_select_action_returns_action_and_logprob():
    ag = REINFORCE(seed=1)
    a, logp = ag.select_action(np.zeros(4))
    assert a in (0, 1)
    assert logp.requires_grad and logp.item() <= 0.0       # log prob <= 0


# ---------------------------- update logic ----------------------------
def test_reinforce_update_raises_logprob_on_positive_return():
    # The core idea of the policy gradient: an action with a positive (normalized) return
    # should become more probable. We check: an update changes the parameters (the gradient
    # is not zero).
    ag = REINFORCE(normalize=False, seed=2)
    before = [p.clone() for p in ag.pi.parameters()]
    # an artificial episode
    logps = [ag.select_action(np.zeros(4))[1] for _ in range(5)]
    ag.update(logps, [1.0, 1.0, 1.0, 1.0, 1.0])
    after = list(ag.pi.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "the update did not change the policy parameters"


def test_actor_critic_update_runs_and_returns_two_losses():
    ag = ActorCritic(seed=3)
    logps, vals, rews = [], [], []
    s = np.zeros(4)
    for _ in range(6):
        a, logp, v = ag.select_action(s); logps.append(logp); vals.append(v); rews.append(1.0)
    al, cl = ag.update(logps, vals, rews)
    assert np.isfinite(al) and np.isfinite(cl)


def test_reinforce_learns_cartpole_partially():
    # A short training: the policy should clearly improve over the start.
    # (Absolute thresholds are unreliable for REINFORCE - the learning curve jumps late and
    #  strongly; that is why we check the *improvement* over the beginning.)
    ag = REINFORCE(seed=0); env = CartPole(seed=0)
    scores = []
    for ep in range(250):
        s = env.reset(); done = False; logps = []; rews = []
        while not done:
            a, logp = ag.select_action(s); s, r, done = env.step(a)
            logps.append(logp); rews.append(r)
        ag.update(logps, rews); scores.append(sum(rews))
    start, end = np.mean(scores[:30]), np.mean(scores[-30:])
    assert end > 2.5 * start, f"barely learned: {start:.1f} -> {end:.1f}"
    assert end > 60.0, f"too weak: {end:.1f}"


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("All tests passed.")
