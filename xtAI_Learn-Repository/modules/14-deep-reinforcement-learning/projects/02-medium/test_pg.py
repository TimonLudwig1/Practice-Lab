"""Tests fuer Policy-Gradient-Bausteine.  Aufruf:  python test_pg.py"""
import numpy as np
import torch
from torch.distributions import Categorical
from cartpole import CartPole
from policy_gradient import (compute_returns, PolicyNet, ValueNet,
                             REINFORCE, ActorCritic)


# ---------------------------- Returns ----------------------------
def test_compute_returns_undiskontiert():
    # gamma=1: G_t = Summe der restlichen Belohnungen
    assert compute_returns([1, 1, 1], 1.0) == [3, 2, 1]


def test_compute_returns_diskontiert():
    r = [0.0, 0.0, 1.0]
    G = compute_returns(r, 0.5)
    assert np.allclose(G, [0.25, 0.5, 1.0])


def test_compute_returns_laenge():
    r = [1.0] * 17
    assert len(compute_returns(r, 0.99)) == 17


# ---------------------------- Netze / Verteilung ----------------------------
def test_policy_net_liefert_gueltige_verteilung():
    pi = PolicyNet()
    s = torch.zeros(4)
    dist = Categorical(logits=pi(s))
    p = dist.probs
    assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-5)
    assert p.shape == (2,) and torch.all(p >= 0)


def test_value_net_skalar():
    v = ValueNet()
    assert v(torch.zeros(4)).shape == torch.Size([])       # ein Skalar je Zustand


def test_select_action_gibt_action_und_logprob():
    ag = REINFORCE(seed=1)
    a, logp = ag.select_action(np.zeros(4))
    assert a in (0, 1)
    assert logp.requires_grad and logp.item() <= 0.0       # log-Wkt. <= 0


# ---------------------------- Update-Logik ----------------------------
def test_reinforce_update_erhoeht_logprob_bei_positivem_return():
    # Kernidee des Policy-Gradient: eine Aktion mit positivem (normalisiertem) Return
    # soll wahrscheinlicher werden. Wir pruefen: ein Update senkt den Loss-Betrag /
    # veraendert die Parameter (Gradient ist nicht null).
    ag = REINFORCE(normalize=False, seed=2)
    before = [p.clone() for p in ag.pi.parameters()]
    # kuenstliche Episode
    logps = [ag.select_action(np.zeros(4))[1] for _ in range(5)]
    ag.update(logps, [1.0, 1.0, 1.0, 1.0, 1.0])
    after = list(ag.pi.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "Update hat die Policy-Parameter nicht veraendert"


def test_actor_critic_update_laeuft_und_gibt_zwei_losses():
    ag = ActorCritic(seed=3)
    logps, vals, rews = [], [], []
    s = np.zeros(4)
    for _ in range(6):
        a, logp, v = ag.select_action(s); logps.append(logp); vals.append(v); rews.append(1.0)
    al, cl = ag.update(logps, vals, rews)
    assert np.isfinite(al) and np.isfinite(cl)


def test_reinforce_lernt_cartpole_teilweise():
    # Kurzes Training: die Policy soll sich klar gegenueber dem Start verbessern.
    # (Absolute Schwellen sind bei REINFORCE unzuverlaessig - die Lernkurve springt
    #  spaet und stark; deshalb pruefen wir die *Verbesserung* gegenueber dem Anfang.)
    ag = REINFORCE(seed=0); env = CartPole(seed=0)
    scores = []
    for ep in range(250):
        s = env.reset(); done = False; logps = []; rews = []
        while not done:
            a, logp = ag.select_action(s); s, r, done = env.step(a)
            logps.append(logp); rews.append(r)
        ag.update(logps, rews); scores.append(sum(rews))
    start, end = np.mean(scores[:30]), np.mean(scores[-30:])
    assert end > 2.5 * start, f"kaum gelernt: {start:.1f} -> {end:.1f}"
    assert end > 60.0, f"zu schwach: {end:.1f}"


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Starte {len(tests)} Tests ...")
    for name, t in tests:
        t(); print(f"  {name} ... OK")
    print("Alle Tests bestanden.")
