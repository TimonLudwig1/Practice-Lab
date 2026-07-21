"""Tests fuer Bestandsmanagement-MDP, DP-Referenz und modellfreie Lerner.

Aufruf:  python test_inventory.py     (kein pytest noetig)
"""
import numpy as np
from inventory_env import InventoryEnv, poisson_pmf
from dp_reference import value_iteration, policy_value, order_up_to_levels
from agents import QLearning, ExpectedSarsa


# ------------------------------- Umgebung -------------------------------
def test_poisson_pmf_normiert():
    pmf = poisson_pmf(8.0, 30)
    assert np.isclose(pmf.sum(), 1.0)
    assert np.all(pmf >= 0)


def test_reward_formel():
    env = InventoryEnv(K=10, c=2, h=1, p=6)
    # i=3, a=5 -> x=8; d=10 -> lost=2, ending=0
    r = env._reward(3, 5, 10)
    assert np.isclose(r, -(10 + 2 * 5 + 1 * 0 + 6 * 2))
    # d=2 -> ending=6, lost=0
    r2 = env._reward(3, 5, 2)
    assert np.isclose(r2, -(10 + 2 * 5 + 1 * 6 + 6 * 0))
    # a=0 -> keine Fixkosten
    r3 = env._reward(3, 0, 1)
    assert np.isclose(r3, -(0 + 0 + 1 * 2 + 0))


def test_valid_actions_und_maske():
    env = InventoryEnv(capacity=20)
    assert list(env.valid_actions(18)) == [0, 1, 2]     # darf hoechstens auf 20 auffuellen
    assert env.action_mask(18).sum() == 3
    assert env.action_mask(0).sum() == 21


def test_transition_probs_normiert_und_konsistent():
    env = InventoryEnv()
    P = env.transition_probs(5, 7)                        # x=12
    assert np.isclose(P.sum(), 1.0)
    # Endbestand kann nie groesser als x=12 sein
    assert np.all(P[13:] == 0.0)


def test_step_respektiert_kapazitaet():
    env = InventoryEnv(capacity=20, seed=3)
    env.reset(state=19)
    try:
        env.step(5)          # 19+5 > 20 -> unzulaessig
        assert False, "haette AssertionError werfen muessen"
    except AssertionError as e:
        assert "unzulaessig" in str(e)


# ------------------------------- DP-Referenz -------------------------------
def test_vi_hat_sS_struktur():
    env = InventoryEnv()
    res = value_iteration(env)
    x = order_up_to_levels(env, res["policy"])
    orders = res["policy"] > 0
    # S = Ziel-Lagerstand, auf den in den *bestellenden* Zustaenden aufgefuellt wird;
    # dieses Niveau muss konstant sein (das "S" in (s,S)).
    S = x[orders].max()
    assert np.all(x[orders] == S), x[orders]
    # in Zustand 0 wird bestellt, und zwar auf S
    assert res["policy"][0] > 0 and x[0] == S
    # oberhalb des Bestellpunkts wird nichts bestellt -> x == i dort
    assert np.all(x[~orders] == np.arange(env.n_states)[~orders])
    # es gibt einen eindeutigen Bestellpunkt s: ab dort wird nichts mehr bestellt
    # keine "Loecher": bestellte Zustaende bilden ein Anfangsstueck 0..s
    s = np.flatnonzero(orders).max()
    assert np.all(orders[: s + 1]) and not np.any(orders[s + 1:])


def test_policy_value_stimmt_mit_VI_ueberein():
    env = InventoryEnv()
    res = value_iteration(env, gamma=0.95)
    V_pi = policy_value(env, res["policy"], gamma=0.95)
    assert np.allclose(V_pi, res["V"], atol=1e-4)


# ------------------------------- modellfreie Lerner -------------------------------
def test_qlearning_nahe_am_optimum():
    env = InventoryEnv(seed=0)
    res = value_iteration(env, gamma=0.95)
    ag = QLearning(env, alpha=0.1, gamma=0.95, epsilon=0.1, seed=1)
    ag.train(n_steps=150_000)
    V = policy_value(env, ag.greedy_policy(), gamma=0.95)
    gap = (res["V"].mean() - V.mean()) / abs(res["V"].mean())
    assert gap < 0.12, f"Optimalitaetsluecke zu gross: {gap:.1%}"


def test_greedy_prob_ist_verteilung():
    env = InventoryEnv()
    ag = ExpectedSarsa(env, epsilon=0.1)
    for s in (0, 5, 18):
        p = ag.greedy_prob(s)
        assert np.isclose(p.sum(), 1.0)
        assert np.all(p[~ag._masks[s]] == 0.0)          # unzulaessige Aktionen: 0


def test_masken_bleiben_minus_inf():
    env = InventoryEnv()
    ag = QLearning(env)
    ag.train(n_steps=5_000)
    for i in range(env.n_states):
        invalid = ~env.action_mask(i)
        assert np.all(ag.Q[i, invalid] == -np.inf)      # unzulaessige Aktionen nie aktualisiert


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Starte {len(tests)} Tests ...")
    for name, t in tests:
        t()
        print(f"  {name} ... OK")
    print("Alle Tests bestanden.")
