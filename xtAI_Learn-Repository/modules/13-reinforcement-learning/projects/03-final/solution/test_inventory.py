"""Tests for the inventory-management MDP, the DP reference and the model-free learners.

Call:  python test_inventory.py     (no pytest needed)
"""
import numpy as np
from inventory_env import InventoryEnv, poisson_pmf
from dp_reference import value_iteration, policy_value, order_up_to_levels
from agents import QLearning, ExpectedSarsa


# ------------------------------- environment -------------------------------
def test_poisson_pmf_normalized():
    pmf = poisson_pmf(8.0, 30)
    assert np.isclose(pmf.sum(), 1.0)
    assert np.all(pmf >= 0)


def test_reward_formula():
    env = InventoryEnv(K=10, c=2, h=1, p=6)
    # i=3, a=5 -> x=8; d=10 -> lost=2, ending=0
    r = env._reward(3, 5, 10)
    assert np.isclose(r, -(10 + 2 * 5 + 1 * 0 + 6 * 2))
    # d=2 -> ending=6, lost=0
    r2 = env._reward(3, 5, 2)
    assert np.isclose(r2, -(10 + 2 * 5 + 1 * 6 + 6 * 0))
    # a=0 -> no fixed cost
    r3 = env._reward(3, 0, 1)
    assert np.isclose(r3, -(0 + 0 + 1 * 2 + 0))


def test_valid_actions_and_mask():
    env = InventoryEnv(capacity=20)
    assert list(env.valid_actions(18)) == [0, 1, 2]     # may fill up to at most 20
    assert env.action_mask(18).sum() == 3
    assert env.action_mask(0).sum() == 21


def test_transition_probs_normalized_and_consistent():
    env = InventoryEnv()
    P = env.transition_probs(5, 7)                        # x=12
    assert np.isclose(P.sum(), 1.0)
    # the ending inventory can never be greater than x=12
    assert np.all(P[13:] == 0.0)


def test_step_respects_capacity():
    env = InventoryEnv(capacity=20, seed=3)
    env.reset(state=19)
    try:
        env.step(5)          # 19+5 > 20 -> inadmissible
        assert False, "should have raised an AssertionError"
    except AssertionError as e:
        assert "inadmissible" in str(e)


# ------------------------------- DP reference -------------------------------
def test_vi_has_sS_structure():
    env = InventoryEnv()
    res = value_iteration(env)
    x = order_up_to_levels(env, res["policy"])
    orders = res["policy"] > 0
    # S = the target inventory level up to which the *ordering* states fill;
    # this level must be constant (the "S" in (s,S)).
    S = x[orders].max()
    assert np.all(x[orders] == S), x[orders]
    # in state 0 one orders, and up to S
    assert res["policy"][0] > 0 and x[0] == S
    # above the order point nothing is ordered -> x == i there
    assert np.all(x[~orders] == np.arange(env.n_states)[~orders])
    # there is a unique order point s: from there on nothing is ordered
    # no "holes": the ordering states form an initial segment 0..s
    s = np.flatnonzero(orders).max()
    assert np.all(orders[: s + 1]) and not np.any(orders[s + 1:])


def test_policy_value_matches_VI():
    env = InventoryEnv()
    res = value_iteration(env, gamma=0.95)
    V_pi = policy_value(env, res["policy"], gamma=0.95)
    assert np.allclose(V_pi, res["V"], atol=1e-4)


# ------------------------------- model-free learners -------------------------------
def test_qlearning_close_to_optimum():
    env = InventoryEnv(seed=0)
    res = value_iteration(env, gamma=0.95)
    ag = QLearning(env, alpha=0.1, gamma=0.95, epsilon=0.1, seed=1)
    ag.train(n_steps=150_000)
    V = policy_value(env, ag.greedy_policy(), gamma=0.95)
    gap = (res["V"].mean() - V.mean()) / abs(res["V"].mean())
    assert gap < 0.12, f"optimality gap too large: {gap:.1%}"


def test_greedy_prob_is_distribution():
    env = InventoryEnv()
    ag = ExpectedSarsa(env, epsilon=0.1)
    for s in (0, 5, 18):
        p = ag.greedy_prob(s)
        assert np.isclose(p.sum(), 1.0)
        assert np.all(p[~ag._masks[s]] == 0.0)          # inadmissible actions: 0


def test_masks_stay_minus_inf():
    env = InventoryEnv()
    ag = QLearning(env)
    ag.train(n_steps=5_000)
    for i in range(env.n_states):
        invalid = ~env.action_mask(i)
        assert np.all(ag.Q[i, invalid] == -np.inf)      # inadmissible actions never updated


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests ...")
    for name, t in tests:
        t()
        print(f"  {name} ... OK")
    print("All tests passed.")
