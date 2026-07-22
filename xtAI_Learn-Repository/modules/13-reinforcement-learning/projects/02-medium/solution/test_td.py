"""Tests for the cliff-walking environment and TD control.

Call:  python test_td.py        (no pytest needed)
   or:  python -m pytest -q     (if pytest is installed)
"""
import numpy as np
from cliff_walking import CliffWalking
from td_control import TDAgent, train, rollout_greedy


# ----------------------------- environment -----------------------------
def test_encode_decode_roundtrip():
    env = CliffWalking()
    for s in range(env.n_states):
        assert env.encode(env.decode(s)) == s


def test_step_cost_and_bounds():
    env = CliffWalking()
    env.reset()
    # left into the wall -> stays at start, cost -1
    s, r, done = env.step(3)   # left
    assert r == -1.0 and not done
    assert env.decode(s) == env.start


def test_cliff_resets_to_start():
    env = CliffWalking()
    env.reset()
    # from the start (3,0) to the right -> (3,1) is the cliff
    s, r, done = env.step(1)   # right
    assert r == -100.0 and not done
    assert env.decode(s) == env.start


def test_goal_terminates():
    env = CliffWalking()
    # artificially place next to the goal: (2,11), then down -> goal (3,11)
    env._state = (2, env.cols - 1)
    s, r, done = env.step(2)   # down
    assert done and r == -1.0
    assert env.decode(s) == env.goal


def test_optimal_path_return_is_minus_13():
    # shortest safe route: up, 11x right, down = 13 steps, return -13
    env = CliffWalking()
    env.reset()
    total, done = 0.0, False
    for a in [0] + [1] * 11 + [2]:
        _, r, done = env.step(a)
        total += r
    assert done and total == -13.0


# ----------------------------- agents -----------------------------
def test_update_targets_differ():
    # SARSA uses Q[s',a'], Q-learning uses max_a Q[s',.]
    for algo, exp in [("sarsa", 0.5 * (1 + 1.0 * 2.0)),
                      ("qlearning", 0.5 * (1 + 1.0 * 5.0))]:
        ag = TDAgent(3, 2, algo=algo, alpha=0.5, gamma=1.0, epsilon=0.0)
        ag.Q[1] = np.array([5.0, 2.0])     # max=5 (a=0), a'=1 -> 2.0
        ag.update(s=0, a=0, r=1.0, s_next=1, a_next=1, done=False)
        assert np.isclose(ag.Q[0, 0], exp), (algo, ag.Q[0, 0], exp)


def test_terminal_update_ignores_bootstrap():
    ag = TDAgent(3, 2, algo="qlearning", alpha=1.0, gamma=1.0)
    ag.Q[1] = np.array([99.0, 99.0])       # should be ignored on done
    ag.update(s=0, a=0, r=-1.0, s_next=1, a_next=0, done=True)
    assert np.isclose(ag.Q[0, 0], -1.0)


def test_epsilon_zero_is_greedy():
    ag = TDAgent(1, 3, epsilon=0.0)
    ag.Q[0] = np.array([0.0, 9.0, 1.0])
    assert all(ag.select_action(0) == 1 for _ in range(20))


def test_qlearning_finds_optimal_greedy_path():
    # after training the greedy policy should yield the optimal return -13
    env = CliffWalking()
    ag = TDAgent(env.n_states, env.n_actions, algo="qlearning",
                 alpha=0.5, gamma=1.0, epsilon=0.1, seed=1)
    train(env, ag, n_episodes=500)
    assert rollout_greedy(env, ag) == -13.0


def test_sarsa_online_beats_qlearning_online():
    # SARSA has the better online return during training (with exploration)
    env = CliffWalking()
    def mean_online(algo, seed):
        ag = TDAgent(env.n_states, env.n_actions, algo=algo,
                     alpha=0.5, gamma=1.0, epsilon=0.1, seed=seed)
        ret = train(env, ag, n_episodes=500)
        return ret[-200:].mean()
    s = np.mean([mean_online("sarsa", i) for i in range(5)])
    q = np.mean([mean_online("qlearning", i) for i in range(5)])
    assert s > q, (s, q)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests ...")
    for name, t in tests:
        t()
        print(f"  {name} ... OK")
    print("All tests passed.")
