"""Tests fuer Cliff-Walking-Umgebung und TD-Kontrolle.

Aufruf:  python test_td.py        (kein pytest noetig)
   oder:  python -m pytest -q     (falls pytest installiert ist)
"""
import numpy as np
from cliff_walking import CliffWalking
from td_control import TDAgent, train, rollout_greedy


# ----------------------------- Umgebung -----------------------------
def test_encode_decode_roundtrip():
    env = CliffWalking()
    for s in range(env.n_states):
        assert env.encode(env.decode(s)) == s


def test_step_cost_and_bounds():
    env = CliffWalking()
    env.reset()
    # links an der Wand -> bleibt Start, Kosten -1
    s, r, done = env.step(3)   # links
    assert r == -1.0 and not done
    assert env.decode(s) == env.start


def test_cliff_resets_to_start():
    env = CliffWalking()
    env.reset()
    # vom Start (3,0) nach rechts -> (3,1) ist Klippe
    s, r, done = env.step(1)   # rechts
    assert r == -100.0 and not done
    assert env.decode(s) == env.start


def test_goal_terminates():
    env = CliffWalking()
    # kuenstlich neben das Ziel setzen: (2,11), dann runter -> Ziel (3,11)
    env._state = (2, env.cols - 1)
    s, r, done = env.step(2)   # runter
    assert done and r == -1.0
    assert env.decode(s) == env.goal


def test_optimal_path_return_is_minus_13():
    # kuerzeste sichere Route: hoch, 11x rechts, runter = 13 Schritte, Ertrag -13
    env = CliffWalking()
    env.reset()
    total, done = 0.0, False
    for a in [0] + [1] * 11 + [2]:
        _, r, done = env.step(a)
        total += r
    assert done and total == -13.0


# ----------------------------- Agenten -----------------------------
def test_update_targets_differ():
    # SARSA nutzt Q[s',a'], Q-Learning nutzt max_a Q[s',.]
    for algo, exp in [("sarsa", 0.5 * (1 + 1.0 * 2.0)),
                      ("qlearning", 0.5 * (1 + 1.0 * 5.0))]:
        ag = TDAgent(3, 2, algo=algo, alpha=0.5, gamma=1.0, epsilon=0.0)
        ag.Q[1] = np.array([5.0, 2.0])     # max=5 (a=0), a'=1 -> 2.0
        ag.update(s=0, a=0, r=1.0, s_next=1, a_next=1, done=False)
        assert np.isclose(ag.Q[0, 0], exp), (algo, ag.Q[0, 0], exp)


def test_terminal_update_ignores_bootstrap():
    ag = TDAgent(3, 2, algo="qlearning", alpha=1.0, gamma=1.0)
    ag.Q[1] = np.array([99.0, 99.0])       # sollte ignoriert werden bei done
    ag.update(s=0, a=0, r=-1.0, s_next=1, a_next=0, done=True)
    assert np.isclose(ag.Q[0, 0], -1.0)


def test_epsilon_zero_is_greedy():
    ag = TDAgent(1, 3, epsilon=0.0)
    ag.Q[0] = np.array([0.0, 9.0, 1.0])
    assert all(ag.select_action(0) == 1 for _ in range(20))


def test_qlearning_finds_optimal_greedy_path():
    # nach Training soll die greedy-Policy den optimalen Ertrag -13 liefern
    env = CliffWalking()
    ag = TDAgent(env.n_states, env.n_actions, algo="qlearning",
                 alpha=0.5, gamma=1.0, epsilon=0.1, seed=1)
    train(env, ag, n_episodes=500)
    assert rollout_greedy(env, ag) == -13.0


def test_sarsa_online_beats_qlearning_online():
    # SARSA hat waehrend des Trainings (mit Exploration) den besseren Online-Ertrag
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
    print(f"Starte {len(tests)} Tests ...")
    for name, t in tests:
        t()
        print(f"  {name} ... OK")
    print("Alle Tests bestanden.")
