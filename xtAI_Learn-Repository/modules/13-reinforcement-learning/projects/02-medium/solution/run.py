"""Comparison SARSA vs. Q-learning on cliff walking.

Reproduces the classic figure (Sutton & Barto, example 6.6):
- Q-learning learns the *optimal* route right along the cliff edge (greedy return -13), but
  achieves less online reward during training because epsilon-exploration occasionally makes
  it fall off.
- SARSA learns a *safer* route with distance to the cliff and has the better online return
  during training.

Call:  python run.py        (with a plot, if matplotlib is available)
"""
from __future__ import annotations
import numpy as np
from cliff_walking import CliffWalking
from td_control import TDAgent, train, rollout_greedy


def moving_average(x, w=10):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main(n_episodes=500, n_runs=50, seed=0):
    env = CliffWalking()
    seeder = np.random.default_rng(seed)
    curves = {}
    final_agents = {}
    for algo in ("sarsa", "qlearning"):
        allret = np.zeros((n_runs, n_episodes))
        for run in range(n_runs):
            agent = TDAgent(env.n_states, env.n_actions, algo=algo,
                            alpha=0.5, gamma=1.0, epsilon=0.1,
                            seed=int(seeder.integers(1 << 30)))
            allret[run] = train(env, agent, n_episodes=n_episodes)
            if run == n_runs - 1:
                final_agents[algo] = agent
        curves[algo] = allret.mean(axis=0)

    print(f"{'algorithm':12s} {'online return (last 100 ep., mean)':>38s}  {'greedy return':>14s}")
    for algo in ("sarsa", "qlearning"):
        online = curves[algo][-100:].mean()
        greedy = rollout_greedy(env, final_agents[algo])
        print(f"{algo:12s} {online:38.1f}  {greedy:14.0f}")

    print("\n--- greedy policy SARSA ---")
    print(env.render_policy(final_agents["sarsa"].greedy_policy()))
    print("\n--- greedy policy Q-learning ---")
    print(env.render_policy(final_agents["qlearning"].greedy_policy()))

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.5))
        for algo, label in [("sarsa", "SARSA (on-policy)"),
                            ("qlearning", "Q-learning (off-policy)")]:
            plt.plot(moving_average(curves[algo], 10), label=label, lw=1.5)
        plt.ylim(-100, 0)
        plt.xlabel("episode"); plt.ylabel("reward sum per episode (moving average)")
        plt.title("Cliff walking: SARSA vs. Q-learning (online, during training)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        import os
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/cliff_comparison.png", dpi=110)
        print("\nPlot saved: results/cliff_comparison.png")
    except Exception as e:
        print("(no plot:", e, ")")


if __name__ == "__main__":
    main()
