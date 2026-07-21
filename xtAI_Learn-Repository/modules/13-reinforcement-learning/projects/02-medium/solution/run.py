"""Vergleich SARSA vs. Q-Learning auf Cliff Walking.

Reproduziert die klassische Figur (Sutton & Barto, Beispiel 6.6):
- Q-Learning lernt die *optimale* Route direkt an der Klippenkante (greedy-Ertrag -13),
  erzielt aber waehrend des Trainings weniger Online-Belohnung, weil epsilon-Exploration
  es gelegentlich abstuerzen laesst.
- SARSA lernt eine *sicherere* Route mit Abstand zur Klippe und hat waehrend des Trainings
  den besseren Online-Ertrag.

Aufruf:  python run.py        (mit Plot, falls matplotlib vorhanden)
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

    print(f"{'Algorithmus':12s} {'Online-Ertrag (letzte 100 Ep., Mittel)':>38s}  {'greedy-Ertrag':>14s}")
    for algo in ("sarsa", "qlearning"):
        online = curves[algo][-100:].mean()
        greedy = rollout_greedy(env, final_agents[algo])
        print(f"{algo:12s} {online:38.1f}  {greedy:14.0f}")

    print("\n--- greedy-Policy SARSA ---")
    print(env.render_policy(final_agents["sarsa"].greedy_policy()))
    print("\n--- greedy-Policy Q-Learning ---")
    print(env.render_policy(final_agents["qlearning"].greedy_policy()))

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.5))
        for algo, label in [("sarsa", "SARSA (on-policy)"),
                            ("qlearning", "Q-Learning (off-policy)")]:
            plt.plot(moving_average(curves[algo], 10), label=label, lw=1.5)
        plt.ylim(-100, 0)
        plt.xlabel("Episode"); plt.ylabel("Belohnungssumme je Episode (gleitendes Mittel)")
        plt.title("Cliff Walking: SARSA vs. Q-Learning (online, waehrend Training)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        import os
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/cliff_comparison.png", dpi=110)
        print("\nPlot gespeichert: results/cliff_comparison.png")
    except Exception as e:
        print("(kein Plot:", e, ")")


if __name__ == "__main__":
    main()
