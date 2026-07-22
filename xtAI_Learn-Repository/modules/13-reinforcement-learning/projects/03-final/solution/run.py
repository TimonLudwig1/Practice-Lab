"""Main experiment: model-free RL vs. the exact DP optimal policy on inventory management.

Flow:
  1. Value iteration on the known model -> the optimal (s,S) policy + V* (reference).
  2. Train MC / SARSA / Q-learning / expected SARSA model-free.
  3. Evaluate each learned greedy policy EXACTLY (policy_value) and compare with V*:
     - quality    = mean(V^pi) / mean(V*)   (1.0 = optimal)
     - agreement of the actions with the DP policy.
  4. Plots: learning curves + target inventory-level curves (learned vs. optimal).

Call:  python run.py            (seconds to ~1 min on the CPU)
"""
from __future__ import annotations
import numpy as np
from inventory_env import InventoryEnv
from dp_reference import value_iteration, policy_value, order_up_to_levels
from agents import AGENTS

GAMMA = 0.95
N_STEPS = 300_000


def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def evaluate(env, policy, opt_policy, V_star):
    V = policy_value(env, policy, gamma=GAMMA)
    # optimality gap in %: how much worse than the DP optimum (0% = optimal).
    gap = (V_star.mean() - V.mean()) / abs(V_star.mean()) * 100.0
    # action agreement only over the *relevant* states (inventory <= S),
    # because high inventories are practically never reached under a good policy.
    S = int(order_up_to_levels(env, opt_policy).max())
    rel = slice(0, S + 1)
    agree = float(np.mean(policy[rel] == opt_policy[rel]))
    return V.mean(), gap, agree


def main():
    env = InventoryEnv(seed=0)
    print("Parameters: M=%d, lam=%.1f, c=%.1f, K=%.1f, h=%.1f, p=%.1f, gamma=%.2f"
          % (env.M, env.lam, env.c, env.K, env.h, env.p, GAMMA))

    # 1) DP reference
    dp = value_iteration(env, gamma=GAMMA)
    opt_policy, V_star = dp["policy"], dp["V"]
    print(f"\n[DP] value iteration: {dp['iters']} iter., mean V* = {V_star.mean():.2f}")
    print("     optimal (s,S) policy, target inventory level x=i+a:",
          order_up_to_levels(env, opt_policy).tolist())

    # 2)+3) model-free learners
    print(f"\n[model-free] training over {N_STEPS:,} steps per method:")
    print(f"{'method':16s} {'mean V^pi':>12s} {'opt. gap':>12s} {'match(i<=S)':>12s}")
    curves = {}
    policies = {}
    for name, Agent in AGENTS.items():
        agent = Agent(env, gamma=GAMMA, epsilon=0.1, seed=1) if name == "MC" else \
                Agent(env, alpha=0.1, gamma=GAMMA, epsilon=0.1, seed=1)
        hist = agent.train(n_steps=N_STEPS)
        pol = agent.greedy_policy()
        vmean, gap, agree = evaluate(env, pol, opt_policy, V_star)
        curves[name] = hist
        policies[name] = pol
        print(f"{name:16s} {vmean:12.2f} {gap:11.2f}% {agree*100:11.0f}%")

    # 4) hyperparameter study: the learning rate alpha for Q-learning
    print("\n[study] Q-learning: the influence of the learning rate alpha (opt. gap, 0%=optimal):")
    for alpha in (0.02, 0.05, 0.1, 0.3, 0.6):
        ag = AGENTS["Q-Learning"](env, alpha=alpha, gamma=GAMMA, epsilon=0.1, seed=2)
        ag.train(n_steps=N_STEPS)
        _, gap, _ = evaluate(env, ag.greedy_policy(), opt_policy, V_star)
        print(f"   alpha={alpha:>4}:  opt. gap={gap:.2f}%")

    # ---- plots ----
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        w = 5000
        for name, hist in curves.items():
            ax1.plot(moving_average(hist, w), label=name, lw=1.2)
        ax1.set(xlabel="step", ylabel=f"mean reward (moving, w={w})",
                title="Learning curves (online costs, higher=better)")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        xs = np.arange(env.n_states)
        ax2.step(xs, order_up_to_levels(env, opt_policy), where="mid",
                 label="DP optimum (s,S)", color="k", lw=2)
        for name in ("Q-Learning", "SARSA"):
            ax2.step(xs, order_up_to_levels(env, policies[name]), where="mid",
                     label=name, lw=1.3, alpha=0.8)
        ax2.plot(xs, xs, ":", color="gray", lw=1, label="order nothing (x=i)")
        ax2.set(xlabel="inventory i", ylabel="target inventory level x = i + a",
                title="Learned vs. optimal policy")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig("results/inventory_rl.png", dpi=110)
        print("\nPlot saved: results/inventory_rl.png")
    except Exception as e:
        print("(no plot:", e, ")")


if __name__ == "__main__":
    main()
