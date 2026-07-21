"""Hauptexperiment: modellfreies RL vs. exakte DP-Optimalpolitik auf Bestandsmanagement.

Ablauf:
  1. Value Iteration auf dem bekannten Modell -> optimale (s,S)-Politik + V* (Referenz).
  2. MC / SARSA / Q-Learning / Expected SARSA modellfrei trainieren.
  3. Jede gelernte greedy-Politik EXAKT bewerten (policy_value) und mit V* vergleichen:
     - Qualitaet  = mean(V^pi) / mean(V*)   (1.0 = optimal)
     - Uebereinstimmung der Aktionen mit der DP-Politik.
  4. Plots: Lernkurven + Ziel-Lagerstand-Kurven (learned vs optimal).

Aufruf:  python run.py            (Sekunden bis ~1 min auf der CPU)
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
    # Optimalitaetsluecke in %: wie viel schlechter als das DP-Optimum (0% = optimal).
    gap = (V_star.mean() - V.mean()) / abs(V_star.mean()) * 100.0
    # Aktions-Uebereinstimmung nur ueber die *relevanten* Zustaende (Bestand <= S),
    # denn hohe Bestaende werden unter einer guten Politik praktisch nie erreicht.
    S = int(order_up_to_levels(env, opt_policy).max())
    rel = slice(0, S + 1)
    agree = float(np.mean(policy[rel] == opt_policy[rel]))
    return V.mean(), gap, agree


def main():
    env = InventoryEnv(seed=0)
    print("Parameter: M=%d, lam=%.1f, c=%.1f, K=%.1f, h=%.1f, p=%.1f, gamma=%.2f"
          % (env.M, env.lam, env.c, env.K, env.h, env.p, GAMMA))

    # 1) DP-Referenz
    dp = value_iteration(env, gamma=GAMMA)
    opt_policy, V_star = dp["policy"], dp["V"]
    print(f"\n[DP] Value Iteration: {dp['iters']} Iter., mittleres V* = {V_star.mean():.2f}")
    print("     optimale (s,S)-Politik, Ziel-Lagerstand x=i+a:",
          order_up_to_levels(env, opt_policy).tolist())

    # 2)+3) modellfreie Lerner
    print(f"\n[modellfrei] Training ueber {N_STEPS:,} Schritte je Verfahren:")
    print(f"{'Verfahren':16s} {'mittl. V^pi':>12s} {'Opt.-Luecke':>12s} {'Match(i<=S)':>12s}")
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

    # 4) Hyperparameter-Studie: Lernrate alpha fuer Q-Learning
    print("\n[Studie] Q-Learning: Einfluss der Lernrate alpha (Opt.-Luecke, 0%=optimal):")
    for alpha in (0.02, 0.05, 0.1, 0.3, 0.6):
        ag = AGENTS["Q-Learning"](env, alpha=alpha, gamma=GAMMA, epsilon=0.1, seed=2)
        ag.train(n_steps=N_STEPS)
        _, gap, _ = evaluate(env, ag.greedy_policy(), opt_policy, V_star)
        print(f"   alpha={alpha:>4}:  Opt.-Luecke={gap:.2f}%")

    # ---- Plots ----
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        w = 5000
        for name, hist in curves.items():
            ax1.plot(moving_average(hist, w), label=name, lw=1.2)
        ax1.set(xlabel="Schritt", ylabel=f"mittlere Belohnung (gleitend, w={w})",
                title="Lernkurven (Online-Kosten, hoeher=besser)")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        xs = np.arange(env.n_states)
        ax2.step(xs, order_up_to_levels(env, opt_policy), where="mid",
                 label="DP-Optimum (s,S)", color="k", lw=2)
        for name in ("Q-Learning", "SARSA"):
            ax2.step(xs, order_up_to_levels(env, policies[name]), where="mid",
                     label=name, lw=1.3, alpha=0.8)
        ax2.plot(xs, xs, ":", color="gray", lw=1, label="nichts bestellen (x=i)")
        ax2.set(xlabel="Bestand i", ylabel="Ziel-Lagerstand x = i + a",
                title="Gelernte vs. optimale Politik")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig("results/inventory_rl.png", dpi=110)
        print("\nPlot gespeichert: results/inventory_rl.png")
    except Exception as e:
        print("(kein Plot:", e, ")")


if __name__ == "__main__":
    main()
