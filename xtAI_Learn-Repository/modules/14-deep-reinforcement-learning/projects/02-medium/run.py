"""Comparison of three policy-gradient variants on CartPole:
  - REINFORCE WITHOUT a baseline  (high variance -> slow/jittery)
  - REINFORCE WITH a baseline     (return normalized -> markedly more stable)
  - actor-critic (A2C)            (the critic as a learned baseline -> advantage)

Call:  python run.py
"""
from __future__ import annotations
import time
import numpy as np
from cartpole import CartPole
from policy_gradient import REINFORCE, ActorCritic


def run_reinforce(normalize, n_episodes=600, seed=0, solved=475.0):
    agent = REINFORCE(normalize=normalize, seed=seed)
    env = CartPole(seed=seed)
    scores = []
    t0 = time.time()
    for ep in range(n_episodes):
        s = env.reset(); done = False
        log_probs, rewards = [], []
        while not done:
            a, logp = agent.select_action(s)
            s, r, done = env.step(a)
            log_probs.append(logp); rewards.append(r)
        agent.update(log_probs, rewards)
        scores.append(sum(rewards))
        if len(scores) >= 20 and np.mean(scores[-20:]) >= solved:
            break
    return scores, time.time() - t0


def run_actor_critic(n_episodes=600, seed=0, solved=475.0):
    agent = ActorCritic(seed=seed)
    env = CartPole(seed=seed)
    scores = []
    t0 = time.time()
    for ep in range(n_episodes):
        s = env.reset(); done = False
        log_probs, values, rewards = [], [], []
        while not done:
            a, logp, v = agent.select_action(s)
            s, r, done = env.step(a)
            log_probs.append(logp); values.append(v); rewards.append(r)
        agent.update(log_probs, values, rewards)
        scores.append(sum(rewards))
        if len(scores) >= 20 and np.mean(scores[-20:]) >= solved:
            break
    return scores, time.time() - t0


def moving_average(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid")


SEEDS = (0, 1, 2, 3, 4)


def main():
    # Deep RL fluctuates STRONGLY over seeds -> a single run says almost nothing.
    # That is why we average over several seeds (script 5: "reproducibility").
    methods = {
        "REINFORCE (no baseline)": lambda sd: run_reinforce(False, seed=sd),
        "REINFORCE (with baseline)": lambda sd: run_reinforce(True, seed=sd),
        "actor-critic (A2C)":       lambda sd: run_actor_critic(seed=sd),
    }
    print(f"Averaging over {len(SEEDS)} seeds {SEEDS}\n")
    print(f"{'method':28s} {'episodes until solved (mean±std)':>34s} {'time':>7s}")
    runs = {}
    for label, fn in methods.items():
        eps_to_solve, curves, t_total = [], [], 0.0
        for sd in SEEDS:
            scores, dt = fn(sd)
            t_total += dt
            curves.append(scores)
            eps_to_solve.append(len(scores) if np.mean(scores[-20:]) >= 475 else np.nan)
        runs[label] = curves
        m, s = np.nanmean(eps_to_solve), np.nanstd(eps_to_solve)
        n_solved = int(np.sum(~np.isnan(eps_to_solve)))
        print(f"{label:28s} {m:20.0f} ± {s:<4.0f} ({n_solved}/{len(SEEDS)}) {t_total:6.0f}s")

    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        plt.figure(figsize=(8, 4.6))
        for label, curves in runs.items():
            # bring the curves to the same length (repeat the last value), then average
            L = max(len(c) for c in curves)
            padded = np.array([np.pad(c, (0, L - len(c)), mode="edge") for c in curves])
            mean_curve = padded.mean(axis=0)
            plt.plot(range(19, L), moving_average(mean_curve, 20), lw=1.6, label=label)
        plt.axhline(500, ls="--", color="gray", lw=1)
        plt.xlabel("episode"); plt.ylabel(f"return (mean over {len(SEEDS)} seeds, moving 20)")
        plt.title("Policy gradient on CartPole: variance reduction via baseline/critic")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("results/policy_gradient.png", dpi=110)
        print("\nPlot saved: results/policy_gradient.png")
    except Exception as e:
        print("(no plot:", e, ")")


if __name__ == "__main__":
    main()
