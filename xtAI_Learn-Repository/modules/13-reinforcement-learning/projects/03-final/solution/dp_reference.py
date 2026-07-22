"""Exact reference solution via value iteration (model KNOWN).

This is the bridge to module 07: there you built value iteration on a gridworld. Here the
same idea runs on the inventory-management MDP. Because we know the model (P, R) from
inventory_env.py, VI yields the *true* optimal policy and the optimal Q values — the benchmark
that the model-free learners (Q-learning etc.) have to measure up to.

Bellman optimality update (discounted, continuing task):
    Q*(i,a) = R(i,a) + gamma * sum_{i'} P(i'|i,a) * max_{a' admissible} Q*(i', a')
"""
from __future__ import annotations
import numpy as np


def value_iteration(env, gamma=0.95, theta=1e-9, max_iter=10_000):
    nS, nA = env.n_states, env.n_actions
    # mask inadmissible actions with -inf so they never win in the max/argmax
    R = np.full((nS, nA), -np.inf)
    P = np.zeros((nS, nA, nS))
    for i in range(nS):
        for a in env.valid_actions(i):
            R[i, a] = env.expected_reward(i, a)
            P[i, a] = env.transition_probs(i, a)

    V = np.zeros(nS)
    for it in range(max_iter):
        Q = np.where(np.isfinite(R), R + gamma * (P @ V), -np.inf)
        V_new = Q.max(axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            break
    Q = np.where(np.isfinite(R), R + gamma * (P @ V), -np.inf)
    policy = Q.argmax(axis=1)          # the optimal order quantity per inventory level
    return {"V": V, "Q": Q, "policy": policy, "iters": it + 1}


def policy_value(env, policy, gamma=0.95):
    """Exact discounted value V^pi of a (deterministic) policy by directly solving the linear
    system V = R_pi + gamma * P_pi V. Uses the known model — only for *evaluating* learned
    policies, not for learning."""
    nS = env.n_states
    R_pi = np.array([env.expected_reward(i, int(policy[i])) for i in range(nS)])
    P_pi = np.array([env.transition_probs(i, int(policy[i])) for i in range(nS)])
    # (I - gamma P) V = R
    V = np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)
    return V


def order_up_to_levels(env, policy):
    """Computes the (inventory -> target inventory level x=i+a) curve. With an (s,S) structure
    x is constant = S for all i below the order point s and = i above it (order nothing)."""
    return np.array([i + policy[i] for i in range(env.n_states)])


def greedy_from_Q(Q, mask_fn, n_states):
    """Greedy policy from an arbitrary Q table, respecting the action masks."""
    pol = np.zeros(n_states, dtype=int)
    for i in range(n_states):
        q = Q[i].copy()
        q[~mask_fn(i)] = -np.inf
        pol[i] = int(np.argmax(q))
    return pol


if __name__ == "__main__":
    from inventory_env import InventoryEnv
    env = InventoryEnv()
    res = value_iteration(env)
    print(f"Value iteration converges in {res['iters']} iterations.")
    print("Inventory i :", list(range(env.n_states)))
    print("Order       :", res["policy"].tolist())
    print("Target x=i+a:", order_up_to_levels(env, res["policy"]).tolist())
