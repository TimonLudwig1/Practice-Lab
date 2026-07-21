"""Exakte Referenzloesung per Value Iteration (Modell BEKANNT).

Das ist die Bruecke zu Modul 07: dort hast du Value Iteration auf einer Gridworld gebaut.
Hier laeuft dieselbe Idee auf dem Bestandsmanagement-MDP. Weil wir das Modell (P, R) aus
inventory_env.py kennen, liefert VI die *wahre* optimale Politik und die optimalen Q-Werte —
die Messlatte, an der sich die modellfreien Lerner (Q-Learning etc.) messen muessen.

Bellman-Optimalitaets-Update (diskontiert, fortlaufende Aufgabe):
    Q*(i,a) = R(i,a) + gamma * sum_{i'} P(i'|i,a) * max_{a' zulaessig} Q*(i', a')
"""
from __future__ import annotations
import numpy as np


def value_iteration(env, gamma=0.95, theta=1e-9, max_iter=10_000):
    nS, nA = env.n_states, env.n_actions
    # unzulaessige Aktionen mit -inf maskieren, damit sie nie im max/argmax gewinnen
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
    policy = Q.argmax(axis=1)          # optimale Bestellmenge je Bestand
    return {"V": V, "Q": Q, "policy": policy, "iters": it + 1}


def policy_value(env, policy, gamma=0.95):
    """Exakter diskontierter Wert V^pi einer (deterministischen) Politik durch direktes
    Loesen des linearen Gleichungssystems V = R_pi + gamma * P_pi V. Nutzt das bekannte
    Modell — nur zur *Bewertung* gelernter Politiken, nicht zum Lernen."""
    nS = env.n_states
    R_pi = np.array([env.expected_reward(i, int(policy[i])) for i in range(nS)])
    P_pi = np.array([env.transition_probs(i, int(policy[i])) for i in range(nS)])
    # (I - gamma P) V = R
    V = np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)
    return V


def order_up_to_levels(env, policy):
    """Rechnet die (Bestand -> Ziel-Lagerstand x=i+a)-Kurve aus. Bei (s,S)-Struktur ist x
    konstant = S fuer alle i unterhalb des Bestellpunkts s und = i darueber (nichts bestellen)."""
    return np.array([i + policy[i] for i in range(env.n_states)])


def greedy_from_Q(Q, mask_fn, n_states):
    """Greedy-Politik aus beliebiger Q-Tabelle unter Beachtung der Aktionsmasken."""
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
    print(f"Value Iteration konvergiert in {res['iters']} Iterationen.")
    print("Bestand i :", list(range(env.n_states)))
    print("Bestellung:", res["policy"].tolist())
    print("Ziel x=i+a:", order_up_to_levels(env, res["policy"]).tolist())
