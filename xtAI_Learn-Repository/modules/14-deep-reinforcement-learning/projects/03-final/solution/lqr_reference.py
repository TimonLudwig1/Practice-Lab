"""Exact optimal control: LQR via the Riccati equation (model KNOWN).

This is the bridge module 13 -> 14: there **value iteration** provided the exact reference for a
discrete MDP; here the **LQR** provides the exact reference for a continuous control problem. Both
are the same Bellman idea — only in the linear-quadratic case it is solvable in closed form.

Discrete algebraic Riccati equation (DARE):
    P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
Optimal feedback (linear!):
    K = (R + B^T P B)^{-1} B^T P A          ->  u* = -K x
Optimal cost-to-go (infinite horizon, deterministic):
    J*(x0) = x0^T P x0

We solve the DARE by fixed-point iteration — this is, in content, **value iteration on the
quadratic value function** V(x) = x^T P x (instead of on a table).
"""
from __future__ import annotations
import numpy as np


def solve_dare(A, B, Q, R, tol=1e-12, max_iter=10_000):
    """Riccati fixed-point iteration. Returns: (P, K, iters)."""
    P = Q.copy()
    iters = max_iter
    for i in range(max_iter):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            iters = i + 1
            break
        P = P_new
    S = R + B.T @ P @ B
    K = np.linalg.solve(S, B.T @ P @ A)
    return P, K, iters


def lqr_gain(env, **kw):
    """The optimal feedback matrix K for the system in `env` (u* = -K x)."""
    P, K, iters = solve_dare(env.A, env.B, env.Q, env.R, **kw)
    return P, K, iters


def closed_loop_eigenvalues(env, K):
    """Eigenvalues of (A - B K). Magnitude < 1 => stable (discrete time)."""
    return np.linalg.eigvals(env.A - env.B @ K)


def optimal_cost(P, x0):
    """J*(x0) = x0^T P x0 (infinite horizon, deterministic)."""
    x0 = np.asarray(x0, float)
    return float(x0 @ P @ x0)


if __name__ == "__main__":
    from lqr_env import LinearQuadraticSystem
    env = LinearQuadraticSystem()
    P, K, iters = lqr_gain(env)
    ev = closed_loop_eigenvalues(env, K)
    print(f"Riccati converges after {iters} iterations.")
    print("P =\n", np.round(P, 4))
    print("K =", np.round(K.ravel(), 4), " -> the optimal policy u* = -K x")
    print("closed-loop eigenvalues:", np.round(ev, 4), "| magnitudes:", np.round(np.abs(ev), 4),
          "(< 1 => stable)")
    x0 = np.array([1.0, 0.0])
    sim_cost, _ = env.rollout_linear(-K.ravel(), x0, horizon=400)
    print(f"\nCheck: simulated cost of u=-Kx from x0={x0}: {sim_cost:.4f}")
    print(f"       theory  J*(x0) = x0^T P x0          : {optimal_cost(P, x0):.4f}")
