"""Exakte Optimalsteuerung: LQR via Riccati-Gleichung (Modell BEKANNT).

Das ist die Bruecke Modul 13 -> 14: dort lieferte **Value Iteration** die exakte Referenz fuer
einen diskreten MDP; hier liefert der **LQR** die exakte Referenz fuer ein kontinuierliches
Steuerungsproblem. Beides ist derselbe Bellman-Gedanke — nur ist er im linear-quadratischen
Fall in geschlossener Form loesbar.

Diskrete algebraische Riccati-Gleichung (DARE):
    P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
Optimale Rueckfuehrung (linear!):
    K = (R + B^T P B)^{-1} B^T P A          ->  u* = -K x
Optimale Kosten-to-go (unendlicher Horizont, deterministisch):
    J*(x0) = x0^T P x0

Wir loesen die DARE per Fixpunkt-Iteration — das ist inhaltlich **Value Iteration auf der
quadratischen Wertfunktion** V(x) = x^T P x (statt auf einer Tabelle).
"""
from __future__ import annotations
import numpy as np


def solve_dare(A, B, Q, R, tol=1e-12, max_iter=10_000):
    """Riccati-Fixpunktiteration. Rueckgabe: (P, K, iters)."""
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
    """Optimale Rueckfuehrmatrix K fuer das System in `env` (u* = -K x)."""
    P, K, iters = solve_dare(env.A, env.B, env.Q, env.R, **kw)
    return P, K, iters


def closed_loop_eigenvalues(env, K):
    """Eigenwerte von (A - B K). Betrag < 1 => stabil (diskrete Zeit)."""
    return np.linalg.eigvals(env.A - env.B @ K)


def optimal_cost(P, x0):
    """J*(x0) = x0^T P x0 (unendlicher Horizont, deterministisch)."""
    x0 = np.asarray(x0, float)
    return float(x0 @ P @ x0)


if __name__ == "__main__":
    from lqr_env import LinearQuadraticSystem
    env = LinearQuadraticSystem()
    P, K, iters = lqr_gain(env)
    ev = closed_loop_eigenvalues(env, K)
    print(f"Riccati konvergiert nach {iters} Iterationen.")
    print("P =\n", np.round(P, 4))
    print("K =", np.round(K.ravel(), 4), " -> optimale Politik u* = -K x")
    print("closed-loop Eigenwerte:", np.round(ev, 4), "| Betraege:", np.round(np.abs(ev), 4),
          "(< 1 => stabil)")
    x0 = np.array([1.0, 0.0])
    sim_cost, _ = env.rollout_linear(-K.ravel(), x0, horizon=400)
    print(f"\nProbe: simulierte Kosten von u=-Kx ab x0={x0}: {sim_cost:.4f}")
    print(f"       Theorie  J*(x0) = x0^T P x0        : {optimal_cost(P, x0):.4f}")
