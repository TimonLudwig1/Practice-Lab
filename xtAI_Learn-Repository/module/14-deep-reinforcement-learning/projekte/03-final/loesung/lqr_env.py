"""Lineares System mit quadratischen Kosten — der klassische Optimal-Control-Aufbau.

System (doppelter Integrator = Masse auf reibungsfreier Schiene):
    Zustand  x = [Position, Geschwindigkeit],  Stellgroesse u = Kraft (kontinuierlich, skalar)
    Dynamik  x_{t+1} = A x_t + B u_t
             A = [[1, dt], [0, 1]],  B = [[dt^2/2], [dt]]
Kosten je Schritt (quadratisch):
    c(x,u) = x^T Q x + u^T R u          -> Belohnung r = -c(x,u)

Ziel: den Wagen aus einem zufaelligen Startzustand moeglichst schnell UND
energiesparend in den Ursprung (x=0, v=0) bringen. Q gewichtet den Zustandsfehler,
R die Stellenergie.

WICHTIG: A, B, Q, R sind hier als Attribute sichtbar — aber NUR die Referenzloesung
(lqr_reference.py) darf sie benutzen. Der modellfreie Lerner sieht ausschliesslich
(x, u, r, x') aus reset()/step().
"""
from __future__ import annotations
import numpy as np


class LinearQuadraticSystem:
    def __init__(self, dt=0.1, q_pos=1.0, q_vel=0.1, r_ctrl=0.1, horizon=50, seed=None):
        self.dt = dt
        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.B = np.array([[0.5 * dt * dt], [dt]])
        self.Q = np.diag([q_pos, q_vel])
        self.R = np.array([[r_ctrl]])
        self.horizon = horizon
        self.n_states = 2
        self.n_actions = 1
        self.rng = np.random.default_rng(seed)
        self._x = np.zeros(2)
        self._t = 0

    def reset(self, x0=None):
        self._x = self.rng.normal(0.0, 1.0, 2) if x0 is None else np.asarray(x0, float).copy()
        self._t = 0
        return self._x.copy()

    def cost(self, x, u):
        """Quadratische Schrittkosten c = x^T Q x + u^T R u (u skalar)."""
        u = float(np.asarray(u).ravel()[0])
        return float(x @ self.Q @ x + u * self.R[0, 0] * u)

    def step(self, u):
        """Rueckgabe: (x_next, reward, done) mit reward = -cost."""
        u = float(np.asarray(u).ravel()[0])
        r = -self.cost(self._x, u)
        self._x = self.A @ self._x + self.B.ravel() * u
        self._t += 1
        done = self._t >= self.horizon
        return self._x.copy(), r, done

    # ---- Hilfsfunktion: eine lineare Rueckfuehrung u = w^T x ausrollen ----
    def rollout_linear(self, w, x0, horizon=None):
        """Simuliert u = w @ x deterministisch. Rueckgabe: (Gesamtkosten, Trajektorie)."""
        H = self.horizon if horizon is None else horizon
        x = np.asarray(x0, float).copy()
        total = 0.0
        traj = [x.copy()]
        for _ in range(H):
            u = float(np.asarray(w).ravel() @ x)
            total += self.cost(x, u)
            x = self.A @ x + self.B.ravel() * u
            traj.append(x.copy())
        return total, np.array(traj)
