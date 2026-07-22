"""Linear system with quadratic costs — the classical optimal-control setup.

System (double integrator = a mass on a frictionless rail):
    state    x = [position, velocity],  control input u = force (continuous, scalar)
    dynamics x_{t+1} = A x_t + B u_t
             A = [[1, dt], [0, 1]],  B = [[dt^2/2], [dt]]
Cost per step (quadratic):
    c(x,u) = x^T Q x + u^T R u          -> reward r = -c(x,u)

Goal: bring the cart from a random start state into the origin (x=0, v=0) as quickly AND
energy-efficiently as possible. Q weights the state error, R the control energy.

IMPORTANT: A, B, Q, R are visible here as attributes — but ONLY the reference solution
(lqr_reference.py) may use them. The model-free learner sees exclusively (x, u, r, x') from
reset()/step().
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
        """Quadratic step cost c = x^T Q x + u^T R u (u scalar)."""
        u = float(np.asarray(u).ravel()[0])
        return float(x @ self.Q @ x + u * self.R[0, 0] * u)

    def step(self, u):
        """Returns: (x_next, reward, done) with reward = -cost."""
        u = float(np.asarray(u).ravel()[0])
        r = -self.cost(self._x, u)
        self._x = self.A @ self._x + self.B.ravel() * u
        self._t += 1
        done = self._t >= self.horizon
        return self._x.copy(), r, done

    # ---- helper: roll out a linear feedback u = w^T x ----
    def rollout_linear(self, w, x0, horizon=None):
        """Simulates u = w @ x deterministically. Returns: (total cost, trajectory)."""
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
