"""Model Predictive Control from scratch: condensed QP, receding horizon, constraints.
(Reference solution P03-final)   Module 23 — Advanced Automation.

Linear plant x_{k+1} = A x_k + B u_k. At each step MPC solves the finite-horizon constrained
optimal-control problem, condensed into a convex QP in the stacked input U, and applies u_0.
Unconstrained (with the LQR terminal cost) it equals the LQR of module 14. numpy + scipy only.
"""
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize, LinearConstraint


# ==========================================================================
# Plant and LQR baseline
# ==========================================================================
def double_integrator(dt=0.1):
    """An automated positioning stage: state (position, velocity), input force.
    Returns A, B, Q, R."""
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt], [dt]])
    Q = np.diag([1.0, 0.1])
    R = np.array([[0.1]])
    return A, B, Q, R


def lqr(A, B, Q, R):
    """Discrete infinite-horizon LQR: returns the gain K (u = -K x) and cost-to-go P (Riccati)."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


# ==========================================================================
# Condensation: predicted states as an affine function of the input sequence
# ==========================================================================
def prediction_matrices(A, B, N):
    """X = Sx x0 + Su U, where X stacks x_1..x_N and U stacks u_0..u_{N-1}."""
    n = A.shape[0]; m = B.shape[1]
    Sx = np.zeros((N * n, n))
    Su = np.zeros((N * n, N * m))
    for i in range(N):
        Sx[i * n:(i + 1) * n, :] = np.linalg.matrix_power(A, i + 1)
        for j in range(i + 1):
            Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = np.linalg.matrix_power(A, i - j) @ B
    return Sx, Su


def condense(A, B, Q, R, P, N):
    """Turn the MPC cost into the QP  1/2 U' H U + x0' F' U.  Returns H, F, Sx, Su."""
    n = A.shape[0]; m = B.shape[1]
    Sx, Su = prediction_matrices(A, B, N)
    Qbar = np.kron(np.eye(N), Q)
    Qbar[(N - 1) * n:, (N - 1) * n:] = P          # terminal weight on x_N
    Rbar = np.kron(np.eye(N), R)
    H = Su.T @ Qbar @ Su + Rbar
    H = 0.5 * (H + H.T)                            # symmetrise (numerical hygiene)
    F = Su.T @ Qbar @ Sx
    return H, F, Sx, Su


# ==========================================================================
# The MPC controller
# ==========================================================================
class MPC:
    def __init__(self, A, B, Q, R, N=20, u_bound=None, v_bound=None, terminal="lqr"):
        """u_bound: scalar |u|<=u_bound; v_bound: scalar speed limit |x[1]|<=v_bound.
        terminal='lqr' uses the Riccati cost-to-go as the terminal weight P (=> MPC==LQR when
        unconstrained, for any horizon); terminal='stage' uses P=Q instead."""
        self.A, self.B, self.N = A, B, N
        self.n = A.shape[0]; self.m = B.shape[1]
        _, P_lqr = lqr(A, B, Q, R)
        P = P_lqr if terminal == "lqr" else Q
        self.H, self.F, self.Sx, self.Su = condense(A, B, Q, R, P, N)
        self.u_bound, self.v_bound = u_bound, v_bound

    def solve(self, x0):
        """Solve the QP for the whole input sequence U at state x0."""
        x0 = np.asarray(x0, float)
        f = self.F @ x0
        if self.u_bound is None and self.v_bound is None:
            return -np.linalg.solve(self.H, f)        # unconstrained: closed form (== LQR)
        bounds = [(-self.u_bound, self.u_bound)] * (self.N * self.m) if self.u_bound is not None else None
        constraints = []
        if self.v_bound is not None:
            # velocity is component 1 of each predicted state x_{i+1}: rows 1,3,... of X
            rows = np.array([self.Su[i * self.n + 1, :] for i in range(self.N)])
            offs = np.array([(self.Sx @ x0)[i * self.n + 1] for i in range(self.N)])
            constraints = [LinearConstraint(rows, -self.v_bound - offs, self.v_bound - offs)]
        res = minimize(lambda U: 0.5 * U @ self.H @ U + f @ U, np.zeros(self.N * self.m),
                       jac=lambda U: self.H @ U + f, bounds=bounds, constraints=constraints,
                       method="SLSQP", options=dict(maxiter=300, ftol=1e-11))
        return res.x

    def control(self, x0):
        """Receding horizon: solve, apply only the first input."""
        return self.solve(x0)[:self.m]


# ==========================================================================
# Closed-loop simulation
# ==========================================================================
def simulate(controller, A, B, x0, T=80, disturbance=0.0, clip=None):
    """Run the closed loop. `controller(x) -> u`. `disturbance` is a constant added to the velocity.
    `clip` (scalar) saturates the applied input (used to model a naive LQR hitting the actuator limit).
    Returns xs (T+1, n), us (T, m)."""
    x = np.asarray(x0, float).copy()
    n = A.shape[0]
    xs = [x.copy()]; us = []
    d = np.zeros(n); d[-1] = disturbance
    for _ in range(T):
        u = np.atleast_1d(controller(x))
        if clip is not None:
            u = np.clip(u, -clip, clip)
        x = A @ x + (B @ u) + d
        xs.append(x.copy()); us.append(u)
    return np.array(xs), np.array(us)


def lqr_cost(xs, us, Q, R):
    """Closed-loop quadratic cost sum x'Qx + u'Ru (the objective MPC and LQR both target)."""
    c = sum(x @ Q @ x for x in xs[:-1])
    c += sum(float(u @ R @ u) for u in us)
    return float(c)
