"""Model-based control of a 2-link arm: PD+gravity vs. computed torque.
(Reference solution P02-medium)   Module 22 — Robotics 2.

The dynamics model (mass matrix, Coriolis, gravity, forward dynamics) is given — you built it in
P01. Here you implement the two CONTROLLERS and study why cancelling the dynamics turns a non-linear,
coupled arm into a set of trivial linear systems.
"""
import numpy as np


# ==========================================================================
# Arm dynamics model (given, from P01). A `scale` on the masses lets the
# CONTROLLER use a deliberately wrong model (experiment 4).
# ==========================================================================
class ArmModel:
    def __init__(self, l1=1.0, l2=1.0, m1=1.0, m2=1.0, g=9.81):
        self.l1, self.l2, self.m1, self.m2, self.g = l1, l2, m1, m2, g

    def mass_matrix(self, q):
        l1, l2, m1, m2 = self.l1, self.l2, self.m1, self.m2
        c2 = np.cos(q[1])
        m12 = m2 * (l2**2 + l1 * l2 * c2)
        return np.array([[m1 * l1**2 + m2 * (l1**2 + l2**2 + 2 * l1 * l2 * c2), m12],
                         [m12, m2 * l2**2]])

    def coriolis(self, q, dq):
        h = self.m2 * self.l1 * self.l2 * np.sin(q[1])
        return np.array([[-h * dq[1], -h * (dq[0] + dq[1])],
                         [h * dq[0], 0.0]])

    def gravity(self, q):
        l1, l2, m1, m2, g = self.l1, self.l2, self.m1, self.m2, self.g
        q1, q2 = q
        return np.array([(m1 + m2) * g * l1 * np.cos(q1) + m2 * g * l2 * np.cos(q1 + q2),
                         m2 * g * l2 * np.cos(q1 + q2)])

    def forward_dynamics(self, q, dq, tau):
        return np.linalg.solve(self.mass_matrix(q), tau - self.coriolis(q, dq) @ dq - self.gravity(q))


TRUE = ArmModel()          # the real robot (the plant)


# ==========================================================================
# Reference trajectory (given): a smooth sinusoid per joint
# ==========================================================================
def reference(t, w=2.0):
    """Desired (qd, dqd, ddqd) at time t. Higher w = faster motion."""
    qd = np.array([0.8 * np.sin(w * t), 0.6 * np.sin(w * t + 0.5)])
    dqd = np.array([0.8 * w * np.cos(w * t), 0.6 * w * np.cos(w * t + 0.5)])
    ddqd = np.array([-0.8 * w**2 * np.sin(w * t), -0.6 * w**2 * np.sin(w * t + 0.5)])
    return qd, dqd, ddqd


# ==========================================================================
# The two controllers
# ==========================================================================
def pd_gravity_control(model, q, dq, qd, dqd, ddqd, Kp, Kd):
    """PD + gravity compensation: tau = -Kp e - Kd edot + g(q), with e = q - qd.

    Cancels gravity and pulls the error to zero like a spring-damper. It ignores ddqd and the
    inertia/Coriolis coupling, so it REGULATES a setpoint well but LAGS on a fast trajectory.
    """
    e = q - qd
    edot = dq - dqd
    return -Kp @ e - Kd @ edot + model.gravity(q)


def computed_torque_control(model, q, dq, qd, dqd, ddqd, Kp, Kd):
    """Computed torque (inverse-dynamics control):
        tau = M(q) (ddqd - Kd edot - Kp e) + C(q,dq) dq + g(q),  e = q - qd.

    Substituting into the manipulator equation cancels C and g exactly and leaves the linear,
    decoupled error dynamics  edot-dot + Kd edot + Kp e = 0  -> exact tracking.
    """
    e = q - qd
    edot = dq - dqd
    v = ddqd - Kd @ edot - Kp @ e
    return model.mass_matrix(q) @ v + model.coriolis(q, dq) @ dq + model.gravity(q)


# ==========================================================================
# Closed-loop simulation (given): RK4 on the TRUE plant, torque from the controller
# ==========================================================================
def simulate(controller, q0, dq0, Kp, Kd, ref_fn=reference, ctrl_model=TRUE, plant=TRUE,
             T=8.0, dt=1e-3):
    """Run the closed loop. The controller may use `ctrl_model` (possibly wrong); the arm moves
    according to `plant` (the true dynamics). Returns ts, Q, QD, err (||q - qd|| per step)."""
    n = int(T / dt)
    s = np.array([*q0, *dq0], float)
    Q = np.zeros((n, 2)); QD = np.zeros((n, 2)); err = np.zeros(n); ts = np.arange(n) * dt

    def state_deriv(state, tau):
        q, dq = state[:2], state[2:]
        return np.concatenate([dq, plant.forward_dynamics(q, dq, tau)])

    for k in range(n):
        t = k * dt
        qd, dqd, ddqd = ref_fn(t)
        tau = controller(ctrl_model, s[:2], s[2:], qd, dqd, ddqd, Kp, Kd)
        k1 = state_deriv(s, tau)
        k2 = state_deriv(s + 0.5 * dt * k1, tau)
        k3 = state_deriv(s + 0.5 * dt * k2, tau)
        k4 = state_deriv(s + dt * k3, tau)
        s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q[k] = s[:2]; QD[k] = s[2:]; err[k] = np.linalg.norm(s[:2] - qd)
    return ts, Q, QD, err


def rms_second_half(err):
    """RMS tracking error over the second half (after the initial transient has decayed)."""
    return float(np.sqrt(np.mean(err[len(err) // 2:] ** 2)))
