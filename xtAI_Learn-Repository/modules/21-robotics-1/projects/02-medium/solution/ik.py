"""Inverse kinematics: analytic and numerical (J^T / pseudoinverse / DLS).
(Reference solution P02-medium)   Module 21 — Robotics 1.

Planar arm with n revolute joints, link lengths l_1..l_n. The angles add up along the
chain (script ch. 4), so with the cumulative angles c_i = q_1+...+q_i:

    x = sum_i l_i cos(c_i),      y = sum_i l_i sin(c_i)

    J[0,k] = -sum_{i>=k} l_i sin(c_i)
    J[1,k] =  sum_{i>=k} l_i cos(c_i)
"""
import numpy as np


# ==========================================================================
# Forward kinematics + Jacobian (general, n links)
# ==========================================================================
def fk(q, lengths):
    """End effector position (2,) of the planar arm."""
    c = np.cumsum(np.asarray(q, float))
    L = np.asarray(lengths, float)
    return np.array([np.sum(L * np.cos(c)), np.sum(L * np.sin(c))])


def joints(q, lengths):
    """All joint positions (n+1, 2) incl. the base — for plots."""
    c = np.cumsum(np.asarray(q, float))
    L = np.asarray(lengths, float)
    xs = np.concatenate([[0.0], np.cumsum(L * np.cos(c))])
    ys = np.concatenate([[0.0], np.cumsum(L * np.sin(c))])
    return np.column_stack([xs, ys])


def jacobian(q, lengths):
    """Position Jacobian (2, n): J[:,k] = derivative of the EE position w.r.t. q_k."""
    c = np.cumsum(np.asarray(q, float))
    L = np.asarray(lengths, float)
    n = len(L)
    J = np.zeros((2, n))
    for k in range(n):
        J[0, k] = -np.sum(L[k:] * np.sin(c[k:]))
        J[1, k] = np.sum(L[k:] * np.cos(c[k:]))
    return J


# ==========================================================================
# Analytic IK for the 2-joint arm
# ==========================================================================
def analytic_ik_2link(target, lengths):
    """All exact solutions (elbow up/down). Empty list if unreachable.

    Law of cosines:  cos q2 = (r^2 - l1^2 - l2^2) / (2 l1 l2),   q2 = +- arccos(.)
                     q1 = atan2(y,x) - atan2(l2 sin q2, l1 + l2 cos q2)
    Reachable only for |l1-l2| <= r <= l1+l2.
    """
    x, y = target
    l1, l2 = lengths
    r = np.hypot(x, y)
    if r > l1 + l2 + 1e-12 or r < abs(l1 - l2) - 1e-12:
        return []                                   # outside the workspace
    cq2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cq2 = np.clip(cq2, -1.0, 1.0)                   # catch rounding errors
    sols = []
    for sign in (+1.0, -1.0):
        q2 = sign * np.arccos(cq2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        sols.append(np.array([q1, q2]))
    # at q2=0 or pi both solutions coincide -> remove the duplicate
    if len(sols) == 2 and np.allclose(sols[0], sols[1], atol=1e-9):
        sols = [sols[0]]
    return sols


# ==========================================================================
# Numerical IK: J^T / pseudoinverse / damped least squares
# ==========================================================================
def numeric_ik(target, lengths, q0, method="dls", lam=0.1, max_iter=200, tol=1e-6,
               max_step=None):
    """Iterative IK. Returns (q, history_err, converged, max_step_norm).

    method:
      'transpose' : dq = alpha J^T e   with the optimal step size alpha
                    (alpha = <e, JJ^T e> / <JJ^T e, JJ^T e>)  -> gradient descent
      'pinv'      : dq = J^+ e         -> fast, EXPLODES near singularities
      'dls'       : dq = J^T (J J^T + lam^2 I)^-1 e  -> always well defined
    """
    q = np.array(q0, float)
    target = np.asarray(target, float)
    hist = []
    max_step_norm = 0.0
    converged = False
    for _ in range(max_iter):
        e = target - fk(q, lengths)
        err = np.linalg.norm(e)
        hist.append(err)
        if err < tol:
            converged = True
            break
        J = jacobian(q, lengths)
        if method == "transpose":
            JJt_e = J @ (J.T @ e)
            denom = JJt_e @ JJt_e
            alpha = (e @ JJt_e) / denom if denom > 1e-30 else 0.0
            dq = alpha * (J.T @ e)
        elif method == "pinv":
            dq = np.linalg.pinv(J) @ e
        elif method == "dls":
            A = J @ J.T + (lam**2) * np.eye(2)
            dq = J.T @ np.linalg.solve(A, e)
        else:
            raise ValueError(f"unknown method: {method}")
        max_step_norm = max(max_step_norm, float(np.linalg.norm(dq)))
        if max_step is not None:                    # optional step limiting
            nrm = np.linalg.norm(dq)
            if nrm > max_step:
                dq = dq * (max_step / nrm)
        q = q + dq
    return q, np.array(hist), converged, max_step_norm


def nullspace_step(q, lengths, z):
    """Null-space projection (I - J^+ J) z: a joint motion that does NOT move the end effector.
    Non-zero only for redundant arms (n > 2)."""
    J = jacobian(q, lengths)
    n = len(q)
    return (np.eye(n) - np.linalg.pinv(J) @ J) @ np.asarray(z, float)
