"""Inverse kinematics: analytic and numerical (J^T / pseudoinverse / DLS).

Module 21 — Robotics 1.

>>> YOUR TASK <<<
Forward kinematics and the Jacobian are given (from P01). Implement the three functions marked
with `# TODO`: `analytic_ik_2link`, the update rules in `numeric_ik` and `nullspace_step`.
The solution is in solution/.

Planar arm with n revolute joints; the angles add up along the chain, so with the cumulative
angles c_i = q_1+...+q_i:
    x = sum_i l_i cos(c_i),   y = sum_i l_i sin(c_i)
"""
import numpy as np


# ==========================================================================
# Forward kinematics + Jacobian   [given, from P01]
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
    J = np.zeros((2, len(L)))
    for k in range(len(L)):
        J[0, k] = -np.sum(L[k:] * np.sin(c[k:]))
        J[1, k] = np.sum(L[k:] * np.cos(c[k:]))
    return J


# ==========================================================================
# Analytic IK for the 2-joint arm   >>> YOUR TURN <<<
# ==========================================================================
def analytic_ik_2link(target, lengths):
    """All exact solutions (elbow up/down) as a list of (2,) arrays.
    Empty list if the target lies outside the workspace.

    Steps (script ch. 6a):
      1. r = hypot(x, y). Reachable only for |l1-l2| <= r <= l1+l2 -> otherwise return [].
      2. cos q2 = (r^2 - l1^2 - l2^2) / (2 l1 l2)   (clip to [-1,1] against rounding errors!)
      3. For BOTH signs: q2 = +- arccos(...)
         and  q1 = atan2(y, x) - atan2(l2 sin q2, l1 + l2 cos q2).
         (atan2 instead of atan — otherwise the quadrant is wrong!)
      4. At q2 = 0 or pi both solutions coincide -> remove the duplicate.
    """
    # TODO: implement the 4 steps
    raise NotImplementedError


# ==========================================================================
# Numerical IK   >>> YOUR TURN (the three update rules) <<<
# ==========================================================================
def numeric_ik(target, lengths, q0, method="dls", lam=0.1, max_iter=200, tol=1e-6,
               max_step=None):
    """Iterative IK. Returns (q, history_err, converged, max_step_norm).

    Update rules (script ch. 6b) — with e = target - fk(q) and J = jacobian(q):
      'transpose' : dq = alpha * J^T e,  the optimal step size
                    alpha = <e, J J^T e> / <J J^T e, J J^T e>   (guard against 0-division)
      'pinv'      : dq = J^+ e            (np.linalg.pinv)
      'dls'       : dq = J^T (J J^T + lam^2 I)^-1 e   (np.linalg.solve instead of inv!)
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
        # TODO: compute dq depending on `method` (the three rules above)
        raise NotImplementedError
        max_step_norm = max(max_step_norm, float(np.linalg.norm(dq)))
        if max_step is not None:
            nrm = np.linalg.norm(dq)
            if nrm > max_step:
                dq = dq * (max_step / nrm)
        q = q + dq
    return q, np.array(hist), converged, max_step_norm


def nullspace_step(q, lengths, z):
    """Null-space projection: a joint motion that does NOT move the end effector.
    >>> YOUR TURN <<<

    Formula (script ch. 6c):  dq = (I - J^+ J) z
    Non-zero only for redundant arms (n > 2).
    """
    # TODO: implement the null-space projection
    raise NotImplementedError
