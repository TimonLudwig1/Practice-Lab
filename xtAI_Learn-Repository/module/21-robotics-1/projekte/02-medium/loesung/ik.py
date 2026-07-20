"""Inverse Kinematik: analytisch und numerisch (J^T / Pseudoinverse / DLS).
(Referenzloesung P02-medium)   Modul 21 — Robotics 1.

Planarer Arm mit n Drehgelenken, Gliedlaengen l_1..l_n. Die Winkel addieren sich entlang der
Kette (Skript Kap. 4), daher mit den kumulierten Winkeln c_i = q_1+...+q_i:

    x = sum_i l_i cos(c_i),      y = sum_i l_i sin(c_i)

    J[0,k] = -sum_{i>=k} l_i sin(c_i)
    J[1,k] =  sum_{i>=k} l_i cos(c_i)
"""
import numpy as np


# ==========================================================================
# Vorwaertskinematik + Jacobi (allgemein, n Glieder)
# ==========================================================================
def fk(q, lengths):
    """Endeffektor-Position (2,) des planaren Arms."""
    c = np.cumsum(np.asarray(q, float))
    L = np.asarray(lengths, float)
    return np.array([np.sum(L * np.cos(c)), np.sum(L * np.sin(c))])


def joints(q, lengths):
    """Alle Gelenkpositionen (n+1, 2) inkl. Basis — fuer Plots."""
    c = np.cumsum(np.asarray(q, float))
    L = np.asarray(lengths, float)
    xs = np.concatenate([[0.0], np.cumsum(L * np.cos(c))])
    ys = np.concatenate([[0.0], np.cumsum(L * np.sin(c))])
    return np.column_stack([xs, ys])


def jacobian(q, lengths):
    """Positions-Jacobi (2, n): J[:,k] = Ableitung der EE-Position nach q_k."""
    c = np.cumsum(np.asarray(q, float))
    L = np.asarray(lengths, float)
    n = len(L)
    J = np.zeros((2, n))
    for k in range(n):
        J[0, k] = -np.sum(L[k:] * np.sin(c[k:]))
        J[1, k] = np.sum(L[k:] * np.cos(c[k:]))
    return J


# ==========================================================================
# Analytische IK fuer den 2-Gelenk-Arm
# ==========================================================================
def analytic_ik_2link(target, lengths):
    """Alle exakten Loesungen (Ellbogen oben/unten). Leere Liste, wenn unerreichbar.

    Kosinussatz:  cos q2 = (r^2 - l1^2 - l2^2) / (2 l1 l2),   q2 = +- arccos(.)
                  q1 = atan2(y,x) - atan2(l2 sin q2, l1 + l2 cos q2)
    Erreichbar nur fuer |l1-l2| <= r <= l1+l2.
    """
    x, y = target
    l1, l2 = lengths
    r = np.hypot(x, y)
    if r > l1 + l2 + 1e-12 or r < abs(l1 - l2) - 1e-12:
        return []                                   # ausserhalb des Arbeitsraums
    cq2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cq2 = np.clip(cq2, -1.0, 1.0)                   # Rundungsfehler abfangen
    sols = []
    for sign in (+1.0, -1.0):
        q2 = sign * np.arccos(cq2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        sols.append(np.array([q1, q2]))
    # Bei q2=0 oder pi fallen beide Loesungen zusammen -> Duplikat entfernen
    if len(sols) == 2 and np.allclose(sols[0], sols[1], atol=1e-9):
        sols = [sols[0]]
    return sols


# ==========================================================================
# Numerische IK: J^T / Pseudoinverse / Damped Least Squares
# ==========================================================================
def numeric_ik(target, lengths, q0, method="dls", lam=0.1, max_iter=200, tol=1e-6,
               max_step=None):
    """Iterative IK. Gibt (q, history_err, converged, max_step_norm) zurueck.

    method:
      'transpose' : dq = alpha J^T e   mit optimaler Schrittweite alpha
                    (alpha = <e, JJ^T e> / <JJ^T e, JJ^T e>)  -> Gradientenabstieg
      'pinv'      : dq = J^+ e         -> schnell, EXPLODIERT nahe Singularitaeten
      'dls'       : dq = J^T (J J^T + lam^2 I)^-1 e  -> immer wohldefiniert
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
            raise ValueError(f"unbekannte Methode: {method}")
        max_step_norm = max(max_step_norm, float(np.linalg.norm(dq)))
        if max_step is not None:                    # optionale Schrittbegrenzung
            nrm = np.linalg.norm(dq)
            if nrm > max_step:
                dq = dq * (max_step / nrm)
        q = q + dq
    return q, np.array(hist), converged, max_step_norm


def nullspace_step(q, lengths, z):
    """Nullraum-Projektion (I - J^+ J) z: Gelenkbewegung, die den Endeffektor NICHT bewegt.
    Nur bei redundanten Armen (n > 2) ungleich null."""
    J = jacobian(q, lengths)
    n = len(q)
    return (np.eye(n) - np.linalg.pinv(J) @ J) @ np.asarray(z, float)
