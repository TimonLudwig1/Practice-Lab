"""IK experiments: analytic double solution, comparison of methods, singularity, redundancy.
(Reference solution P02-medium)   /Users/.../.venv/bin/python run.py   Plots -> results/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ik import fk, joints, jacobian, analytic_ik_2link, numeric_ik, nullspace_step

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
L2 = [1.0, 1.0]


def exp_analytic():
    print("=" * 68)
    print("EXPERIMENT 1 — analytic IK: two solutions (elbow up/down)")
    print("=" * 68)
    for tgt in [(1.0, 1.0), (1.5, 0.3), (0.0, 2.0), (2.5, 0.0)]:
        sols = analytic_ik_2link(tgt, L2)
        if not sols:
            print(f"  target {tgt}: UNREACHABLE (r={np.hypot(*tgt):.3f} > l1+l2={sum(L2)})")
            continue
        txt = []
        for s in sols:
            back = fk(s, L2)
            txt.append(f"q=({np.degrees(s[0]):+7.2f},{np.degrees(s[1]):+7.2f}) deg "
                       f"-> FK={back.round(6)}")
        print(f"  target {tgt}: {len(sols)} solution(s)")
        for t in txt:
            print(f"      {t}")
    print("  => Both solutions hit the target EXACTLY; they differ in the sign")
    print("     of q2 (elbow up vs down). At q2=0/pi they coincide.")

    fig, ax = plt.subplots(figsize=(6, 6))
    tgt = (1.0, 1.0)
    for s, col, lab in zip(analytic_ik_2link(tgt, L2), ["steelblue", "darkorange"],
                           ["elbow down", "elbow up"]):
        P = joints(s, L2)
        ax.plot(P[:, 0], P[:, 1], "o-", lw=2.5, ms=7, color=col, label=lab)
    ax.plot(*tgt, "r*", ms=16, label="target")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("two exact IK solutions for the same target")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "analytic_ik.png"), dpi=110); plt.close(fig)


def exp_methods():
    print("\n" + "=" * 68)
    print("EXPERIMENT 2 — numerical IK: J^T vs pseudoinverse vs DLS (200 random targets)")
    print("=" * 68)
    rng = np.random.default_rng(0)
    print(f"  {'method':>10} | {'success':>7} | {'iter (median)':>13} | {'max |dq|':>10}")
    print("  " + "-" * 48)
    results = {}
    for method in ["transpose", "pinv", "dls"]:
        succ, iters, steps = 0, [], []
        for _ in range(200):
            # a well-reachable target (radius 0.3..1.8), random start
            r = rng.uniform(0.3, 1.8); th = rng.uniform(-np.pi, np.pi)
            tgt = np.array([r*np.cos(th), r*np.sin(th)])
            q0 = rng.uniform(-np.pi, np.pi, 2)
            q, hist, conv, ms = numeric_ik(tgt, L2, q0, method=method, lam=0.1)
            succ += conv
            if conv:
                iters.append(len(hist))
            steps.append(ms)
        results[method] = (succ/200, np.median(iters) if iters else np.nan, np.max(steps))
        print(f"  {method:>10} | {succ/200:7.2f} | {results[method][1]:13.0f} | "
              f"{results[method][2]:10.1f}")
    print("  => Pseudoinverse and DLS converge in a few iterations, J^T needs many")
    print("     (pure gradient descent). Watch max|dq| — see experiment 3.")
    return results


def exp_singularity():
    print("\n" + "=" * 68)
    print("EXPERIMENT 3 — singularity: why the pseudoinverse explodes")
    print("=" * 68)
    print("The singularity depends on the CONFIGURATION, not the target: det J = l1*l2*sin(q2).")
    print("So we start the iteration ever closer to the stretched arm (q2 -> 0) and measure")
    print("how large the first/largest joint step becomes.\n")
    tgt = np.array([0.6, 0.9])          # well-reachable target, lateral to the start
    print(f"  {'q2_start':>9} | {'det J(q0)':>10} | {'pinv max|dq|':>13} | {'DLS max|dq|':>12}")
    print("  " + "-" * 52)
    detjs, ms_ps, ms_ds = [], [], []
    for q2s in [0.5, 0.1, 0.01, 0.001, 0.0001]:
        q0 = np.array([0.2, q2s])
        detJ = np.linalg.det(jacobian(q0, L2))
        _, _, _, ms_p = numeric_ik(tgt, L2, q0, method="pinv", max_iter=100)
        _, _, _, ms_d = numeric_ik(tgt, L2, q0, method="dls", lam=0.1, max_iter=100)
        detjs.append(detJ); ms_ps.append(ms_p); ms_ds.append(ms_d)
        print(f"  {q2s:9.4f} | {detJ:10.6f} | {ms_p:13.1f} | {ms_d:12.2f}")
    print("  => The closer to the singularity (det J -> 0), the larger the")
    print("     pseudoinverse steps (unrealistic joint jumps, ~1/det J).")
    print("     DLS stays BOUNDED thanks to the damping — that is the whole point.")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.loglog(detjs, ms_ps, "o-", color="crimson", label="pseudoinverse")
    ax.loglog(detjs, ms_ds, "s-", color="steelblue", label="DLS (lambda=0.1)")
    ax.set_xlabel("det J at start (smaller = more singular)"); ax.set_ylabel("max |dq|")
    ax.invert_xaxis(); ax.grid(alpha=0.3, which="both"); ax.legend()
    ax.set_title("proximity to the singularity: pinv explodes, DLS stays bounded")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "singularity.png"), dpi=110); plt.close(fig)

    # lambda trade-off: accuracy vs step size
    print("\n  lambda trade-off (target r=1.99, start almost stretched):")
    print(f"  {'lambda':>8} | {'residual err':>12} | {'max |dq|':>10} | {'iterations':>11}")
    print("  " + "-" * 49)
    tgt = np.array([1.99, 0.0])
    lams, errs, steps = [], [], []
    for lam in [0.001, 0.01, 0.05, 0.2, 0.5, 1.0]:
        q, hist, conv, ms = numeric_ik(tgt, L2, q0, method="dls", lam=lam, max_iter=300)
        err = np.linalg.norm(tgt - fk(q, L2))
        lams.append(lam); errs.append(err); steps.append(ms)
        print(f"  {lam:8.3f} | {err:12.2e} | {ms:10.2f} | {len(hist):11d}")
    print("  => Small lambda: accurate, but large (risky) steps. Large lambda: smooth,")
    print("     but slower/less accurate. That is the DLS trade-off.")

    fig, ax1 = plt.subplots(figsize=(7, 4.3))
    ax1.loglog(lams, np.maximum(errs, 1e-16), "o-", color="crimson", label="residual err")
    ax1.set_xlabel("lambda (log)"); ax1.set_ylabel("residual err", color="crimson")
    ax2 = ax1.twinx()
    ax2.loglog(lams, steps, "s-", color="steelblue", label="max |dq|")
    ax2.set_ylabel("max |dq|", color="steelblue")
    ax1.set_title("DLS: damping trades accuracy for stability")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "dls_tradeoff.png"), dpi=110); plt.close(fig)


def exp_redundancy():
    print("\n" + "=" * 68)
    print("EXPERIMENT 4 — redundancy: 3 joints, 2 task dimensions -> null space")
    print("=" * 68)
    L3 = [0.9, 0.7, 0.5]
    q = np.array([0.4, 0.6, -0.3])
    x0 = fk(q, L3)
    J = jacobian(q, L3)
    print(f"  arm with n=3 joints, task m=2 -> Jacobian is {J.shape}, null space dim = "
          f"{3 - np.linalg.matrix_rank(J)}")
    rng = np.random.default_rng(1)
    max_drift = 0.0
    for _ in range(200):
        z = rng.normal(size=3)
        dq = nullspace_step(q, L3, z)
        dq = 0.01 * dq / (np.linalg.norm(dq) + 1e-12)   # small step in the null space
        max_drift = max(max_drift, float(np.linalg.norm(fk(q + dq, L3) - x0)))
    print(f"  200 random null-space steps (|dq|=0.01): max EE drift = {max_drift:.2e}")
    print("  => The joints move, the end effector stays put. Exactly this freedom is")
    print("     used for secondary goals (avoiding obstacles, respecting joint limits).")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    qq = q.copy()
    for i in range(12):
        P = joints(qq, L3)
        ax.plot(P[:, 0], P[:, 1], "o-", alpha=0.35 + 0.05*i, color="steelblue", ms=4, lw=1.5)
        dq = nullspace_step(qq, L3, np.array([1.0, -0.5, 0.3]))
        nrm = np.linalg.norm(dq)
        if nrm < 1e-12:
            break
        qq = qq + 0.12 * dq / nrm
    ax.plot(*x0, "r*", ms=18, label="end effector (stays fixed)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("null-space motion: the arm changes, the hand stays")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "nullspace.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    exp_analytic()
    exp_methods()
    exp_singularity()
    exp_redundancy()
    print("\nPlots in results/. Done.")
