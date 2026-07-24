"""MPC experiments: MPC == LQR, input constraints, state constraints.
(Reference solution P03-final)   /Users/.../.venv/bin/python run.py   Plots -> results/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpc import double_integrator, lqr, MPC, simulate, lqr_cost

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

A, B, Q, R = double_integrator(dt=0.1)
X0 = np.array([2.0, 0.0])          # start 2 m from the target, at rest


def exp_mpc_equals_lqr():
    print("=" * 72)
    print("EXPERIMENT 1 — unconstrained MPC == LQR (and why the terminal cost matters)")
    print("=" * 72)
    K, _ = lqr(A, B, Q, R)
    u_lqr = (-K @ X0).item()
    print(f"  LQR first input at x0=(2,0):  u = -K x0 = {u_lqr:.6f}")

    mpc_lqr_term = MPC(A, B, Q, R, N=5, terminal="lqr")
    print(f"  MPC with LQR terminal cost, N=5:  u0 = {mpc_lqr_term.control(X0).item():.6f}  "
          f"(diff {abs(mpc_lqr_term.control(X0).item() - u_lqr):.1e})")
    print("  => With the Riccati cost-to-go as the terminal weight, MPC == LQR to machine precision")
    print("     for ANY horizon (the terminal cost encodes the infinite tail).\n")

    print("  With only the stage cost as terminal weight (P=Q), MPC -> LQR as the horizon grows:")
    print(f"  {'N':>4} | {'MPC u0':>10} | {'error vs LQR':>13}")
    print("  " + "-" * 34)
    Ns, us = [], []
    for N in [1, 3, 5, 10, 20, 40, 60]:
        mpc = MPC(A, B, Q, R, N=N, terminal="stage")
        u0 = mpc.control(X0).item(); Ns.append(N); us.append(u0)
        print(f"  {N:4d} | {u0:10.5f} | {abs(u0 - u_lqr):13.2e}")
    print("  => A short-horizon MPC is myopic; lengthening the horizon (or using the LQR terminal")
    print("     cost) recovers the optimal infinite-horizon LQR action.")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.axhline(u_lqr, ls="--", color="black", label="LQR optimum")
    ax.plot(Ns, us, "o-", color="steelblue", label="MPC (terminal P=Q)")
    ax.set_xlabel("horizon N"); ax.set_ylabel("first input u0")
    ax.set_title("MPC converges to LQR as the horizon grows"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "mpc_vs_lqr.png"), dpi=110); plt.close(fig)


def exp_input_constraint():
    print("\n" + "=" * 72)
    print("EXPERIMENT 2 — input constraint |u| <= u_max: MPC plans within, LQR saturates")
    print("=" * 72)
    u_max = 0.5
    K, _ = lqr(A, B, Q, R)
    print(f"  LQR's UNCLIPPED command at the start: |u| = {abs((-K @ X0).item()):.2f}  "
          f"(the actuator limit is {u_max}) -> {abs((-K @ X0).item()) / u_max:.0f}x over the limit")

    # LQR must be naively clipped to the actuator limit; MPC optimises within it
    xs_l, us_l = simulate(lambda x: -K @ x, A, B, X0, T=80, clip=u_max)
    mpc = MPC(A, B, Q, R, N=25, u_bound=u_max)
    xs_m, us_m = simulate(mpc.control, A, B, X0, T=80)
    print(f"  applied max |u|:   LQR(clipped) {np.abs(us_l).max():.3f}   MPC {np.abs(us_m).max():.3f}")
    print(f"  closed-loop cost:  LQR(clipped) {lqr_cost(xs_l, us_l, Q, R):.3f}   "
          f"MPC {lqr_cost(xs_m, us_m, Q, R):.3f}")
    print(f"  final position:    LQR {xs_l[-1, 0]:+.4f}   MPC {xs_m[-1, 0]:+.4f}")
    print("  => LQR would demand ~11x the actuator limit and gets naively clipped -> suboptimal.")
    print("     MPC KNOWS the limit and plans the best feasible input sequence -> lower cost.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    ax1.plot(xs_l[:, 0], color="crimson", label="LQR (clipped)")
    ax1.plot(xs_m[:, 0], color="steelblue", label="MPC")
    ax1.axhline(0, color="gray", lw=0.6); ax1.set_xlabel("step"); ax1.set_ylabel("position")
    ax1.set_title("position"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(us_l[:, 0], color="crimson", label="LQR (clipped)")
    ax2.plot(us_m[:, 0], color="steelblue", label="MPC")
    ax2.axhline(u_max, ls="--", color="black"); ax2.axhline(-u_max, ls="--", color="black")
    ax2.set_xlabel("step"); ax2.set_ylabel("input u"); ax2.set_title("input (dashed = limit)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "input_constraint.png"), dpi=110); plt.close(fig)


def exp_state_constraint():
    print("\n" + "=" * 72)
    print("EXPERIMENT 3 — state (speed) constraint |v| <= v_max: MPC respects it, LQR violates it")
    print("=" * 72)
    v_max = 0.6
    K, _ = lqr(A, B, Q, R)
    xs_l, _ = simulate(lambda x: -K @ x, A, B, X0, T=80)          # LQR, no state awareness
    mpc = MPC(A, B, Q, R, N=25, v_bound=v_max)
    xs_m, us_m = simulate(mpc.control, A, B, X0, T=80)
    print(f"  speed limit |v| <= {v_max}")
    print(f"  LQR peak speed: {np.abs(xs_l[:, 1]).max():.3f}   -> VIOLATES the limit")
    print(f"  MPC peak speed: {np.abs(xs_m[:, 1]).max():.3f}   -> respects it (to solver tolerance)")
    print(f"  both still reach the target: LQR final pos {xs_l[-1, 0]:+.4f}, MPC {xs_m[-1, 0]:+.4f}")
    print("  => This is the decisive advantage. LQR knows no bounds, so on a big setpoint change it")
    print("     races past the speed limit; MPC forecasts the trajectory and shapes the inputs to")
    print("     stay under the limit the whole way — impossible for an unconstrained controller.")

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(xs_l[:, 1], color="crimson", label="LQR velocity")
    ax.plot(xs_m[:, 1], color="steelblue", label="MPC velocity")
    ax.axhline(v_max, ls="--", color="black", label="speed limit")
    ax.axhline(-v_max, ls="--", color="black")
    ax.set_xlabel("step"); ax.set_ylabel("velocity")
    ax.set_title("MPC honours the speed limit; LQR overshoots it")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "state_constraint.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    exp_mpc_equals_lqr()
    exp_input_constraint()
    exp_state_constraint()
    print("\nPlots in results/. Done.")
