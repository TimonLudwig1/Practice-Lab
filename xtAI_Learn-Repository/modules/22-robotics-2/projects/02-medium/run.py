"""Control experiments: regulation, tracking, feedback-linearisation, model error.
/Users/.../.venv/bin/python run.py   Plots -> results/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from control import (ArmModel, TRUE, reference, pd_gravity_control, computed_torque_control,
                     simulate, rms_second_half)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

KP = 100.0 * np.eye(2)      # -> natural frequency w0 = sqrt(100) = 10 rad/s
KD = 20.0 * np.eye(2)       # -> critical damping (Kd = 2 w0)
W0 = 10.0


def exp_regulation():
    print("=" * 70)
    print("EXPERIMENT 1 — regulation (reach a fixed setpoint): both controllers suffice")
    print("=" * 70)
    qd = np.array([0.6, -0.8])
    const_ref = lambda t: (qd, np.zeros(2), np.zeros(2))
    for name, ctrl in [("PD + gravity", pd_gravity_control),
                       ("computed torque", computed_torque_control)]:
        ts, Q, QD, err = simulate(ctrl, [0.0, 0.0], [0.0, 0.0], KP, KD, ref_fn=const_ref, T=3.0)
        print(f"  {name:16s}: final position error {err[-1]:.2e}")
    print("  => For a CONSTANT setpoint, PD+gravity is enough — gravity is cancelled and the PD")
    print("     term pulls the error to zero. The difference only shows up when the target MOVES.")


def exp_tracking():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — tracking a moving trajectory: computed torque wins, gap grows with speed")
    print("=" * 70)
    print(f"  {'speed w':>8} | {'PD+gravity RMS':>15} | {'computed-torque RMS':>20} | {'ratio':>8}")
    print("  " + "-" * 60)
    ws, pd_rms, ct_rms = [], [], []
    for w in [1.0, 2.0, 3.0, 4.0]:
        ref_w = lambda t, w=w: reference(t, w)
        _, _, _, ep = simulate(pd_gravity_control, [0, 0], [0, 0], KP, KD, ref_fn=ref_w)
        _, _, _, ec = simulate(computed_torque_control, [0, 0], [0, 0], KP, KD, ref_fn=ref_w)
        rp, rc = rms_second_half(ep), rms_second_half(ec)
        ws.append(w); pd_rms.append(rp); ct_rms.append(rc)
        print(f"  {w:8.1f} | {rp:15.4f} | {rc:20.2e} | {rp/rc:8.0f}x")
    print("  => PD+gravity's error grows with speed (it ignores the inertia term M ddqd and the")
    print("     Coriolis coupling); computed torque stays near zero — it feeds those terms forward.")

    # tracking plot at w=3
    ref_w = lambda t: reference(t, 3.0)
    ts, Qp, _, _ = simulate(pd_gravity_control, [0, 0], [0, 0], KP, KD, ref_fn=ref_w)
    _, Qc, _, _ = simulate(computed_torque_control, [0, 0], [0, 0], KP, KD, ref_fn=ref_w)
    qd_traj = np.array([reference(t, 3.0)[0] for t in ts])
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(ts, qd_traj[:, 0], "k--", lw=1.5, label="desired q1")
    ax.plot(ts, Qp[:, 0], color="crimson", lw=1.2, label="PD+gravity q1")
    ax.plot(ts, Qc[:, 0], color="steelblue", lw=1.2, label="computed torque q1")
    ax.set_xlim(0, 4); ax.set_xlabel("t [s]"); ax.set_ylabel("joint 1 angle [rad]")
    ax.set_title("trajectory tracking at w=3: computed torque follows, PD+gravity lags")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "tracking.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.semilogy(ws, pd_rms, "o-", color="crimson", label="PD + gravity")
    ax.semilogy(ws, ct_rms, "s-", color="steelblue", label="computed torque")
    ax.set_xlabel("trajectory speed w [rad/s]"); ax.set_ylabel("RMS tracking error [rad] (log)")
    ax.set_title("tracking error vs. speed"); ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "rms_vs_speed.png"), dpi=110); plt.close(fig)


def exp_feedback_linearisation():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — feedback linearisation: computed torque gives EXACT linear error decay")
    print("=" * 70)
    print("Computed torque should make the error obey  edot-dot + Kd edot + Kp e = 0, i.e. a")
    print("critically-damped 2nd-order decay  e(t) = e0 (1 + w0 t) exp(-w0 t),  independent of pose.\n")
    e0 = np.array([0.5, -0.5])
    for label, qd in [("config A", np.array([0.0, 0.0])), ("config B", np.array([1.0, 0.8]))]:
        const_ref = lambda t, qd=qd: (qd, np.zeros(2), np.zeros(2))
        ts, Q, _, _ = simulate(computed_torque_control, qd + e0, [0, 0], KP, KD,
                               ref_fn=const_ref, T=1.5, dt=1e-4)
        E = Q - qd
        analytic = np.outer((1 + W0 * ts) * np.exp(-W0 * ts), e0)
        dev = np.abs(E - analytic).max()
        print(f"  computed torque, {label}: max deviation from the analytic decay = {dev:.2e}")
    # PD+gravity: the decay depends on the configuration (through M) -> not the clean linear curve
    ts, Qp1, _, _ = simulate(pd_gravity_control, np.array([0., 0.]) + e0, [0, 0], KP, KD,
                             ref_fn=lambda t: (np.array([0., 0.]), np.zeros(2), np.zeros(2)),
                             T=1.5, dt=1e-4)
    ts, Qp2, _, _ = simulate(pd_gravity_control, np.array([1., 0.8]) + e0, [0, 0], KP, KD,
                             ref_fn=lambda t: (np.array([1., 0.8]), np.zeros(2), np.zeros(2)),
                             T=1.5, dt=1e-4)
    diff = np.abs((Qp1 - np.array([0., 0.])) - (Qp2 - np.array([1., 0.8]))).max()
    print(f"  PD + gravity: same initial error, two configs -> decays differ by {diff:.3f}")
    print("  => Computed torque's decay is IDENTICAL in both configurations and matches the linear")
    print("     prediction to ~1e-4: the model cancellation truly linearises and decouples the arm.")
    print("     PD+gravity's decay is configuration-dependent (M is not cancelled).")

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(ts, np.abs(E[:, 0]), color="steelblue", lw=2, label="computed torque |e1| (config B)")
    ax.plot(ts, np.abs(analytic[:, 0]), "k--", lw=1.2, label="analytic (1+w0 t)e^{-w0 t}")
    ax.plot(ts, np.abs(Qp2[:, 0] - 1.0), color="crimson", lw=1.2, label="PD+gravity |e1| (config B)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("|joint-1 error| [rad]")
    ax.set_title("error decay: computed torque == analytic linear system")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "error_decay.png"), dpi=110); plt.close(fig)


def exp_model_error():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 — model error: computed torque relies on knowing M, C, g")
    print("=" * 70)
    print(f"  {'model mass error':>16} | {'computed-torque RMS':>20}")
    print("  " + "-" * 40)
    ref_w = lambda t: reference(t, 2.0)
    scales, rmss = [], []
    for pct in [0, 10, 20, 30, 50]:
        wrong = ArmModel(m1=1.0 * (1 + pct / 100), m2=1.0 * (1 + pct / 100))
        _, _, _, err = simulate(computed_torque_control, [0, 0], [0, 0], KP, KD,
                                ref_fn=ref_w, ctrl_model=wrong, plant=TRUE)
        r = rms_second_half(err)
        scales.append(pct); rmss.append(r)
        print(f"  {('+' + str(pct) + ' %'):>16} | {r:20.4f}")
    print("  => With the exact model the tracking is near-perfect; as the controller's mass estimate")
    print("     drifts from the truth, the cancellation is incomplete and the error grows. This is")
    print("     why real systems add robust/adaptive control on top of computed torque.")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(scales, rmss, "o-", color="purple")
    ax.set_xlabel("mass error in the controller's model [%]"); ax.set_ylabel("RMS tracking error [rad]")
    ax.set_title("computed torque degrades with model error"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "model_error.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    exp_regulation()
    exp_tracking()
    exp_feedback_linearisation()
    exp_model_error()
    print("\nPlots in results/. Done.")
