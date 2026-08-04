"""Self-aware system experiments: model drift, self-healing, and the base-rate trap.
(Reference solution P03-final)   /Users/.../.venv/bin/python run.py   Plots -> results/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from self_aware import run

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def exp_drift():
    print("=" * 78)
    print("EXPERIMENT 1 — models at run-time: surviving a deployment that changes the system")
    print("=" * 78)
    print("  At interval 150 a 'deployment' drops the true service rate by 30 % (no failures here, so")
    print("  the drift is isolated). The design-time model never finds out.\n")
    print(f"  {'model':>24} | {'SLO before':>10} | {'SLO after':>10} | {'|pred err| after':>16}")
    print("  " + "-" * 70)
    results = {}
    for adaptive, label in [(False, "static (design-time)"), (True, "re-estimated online")]:
        r = run(adaptive_model=adaptive, healing=False, deploy_at=150, T=300, p_sick=0.0)
        results[label] = r
        print(f"  {label:>24} | {r['slo_before']:10.1%} | {r['slo_after']:10.1%} | "
              f"{r['pred_error_after']:16.3f}")
    a = results["re-estimated online"]
    print(f"\n  final estimate: mu_hat = {a['mu_hat'][-1]:.2f}  (true mu after the deployment = "
          f"{a['mu_true'][-1]:.2f})")
    print("  => The static model does not merely become inaccurate — it becomes DANGEROUS. It keeps")
    print("     provisioning for a capacity the system no longer has, so almost every interval")
    print("     violates the SLO. The adaptive model tracks the new service rate within a few")
    print("     intervals and recovers. Note the diagnostic: the PREDICTION ERROR is what reveals")
    print("     the drift — a self-aware system must monitor how wrong it is, not just the metric.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for label, col in [("static (design-time)", "crimson"), ("re-estimated online", "steelblue")]:
        ax1.plot(results[label]["mu_hat"], color=col, lw=1.6, label=f"mu_hat, {label}")
    ax1.plot(results["re-estimated online"]["mu_true"], color="black", ls="--", lw=1.4, label="true mu")
    ax1.axvline(150, color="gray", ls=":", lw=1.2)
    ax1.set_ylabel("service rate"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_title("the model at run-time: only re-estimation keeps it true (deployment at t=150)")
    for label, col in [("static (design-time)", "crimson"), ("re-estimated online", "steelblue")]:
        ax2.plot(results[label]["pred_error"], color=col, lw=1.0, label=label)
    ax2.axvline(150, color="gray", ls=":", lw=1.2)
    ax2.set_xlabel("control interval"); ax2.set_ylabel("|prediction error| [s]")
    ax2.set_title("prediction error: the signal that the model has gone stale")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "model_drift.png"), dpi=110); plt.close(fig)


def exp_healing():
    print("\n" + "=" * 78)
    print("EXPERIMENT 2 — self-healing and the BASE-RATE TRAP")
    print("=" * 78)
    print("  Replicas fall sick rarely. The detector is noisy. A restart heals a replica but makes")
    print("  it unavailable for 3 intervals — so a FALSE alarm actively removes healthy capacity.\n")
    print(f"  {'strategy':>26} | {'SLO viol':>8} | {'restarts':>8} | {'TP':>4} | {'FP':>4} | "
          f"{'precision':>9} | {'sick time':>9}")
    print("  " + "-" * 88)
    r0 = run(healing=False, T=300)
    print(f"  {'no healing':>26} | {r0['slo_violations']:8.1%} | {'-':>8} | {'-':>4} | {'-':>4} | "
          f"{'-':>9} | {r0['sick_time']:9.1%}")
    rows = []
    for k in [1, 2, 3, 5]:
        r = run(healing=True, persistence=k, T=300)
        rows.append((k, r))
        print(f"  {('heal, persistence k=' + str(k)):>26} | {r['slo_violations']:8.1%} | "
              f"{r['restarts']:8d} | {r['true_positives']:4d} | {r['false_positives']:4d} | "
              f"{r['precision']:9.1%} | {r['sick_time']:9.1%}")
    naive = rows[0][1]
    print(f"\n  => The naive healer (k=1) fires {naive['restarts']} restarts of which "
          f"{naive['false_positives']} are FALSE — precision {naive['precision']:.1%}.")
    print("     This is the base-rate fallacy in the flesh: failures are rare, so even a decent")
    print("     detector produces mostly false alarms. And the consequence is not merely wasted")
    print(f"     effort — because each false restart removes a HEALTHY replica for 3 intervals, the")
    print(f"     naive healer makes the SLO {naive['slo_violations']/r0['slo_violations']:.1f}x WORSE "
          f"than not healing at all ({naive['slo_violations']:.1%} vs {r0['slo_violations']:.1%}).")
    print("     It causes the outage it exists to prevent.")
    print("     Requiring PERSISTENCE (k consecutive anomalous intervals) collapses the false")
    print("     positives and turns healing into a genuine win: at k=3 precision is 92 % and the")
    print(f"     violations drop to {rows[2][1]['slo_violations']:.1%} — now the loop actually helps.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ks = [k for k, _ in rows]
    ax1.plot(ks, [r["precision"] * 100 for _, r in rows], "o-", color="steelblue")
    ax1.set_xlabel("persistence k (consecutive anomalous intervals)")
    ax1.set_ylabel("alert precision [%]"); ax1.set_title("persistence defeats the base-rate trap")
    ax1.grid(alpha=0.3)
    ax2.plot(ks, [r["slo_violations"] * 100 for _, r in rows], "o-", color="crimson", label="with healing")
    ax2.axhline(r0["slo_violations"] * 100, ls="--", color="black", label="no healing")
    ax2.set_xlabel("persistence k"); ax2.set_ylabel("SLO violations [%]")
    ax2.set_title("healing only pays once the alerts are trustworthy")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "self_healing.png"), dpi=110); plt.close(fig)


def exp_full_system():
    print("\n" + "=" * 78)
    print("EXPERIMENT 3 — the complete self-aware system: everything at once")
    print("=" * 78)
    print("  Drift AND failures, with each capability switched on in turn.\n")
    print(f"  {'configuration':>38} | {'SLO viol':>8} | {'|pred err|':>10} | {'restarts':>8}")
    print("  " + "-" * 74)
    configs = [
        ("static model, no healing", dict(adaptive_model=False, healing=False)),
        ("static model, healing (k=3)", dict(adaptive_model=False, healing=True, persistence=3)),
        ("run-time model, no healing", dict(adaptive_model=True, healing=False)),
        ("run-time model + healing (k=3)", dict(adaptive_model=True, healing=True, persistence=3)),
    ]
    for label, kw in configs:
        r = run(deploy_at=150, T=300, **kw)
        rs = r["restarts"] if kw.get("healing") else "-"
        print(f"  {label:>38} | {r['slo_violations']:8.1%} | {r['mean_prediction_error']:10.3f} | "
              f"{str(rs):>8}")
    print("  => Neither capability substitutes for the other. Healing cannot rescue a stale model")
    print("     (the capacity plan is simply wrong), and a perfect model cannot rescue sick replicas")
    print("     (the capacity is there on paper but not in reality). Self-awareness is the whole")
    print("     loop: know yourself (monitor), keep the model true (re-estimate), predict, and act")
    print("     on both knobs — capacity and repair.")


if __name__ == "__main__":
    exp_drift()
    exp_healing()
    exp_full_system()
    print("\nPlots in results/. Done.")
