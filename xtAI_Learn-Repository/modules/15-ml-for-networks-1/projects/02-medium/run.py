"""Intrusion detection under a realistic base rate — the whole inconvenient truth.

Four analyses:
  1. Train the detector, ROC-AUC/PR-AUC on the test set (looks great).
  2. EMPIRICAL: lower the base rate by thinning out the attacks -> ROC-AUC holds, PR-AUC collapses.
  3. ANALYTICAL (Bayes): PPV over the base rate + alarms per day -> the base-rate fallacy.
  4. Operating point: cost-minimal threshold + which FPR you WOULD NEED.

Plus a cross-check: theory (Bayes) and empiricism (subsampling) have to agree.

Call:  python run.py     (~20 s, ~2 MB download of the dataset on the first run)
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_score)

from flow_data import load_flows, build_xy, subsample_to_base_rate
from base_rate import (ppv_at_base_rate, required_fpr, alarms_per_day,
                       expected_cost, best_operating_point)

FLOWS_PER_DAY = 10_000_000      # a medium-sized campus/company network
RANDOM_STATE = 0


def main():
    # ---------- 1) the detector ----------
    df = load_flows()
    X, y = build_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=RANDOM_STATE)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(Xtr, ytr)
    scores = model.predict_proba(Xte)[:, 1]

    roc_auc = roc_auc_score(yte, scores)
    pr_auc = average_precision_score(yte, scores)
    print("=== 1) The detector (LogReg on volume/timing features) ===")
    print(f"Test set: {len(yte):,} flows, base rate pi = {yte.mean():.4f} "
          f"({100*yte.mean():.2f} % attacks)")
    print(f"ROC-AUC = {roc_auc:.4f}   PR-AUC = {pr_auc:.4f}    <- looks excellent!")

    fpr_c, tpr_c, thresholds = roc_curve(yte, scores)
    i99 = int(np.argmax(tpr_c >= 0.99))
    TPR, FPR = tpr_c[i99], fpr_c[i99]
    print(f"\nOperating point 'I want to see 99 % of the attacks': "
          f"TPR = {TPR:.4f}, FPR = {FPR:.5f} ({100*FPR:.2f} %)")

    # ---------- 2) empirical: lower the base rate ----------
    print("\n=== 2) EMPIRICAL: what happens at a more realistic base rate? ===")
    print("(thin out the attacks, leave the detector UNCHANGED)")
    print(f"{'pi':>8s} {'#attacks':>10s} {'ROC-AUC':>9s} {'PR-AUC':>8s}")
    rng = np.random.default_rng(RANDOM_STATE)
    for pi in (0.0336, 0.01, 0.001, 0.0001):
        res = subsample_to_base_rate(yte.values, scores, pi, rng)
        if res is None:
            continue
        yy, ss = res
        print(f"{pi:8.4f} {int(yy.sum()):10d} {roc_auc_score(yy, ss):9.4f} "
              f"{average_precision_score(yy, ss):8.4f}")
    print("-> ROC-AUC stays ~constant (or even rises!), PR-AUC COLLAPSES.")
    print("   The ROC curve is blind to the base rate - which is exactly why it lies here.")
    print("   (Careful: at pi=1e-4 only ~3 attacks remain -> the PR-AUC is very noisy there.)")

    # ---------- 3) analytical: Bayes ----------
    print("\n=== 3) ANALYTICAL (Bayes): the base-rate fallacy ===")
    print(f"At the fixed operating point TPR={TPR:.4f}, FPR={FPR:.5f}:")
    print(f"{'base rate pi':>14s} {'PPV = P(attack|alarm)':>24s} {'share false alarms':>19s}")
    for pi in (0.0336, 0.01, 1e-3, 1e-4, 1e-5):
        p = ppv_at_base_rate(TPR, FPR, pi)
        print(f"{pi:14g} {100*p:23.3f}% {100*(1-p):18.2f}%")

    pi_real = 1e-4
    true_alarms, false_alarms = alarms_per_day(FLOWS_PER_DAY, TPR, FPR, pi_real)
    print(f"\nReality check at {FLOWS_PER_DAY:,} flows/day and pi = {pi_real:g}:")
    print(f"  true alarms  : {true_alarms:,.0f} per day")
    print(f"  FALSE ALARMS : {false_alarms:,.0f} per day   <- no team in the world works through that")

    # cross-check theory vs. empiricism
    print("\n--- Cross-check: does the Bayes formula agree with the empirical result? ---")
    for pi in (0.0336, 0.01, 0.001):
        res = subsample_to_base_rate(yte.values, scores, pi, rng)
        if res is None:
            continue
        yy, ss = res
        emp = precision_score(yy, (ss >= thresholds[i99]).astype(int), zero_division=0)
        theo = ppv_at_base_rate(TPR, FPR, pi)
        print(f"  pi={pi:<7g} empirical precision {100*emp:6.2f}%  vs. Bayes {100*theo:6.2f}%")
    print("  -> Theory and measurement match. The effect is not a quirk of the data,")
    print("     it is pure probability theory.")

    # ---------- 4) what would be needed? / operating point ----------
    print("\n=== 4) What would the detector have to be able to do? ===")
    for target in (0.5, 0.9):
        n_fpr = required_fpr(TPR, pi_real, target)
        print(f"  For PPV = {100*target:.0f} % at pi=1e-4 you would need FPR <= {n_fpr:.3e} "
              f"({FPR/n_fpr:,.0f}x better than now)")
    print("  -> The bottleneck is the FPR, NOT the detection rate.")

    C_FALSE_ALARM, C_MISSED = 5.0, 50_000.0     # EUR: analyst time vs. damage event
    idx, threshold, cost = best_operating_point(
        fpr_c, tpr_c, thresholds, FLOWS_PER_DAY, pi_real, C_FALSE_ALARM, C_MISSED)
    t, f = alarms_per_day(FLOWS_PER_DAY, tpr_c[idx], fpr_c[idx], pi_real)
    print(f"\nCost-optimal operating point (false alarm {C_FALSE_ALARM:.0f} EUR, "
          f"missed attack {C_MISSED:,.0f} EUR):")
    print(f"  threshold {threshold:.4f} -> TPR {tpr_c[idx]:.4f}, FPR {fpr_c[idx]:.5f}")
    print(f"  {t:,.0f} true + {f:,.0f} false alarms/day, expected cost {cost:,.0f} EUR/day")
    print(f"  Comparison with the 'TPR=99 %' point: "
          f"{expected_cost(FLOWS_PER_DAY, TPR, FPR, pi_real, C_FALSE_ALARM, C_MISSED):,.0f} EUR/day")

    # ---------- plots ----------
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

        pis = np.logspace(-6, -1, 200)
        for t_, f_, lab in [(TPR, FPR, f"our detector (FPR={FPR:.3f})"),
                            (0.99, 1e-3, "very good (FPR=1e-3)"),
                            (0.99, 1e-5, "utopian (FPR=1e-5)")]:
            ax1.semilogx(pis, 100*ppv_at_base_rate(t_, f_, pis), lw=1.8, label=lab)
        ax1.axvline(1e-4, ls="--", color="gray", lw=1)
        ax1.annotate("realistic\nbase rate", xy=(1e-4, 60), fontsize=8, color="gray")
        ax1.set(xlabel="base rate $\\pi$ (share of attacks)",
                ylabel="PPV = P(attack | alarm)  [%]",
                title="Base-rate fallacy: good detectors, useless alarms", ylim=(0, 100))
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        pis2 = np.logspace(-5, -1.5, 40)
        alarms = [alarms_per_day(FLOWS_PER_DAY, TPR, FPR, p) for p in pis2]
        ax2.loglog(pis2, [a[0] for a in alarms], lw=1.8, label="true alarms/day")
        ax2.loglog(pis2, [a[1] for a in alarms], lw=1.8, label="false alarms/day")
        ax2.axhline(100, ls=":", color="k", lw=1, label="what a team can handle (~100/day)")
        ax2.set(xlabel="base rate $\\pi$", ylabel="alarms per day",
                title=f"Absolute numbers at {FLOWS_PER_DAY:,} flows/day")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")

        plt.tight_layout(); plt.savefig("results/base_rate.png", dpi=110)
        print("\nPlot saved: results/base_rate.png")
    except Exception as e:
        print("(no plot:", e, ")")


if __name__ == "__main__":
    main()
