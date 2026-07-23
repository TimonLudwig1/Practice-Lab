"""Analysis of the XR user study: 3 DoF vs. 6 DoF.

The common thread is METHODOLOGY, not "6 DoF is better" (we know that in advance). The point is
to evaluate a study SO that the conclusions hold:
  1. descriptives + check the design (was the counterbalancing kept?)
  2. the right test (ordinal -> Wilcoxon) and the EFFECT SIZE (not only p)
  3. correct for multiple comparisons (Bonferroni/Holm)
  4. demonstrate the order effects - and show what would have happened without counterbalancing

Call:  python run.py       (~1 s)
"""
from __future__ import annotations
import numpy as np

from generate_study import (generate_study, generate_naive_without_counterbalancing, TRUTH)
from stats_tools import (paired_comparison, bonferroni, holm_bonferroni, order_effect)

# (name, column, which direction is "better for 6 DoF"?)
OUTCOMES = [
    ("Presence (IPQ)", "presence", "higher"),
    ("Sickness (SSQ)", "sickness", "lower"),
    ("Task time [s]",  "time",     "lower"),
    ("Comfort",        "comfort",  "higher"),
]


def pairs(df, column):
    a = df[df.condition == "3DoF"].sort_values("participant")[column].values
    b = df[df.condition == "6DoF"].sort_values("participant")[column].values
    return a, b


def main():
    df = generate_study(n_participants=24, seed=3)
    n = df.participant.nunique()

    print("=== Study: 3 DoF vs. 6 DoF, within-subject ===")
    print(f"N = {n} participants, each tests BOTH conditions.")
    cb = df[df.position == 1].condition.value_counts().to_dict()
    print(f"Counterbalancing: {cb} start with this condition respectively "
          f"({'balanced' if len(set(cb.values())) == 1 else 'UNBALANCED!'}).")

    # ---------- 1)+2) the comparisons with effect sizes ----------
    print("\n=== Comparison per outcome (ordinal -> Wilcoxon; ALWAYS add the effect size) ===")
    print(f"{'Outcome':18s} {'3DoF':>7s} {'6DoF':>7s} {'t-p':>8s} {'Wilcox-p':>9s} "
          f"{'dz':>6s} {'r':>6s}")
    pvals = []
    for name, column, _ in OUTCOMES:
        a, b = pairs(df, column)
        r = paired_comparison(a, b)
        pvals.append(r["wilcoxon_p"])
        print(f"{name:18s} {r['mean_x']:7.1f} {r['mean_y']:7.1f} {r['t_p']:8.4f} "
              f"{r['wilcoxon_p']:9.4f} {r['cohen_dz']:6.2f} {r['rank_biserial']:6.2f}")
    print("(dz = Cohen's dz, parametric; r = rank-biserial for the Wilcoxon test. |0.2| small, "
          "0.5 medium, 0.8 large.)")

    # ---------- 3) multiple comparisons ----------
    print("\n=== Correcting for multiple comparisons (4 tests!) ===")
    threshold, bonf = bonferroni(pvals, alpha=0.05)
    holm = holm_bonferroni(pvals, alpha=0.05)
    print(f"{'Outcome':18s} {'Wilcox-p':>9s} {'raw<0.05':>9s} "
          f"{'Bonf<'+format(threshold,'.4f'):>13s} {'Holm':>6s}")
    for i, (name, _, _) in enumerate(OUTCOMES):
        raw = "sig" if pvals[i] < 0.05 else "n.s."
        b = "sig" if bonf[i] else "n.s."
        h = "sig" if holm[i] else "n.s."
        flag = "   <-- tips!" if raw != b else ""
        print(f"{name:18s} {pvals[i]:9.4f} {raw:>9s} {b:>13s} {h:>6s}{flag}")
    print("-> Comfort is significant raw (p<0.05), but tips under Bonferroni - and it is the")
    print("   ONLY one with a small effect size (dz 0.50 vs. >1.1 for the others). Both say:")
    print("   this is the SHAKIEST finding. (Holm is less conservative and keeps it narrowly -")
    print("   which is exactly why you ALWAYS report the effect size alongside, not only the")
    print("   significance.) The underlying idea is the same as the base-rate fallacy (module 15):")
    print("   many tests x a small error rate = many false alarms.")

    # ---------- 4) order effects + the counter-check ----------
    print("\n=== Why counterbalancing was necessary (order effects) ===")
    for name, column in [("Sickness", "sickness"), ("Task time", "time")]:
        oe = order_effect(df, column)
        print(f"  {name:12s}: 1st session {oe['mean_first']:6.1f}  vs  2nd session "
              f"{oe['mean_second']:6.1f}  (p={oe['p']:.3f})")
    print("  -> There ARE order effects (carryover nausea, a learning effect for the time).")
    print("     Because counterbalancing was used, they average out across the conditions.")

    print("\n--- The counter-check: what would have happened without counterbalancing ---")
    df_naive = generate_naive_without_counterbalancing(n_participants=24, seed=3)
    a, b = pairs(df_naive, "sickness")
    measured = a.mean() - b.mean()
    true_value = -TRUTH["sickness_effect"]      # 3 DoF is sicker by 12
    print(f"  If EVERYBODY does 3 DoF first and 6 DoF second (no counterbalancing):")
    print(f"  the measured sickness difference is {measured:.1f} instead of the true "
          f"{true_value:.0f} points.")
    print(f"  -> The carryover (+{TRUTH['carryover_sickness']:.0f} onto the 2nd, i.e. the 6 DoF "
          f"session)")
    print(f"     MASKS half of the real effect. Without counterbalancing you would have")
    print(f"     wrongly classified 6 DoF as barely better.")

    # ---------- plots ----------
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        for ax, (name, column, direction) in zip(axes, OUTCOMES):
            a, b = pairs(df, column)
            # paired lines: every person is one line between the conditions
            for pa, pb in zip(a, b):
                ax.plot([0, 1], [pa, pb], color="gray", alpha=0.35, lw=0.8)
            ax.plot([0, 1], [a.mean(), b.mean()], "o-", color="crimson", lw=2.5, ms=8)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["3 DoF", "6 DoF"])
            ax.set_title(f"{name}\n({direction}=better)", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
        plt.suptitle("Paired data: every grey line is one participant", fontsize=10)
        plt.tight_layout(); plt.savefig("results/study_results.png", dpi=110)
        print("\nPlot saved: results/study_results.png")
    except Exception as e:
        print("(no plot:", e, ")")


if __name__ == "__main__":
    main()
