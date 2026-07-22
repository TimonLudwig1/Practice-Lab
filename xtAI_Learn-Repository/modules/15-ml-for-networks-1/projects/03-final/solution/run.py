"""Zero-day detection: misuse vs. anomaly — the final experiment.

Scenario: in training ONLY the attack 'smurf' is known. All other attack types
(neptune, satan, teardrop, ...) only appear in production = zero-days.

Question: who finds them - the supervised detector (knows attacks, but only the known ones)
or the anomaly detector (knows NO attacks at all, only normal traffic)?

At the end: the reality check with the base rate from project 02.

Call:  python run.py     (~1-2 min)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from flow_data import ZeroDayScenario
from detectors import (SupervisedDetector, IsolationForestDetector,
                       MahalanobisDetector, OneClassSVMDetector, LOFDetector)

FLOWS_PER_DAY = 10_000_000
PI_REAL = 1e-4


def ppv(tpr, fpr, pi):
    return tpr * pi / (tpr * pi + fpr * (1 - pi)) if (tpr * pi + fpr * (1 - pi)) > 0 else 0.0


def main():
    scen = ZeroDayScenario(known_attacks=("smurf",), random_state=0)
    print("=== Scenario ===")
    print(scen.overview())
    print("\nOnly 'smurf' is known in training. Everything else is a zero-day.\n")

    # ---------- train the detectors ----------
    Xtr_sup, ytr_sup = scen.supervised_training_data()
    detectors = []

    sup = SupervisedDetector().fit(Xtr_sup, ytr_sup)
    detectors.append(sup)
    for D in (IsolationForestDetector(), MahalanobisDetector(),
              OneClassSVMDetector(), LOFDetector()):
        detectors.append(D.fit(scen.normal_train))      # ONLY normal traffic!

    # ---------- evaluation per attack type ----------
    rows = []
    for det in detectors:
        row = {"Detector": det.name}
        row["FPR (normal)"] = det.predict(scen.normal_test).mean()
        row["smurf (known)"] = det.predict(scen.known_test).mean()
        for a, Xa in scen.zeroday.items():
            if len(Xa) >= 10:                            # types that are too small are pure noise
                row[f"{a} (0-day)"] = det.predict(Xa).mean()
        rows.append(row)

    tab = pd.DataFrame(rows).set_index("Detector")
    print("=== Recall per attack type (share of detected flows) ===")
    print("(the column 'FPR (normal)' is the PRICE: false alarms on harmless traffic)\n")
    print(tab.round(3).to_string())

    # ---------- the core statement ----------
    zd_columns = [c for c in tab.columns if "0-day" in c]
    print("\n=== Core statement: average zero-day recall ===")
    for name, row in tab.iterrows():
        print(f"  {name:44s} zero-day recall {row[zd_columns].mean():.3f} "
              f"| FPR {row['FPR (normal)']:.4f}")

    print("\n--> The supervised detector recognizes 'smurf' perfectly and is COMPLETELY BLIND")
    print("    to 'neptune' (a 928-flow SYN flood!) - it has simply never seen it.")
    print("    The anomaly detectors have NEVER seen an attack and still find it.")
    print("    The price: they also raise alarms on harmless traffic (FPR > 0).")

    # ---------- reality check: base rate (project 02) ----------
    print("\n=== Reality check: what does this mean in production? (project 02) ===")
    print(f"At {FLOWS_PER_DAY:,} flows/day and a base rate of pi={PI_REAL:g}:")
    print(f"{'Detector':44s} {'PPV':>8s} {'false alarms/day':>18s}")
    for name, row in tab.iterrows():
        tpr = row[zd_columns].mean()
        fpr = row["FPR (normal)"]
        p = ppv(tpr, fpr, PI_REAL)
        fa = fpr * (1 - PI_REAL) * FLOWS_PER_DAY
        print(f"{name:44s} {100*p:7.3f}% {fa:18,.0f}")
    print("\n--> This is exactly where anomaly detection dies in practice: 1 % FPR sounds tiny,")
    print("    but means ~100,000 false alarms per day. The supervised detector has FPR~0")
    print("    - but sees nothing new in return. THAT is the actual trade-off.")

    # ---------- plot ----------
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        plot_cols = ["smurf (known)"] + zd_columns
        ax = tab[plot_cols].T.plot.bar(figsize=(12, 4.8), width=0.8)
        ax.set(ylabel="recall", title="Zero-day detection: misuse vs. anomaly "
                                      "(only 'smurf' was known in training)", ylim=(0, 1.05))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3, axis="y")
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.annotate("known |  zero-day", xy=(0.52, 1.0), fontsize=8)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout(); plt.savefig("results/zeroday.png", dpi=110)
        print("\nPlot saved: results/zeroday.png")
    except Exception as e:
        print("(no plot:", e, ")")


if __name__ == "__main__":
    main()
