"""Auto-scaling experiments: the elasticity triple, the Pareto front, and flapping.
(Reference solution P02-medium)   /Users/.../.venv/bin/python run.py   Plots -> results/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scaling import simulate, oracle, workload, min_replicas

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def exp_triple():
    print("=" * 74)
    print("EXPERIMENT 1 — the elasticity triple: SLO violations, cost, flapping")
    print("=" * 74)
    print("  A single number never characterises an auto-scaler. Herbst/Kounev's triple:")
    print("  under-provisioning (SLO), over-provisioning (cost), instability (adaptations).\n")
    print(f"  {'policy':>12} | {'SLO viol':>9} | {'avg replicas':>12} | {'adaptations':>11} | {'over':>6}")
    print("  " + "-" * 62)
    results = {}
    for pol in ["reactive", "control", "predictive"]:
        r = simulate(pol)
        results[pol] = r
        print(f"  {pol:>12} | {r['slo_violations']:9.1%} | {r['mean_replicas']:12.2f} | "
              f"{r['adaptations']:11d} | {r['over']:6.2f}")
    o = oracle()
    print(f"  {'oracle':>12} | {o['slo_violations']:9.1%} | {o['mean_replicas']:12.2f} | "
          f"{'-':>11} | {0.0:6.2f}   <- perfect knowledge, no delay")
    print("\n  => Read the whole row, never one column. `reactive` looks best on SLO but only")
    print("     because it over-provisions (it buys the SLO with capacity). `predictive` runs")
    print("     closest to the oracle's cost — it is the only one that provisions almost exactly")
    print("     what is needed, because it inverts the model instead of reacting to a threshold.")

    # time-series plot around the flash crowd
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(results["reactive"]["needed"], color="black", lw=2, ls="--", label="needed (oracle)")
    for pol, col in [("reactive", "crimson"), ("control", "orange"), ("predictive", "steelblue")]:
        ax1.step(range(len(results[pol]["c"])), results[pol]["c"], color=col, lw=1.4, label=pol)
    ax1.set_ylabel("replicas"); ax1.legend(fontsize=8, ncol=4); ax1.grid(alpha=0.3)
    ax1.set_title("capacity vs. demand (the flash crowd is around t=125-140)")
    for pol, col in [("reactive", "crimson"), ("control", "orange"), ("predictive", "steelblue")]:
        ax2.plot(results[pol]["latency"], color=col, lw=1.1, label=pol)
    ax2.axhline(0.5, ls="--", color="black", label="SLO")
    ax2.set_ylim(0, 1.6); ax2.set_xlabel("control interval"); ax2.set_ylabel("p95 latency [s]")
    ax2.legend(fontsize=8, ncol=4); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "elasticity.png"), dpi=110); plt.close(fig)
    return results


def exp_pareto():
    print("\n" + "=" * 74)
    print("EXPERIMENT 2 — the honest comparison: the cost/SLO Pareto front")
    print("=" * 74)
    print("  Every policy has a knob trading cost against SLO violations, so comparing them at one")
    print("  setting proves nothing. We sweep each knob and compare the whole fronts.\n")
    sweeps = {
        "reactive":   [0.85, 0.80, 0.75, 0.70, 0.60, 0.50],
        "control":    [0.85, 0.80, 0.75, 0.70, 0.60, 0.50],
        "predictive": [1.00, 1.05, 1.10, 1.20, 1.30, 1.45],
    }
    fronts = {}
    for pol, knobs in sweeps.items():
        pts = []
        print(f"  {pol}:")
        for kn in knobs:
            r = simulate(pol, knob=kn)
            pts.append((r["mean_replicas"], r["slo_violations"], r["adaptations"]))
            print(f"    knob={kn:<5}  cost={r['mean_replicas']:5.2f}  "
                  f"viol={r['slo_violations']:6.1%}  adapt={r['adaptations']:3d}")
        fronts[pol] = np.array(pts)
    o = oracle()
    print(f"\n  oracle: cost={o['mean_replicas']:.2f}, viol=0.0 % (unreachable: it knows the future)")
    print("  => The fronts separate clearly. At EQUAL cost, `predictive` violates least; at EQUAL")
    print("     SLO, it is cheapest. `control` is dominated throughout. And `predictive` at its")
    print("     cheapest setting nearly touches the oracle's cost — the value of a model.")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pol, col in [("reactive", "crimson"), ("control", "orange"), ("predictive", "steelblue")]:
        f = fronts[pol]
        ax.plot(f[:, 0], f[:, 1] * 100, "o-", color=col, label=pol)
    ax.axvline(o["mean_replicas"], ls=":", color="green", label="oracle cost")
    ax.set_xlabel("average replicas (cost)  ->  cheaper is left")
    ax.set_ylabel("SLO violations [%]  ->  better is down")
    ax.set_title("elasticity Pareto front: predictive dominates")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pareto.png"), dpi=110); plt.close(fig)


def exp_flapping():
    print("\n" + "=" * 74)
    print("EXPERIMENT 3 — flapping: what hysteresis and cooldown are for")
    print("=" * 74)
    print(f"  {'configuration':>34} | {'adaptations':>11} | {'SLO viol':>9} | {'cost':>6}")
    print("  " + "-" * 68)
    configs = [
        ("gap 0.40, cooldown 2, serialised", dict(hyst_gap=0.40, cooldown=2, allow_concurrent=False)),
        ("gap 0.05 (weak hysteresis)", dict(hyst_gap=0.05, cooldown=2, allow_concurrent=False)),
        ("gap 0.40, no cooldown, concurrent", dict(hyst_gap=0.40, cooldown=0, allow_concurrent=True)),
        ("gap 0.05, no cooldown, concurrent", dict(hyst_gap=0.05, cooldown=0, allow_concurrent=True)),
    ]
    for label, kw in configs:
        r = simulate("reactive", knob=0.75, **kw)
        print(f"  {label:>34} | {r['adaptations']:11d} | {r['slo_violations']:9.1%} | "
              f"{r['mean_replicas']:6.2f}")
    print("  => Both mechanisms matter, and they are independent. Shrinking the hysteresis gap makes")
    print("     the manager oscillate around the threshold; allowing it to act again before the")
    print("     previous change has taken effect (no cooldown, concurrent changes) makes it chase")
    print("     its own not-yet-visible corrections. Remove both and the adaptation count multiplies")
    print("     — and since every adaptation costs a warm-up in reality, flapping actively degrades")
    print("     the system it is supposed to be improving.")


if __name__ == "__main__":
    exp_triple()
    exp_pareto()
    exp_flapping()
    print("\nPlots in results/. Done.")
