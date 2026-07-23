"""A comparative 3D selection study — the analysis.  (Reference solution P03-final)

    /Users/.../.venv/bin/python run.py
The plots go to results/ (gitignored). Pure CPU seconds.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from selection3d import make_trial, noisy_ray_dir, run_technique, angle_between
from stats_tools import rank_biserial_from_diffs, holm_bonferroni

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

TECHS = ["raycast", "cone", "bubble"]
ORIGIN = np.zeros(3)


def accuracy(L, nd, spread, occ, sig_deg, n=2000, seed=0, r=0.12):
    rng = np.random.default_rng(seed)
    sig = np.deg2rad(sig_deg)
    hit = {t: 0 for t in TECHS}
    for _ in range(n):
        pos, radii, tgt, base = make_trial(rng, L_target=L, n_distractors=nd,
                                           spread_deg=spread, r=r, occlude_frac=occ)
        d = noisy_ray_dir(base, sig, rng)
        for t in TECHS:
            hit[t] += (run_technique(t, pos, radii, ORIGIN, d) == tgt)
    return {t: hit[t] / n for t in TECHS}


def exp_conditions():
    print("=" * 70)
    print("EXPERIMENT A — accuracy over four conditions (2000 trials per cell)")
    print("=" * 70)
    conds = [
        ("ISOLATED-NEAR", dict(L=1.5, nd=1, spread=20, occ=0.0, sig_deg=1.5)),
        ("SPARSE-FAR",     dict(L=8.0, nd=3, spread=14, occ=0.2, sig_deg=1.5)),
        ("DENSE-NEAR",     dict(L=2.0, nd=6, spread=6,  occ=0.15, sig_deg=1.0)),
        ("DENSE-FAR",      dict(L=8.0, nd=6, spread=5,  occ=0.2, sig_deg=1.5)),
    ]
    print(f"  {'condition':15s} | {'raycast':>8} | {'cone':>8} | {'bubble':>8} | best")
    print("  " + "-" * 60)
    results = {}
    for name, kw in conds:
        acc = accuracy(**kw)
        results[name] = acc
        best = max(acc, key=acc.get)
        print(f"  {name:15s} | {acc['raycast']:8.3f} | {acc['cone']:8.3f} | "
              f"{acc['bubble']:8.3f} | {best}")
    print("  => No universal winner:")
    print("     - ISOLATED/NEAR: all ~0.95 (ray-casting fully usable).")
    print("     - FAR: ray-casting collapses (angular shrinkage), cone/bubble capture.")
    print("     - DENSE: bubble over-selects (< cone); cone is the most robust all-rounder.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(conds)); w = 0.25
    for i, t in enumerate(TECHS):
        ax.bar(x + (i - 1) * w, [results[n][t] for n, _ in conds], w, label=t)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in conds], fontsize=9)
    ax.set_ylabel("selection accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("3D selection: no technique wins everywhere"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "conditions.png"), dpi=110); plt.close(fig)
    return results


def exp_distance_grid():
    print("\n" + "=" * 70)
    print("EXPERIMENT A2 — accuracy over distance (a sparse scene, nd=2)")
    print("=" * 70)
    Ls = [1, 2, 4, 8, 16]
    grid = {t: [] for t in TECHS}
    for L in Ls:
        acc = accuracy(L=L, nd=2, spread=16, occ=0.15, sig_deg=1.5, n=1500)
        for t in TECHS:
            grid[t].append(acc[t])
    print(f"  {'L [m]':>6} | " + " | ".join(f"{t:>8}" for t in TECHS))
    print("  " + "-" * 40)
    for i, L in enumerate(Ls):
        print(f"  {L:6d} | " + " | ".join(f"{grid[t][i]:8.3f}" for t in TECHS))
    print("  => Ray-casting falls steeply with distance; cone/bubble stay high (capture).")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    for t in TECHS:
        ax.plot(Ls, grid[t], "o-", label=t)
    ax.set_xlabel("target distance L [m]"); ax.set_ylabel("accuracy")
    ax.set_title("The distance effect (a sparse scene)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "distance_grid.png"), dpi=110); plt.close(fig)


def exp_time_throughput():
    print("\n" + "=" * 70)
    print("EXPERIMENT B — selection time & throughput on ISOLATED targets (angular Fitts)")
    print("=" * 70)
    print("The model (from P02): MT = a + b*log2(A/W_eff + 1). W_eff = the effective capture width:")
    print("  raycast: 2*the target angular radius | cone: 2*the cone half angle | bubble: a large Voronoi width")
    a, b = 0.15, 0.20
    A = np.deg2rad(30.0)               # the angular amplitude
    L, r = 3.0, 0.12
    ang_r = np.arctan2(r, L)           # the target angular radius
    W = {"raycast": 2 * ang_r,
         "cone": 2 * np.deg2rad(8.0),
         "bubble": 2 * np.deg2rad(20.0)}  # isolated -> a large capture area
    print(f"  {'technique':9s} | {'W_eff [deg]':>11} | {'ID [bit]':>8} | {'MT [s]':>7} | {'TP [bit/s]':>10}")
    print("  " + "-" * 54)
    MTs = {}
    for t in TECHS:
        ID = np.log2(A / W[t] + 1.0)
        MT = a + b * ID
        TP = ID / MT
        MTs[t] = MT
        print(f"  {t:9s} | {np.rad2deg(W[t]):11.2f} | {ID:8.2f} | {MT:7.3f} | {TP:10.2f}")
    print(f"  => Bubble is the FASTEST ({MTs['bubble']:.2f}s vs ray-casting {MTs['raycast']:.2f}s,")
    print("     x{:.1f}), because the large capture area lowers the ID. But the throughput is"
          .format(MTs['raycast'] / MTs['bubble']))
    print("     HIGHER for ray-casting (more precise) — the bubble speed is paid for with the")
    print("     density susceptibility from experiment A. Time vs precision vs robustness.")


def exp_statistics():
    print("\n" + "=" * 70)
    print("EXPERIMENT C — within-subject statistics (16 simulated participants)")
    print("=" * 70)
    P = 16
    # SPARSE-FAR: expects cone/bubble >> raycast
    print("  The SPARSE-FAR condition — accuracy per participant (100 trials), the techniques compared pairwise:")
    per = {t: [] for t in TECHS}
    for p in range(P):
        acc = accuracy(L=8, nd=3, spread=14, occ=0.2, sig_deg=1.5, n=100, seed=1000 + p)
        for t in TECHS:
            per[t].append(acc[t])
    per = {t: np.array(v) for t, v in per.items()}
    for t in TECHS:
        print(f"    {t:8s}: mean {per[t].mean():.3f} +/- {per[t].std(ddof=1):.3f}")

    pairs = [("cone", "raycast"), ("bubble", "raycast"), ("cone", "bubble")]
    raw_p, stats = [], []
    for a_, b_ in pairs:
        diffs = per[a_] - per[b_]
        try:
            W, pval = wilcoxon(per[a_], per[b_])
        except ValueError:
            W, pval = np.nan, 1.0
        rb = rank_biserial_from_diffs(diffs)
        raw_p.append(pval); stats.append((a_, b_, per[a_].mean() - per[b_].mean(), rb, pval))
    adj = holm_bonferroni(raw_p)
    print(f"\n  {'comparison':18s} | {'d.mean':>9} | {'rank-bis.':>9} | {'p (raw)':>9} | {'p (Holm)':>9} | sig?")
    print("  " + "-" * 74)
    for (a_, b_, dm, rb, pval), pa in zip(stats, adj):
        sig = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else "n.s."
        print(f"  {a_+' vs '+b_:18s} | {dm:9.3f} | {rb:9.3f} | {pval:9.4f} | {pa:9.4f} | {sig}")
    print("  => cone/bubble beat ray-casting highly significantly (a large effect size);")
    print("     the difference cone vs bubble is small here.")

    # DENSE-NEAR: expects cone > bubble (bubble over-selects)
    print("\n  The DENSE-NEAR condition — the focus is cone vs bubble:")
    perc, perb = [], []
    for p in range(P):
        acc = accuracy(L=2, nd=6, spread=6, occ=0.15, sig_deg=1.0, n=100, seed=2000 + p)
        perc.append(acc["cone"]); perb.append(acc["bubble"])
    perc, perb = np.array(perc), np.array(perb)
    W, pval = wilcoxon(perc, perb)
    rb = rank_biserial_from_diffs(perc - perb)
    print(f"    cone {perc.mean():.3f} vs bubble {perb.mean():.3f} | d.mean {perc.mean()-perb.mean():+.3f}"
          f" | rank-bis {rb:.3f} | p={pval:.4f}")
    print("    => In dense scenes cone is significantly better: bubble captures the nearest")
    print("       neighbour instead of the target ('over-selection').")


if __name__ == "__main__":
    exp_conditions()
    exp_distance_grid()
    exp_time_throughput()
    exp_statistics()
    print("\nThe plots are in results/. Done.")
