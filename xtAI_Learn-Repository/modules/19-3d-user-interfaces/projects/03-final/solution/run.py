"""Vergleichende 3D-Selektionsstudie — Auswertung.  (Referenzloesung P03-final)

    /Users/.../.venv/bin/python run.py
Plots -> results/ (gitignored). Reine CPU-Sekunden.
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
    print("EXPERIMENT A — Genauigkeit ueber vier Bedingungen (2000 Trials je Zelle)")
    print("=" * 70)
    conds = [
        ("ISOLATED-NEAR", dict(L=1.5, nd=1, spread=20, occ=0.0, sig_deg=1.5)),
        ("SPARSE-FAR",     dict(L=8.0, nd=3, spread=14, occ=0.2, sig_deg=1.5)),
        ("DENSE-NEAR",     dict(L=2.0, nd=6, spread=6,  occ=0.15, sig_deg=1.0)),
        ("DENSE-FAR",      dict(L=8.0, nd=6, spread=5,  occ=0.2, sig_deg=1.5)),
    ]
    print(f"  {'Bedingung':15s} | {'raycast':>8} | {'cone':>8} | {'bubble':>8} | bester")
    print("  " + "-" * 60)
    results = {}
    for name, kw in conds:
        acc = accuracy(**kw)
        results[name] = acc
        best = max(acc, key=acc.get)
        print(f"  {name:15s} | {acc['raycast']:8.3f} | {acc['cone']:8.3f} | "
              f"{acc['bubble']:8.3f} | {best}")
    print("  => Kein universeller Sieger:")
    print("     - ISOLIERT/NAH: alle ~0.95 (Ray-Casting voll brauchbar).")
    print("     - FERN: Ray-Casting bricht ein (angulare Schrumpfung), Cone/Bubble fangen ein.")
    print("     - DICHT: Bubble ueber-selektiert (< Cone); Cone ist der robusteste Allrounder.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(conds)); w = 0.25
    for i, t in enumerate(TECHS):
        ax.bar(x + (i - 1) * w, [results[n][t] for n, _ in conds], w, label=t)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in conds], fontsize=9)
    ax.set_ylabel("Selektions-Genauigkeit"); ax.set_ylim(0, 1.05)
    ax.set_title("3D-Selektion: keine Technik gewinnt ueberall"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "bedingungen.png"), dpi=110); plt.close(fig)
    return results


def exp_distance_grid():
    print("\n" + "=" * 70)
    print("EXPERIMENT A2 — Genauigkeit ueber Distanz (duenne Szene, nd=2)")
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
    print("  => Ray-Casting faellt steil mit der Distanz; Cone/Bubble bleiben hoch (Capture).")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    for t in TECHS:
        ax.plot(Ls, grid[t], "o-", label=t)
    ax.set_xlabel("Zieldistanz L [m]"); ax.set_ylabel("Genauigkeit")
    ax.set_title("Distanz-Effekt (duenne Szene)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "distanz_grid.png"), dpi=110); plt.close(fig)


def exp_time_throughput():
    print("\n" + "=" * 70)
    print("EXPERIMENT B — Selektionszeit & Throughput auf ISOLIERTEN Zielen (angulares Fitts)")
    print("=" * 70)
    print("Modell (aus P02): MT = a + b*log2(A/W_eff + 1). W_eff = effektive Fangbreite:")
    print("  raycast: 2*Ziel-Winkelradius | cone: 2*Kegelhalbwinkel | bubble: grosse Voronoi-Breite")
    a, b = 0.15, 0.20
    A = np.deg2rad(30.0)               # angulare Amplitude
    L, r = 3.0, 0.12
    ang_r = np.arctan2(r, L)           # Ziel-Winkelradius
    W = {"raycast": 2 * ang_r,
         "cone": 2 * np.deg2rad(8.0),
         "bubble": 2 * np.deg2rad(20.0)}  # isoliert -> grosse Fangflaeche
    print(f"  {'Technik':8s} | {'W_eff [deg]':>11} | {'ID [bit]':>8} | {'MT [s]':>7} | {'TP [bit/s]':>10}")
    print("  " + "-" * 54)
    MTs = {}
    for t in TECHS:
        ID = np.log2(A / W[t] + 1.0)
        MT = a + b * ID
        TP = ID / MT
        MTs[t] = MT
        print(f"  {t:8s} | {np.rad2deg(W[t]):11.2f} | {ID:8.2f} | {MT:7.3f} | {TP:10.2f}")
    print(f"  => Bubble ist am SCHNELLSTEN ({MTs['bubble']:.2f}s vs Ray-Casting {MTs['raycast']:.2f}s,")
    print("     x{:.1f}), weil die grosse Fangflaeche das ID senkt. Aber der Throughput ist bei"
          .format(MTs['raycast'] / MTs['bubble']))
    print("     Ray-Casting HOEHER (praeziser) — die Bubble-Geschwindigkeit erkauft man mit der")
    print("     Dichte-Anfaelligkeit aus Experiment A. Zeit vs Praezision vs Robustheit.")


def exp_statistics():
    print("\n" + "=" * 70)
    print("EXPERIMENT C — Within-subject-Statistik (16 simulierte Versuchspersonen)")
    print("=" * 70)
    P = 16
    # SPARSE-FAR: erwartet cone/bubble >> raycast
    print("  Bedingung SPARSE-FAR — Genauigkeit je VP (100 Trials), Techniken paarweise verglichen:")
    per = {t: [] for t in TECHS}
    for p in range(P):
        acc = accuracy(L=8, nd=3, spread=14, occ=0.2, sig_deg=1.5, n=100, seed=1000 + p)
        for t in TECHS:
            per[t].append(acc[t])
    per = {t: np.array(v) for t, v in per.items()}
    for t in TECHS:
        print(f"    {t:8s}: Mittel {per[t].mean():.3f} +/- {per[t].std(ddof=1):.3f}")

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
    print(f"\n  {'Vergleich':18s} | {'d.Mittel':>9} | {'rank-bis.':>9} | {'p (roh)':>9} | {'p (Holm)':>9} | sig?")
    print("  " + "-" * 74)
    for (a_, b_, dm, rb, pval), pa in zip(stats, adj):
        sig = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else "n.s."
        print(f"  {a_+' vs '+b_:18s} | {dm:9.3f} | {rb:9.3f} | {pval:9.4f} | {pa:9.4f} | {sig}")
    print("  => cone/bubble schlagen Ray-Casting hochsignifikant (grosse Effektstaerke);")
    print("     der Unterschied cone vs bubble ist hier klein.")

    # DENSE-NEAR: erwartet cone > bubble (Bubble ueber-selektiert)
    print("\n  Bedingung DENSE-NEAR — Fokus cone vs bubble:")
    perc, perb = [], []
    for p in range(P):
        acc = accuracy(L=2, nd=6, spread=6, occ=0.15, sig_deg=1.0, n=100, seed=2000 + p)
        perc.append(acc["cone"]); perb.append(acc["bubble"])
    perc, perb = np.array(perc), np.array(perb)
    W, pval = wilcoxon(perc, perb)
    rb = rank_biserial_from_diffs(perc - perb)
    print(f"    cone {perc.mean():.3f} vs bubble {perb.mean():.3f} | d.Mittel {perc.mean()-perb.mean():+.3f}"
          f" | rank-bis {rb:.3f} | p={pval:.4f}")
    print("    => In dichten Szenen ist Cone signifikant besser: Bubble faengt den naechsten")
    print("       Nachbarn statt des Ziels ('over-selection').")


if __name__ == "__main__":
    exp_conditions()
    exp_distance_grid()
    exp_time_throughput()
    exp_statistics()
    print("\nPlots in results/. Fertig.")
