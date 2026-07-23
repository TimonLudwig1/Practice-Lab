"""Evaluation of the Put-that-there interpreter: ablations, the mu study, the quantification
of mutual disambiguation.  (The reference solution for P03-final)

Run:
    /Users/.../.venv/bin/python run.py
The plots go to results/ (gitignored).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from putthatthere import make_scene, make_command, resolve, resolve_naive_at_word

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

N_CMDS = 3000
MU_TRUE = -0.15
SIGMA = 0.035
TAU = 0.18


def gen_commands(n=N_CMDS, base_seed=10000):
    """A fixed pool of commands over changing scenes (reproducible)."""
    cmds = []
    for k in range(n):
        scene = make_scene(n_objects=7, seed=base_seed + k)
        cmd = make_command(scene, seed=base_seed + 100000 + k,
                           mu_true=MU_TRUE, sigma_point=SIGMA, tau_true=TAU)
        cmds.append(cmd)
    return cmds


def eval_config(cmds, **kw):
    correct = 0
    for cmd in cmds:
        _, pred = resolve(cmd, mu_hat=MU_TRUE, sigma_hat=SIGMA, tau_hat=TAU, **kw)
        correct += (pred == cmd["target"])
    return correct / len(cmds)


def ablation_study(cmds):
    print("=" * 68)
    print("ABLATION — how much does each fusion factor contribute?")
    print("=" * 68)
    configs = [
        ("pointing only (P_point)",           dict(use_sem=False, use_point=True,  use_temp=False)),
        ("semantics only (P_sem)",            dict(use_sem=True,  use_point=False, use_temp=False)),
        ("pointing + semantics (no time)",    dict(use_sem=True,  use_point=True,  use_temp=False)),
        ("pointing + time (no semantics)",    dict(use_sem=False, use_point=True,  use_temp=True)),
        ("FULL FUSION (all three)",           dict(use_sem=True,  use_point=True,  use_temp=True)),
    ]
    results = {}
    for name, kw in configs:
        acc = eval_config(cmds, **kw)
        results[name] = acc
        print(f"  {name:32s}: {acc:.3f}")
    # the naive baseline
    accN = np.mean([resolve_naive_at_word(c) == c["target"] for c in cmds])
    results["naive: nearest object @ t_word"] = accN
    print(f"  {'naive: nearest object @ t_word':32s}: {accN:.3f}")
    print("  => Every factor resolves a different ambiguity; only all three together")
    print("     produce the maximum (complementary, not redundant cues).")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = list(results.keys()); vals = [results[n] for n in names]
    colors = ["gray", "gray", "steelblue", "orange", "green", "lightcoral"]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_xlabel("resolution accuracy")
    ax.set_title("Put-that-there: the contribution of the fusion factors")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.invert_yaxis(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation.png"), dpi=110); plt.close(fig)
    return results


def mutual_disambiguation(cmds):
    print("\n" + "=" * 68)
    print("MUTUAL DISAMBIGUATION — where one modality alone fails, the other rescues it")
    print("=" * 68)
    # the cases in which pointing alone is WRONG but the full fusion is right:
    saved_by_fusion = 0
    point_wrong = 0
    for cmd in cmds:
        _, p_point = resolve(cmd, use_sem=False, use_point=True, use_temp=False,
                             mu_hat=MU_TRUE, sigma_hat=SIGMA, tau_hat=TAU)
        _, p_full = resolve(cmd, mu_hat=MU_TRUE, sigma_hat=SIGMA, tau_hat=TAU)
        if p_point != cmd["target"]:
            point_wrong += 1
            if p_full == cmd["target"]:
                saved_by_fusion += 1
    print(f"  commands for which pointing ALONE is wrong : {point_wrong}/{len(cmds)}")
    print(f"  of those RESCUED by the full fusion        : {saved_by_fusion} "
          f"({100*saved_by_fusion/max(point_wrong,1):.1f} %)")
    print("  => That is mutual disambiguation: the temporal + semantic context")
    print("     corrects the spatial ambiguity of the pointing.")

    # separated by the cause of the error: the decoy (time) vs. the twin (semantics)
    return dict(point_wrong=point_wrong, saved=saved_by_fusion)


def mu_study(cmds):
    print("\n" + "=" * 68)
    print("mu STUDY — why the temporal offset (the gesture leads) has to be modelled")
    print("=" * 68)
    mus = np.linspace(-0.5, 0.3, 17)
    accs = []
    for mu in mus:
        correct = 0
        for cmd in cmds:
            _, pred = resolve(cmd, mu_hat=mu, sigma_hat=SIGMA, tau_hat=TAU)
            correct += (pred == cmd["target"])
        accs.append(correct / len(cmds))
    accs = np.array(accs)
    best = mus[int(accs.argmax())]
    acc_at0 = accs[int(np.argmin(np.abs(mus - 0.0)))]
    print(f"  the true mu (the gesture leads) = {MU_TRUE:+.2f} s")
    print(f"  the best mu_hat                 = {best:+.2f} s  -> accuracy {accs.max():.3f}")
    print(f"  mu_hat = 0 (the naive assumption 'simultaneous'): accuracy {acc_at0:.3f}")
    print(f"  => Whoever ignores the lead of the gesture (mu=0) loses "
          f"{100*(accs.max()-acc_at0):.1f} percentage points.")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(mus, accs, "o-", color="purple")
    ax.axvline(MU_TRUE, ls="--", color="green", label=f"the true mu = {MU_TRUE}")
    ax.axvline(0, ls=":", color="red", label="the naive assumption mu=0")
    ax.set_xlabel("assumed temporal offset mu_hat [s]")
    ax.set_ylabel("resolution accuracy")
    ax.set_title("The temporal offset has to be modelled")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mu_study.png"), dpi=110); plt.close(fig)
    return dict(best=best, acc_best=accs.max(), acc_at0=acc_at0)


if __name__ == "__main__":
    print(f"Generating {N_CMDS} commands ...")
    cmds = gen_commands()
    twin_frac = np.mean([c["twin"] for c in cmds])
    print(f"  ({100*twin_frac:.0f} % of them with a spatial twin)\n")
    ablation_study(cmds)
    mutual_disambiguation(cmds)
    mu_study(cmds)
    print("\nThe plots are in results/. Done.")
