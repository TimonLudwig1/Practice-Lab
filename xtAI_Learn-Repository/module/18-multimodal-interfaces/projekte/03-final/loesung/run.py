"""Evaluation des Put-that-there-Interpreters: Ablationen, mu-Studie,
Mutual-Disambiguation-Quantifizierung.  (Referenzloesung P03-final)

Ausfuehren:
    /Users/.../.venv/bin/python run.py
Plots -> ergebnisse/ (gitignored).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from putthatthere import make_scene, make_command, resolve, resolve_naive_at_word

OUT = os.path.join(os.path.dirname(__file__), "ergebnisse")
os.makedirs(OUT, exist_ok=True)

N_CMDS = 3000
MU_TRUE = -0.15
SIGMA = 0.035
TAU = 0.18


def gen_commands(n=N_CMDS, base_seed=10000):
    """Ein fester Pool von Kommandos ueber wechselnde Szenen (reproduzierbar)."""
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
    print("ABLATION — welcher Fusionsfaktor traegt wie viel bei?")
    print("=" * 68)
    configs = [
        ("nur Zeigen (P_point)",            dict(use_sem=False, use_point=True,  use_temp=False)),
        ("nur Semantik (P_sem)",            dict(use_sem=True,  use_point=False, use_temp=False)),
        ("Zeigen + Semantik (ohne Zeit)",   dict(use_sem=True,  use_point=True,  use_temp=False)),
        ("Zeigen + Zeit (ohne Semantik)",   dict(use_sem=False, use_point=True,  use_temp=True)),
        ("VOLLE FUSION (alle drei)",        dict(use_sem=True,  use_point=True,  use_temp=True)),
    ]
    results = {}
    for name, kw in configs:
        acc = eval_config(cmds, **kw)
        results[name] = acc
        print(f"  {name:32s}: {acc:.3f}")
    # naive Baseline
    accN = np.mean([resolve_naive_at_word(c) == c["target"] for c in cmds])
    results["naiv: naechstes Objekt @ t_word"] = accN
    print(f"  {'naiv: naechstes Objekt @ t_word':32s}: {accN:.3f}")
    print("  => Jeder Faktor loest eine andere Mehrdeutigkeit; erst alle drei zusammen")
    print("     bringen das Maximum (komplementaere, nicht redundante Hinweise).")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = list(results.keys()); vals = [results[n] for n in names]
    colors = ["gray", "gray", "steelblue", "orange", "green", "lightcoral"]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Aufloesungs-Genauigkeit")
    ax.set_title("Put-that-there: Beitrag der Fusionsfaktoren")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.invert_yaxis(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation.png"), dpi=110); plt.close(fig)
    return results


def mutual_disambiguation(cmds):
    print("\n" + "=" * 68)
    print("MUTUAL DISAMBIGUATION — wo eine Modalitaet allein scheitert, rettet die andere")
    print("=" * 68)
    # Faelle, in denen Zeigen-allein DANEBEN liegt, volle Fusion aber richtig:
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
    print(f"  Kommandos, bei denen Zeigen ALLEIN falsch liegt : {point_wrong}/{len(cmds)}")
    print(f"  davon durch volle Fusion GERETTET               : {saved_by_fusion} "
          f"({100*saved_by_fusion/max(point_wrong,1):.1f} %)")
    print("  => Das ist Mutual Disambiguation: der zeitliche + semantische Kontext")
    print("     korrigiert die raeumliche Mehrdeutigkeit des Zeigens.")

    # getrennt nach Fehlerursache: Decoy (Zeit) vs Zwilling (Semantik)
    return dict(point_wrong=point_wrong, saved=saved_by_fusion)


def mu_study(cmds):
    print("\n" + "=" * 68)
    print("mu-STUDIE — warum der zeitliche Versatz (Geste fuehrt) modelliert werden muss")
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
    print(f"  wahres mu (Geste fuehrt) = {MU_TRUE:+.2f} s")
    print(f"  bestes mu_hat            = {best:+.2f} s  -> Genauigkeit {accs.max():.3f}")
    print(f"  mu_hat = 0 (naive Annahme 'gleichzeitig'): Genauigkeit {acc_at0:.3f}")
    print(f"  => Wer den Vorlauf der Geste ignoriert (mu=0), verliert "
          f"{100*(accs.max()-acc_at0):.1f} Prozentpunkte.")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(mus, accs, "o-", color="purple")
    ax.axvline(MU_TRUE, ls="--", color="green", label=f"wahres mu = {MU_TRUE}")
    ax.axvline(0, ls=":", color="red", label="naive Annahme mu=0")
    ax.set_xlabel("angenommener zeitlicher Versatz mu_hat [s]")
    ax.set_ylabel("Aufloesungs-Genauigkeit")
    ax.set_title("Der Zeitversatz muss modelliert werden")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mu_studie.png"), dpi=110); plt.close(fig)
    return dict(best=best, acc_best=accs.max(), acc_at0=acc_at0)


if __name__ == "__main__":
    print(f"Generiere {N_CMDS} Kommandos ...")
    cmds = gen_commands()
    twin_frac = np.mean([c["twin"] for c in cmds])
    print(f"  (davon {100*twin_frac:.0f} % mit raeumlichem Zwilling)\n")
    ablation_study(cmds)
    mutual_disambiguation(cmds)
    mu_study(cmds)
    print("\nPlots in ergebnisse/. Fertig.")
