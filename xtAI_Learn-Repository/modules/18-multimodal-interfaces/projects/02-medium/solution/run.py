"""Runs the two experiments and produces the comparison tables + plots.

Run (from the project root or solution/):
    /Users/.../.venv/bin/python run.py
The plots land in results/ (gitignored).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multimodal import (make_complementary, make_redundant, fit_modality,
                        late_fusion_proba, predict_from_proba, accuracy,
                        early_fusion_fit_predict)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def split(X, frac=0.7, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    k = int(frac * X.shape[0])
    return idx[:k], idx[k:]


def experiment_complementarity():
    print("=" * 70)
    print("EXPERIMENT 1 — complementarity: each modality is ambiguous on its own")
    print("=" * 70)
    XA, XB, y = make_complementary(n_per_class=800, sigma=1.0, seed=0)
    tr, te = split(XA, 0.7, seed=1)
    prior = np.full(4, 0.25)

    mA = fit_modality(XA[tr], y[tr])
    mB = fit_modality(XB[tr], y[tr])
    pA = mA.predict_proba(XA[te])
    pB = mB.predict_proba(XB[te])

    accA = accuracy(y[te], predict_from_proba(pA))
    accB = accuracy(y[te], predict_from_proba(pB))

    # late fusion (the Bayes product)
    pLate = late_fusion_proba([pA, pB], prior)
    accLate = accuracy(y[te], predict_from_proba(pLate))

    # early fusion (feature concatenation)
    yEarly = early_fusion_fit_predict(XA[tr], XB[tr], y[tr], XA[te], XB[te])
    accEarly = accuracy(y[te], yEarly)

    print(f"  modality A only : {accA:.3f}   (theoretical ceiling ~0.50)")
    print(f"  modality B only : {accB:.3f}")
    print(f"  LATE  fusion    : {accLate:.3f}")
    print(f"  EARLY fusion    : {accEarly:.3f}")
    print("  => Each one alone guesses between two classes; fusion resolves the ambiguity.")

    # the missing modality test: modality B drops out at test time
    print("\n  Missing modality test (modality B drops out at test time):")
    # late: simply use A's posterior only -> no rebuilding needed
    accLate_missB = accuracy(y[te], predict_from_proba(pA))
    # early: B has to be imputed (the training mean) -> the model expects 2 features
    XB_imp_tr = np.full_like(XB[tr], XB[tr].mean())
    XB_imp_te = np.full_like(XB[te], XB[tr].mean())
    yEarly_missB = early_fusion_fit_predict(XA[tr], XB[tr], y[tr], XA[te], XB_imp_te)
    accEarly_missB = accuracy(y[te], yEarly_missB)
    print(f"    LATE  fusion with B missing : {accLate_missB:.3f}  (simply uses A's posterior)")
    print(f"    EARLY fusion with B missing : {accEarly_missB:.3f}  (B imputed with the mean)")
    print("    => Late fusion degrades gracefully to the remaining modality;")
    print("       early fusion needs an imputation hack and is tied to a fixed feature count.")

    # plot: the feature space + the confusion per modality
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    for cls in range(4):
        m = y == cls
        axes[0].scatter(XA[m], XB[m], s=6, alpha=0.4, color=colors[cls], label=f"class {cls}")
    axes[0].set_xlabel("modality A (informative about bit1)")
    axes[0].set_ylabel("modality B (informative about bit0)")
    axes[0].set_title("Only together are all 4 classes separable")
    axes[0].legend(markerscale=2, fontsize=8)
    bars = ["A only", "B only", "late", "early"]
    vals = [accA, accB, accLate, accEarly]
    axes[1].bar(bars, vals, color=["gray", "gray", "green", "steelblue"])
    axes[1].axhline(0.25, ls=":", color="k", lw=1, label="chance (0.25)")
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("test accuracy")
    axes[1].set_title("Fusion resolves the ambiguity"); axes[1].legend()
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "complementarity.png"), dpi=110)
    plt.close(fig)
    return dict(accA=accA, accB=accB, accLate=accLate, accEarly=accEarly,
                accLate_missB=accLate_missB, accEarly_missB=accEarly_missB)


def experiment_correlation_trap():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — redundancy & the correlation trap")
    print("=" * 70)
    print("Both modalities estimate THE SAME thing (redundant). Fusion = averaging.")
    print("We vary the noise correlation rho between the modalities.\n")
    print("Two quantities: (1) the error of the averaging fusion; (2) P(both wrong individually) --")
    print("the latter is the actual mutual disambiguation quantity and should equal the")
    print("product eps_A*eps_B at rho=0.\n")
    rhos = np.linspace(0.0, 0.98, 12)
    err_single, err_fused, both_wrong, prod_pred = [], [], [], []
    for rho in rhos:
        XA, XB, y = make_redundant(n=20000, sep=1.0, sigma=1.0, rho=rho, seed=0)
        # a simple classifier: the sign of the measurement (threshold 0)
        predA = (XA[:, 0] > 0).astype(int)
        predB = (XB[:, 0] > 0).astype(int)
        wrongA = predA != y
        wrongB = predB != y
        eA, eB = wrongA.mean(), wrongB.mean()
        # the fusion of redundant measurements = the mean (inverse variance, equal reliability here)
        fused = ((XA[:, 0] + XB[:, 0]) / 2.0 > 0).astype(int)
        eF = np.mean(fused != y)
        err_single.append(0.5 * (eA + eB))
        err_fused.append(eF)
        both_wrong.append(np.mean(wrongA & wrongB))
        prod_pred.append(eA * eB)  # correct only under independence

    err_single = np.array(err_single); err_fused = np.array(err_fused)
    both_wrong = np.array(both_wrong); prod_pred = np.array(prod_pred)

    print(f"  {'rho':>5} | {'error single':>14} | {'error fusion':>13} | {'P(both wrong)':>15} | {'eps_A*eps_B':>11}")
    print("  " + "-" * 70)
    for i in range(len(rhos)):
        print(f"  {rhos[i]:5.2f} | {err_single[i]:14.4f} | {err_fused[i]:13.4f} | "
              f"{both_wrong[i]:15.4f} | {prod_pred[i]:11.4f}")

    gain0 = err_single[0] - err_fused[0]
    gainR = err_single[-1] - err_fused[-1]
    print(f"\n  (1) Averaging fusion: the gain at rho=0.00 : {gain0:.4f}  (the error is ~halved)")
    print(f"                        the gain at rho=0.98 : {gainR:.4f}  (fusion brings almost NOTHING)")
    print(f"  (2) Mutual disambiguation: at rho=0.00, P(both wrong)={both_wrong[0]:.4f}")
    print(f"      == eps_A*eps_B={prod_pred[0]:.4f} (independence). At rho=0.98,")
    print(f"      P(both wrong) rises to {both_wrong[-1]:.4f} ~ the individual eps: the errors coincide.")
    print("  => Both effects show: fusion lives off the INDEPENDENCE of the errors.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(rhos, err_single, "o-", label="error single (the mean of A,B)", color="gray")
    ax1.plot(rhos, err_fused, "s-", label="error of the averaging fusion", color="green")
    ax1.set_xlabel("noise correlation rho"); ax1.set_ylabel("error rate")
    ax1.set_title("(1) The fusion gain vanishes with correlation")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(rhos, both_wrong, "s-", label="measured: P(both wrong)", color="crimson")
    ax2.plot(rhos, prod_pred, "^--", label="prediction eps_A*eps_B (indep.)", color="orange")
    ax2.set_xlabel("noise correlation rho"); ax2.set_ylabel("probability")
    ax2.set_title("(2) Mutual disambiguation only at rho~0")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "correlation_trap.png"), dpi=110)
    plt.close(fig)
    return dict(rhos=rhos, err_single=err_single, err_fused=err_fused,
                both_wrong=both_wrong, prod_pred=prod_pred)


if __name__ == "__main__":
    r1 = experiment_complementarity()
    r2 = experiment_correlation_trap()
    print("\nPlots saved in results/. Done.")
