"""Fuehrt die beiden Experimente aus und erzeugt die Vergleichstabellen + Plots.

Ausfuehren (aus dem Projekt-Root oder loesung/):
    /Users/.../.venv/bin/python run.py
Plots landen in ergebnisse/ (gitignored).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multimodal import (make_complementary, make_redundant, fit_modality,
                        late_fusion_proba, predict_from_proba, accuracy,
                        early_fusion_fit_predict)

OUT = os.path.join(os.path.dirname(__file__), "ergebnisse")
os.makedirs(OUT, exist_ok=True)


def split(X, frac=0.7, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    k = int(frac * X.shape[0])
    return idx[:k], idx[k:]


def experiment_complementarity():
    print("=" * 70)
    print("EXPERIMENT 1 — Komplementaritaet: jede Modalitaet allein mehrdeutig")
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

    # Late fusion (Bayes-Produkt)
    pLate = late_fusion_proba([pA, pB], prior)
    accLate = accuracy(y[te], predict_from_proba(pLate))

    # Early fusion (Feature-Konkatenation)
    yEarly = early_fusion_fit_predict(XA[tr], XB[tr], y[tr], XA[te], XB[te])
    accEarly = accuracy(y[te], yEarly)

    print(f"  nur Modalitaet A : {accA:.3f}   (theoret. Obergrenze ~0.50)")
    print(f"  nur Modalitaet B : {accB:.3f}")
    print(f"  LATE  Fusion     : {accLate:.3f}")
    print(f"  EARLY Fusion     : {accEarly:.3f}")
    print("  => Jede allein raet zwischen zwei Klassen; Fusion loest die Mehrdeutigkeit.")

    # Missing-Modality-Test: Modalitaet B faellt zur Testzeit aus
    print("\n  Missing-Modality-Test (Modalitaet B faellt zur Testzeit aus):")
    # late: benutze einfach nur A's Posterior -> kein Umbau noetig
    accLate_missB = accuracy(y[te], predict_from_proba(pA))
    # early: B muss imputiert werden (Trainingsmittel) -> Modell erwartet 2 Features
    XB_imp_tr = np.full_like(XB[tr], XB[tr].mean())
    XB_imp_te = np.full_like(XB[te], XB[tr].mean())
    yEarly_missB = early_fusion_fit_predict(XA[tr], XB[tr], y[tr], XA[te], XB_imp_te)
    accEarly_missB = accuracy(y[te], yEarly_missB)
    print(f"    LATE  Fusion mit fehlendem B : {accLate_missB:.3f}  (nutzt einfach A's Posterior)")
    print(f"    EARLY Fusion mit fehlendem B : {accEarly_missB:.3f}  (B mit Mittelwert imputiert)")
    print("    => Late Fusion degradiert sauber auf die verbleibende Modalitaet;")
    print("       Early Fusion braucht einen Imputations-Hack und ist an feste Featurezahl gebunden.")

    # Plot: Feature-Raum + Konfusion je Modalitaet
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    for cls in range(4):
        m = y == cls
        axes[0].scatter(XA[m], XB[m], s=6, alpha=0.4, color=colors[cls], label=f"Klasse {cls}")
    axes[0].set_xlabel("Modalitaet A (informativ ueber Bit1)")
    axes[0].set_ylabel("Modalitaet B (informativ ueber Bit0)")
    axes[0].set_title("Nur gemeinsam sind alle 4 Klassen trennbar")
    axes[0].legend(markerscale=2, fontsize=8)
    bars = ["nur A", "nur B", "late", "early"]
    vals = [accA, accB, accLate, accEarly]
    axes[1].bar(bars, vals, color=["gray", "gray", "green", "steelblue"])
    axes[1].axhline(0.25, ls=":", color="k", lw=1, label="Zufall (0.25)")
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("Test-Genauigkeit")
    axes[1].set_title("Fusion loest die Mehrdeutigkeit"); axes[1].legend()
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "komplementaritaet.png"), dpi=110)
    plt.close(fig)
    return dict(accA=accA, accB=accB, accLate=accLate, accEarly=accEarly,
                accLate_missB=accLate_missB, accEarly_missB=accEarly_missB)


def experiment_correlation_trap():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — Redundanz & Korrelations-Falle")
    print("=" * 70)
    print("Beide Modalitaeten schaetzen DASSELBE (redundant). Fusion = Mittelung.")
    print("Wir variieren die Rausch-Korrelation rho zwischen den Modalitaeten.\n")
    print("Zwei Groessen: (1) Fehler der Mittelungs-Fusion; (2) P(beide einzeln falsch) --")
    print("letztere ist die eigentliche Mutual-Disambiguation-Groesse und sollte bei rho=0")
    print("dem Produkt eps_A*eps_B gleichen.\n")
    rhos = np.linspace(0.0, 0.98, 12)
    err_single, err_fused, both_wrong, prod_pred = [], [], [], []
    for rho in rhos:
        XA, XB, y = make_redundant(n=20000, sep=1.0, sigma=1.0, rho=rho, seed=0)
        # einfacher Klassifikator: Vorzeichen der Messung (Schwelle 0)
        predA = (XA[:, 0] > 0).astype(int)
        predB = (XB[:, 0] > 0).astype(int)
        wrongA = predA != y
        wrongB = predB != y
        eA, eB = wrongA.mean(), wrongB.mean()
        # Fusion redundanter Messungen = Mittelwert (inverse-Varianz, hier gleiche Reliabilitaet)
        fused = ((XA[:, 0] + XB[:, 0]) / 2.0 > 0).astype(int)
        eF = np.mean(fused != y)
        err_single.append(0.5 * (eA + eB))
        err_fused.append(eF)
        both_wrong.append(np.mean(wrongA & wrongB))
        prod_pred.append(eA * eB)  # nur bei Unabhaengigkeit korrekt

    err_single = np.array(err_single); err_fused = np.array(err_fused)
    both_wrong = np.array(both_wrong); prod_pred = np.array(prod_pred)

    print(f"  {'rho':>5} | {'Fehler einzeln':>14} | {'Fehler Fusion':>13} | {'P(beide falsch)':>15} | {'eps_A*eps_B':>11}")
    print("  " + "-" * 70)
    for i in range(len(rhos)):
        print(f"  {rhos[i]:5.2f} | {err_single[i]:14.4f} | {err_fused[i]:13.4f} | {both_wrong[i]:15.4f} | {prod_pred[i]:11.4f}")

    gain0 = err_single[0] - err_fused[0]
    gainR = err_single[-1] - err_fused[-1]
    print(f"\n  (1) Mittelungs-Fusion: Gewinn bei rho=0.00 : {gain0:.4f}  (Fehler ~ halbiert)")
    print(f"                          Gewinn bei rho=0.98 : {gainR:.4f}  (Fusion bringt fast NICHTS)")
    print(f"  (2) Mutual Disambiguation: bei rho=0.00 ist P(beide falsch)={both_wrong[0]:.4f}")
    print(f"      == eps_A*eps_B={prod_pred[0]:.4f} (Unabhaengigkeit). Bei rho=0.98 steigt")
    print(f"      P(beide falsch) auf {both_wrong[-1]:.4f} ~ eps einzeln: die Fehler fallen zusammen.")
    print("  => Beide Effekte belegen: Fusion lebt von der UNABHAENGIGKEIT der Fehler.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(rhos, err_single, "o-", label="Fehler einzeln (Mittel A,B)", color="gray")
    ax1.plot(rhos, err_fused, "s-", label="Fehler Mittelungs-Fusion", color="green")
    ax1.set_xlabel("Rausch-Korrelation rho"); ax1.set_ylabel("Fehlerrate")
    ax1.set_title("(1) Fusionsgewinn schwindet mit Korrelation")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(rhos, both_wrong, "s-", label="gemessen: P(beide falsch)", color="crimson")
    ax2.plot(rhos, prod_pred, "^--", label="Vorhersage eps_A*eps_B (Unabh.)", color="orange")
    ax2.set_xlabel("Rausch-Korrelation rho"); ax2.set_ylabel("Wahrscheinlichkeit")
    ax2.set_title("(2) Mutual Disambiguation nur bei rho~0")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "korrelations_falle.png"), dpi=110)
    plt.close(fig)
    return dict(rhos=rhos, err_single=err_single, err_fused=err_fused,
                both_wrong=both_wrong, prod_pred=prod_pred)


if __name__ == "__main__":
    r1 = experiment_complementarity()
    r2 = experiment_correlation_trap()
    print("\nPlots gespeichert in ergebnisse/. Fertig.")
