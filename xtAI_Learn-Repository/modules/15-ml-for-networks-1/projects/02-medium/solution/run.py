"""Intrusion Detection unter realistischer Basisrate — die ganze unbequeme Wahrheit.

Vier Analysen:
  1. Detektor trainieren, ROC-AUC/PR-AUC auf der Testmenge (sieht super aus).
  2. EMPIRISCH: Basisrate durch Ausduennen der Angriffe senken -> ROC-AUC bleibt, PR-AUC bricht ein.
  3. ANALYTISCH (Bayes): PPV ueber der Basisrate + Alarme/Tag -> der Base-Rate-Fallacy.
  4. Betriebspunkt: kostenminimale Schwelle + welche FPR man BRAUCHTE.

Und ein Gegencheck: Theorie (Bayes) und Empirie (Subsampling) muessen uebereinstimmen.

Aufruf:  python run.py     (~20 s, Datensatz beim ersten Mal ~2 MB Download)
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_score)

from flow_data import lade_flows, baue_xy, subsample_zu_basisrate
from base_rate import (ppv_bei_basisrate, benoetigte_fpr, alarme_pro_tag,
                       erwartete_kosten, bester_betriebspunkt)

FLOWS_PRO_TAG = 10_000_000      # mittelgrosses Campus-/Firmennetz
RANDOM_STATE = 0


def main():
    # ---------- 1) Detektor ----------
    df = lade_flows()
    X, y = baue_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=RANDOM_STATE)
    modell = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    modell.fit(Xtr, ytr)
    scores = modell.predict_proba(Xte)[:, 1]

    roc_auc = roc_auc_score(yte, scores)
    pr_auc = average_precision_score(yte, scores)
    print("=== 1) Der Detektor (LogReg auf Volumen-/Timing-Features) ===")
    print(f"Testmenge: {len(yte):,} Flows, Basisrate pi = {yte.mean():.4f} "
          f"({100*yte.mean():.2f} % Angriffe)")
    print(f"ROC-AUC = {roc_auc:.4f}   PR-AUC = {pr_auc:.4f}    <- sieht hervorragend aus!")

    fpr_k, tpr_k, schwellen = roc_curve(yte, scores)
    i99 = int(np.argmax(tpr_k >= 0.99))
    TPR, FPR = tpr_k[i99], fpr_k[i99]
    print(f"\nBetriebspunkt 'ich will 99 % der Angriffe sehen': "
          f"TPR = {TPR:.4f}, FPR = {FPR:.5f} ({100*FPR:.2f} %)")

    # ---------- 2) empirisch: Basisrate senken ----------
    print("\n=== 2) EMPIRISCH: Was passiert bei realistischerer Basisrate? ===")
    print("(Angriffe ausduennen, Detektor UNVERAENDERT lassen)")
    print(f"{'pi':>8s} {'#Angriffe':>10s} {'ROC-AUC':>9s} {'PR-AUC':>8s}")
    rng = np.random.default_rng(RANDOM_STATE)
    for pi in (0.0336, 0.01, 0.001, 0.0001):
        res = subsample_zu_basisrate(yte.values, scores, pi, rng)
        if res is None:
            continue
        yy, ss = res
        print(f"{pi:8.4f} {int(yy.sum()):10d} {roc_auc_score(yy, ss):9.4f} "
              f"{average_precision_score(yy, ss):8.4f}")
    print("-> ROC-AUC bleibt ~konstant (oder steigt sogar!), PR-AUC BRICHT EIN.")
    print("   Die ROC-Kurve ist blind fuer die Basisrate - genau deshalb luegt sie hier.")
    print("   (Vorsicht: bei pi=1e-4 bleiben nur ~3 Angriffe -> PR-AUC ist dort sehr verrauscht.)")

    # ---------- 3) analytisch: Bayes ----------
    print("\n=== 3) ANALYTISCH (Bayes): der Base-Rate-Fallacy ===")
    print(f"Bei festem Betriebspunkt TPR={TPR:.4f}, FPR={FPR:.5f}:")
    print(f"{'Basisrate pi':>14s} {'PPV = P(Angriff|Alarm)':>24s} {'Fehlalarm-Anteil':>18s}")
    for pi in (0.0336, 0.01, 1e-3, 1e-4, 1e-5):
        p = ppv_bei_basisrate(TPR, FPR, pi)
        print(f"{pi:14g} {100*p:23.3f}% {100*(1-p):17.2f}%")

    pi_real = 1e-4
    echt, falsch = alarme_pro_tag(FLOWS_PRO_TAG, TPR, FPR, pi_real)
    print(f"\nReality-Check bei {FLOWS_PRO_TAG:,} Flows/Tag und pi = {pi_real:g}:")
    print(f"  echte Alarme : {echt:,.0f} pro Tag")
    print(f"  FEHLALARME   : {falsch:,.0f} pro Tag   <- kein Team der Welt bearbeitet das")

    # Gegencheck Theorie vs. Empirie
    print("\n--- Gegencheck: stimmt die Bayes-Formel mit der Empirie ueberein? ---")
    for pi in (0.0336, 0.01, 0.001):
        res = subsample_zu_basisrate(yte.values, scores, pi, rng)
        if res is None:
            continue
        yy, ss = res
        emp = precision_score(yy, (ss >= schwellen[i99]).astype(int), zero_division=0)
        theo = ppv_bei_basisrate(TPR, FPR, pi)
        print(f"  pi={pi:<7g} empirische Precision {100*emp:6.2f}%  vs. Bayes {100*theo:6.2f}%")
    print("  -> Theorie und Messung passen zusammen. Der Effekt ist keine Marotte der Daten,")
    print("     sondern reine Wahrscheinlichkeitsrechnung.")

    # ---------- 4) Was waere noetig? / Betriebspunkt ----------
    print("\n=== 4) Was muesste der Detektor koennen? ===")
    for ziel in (0.5, 0.9):
        n_fpr = benoetigte_fpr(TPR, pi_real, ziel)
        print(f"  Fuer PPV = {100*ziel:.0f} % bei pi=1e-4 braeuchte man FPR <= {n_fpr:.3e} "
              f"({FPR/n_fpr:,.0f}x besser als jetzt)")
    print("  -> Der Engpass ist die FPR, NICHT die Erkennungsrate.")

    K_FEHLALARM, K_VERPASST = 5.0, 50_000.0     # EUR: Analystenzeit vs. Schadensfall
    idx, schwelle, kosten = bester_betriebspunkt(
        fpr_k, tpr_k, schwellen, FLOWS_PRO_TAG, pi_real, K_FEHLALARM, K_VERPASST)
    e, f = alarme_pro_tag(FLOWS_PRO_TAG, tpr_k[idx], fpr_k[idx], pi_real)
    print(f"\nKostenoptimaler Betriebspunkt (Fehlalarm {K_FEHLALARM:.0f} EUR, "
          f"verpasster Angriff {K_VERPASST:,.0f} EUR):")
    print(f"  Schwelle {schwelle:.4f} -> TPR {tpr_k[idx]:.4f}, FPR {fpr_k[idx]:.5f}")
    print(f"  {e:,.0f} echte + {f:,.0f} falsche Alarme/Tag, erwartete Kosten {kosten:,.0f} EUR/Tag")
    print(f"  Vergleich 'TPR=99 %'-Punkt: "
          f"{erwartete_kosten(FLOWS_PRO_TAG, TPR, FPR, pi_real, K_FEHLALARM, K_VERPASST):,.0f} EUR/Tag")

    # ---------- Plots ----------
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

        pis = np.logspace(-6, -1, 200)
        for t, f_, lab in [(TPR, FPR, f"unser Detektor (FPR={FPR:.3f})"),
                           (0.99, 1e-3, "sehr gut (FPR=1e-3)"),
                           (0.99, 1e-5, "utopisch (FPR=1e-5)")]:
            ax1.semilogx(pis, 100*ppv_bei_basisrate(t, f_, pis), lw=1.8, label=lab)
        ax1.axvline(1e-4, ls="--", color="gray", lw=1)
        ax1.annotate("realistische\nBasisrate", xy=(1e-4, 60), fontsize=8, color="gray")
        ax1.set(xlabel="Basisrate $\\pi$ (Anteil Angriffe)",
                ylabel="PPV = P(Angriff | Alarm)  [%]",
                title="Base-Rate-Fallacy: gute Detektoren, nutzlose Alarme", ylim=(0, 100))
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        pis2 = np.logspace(-5, -1.5, 40)
        alarme = [alarme_pro_tag(FLOWS_PRO_TAG, TPR, FPR, p) for p in pis2]
        ax2.loglog(pis2, [a[0] for a in alarme], lw=1.8, label="echte Alarme/Tag")
        ax2.loglog(pis2, [a[1] for a in alarme], lw=1.8, label="Fehlalarme/Tag")
        ax2.axhline(100, ls=":", color="k", lw=1, label="was ein Team schafft (~100/Tag)")
        ax2.set(xlabel="Basisrate $\\pi$", ylabel="Alarme pro Tag",
                title=f"Absolute Zahlen bei {FLOWS_PRO_TAG:,} Flows/Tag")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")

        plt.tight_layout(); plt.savefig("results/base_rate.png", dpi=110)
        print("\nPlot gespeichert: results/base_rate.png")
    except Exception as e:
        print("(kein Plot:", e, ")")


if __name__ == "__main__":
    main()
