"""Auswertung der XR-Nutzerstudie: 3 DoF vs. 6 DoF.

Der rote Faden ist METHODIK, nicht "6 DoF ist besser" (das wissen wir vorher). Es geht darum,
eine Studie SO auszuwerten, dass die Schluesse tragen:
  1. deskriptiv + Design pruefen (Counterbalancing eingehalten?)
  2. der richtige Test (ordinal -> Wilcoxon) und die EFFEKTSTAERKE (nicht nur p)
  3. Mehrfachvergleiche korrigieren (Bonferroni/Holm)
  4. Reihenfolge-Effekte belegen - und zeigen, was ohne Counterbalancing passiert waere

Aufruf:  python run.py       (~1 s)
"""
from __future__ import annotations
import numpy as np

from generate_study import (generate_study, generate_naiv_ohne_counterbalancing, WAHRHEIT)
from stats_tools import (paired_comparison, bonferroni, holm_bonferroni, order_effect)

# (Name, Spalte, welche Richtung ist "besser fuer 6 DoF"?)
OUTCOMES = [
    ("Presence (IPQ)",   "presence", "hoeher"),
    ("Sickness (SSQ)",   "sickness", "niedriger"),
    ("Aufgabenzeit [s]", "time",     "niedriger"),
    ("Comfort",          "comfort",  "hoeher"),
]


def paare(df, spalte):
    a = df[df.condition == "3DoF"].sort_values("participant")[spalte].values
    b = df[df.condition == "6DoF"].sort_values("participant")[spalte].values
    return a, b


def main():
    df = generate_study(n_participants=24, seed=3)
    n = df.participant.nunique()

    print("=== Studie: 3 DoF vs. 6 DoF, within-subject ===")
    print(f"N = {n} Probanden, jeder testet BEIDE Bedingungen.")
    cb = df[df.position == 1].condition.value_counts().to_dict()
    print(f"Counterbalancing: {cb} beginnen je mit dieser Bedingung "
          f"({'ausgeglichen' if len(set(cb.values())) == 1 else 'UNAUSGEGLICHEN!'}).")

    # ---------- 1)+2) Vergleiche mit Effektstaerke ----------
    print("\n=== Vergleich je Zielgroesse (ordinal -> Wilcoxon; Effektstaerke IMMER dazu) ===")
    print(f"{'Zielgroesse':18s} {'3DoF':>7s} {'6DoF':>7s} {'t-p':>8s} {'Wilcox-p':>9s} "
          f"{'dz':>6s} {'r':>6s}")
    pvals = []
    for name, spalte, _ in OUTCOMES:
        a, b = paare(df, spalte)
        r = paired_comparison(a, b)
        pvals.append(r["wilcoxon_p"])
        print(f"{name:18s} {r['mean_x']:7.1f} {r['mean_y']:7.1f} {r['t_p']:8.4f} "
              f"{r['wilcoxon_p']:9.4f} {r['cohen_dz']:6.2f} {r['rank_biserial']:6.2f}")
    print("(dz = Cohen's dz parametrisch; r = rank-biserial zum Wilcoxon. |0.2| klein, 0.5 "
          "mittel, 0.8 gross.)")

    # ---------- 3) Mehrfachvergleiche ----------
    print("\n=== Mehrfachvergleiche korrigieren (4 Tests!) ===")
    schwelle, bonf = bonferroni(pvals, alpha=0.05)
    holm = holm_bonferroni(pvals, alpha=0.05)
    print(f"{'Zielgroesse':18s} {'Wilcox-p':>9s} {'roh<0.05':>9s} "
          f"{'Bonf<'+format(schwelle,'.4f'):>13s} {'Holm':>6s}")
    for i, (name, _, _) in enumerate(OUTCOMES):
        roh = "sig" if pvals[i] < 0.05 else "n.s."
        b = "sig" if bonf[i] else "n.s."
        h = "sig" if holm[i] else "n.s."
        flag = "   <-- kippt!" if roh != b else ""
        print(f"{name:18s} {pvals[i]:9.4f} {roh:>9s} {b:>13s} {h:>6s}{flag}")
    print("-> Comfort ist roh signifikant (p<0.05), kippt aber unter Bonferroni - und ist die")
    print("   EINZIGE mit kleiner Effektstaerke (dz 0.50 vs. >1.1 bei den anderen). Beides sagt:")
    print("   das ist der WACKLIGSTE Befund. (Holm ist weniger konservativ und behaelt ihn knapp -")
    print("   genau deshalb berichtet man IMMER die Effektstaerke dazu, nicht nur die Signifikanz.)")
    print("   Grundgedanke wie beim Base-Rate-Fallacy (Modul 15): viele Tests x kleine Fehlerrate")
    print("   = viele Fehlalarme.")

    # ---------- 4) Reihenfolge-Effekte + Gegenprobe ----------
    print("\n=== Warum Counterbalancing noetig war (Reihenfolge-Effekte) ===")
    for name, spalte in [("Sickness", "sickness"), ("Aufgabenzeit", "time")]:
        oe = order_effect(df, spalte)
        print(f"  {name:12s}: 1. Sitzung {oe['mean_erste']:6.1f}  vs  2. Sitzung "
              f"{oe['mean_zweite']:6.1f}  (p={oe['p']:.3f})")
    print("  -> Es GIBT Reihenfolge-Effekte (Carryover-Uebelkeit, Lerneffekt bei der Zeit).")
    print("     Weil counterbalanced wurde, mitteln sie sich ueber die Bedingungen heraus.")

    print("\n--- Gegenprobe: was ohne Counterbalancing passiert waere ---")
    df_naiv = generate_naiv_ohne_counterbalancing(n_participants=24, seed=3)
    a, b = paare(df_naiv, "sickness")
    gemessen = a.mean() - b.mean()
    wahr = -WAHRHEIT["sickness_effekt"]      # 3 DoF ist um 12 kraenker
    print(f"  Wenn ALLE erst 3 DoF, dann 6 DoF machen (kein Counterbalancing):")
    print(f"  gemessener Sickness-Unterschied {gemessen:.1f} statt wahrer {wahr:.0f} Punkte.")
    print(f"  -> Der Carryover (+{WAHRHEIT['carryover_sickness']:.0f} auf die 2., also 6-DoF-Sitzung)")
    print(f"     MASKIERT die Haelfte des echten Effekts. Ohne Counterbalancing haette man")
    print(f"     6 DoF faelschlich als kaum besser eingestuft.")

    # ---------- Plots ----------
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("ergebnisse", exist_ok=True)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        for ax, (name, spalte, richtung) in zip(axes, OUTCOMES):
            a, b = paare(df, spalte)
            # gepaarte Linien: jede Person ist eine Linie zwischen den Bedingungen
            for pa, pb in zip(a, b):
                ax.plot([0, 1], [pa, pb], color="gray", alpha=0.35, lw=0.8)
            ax.plot([0, 1], [a.mean(), b.mean()], "o-", color="crimson", lw=2.5, ms=8)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["3 DoF", "6 DoF"])
            ax.set_title(f"{name}\n({richtung}=besser)", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
        plt.suptitle("Gepaarte Daten: jede graue Linie ist ein Proband", fontsize=10)
        plt.tight_layout(); plt.savefig("ergebnisse/study_results.png", dpi=110)
        print("\nPlot gespeichert: ergebnisse/study_results.png")
    except Exception as e:
        print("(kein Plot:", e, ")")


if __name__ == "__main__":
    main()
