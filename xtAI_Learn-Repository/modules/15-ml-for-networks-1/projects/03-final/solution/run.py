"""Zero-Day-Erkennung: Misuse vs. Anomalie — das Abschlussexperiment.

Szenario: Im Training ist NUR der Angriff 'smurf' bekannt. Alle anderen Angriffstypen
(neptune, satan, teardrop, ...) treten erst im Betrieb auf = Zero-Days.

Frage: Wer findet sie - der ueberwachte Detektor (kennt Angriffe, aber nur die bekannten)
oder der Anomaliedetektor (kennt GAR KEINE Angriffe, nur Normalverkehr)?

Am Ende: der Realitaets-Check mit der Basisrate aus Projekt 02.

Aufruf:  python run.py     (~1-2 min)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from flow_data import ZeroDaySzenario
from detektoren import (SupervisedDetektor, IsolationForestDetektor,
                        MahalanobisDetektor, OneClassSVMDetektor, LOFDetektor)

FLOWS_PRO_TAG = 10_000_000
PI_REAL = 1e-4


def ppv(tpr, fpr, pi):
    return tpr * pi / (tpr * pi + fpr * (1 - pi)) if (tpr * pi + fpr * (1 - pi)) > 0 else 0.0


def main():
    szen = ZeroDaySzenario(bekannte_angriffe=("smurf",), random_state=0)
    print("=== Szenario ===")
    print(szen.uebersicht())
    print("\nNur 'smurf' ist im Training bekannt. Alles andere ist ein Zero-Day.\n")

    # ---------- Detektoren trainieren ----------
    Xtr_sup, ytr_sup = szen.supervised_trainingsdaten()
    detektoren = []

    sup = SupervisedDetektor().fit(Xtr_sup, ytr_sup)
    detektoren.append(sup)
    for D in (IsolationForestDetektor(), MahalanobisDetektor(),
              OneClassSVMDetektor(), LOFDetektor()):
        detektoren.append(D.fit(szen.normal_train))     # NUR Normalverkehr!

    # ---------- Auswertung je Angriffstyp ----------
    zeilen = []
    for det in detektoren:
        zeile = {"Detektor": det.name}
        zeile["FPR (normal)"] = det.predict(szen.normal_test).mean()
        zeile["smurf (bekannt)"] = det.predict(szen.bekannt_test).mean()
        for a, Xa in szen.zeroday.items():
            if len(Xa) >= 10:                            # zu kleine Typen sind reines Rauschen
                zeile[f"{a} (0-day)"] = det.predict(Xa).mean()
        zeilen.append(zeile)

    tab = pd.DataFrame(zeilen).set_index("Detektor")
    print("=== Recall je Angriffstyp (Anteil erkannter Flows) ===")
    print("(Spalte 'FPR (normal)' ist der PREIS: Fehlalarme auf harmlosem Verkehr)\n")
    print(tab.round(3).to_string())

    # ---------- die Kernaussage ----------
    zd_spalten = [c for c in tab.columns if "0-day" in c]
    print("\n=== Kernaussage: mittlerer Zero-Day-Recall ===")
    for name, row in tab.iterrows():
        print(f"  {name:44s} Zero-Day-Recall {row[zd_spalten].mean():.3f} "
              f"| FPR {row['FPR (normal)']:.4f}")

    print("\n--> Der supervised Detektor erkennt 'smurf' perfekt und ist bei 'neptune'")
    print("    (einem 928-Flow-SYN-Flood!) KOMPLETT BLIND - er hat ihn nie gesehen.")
    print("    Die Anomaliedetektoren haben NIE einen Angriff gesehen und finden ihn trotzdem.")
    print("    Preis: sie schlagen auch auf harmlosem Verkehr Alarm (FPR > 0).")

    # ---------- Realitaets-Check: Basisrate (Projekt 02) ----------
    print("\n=== Realitaets-Check: was heisst das im Betrieb? (Projekt 02) ===")
    print(f"Bei {FLOWS_PRO_TAG:,} Flows/Tag und Basisrate pi={PI_REAL:g}:")
    print(f"{'Detektor':44s} {'PPV':>8s} {'Fehlalarme/Tag':>16s}")
    for name, row in tab.iterrows():
        tpr = row[zd_spalten].mean()
        fpr = row["FPR (normal)"]
        p = ppv(tpr, fpr, PI_REAL)
        fa = fpr * (1 - PI_REAL) * FLOWS_PRO_TAG
        print(f"{name:44s} {100*p:7.3f}% {fa:16,.0f}")
    print("\n--> Genau hier stirbt die Anomalieerkennung in der Praxis: 1 % FPR klingt winzig,")
    print("    bedeutet aber ~100.000 Fehlalarme taeglich. Der supervised Detektor hat FPR~0")
    print("    - dafuer sieht er nichts Neues. DAS ist der eigentliche Zielkonflikt.")

    # ---------- Plot ----------
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        plot_sp = ["smurf (bekannt)"] + zd_spalten
        ax = tab[plot_sp].T.plot.bar(figsize=(12, 4.8), width=0.8)
        ax.set(ylabel="Recall", title="Zero-Day-Erkennung: Misuse vs. Anomalie "
                                      "(nur 'smurf' war im Training bekannt)", ylim=(0, 1.05))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3, axis="y")
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.annotate("bekannt |  Zero-Day", xy=(0.52, 1.0), fontsize=8)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout(); plt.savefig("results/zeroday.png", dpi=110)
        print("\nPlot gespeichert: results/zeroday.png")
    except Exception as e:
        print("(kein Plot:", e, ")")


if __name__ == "__main__":
    main()
