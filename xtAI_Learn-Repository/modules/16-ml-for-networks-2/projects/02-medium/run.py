"""GCN vs. Heuristiken vs. MLP — unter ZWEI Evaluationsprotokollen.

Die eigentliche Frage dieses Projekts ist nicht "welches Modell ist besser?", sondern:
**haengt die Antwort davon ab, wie ich messe?** (Spoiler: ja, und zwar dramatisch.)

Aufruf:  python run.py      (~30 s auf der CPU)
"""
from __future__ import annotations
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from graph_data import LinkSplit
from gcn import GCN, MLPOhneStruktur, normalisierte_adjazenz, kanten_score

EPOCHEN = 100
SEED = 0


# ---------------------------- Heuristiken (aus Projekt 01) ----------------------------
def _nb(H, u):
    return set(H[u]) if u in H else set()


def heuristik_scores(split, paare, welche):
    G, kn = split.G_train, split.knoten          # G_train: OHNE Testkanten!
    out = []
    for u, v in paare:
        a, b = _nb(G, kn[u]), _nb(G, kn[v])
        if welche == "Common Neighbors":
            out.append(len(a & b))
        elif welche == "Adamic-Adar":
            out.append(sum(1.0 / np.log(G.degree(w)) for w in a & b if G.degree(w) > 1))
        elif welche == "Pref. Attachment":
            out.append(len(a) * len(b))
    return np.array(out, dtype=float)


# ---------------------------- Training ----------------------------
def trainiere(modell, A_hat, split, epochen=EPOCHEN, lr=0.01, verbose=True):
    opt = torch.optim.Adam(modell.parameters(), lr=lr)
    tp = torch.tensor(split.train_pos)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for ep in range(1, epochen + 1):
        modell.train()
        Z = modell(A_hat)
        # frische Negativbeispiele je Epoche (Standard beim Link-Prediction-Training)
        neg = torch.tensor(split.negative_uniform(len(split.train_pos),
                                                  seed=int(rng.integers(1 << 30))))
        scores = torch.cat([kanten_score(Z, tp), kanten_score(Z, neg)])
        ziele = torch.cat([torch.ones(len(tp)), torch.zeros(len(neg))])
        loss = nn.functional.binary_cross_entropy_with_logits(scores, ziele)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ep % 25 == 0:
            print(f"    Epoche {ep:3d}  Loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")
    return modell


def modell_auc(modell, A_hat, pos, neg):
    modell.eval()
    with torch.no_grad():
        Z = modell(A_hat)
        s = torch.cat([kanten_score(Z, torch.tensor(pos)),
                       kanten_score(Z, torch.tensor(neg))]).numpy()
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, s)


def main():
    split = LinkSplit(test_anteil=0.10, seed=SEED)
    print("=== Daten ===")
    print(split.uebersicht())

    A_hat = normalisierte_adjazenz(split.train_pos, split.n)
    print(f"A_hat: sparse {tuple(A_hat.shape)}, {A_hat._nnz():,} Nicht-Null-Eintraege "
          f"({100*A_hat._nnz()/split.n**2:.4f} % besetzt)")
    print(f"(dicht waeren das {split.n**2*4/1e6:.0f} MB)")

    # ---- zwei Testmengen: derselbe Test-Positivsatz, zwei Arten von Negativen ----
    neg_uniform = split.negative_uniform(len(split.test_pos), seed=99)
    neg_gematcht = split.negative_grad_gematcht(split.test_pos, seed=99)
    print(f"\nNegativbeispiele: uniform {len(neg_uniform):,} | "
          f"grad-gematcht {len(neg_gematcht):,}")
    print("Median Gradprodukt d_u*d_v:")
    print(f"  echte Kanten : {np.median(split.gradprodukt(split.test_pos)):>7.0f}")
    print(f"  uniform      : {np.median(split.gradprodukt(neg_uniform)):>7.0f}  <- Blatt x Blatt!")
    print(f"  grad-gematcht: {np.median(split.gradprodukt(neg_gematcht)):>7.0f}  <- vergleichbar")

    ergebnisse = {}

    # ---- Heuristiken ----
    for h in ("Common Neighbors", "Adamic-Adar", "Pref. Attachment"):
        r = {}
        for tag, neg in (("uniform", neg_uniform), ("grad-gematcht", neg_gematcht)):
            s = np.r_[heuristik_scores(split, split.test_pos, h),
                      heuristik_scores(split, neg, h)]
            y = np.r_[np.ones(len(split.test_pos)), np.zeros(len(neg))]
            r[tag] = roc_auc_score(y, s)
        ergebnisse[h] = r

    # ---- MLP ohne Struktur (Kontrollgruppe) ----
    print("\n=== MLP ohne Struktur (Kontrollgruppe, kein Message Passing) ===")
    mlp = trainiere(MLPOhneStruktur(split.n, seed=SEED), None, split)
    ergebnisse["MLP (ohne Graph)"] = {
        "uniform": modell_auc(mlp, None, split.test_pos, neg_uniform),
        "grad-gematcht": modell_auc(mlp, None, split.test_pos, neg_gematcht),
    }

    # ---- GCN ----
    print("\n=== GCN (2 Schichten, from scratch) ===")
    gcn = trainiere(GCN(split.n, seed=SEED), A_hat, split)
    ergebnisse["GCN (2 Schichten)"] = {
        "uniform": modell_auc(gcn, A_hat, split.test_pos, neg_uniform),
        "grad-gematcht": modell_auc(gcn, A_hat, split.test_pos, neg_gematcht),
    }

    # ---- Die Tabelle, um die es geht ----
    print("\n" + "=" * 64)
    print("ROC-AUC — dasselbe Modell, zwei Evaluationsprotokolle")
    print("=" * 64)
    print(f"{'Verfahren':24s} {'uniform (naiv)':>16s} {'grad-gematcht':>16s}")
    for name, r in ergebnisse.items():
        print(f"{name:24s} {r['uniform']:16.4f} {r['grad-gematcht']:16.4f}")

    bester_naiv = max(ergebnisse, key=lambda k: ergebnisse[k]["uniform"])
    bester_ehrlich = max(ergebnisse, key=lambda k: ergebnisse[k]["grad-gematcht"])
    print(f"\nSieger nach naiver Evaluation : {bester_naiv}")
    print(f"Sieger nach ehrlicher Evaluation: {bester_ehrlich}")
    if bester_naiv != bester_ehrlich:
        print("\n>>> Das Evaluationsprotokoll DREHT die Rangfolge um. <<<")
        print("    Wer uniform evaluiert, verwirft das GNN zugunsten der duemmsten Heuristik.")

    # ---- Plot ----
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        namen = list(ergebnisse)
        x = np.arange(len(namen)); b = 0.38
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(x - b/2, [ergebnisse[n]["uniform"] for n in namen], b, label="uniform (naiv)")
        ax.bar(x + b/2, [ergebnisse[n]["grad-gematcht"] for n in namen], b,
               label="grad-gematcht (ehrlich)")
        ax.axhline(0.5, ls="--", color="k", lw=1, label="Zufall")
        ax.set_xticks(x); ax.set_xticklabels(namen, rotation=20, ha="right")
        ax.set(ylabel="ROC-AUC", title="Link Prediction: das Messprotokoll entscheidet",
               ylim=(0.3, 0.9))
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        plt.tight_layout(); plt.savefig("results/link_prediction.png", dpi=110)
        print("\nPlot gespeichert: results/link_prediction.png")
    except Exception as e:
        print("(kein Plot:", e, ")")


if __name__ == "__main__":
    main()
