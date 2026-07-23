"""GCN vs. heuristics vs. MLP — under TWO evaluation protocols.

The actual question of this project is not "which model is better?", but:
**does the answer depend on how I measure?** (Spoiler: yes, and dramatically so.)

Call:  python run.py      (~30 s on the CPU)
"""
from __future__ import annotations
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from graph_data import LinkSplit
from gcn import GCN, MLPWithoutStructure, normalized_adjacency, edge_score

EPOCHS = 100
SEED = 0


# ---------------------------- heuristics (from project 01) ----------------------------
def _nb(H, u):
    return set(H[u]) if u in H else set()


def heuristic_scores(split, pairs, which):
    G, nd = split.G_train, split.nodes          # G_train: WITHOUT the test edges!
    out = []
    for u, v in pairs:
        a, b = _nb(G, nd[u]), _nb(G, nd[v])
        if which == "Common neighbors":
            out.append(len(a & b))
        elif which == "Adamic-Adar":
            out.append(sum(1.0 / np.log(G.degree(w)) for w in a & b if G.degree(w) > 1))
        elif which == "Pref. attachment":
            out.append(len(a) * len(b))
    return np.array(out, dtype=float)


# ---------------------------- training ----------------------------
def train(model, A_hat, split, epochs=EPOCHS, lr=0.01, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tp = torch.tensor(split.train_pos)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        Z = model(A_hat)
        # fresh negative examples per epoch (standard in link prediction training)
        neg = torch.tensor(split.negative_uniform(len(split.train_pos),
                                                  seed=int(rng.integers(1 << 30))))
        scores = torch.cat([edge_score(Z, tp), edge_score(Z, neg)])
        targets = torch.cat([torch.ones(len(tp)), torch.zeros(len(neg))])
        loss = nn.functional.binary_cross_entropy_with_logits(scores, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ep % 25 == 0:
            print(f"    epoch {ep:3d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")
    return model


def model_auc(model, A_hat, pos, neg):
    model.eval()
    with torch.no_grad():
        Z = model(A_hat)
        s = torch.cat([edge_score(Z, torch.tensor(pos)),
                       edge_score(Z, torch.tensor(neg))]).numpy()
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, s)


def main():
    split = LinkSplit(test_fraction=0.10, seed=SEED)
    print("=== Data ===")
    print(split.overview())

    A_hat = normalized_adjacency(split.train_pos, split.n)
    print(f"A_hat: sparse {tuple(A_hat.shape)}, {A_hat._nnz():,} non-zero entries "
          f"({100*A_hat._nnz()/split.n**2:.4f} % filled)")
    print(f"(dense that would be {split.n**2*4/1e6:.0f} MB)")

    # ---- two test sets: the same positive test set, two kinds of negatives ----
    neg_uniform = split.negative_uniform(len(split.test_pos), seed=99)
    neg_matched = split.negative_degree_matched(split.test_pos, seed=99)
    print(f"\nNegative examples: uniform {len(neg_uniform):,} | "
          f"degree-matched {len(neg_matched):,}")
    print("Median degree product d_u*d_v:")
    print(f"  real edges     : {np.median(split.degree_product(split.test_pos)):>7.0f}")
    print(f"  uniform        : {np.median(split.degree_product(neg_uniform)):>7.0f}  <- leaf x leaf!")
    print(f"  degree-matched : {np.median(split.degree_product(neg_matched)):>7.0f}  <- comparable")

    results = {}

    # ---- heuristics ----
    for h in ("Common neighbors", "Adamic-Adar", "Pref. attachment"):
        r = {}
        for tag, neg in (("uniform", neg_uniform), ("degree-matched", neg_matched)):
            s = np.r_[heuristic_scores(split, split.test_pos, h),
                      heuristic_scores(split, neg, h)]
            y = np.r_[np.ones(len(split.test_pos)), np.zeros(len(neg))]
            r[tag] = roc_auc_score(y, s)
        results[h] = r

    # ---- MLP without structure (the control group) ----
    print("\n=== MLP without structure (control group, no message passing) ===")
    mlp = train(MLPWithoutStructure(split.n, seed=SEED), None, split)
    results["MLP (without graph)"] = {
        "uniform": model_auc(mlp, None, split.test_pos, neg_uniform),
        "degree-matched": model_auc(mlp, None, split.test_pos, neg_matched),
    }

    # ---- GCN ----
    print("\n=== GCN (2 layers, from scratch) ===")
    gcn = train(GCN(split.n, seed=SEED), A_hat, split)
    results["GCN (2 layers)"] = {
        "uniform": model_auc(gcn, A_hat, split.test_pos, neg_uniform),
        "degree-matched": model_auc(gcn, A_hat, split.test_pos, neg_matched),
    }

    # ---- the table this is all about ----
    print("\n" + "=" * 64)
    print("ROC-AUC — the same model, two evaluation protocols")
    print("=" * 64)
    print(f"{'Method':24s} {'uniform (naive)':>16s} {'degree-matched':>16s}")
    for name, r in results.items():
        print(f"{name:24s} {r['uniform']:16.4f} {r['degree-matched']:16.4f}")

    best_naive = max(results, key=lambda k: results[k]["uniform"])
    best_honest = max(results, key=lambda k: results[k]["degree-matched"])
    print(f"\nWinner by the naive evaluation  : {best_naive}")
    print(f"Winner by the honest evaluation : {best_honest}")
    if best_naive != best_honest:
        print("\n>>> The evaluation protocol REVERSES the ranking. <<<")
        print("    Whoever evaluates uniformly discards the GNN in favour of the dumbest heuristic.")

    # ---- plot ----
    try:
        import os
        import matplotlib.pyplot as plt
        os.makedirs("results", exist_ok=True)
        names = list(results)
        x = np.arange(len(names)); b = 0.38
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(x - b/2, [results[n]["uniform"] for n in names], b, label="uniform (naive)")
        ax.bar(x + b/2, [results[n]["degree-matched"] for n in names], b,
               label="degree-matched (honest)")
        ax.axhline(0.5, ls="--", color="k", lw=1, label="chance")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set(ylabel="ROC-AUC", title="Link prediction: the measurement protocol decides",
               ylim=(0.3, 0.9))
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        plt.tight_layout(); plt.savefig("results/link_prediction.png", dpi=110)
        print("\nPlot saved: results/link_prediction.png")
    except Exception as e:
        print("(no plot:", e, ")")


if __name__ == "__main__":
    main()
