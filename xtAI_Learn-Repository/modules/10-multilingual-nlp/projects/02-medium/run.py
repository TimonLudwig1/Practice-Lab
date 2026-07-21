"""Volle Pipeline: Embeddings → Anker → Procrustes-Alignment → Wörterbuch übersetzen.

    python run.py            # nutzt Cache in datasets/ (erster Lauf ~30 s zum Bauen)
    python run.py --rebuild  # Embeddings/Anker neu bauen

Vergleicht **Nearest-Neighbour** vs. **CSLS**-Retrieval per Precision@1 auf dem
handkuratierten Test-Lexikon und zeigt Beispielübersetzungen.
"""
import argparse
import numpy as np

from data import load_pairs, TEST_LEXICON
from embeddings import get_embeddings_and_anchors
from align import (orthogonal_procrustes, nearest_neighbor, csls, precision_at_1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    pairs = load_pairs()
    print(f"{len(pairs):,} Satzpaare geladen.")
    E_en, voc_en, E_de, voc_de, anchors = get_embeddings_and_anchors(pairs, rebuild=args.rebuild)
    stoi_en = {w: i for i, w in enumerate(voc_en)}
    stoi_de = {w: i for i, w in enumerate(voc_de)}
    print(f"Embeddings: EN {E_en.shape}, DE {E_de.shape}. Anker-Lexikon: {len(anchors)} Paare.")
    print("Beispiel-Anker:", anchors[:6])

    # Anker-Matrizen und Procrustes-Alignment (EN -> DE)
    X = np.stack([E_en[stoi_en[a]] for a, b in anchors])
    Y = np.stack([E_de[stoi_de[b]] for a, b in anchors])
    W = orthogonal_procrustes(X, Y)
    print(f"\nW orthogonal? {np.allclose(W @ W.T, np.eye(W.shape[0]), atol=1e-5)}")

    # Test-Lexikon (nur Paare, deren beide Wörter im Vokabular sind)
    test = [(a, b) for a, b in TEST_LEXICON if a in stoi_en and b in stoi_de]
    q_idx = [stoi_en[a] for a, b in test]
    gold = [b for a, b in test]
    proj_all = E_en @ W                          # ganzer Quellraum projiziert
    q_vecs = proj_all[q_idx]

    nn_pred = nearest_neighbor(q_vecs, E_de)
    csls_pred = csls(q_vecs, E_de, proj_all, k=10)
    p1_nn = precision_at_1(nn_pred, gold, voc_de)
    p1_csls = precision_at_1(csls_pred, gold, voc_de)

    print(f"\nTest-Lexikon: {len(test)} Paare")
    print(f"Precision@1  Nearest-Neighbour: {p1_nn:.3f}")
    print(f"Precision@1  CSLS             : {p1_csls:.3f}")

    print("\nBeispielübersetzungen (EN -> DE):  [NN | CSLS | gold]")
    for (a, b), inn, ic in list(zip(test, nn_pred, csls_pred))[:15]:
        mark = "OK " if voc_de[ic] == b else "  x"
        print(f"  {mark} {a:12} -> {voc_de[inn]:14} | {voc_de[ic]:14} | {b}")


if __name__ == "__main__":
    main()
