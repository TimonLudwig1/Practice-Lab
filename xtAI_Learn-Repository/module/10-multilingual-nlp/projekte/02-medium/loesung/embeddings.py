"""Monolinguale Embeddings + automatisches Schürfen eines Anker-Lexikons (vorgegeben).

Das ist die *Infrastruktur* rund um das eigentliche Lernziel (Alignment in align.py):

  build_embeddings : PPMI-Kookkurrenz + truncated SVD  ->  Wort-Embeddings pro Sprache
                     (LSA-Stil; auf Tatoeba ergeben sich brauchbare Vektoren:
                      king~queen, water~boils, dog~cat ...).
  mine_anchors     : aus den *parallelen* Sätzen ein zweisprachiges Anker-Lexikon
                     über Kookkurrenz + Mutual-Nearest-Neighbour extrahieren
                     (~2800 Paare) — dient als Seed fürs Procrustes-Alignment.

Alles wird in daten/ als .npz gecached (Neuaufbau dauert ~30 s).
"""
import os
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import svds

from data import tokenize

CACHE = os.path.join(os.path.dirname(__file__), "daten")


def build_embeddings(sentences, vocab_size=6000, dim=100, window=4):
    """PPMI + truncated SVD. Rückgabe: (emb [V,dim] längen-normiert, vocab, stoi)."""
    freq = Counter(w for s in sentences for w in tokenize(s))
    vocab = [w for w, _ in freq.most_common(vocab_size)]
    stoi = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    counts = defaultdict(float)
    for s in sentences:
        ids = [stoi[w] for w in tokenize(s) if w in stoi]
        for i, wi in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    counts[(wi, ids[j])] += 1.0
    rows, cols, vals = zip(*[(a, b, v) for (a, b), v in counts.items()])
    cooc = coo_matrix((vals, (rows, cols)), shape=(V, V)).tocsr()
    total = cooc.sum()
    r = np.asarray(cooc.sum(1)).ravel()
    c = np.asarray(cooc.sum(0)).ravel()
    cx = cooc.tocoo()
    ppmi = [max(np.log((v * total) / (r[i] * c[j]) + 1e-12), 0.0)
            for v, i, j in zip(cx.data, cx.row, cx.col)]
    M = csr_matrix((ppmi, (cx.row, cx.col)), shape=(V, V))
    U, S, _ = svds(M, k=dim)
    emb = U * np.sqrt(S)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return emb.astype(np.float32), vocab, stoi


def mine_anchors(pairs, stoi_en, stoi_de, min_count=8, min_score=0.10):
    """Zweisprachiges Anker-Lexikon aus parallelen Sätzen (Mutual-NN über Kookkurrenz)."""
    co = defaultdict(float)
    en_c = defaultdict(float)
    de_c = defaultdict(float)
    for en, de in pairs:
        es = set(w for w in tokenize(en) if w in stoi_en)
        ds = set(w for w in tokenize(de) if w in stoi_de)
        for a in es:
            en_c[a] += 1
            for b in ds:
                co[(a, b)] += 1
        for b in ds:
            de_c[b] += 1
    score = defaultdict(dict)
    for (a, b), v in co.items():
        if v >= min_count:
            score[a][b] = v * v / (en_c[a] * de_c[b])   # Dice-artige Assoziation
    best_de = {a: max(dd, key=dd.get) for a, dd in score.items()}
    rev = defaultdict(dict)
    for a, dd in score.items():
        for b, s in dd.items():
            rev[b][a] = s
    best_en = {b: max(dd, key=dd.get) for b, dd in rev.items()}
    anchors = [(a, best_de[a]) for a in best_de
               if best_en.get(best_de[a]) == a and score[a][best_de[a]] > min_score]
    return anchors


def get_embeddings_and_anchors(pairs, rebuild=False):
    """Cached Pipeline: EN/DE-Embeddings + Anker-Lexikon."""
    path = os.path.join(CACHE, "emb_cache.npz")
    meta = os.path.join(CACHE, "emb_meta.npz")
    if not rebuild and os.path.exists(path) and os.path.exists(meta):
        d = np.load(path)
        m = np.load(meta, allow_pickle=True)
        return (d["E_en"], list(m["voc_en"]), d["E_de"], list(m["voc_de"]),
                [tuple(a) for a in m["anchors"]])
    print("Baue Embeddings (~30 s) ...")
    E_en, voc_en, stoi_en = build_embeddings([e for e, d in pairs])
    E_de, voc_de, stoi_de = build_embeddings([d for e, d in pairs])
    anchors = mine_anchors(pairs, stoi_en, stoi_de)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, E_en=E_en, E_de=E_de)
    np.savez(meta, voc_en=np.array(voc_en, dtype=object),
             voc_de=np.array(voc_de, dtype=object),
             anchors=np.array(anchors, dtype=object))
    return E_en, voc_en, E_de, voc_de, anchors
