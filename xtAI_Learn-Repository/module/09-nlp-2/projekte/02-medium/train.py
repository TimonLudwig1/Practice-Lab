"""Trainiert den Mini-Transformer auf dem Negations-Sentiment-Datensatz und
vergleicht mit einer Unigramm-Bag-of-Words-Baseline (vorgegeben).

    python train.py

Erwartung: Der Transformer lernt die Negations-Interaktion (Test-Acc ~0.95+),
die Unigramm-Baseline bleibt nahe am Zufall (~0.5), weil kein einzelnes Wort das
Label verraet. Am Ende werden die Attention-Gewichte eines Beispielsatzes gezeigt:
der Negator 'schaut' auf das folgende Polaritaetswort.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data import make_dataset, build_vocab, encode, PAD
from transformer import TransformerClassifier


def collate(batch, pad_idx=0):
    seqs, labels = zip(*batch)
    maxlen = max(len(s) for s in seqs)
    x = torch.full((len(seqs), maxlen), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return x, torch.tensor(labels, dtype=torch.long)


class SeqDataset(Dataset):
    def __init__(self, data, vocab):
        self.items = [(encode(toks, vocab), y) for toks, y in data]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def bow_baseline(train_data, test_data):
    """Unigramm-Bag-of-Words + logistische Regression (scikit-learn)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = CountVectorizer(analyzer=lambda toks: toks)   # Tokens sind schon gegeben
    Xtr = vec.fit_transform([toks for toks, _ in train_data])
    Xte = vec.transform([toks for toks, _ in test_data])
    ytr = [y for _, y in train_data]
    yte = [y for _, y in test_data]
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return clf.score(Xte, yte)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_data = make_dataset(4000, seed=1)
    test_data = make_dataset(1000, seed=2)
    vocab = build_vocab(train_data)
    pad_idx = vocab[PAD]
    print(f"Vokabular: {len(vocab)} Tokens, Train {len(train_data)}, Test {len(test_data)}")

    train_loader = DataLoader(SeqDataset(train_data, vocab), batch_size=64,
                              shuffle=True, collate_fn=lambda b: collate(b, pad_idx))
    test_loader = DataLoader(SeqDataset(test_data, vocab), batch_size=64,
                             shuffle=False, collate_fn=lambda b: collate(b, pad_idx))

    model = TransformerClassifier(len(vocab), d_model=32, n_heads=4, d_ff=64,
                                  n_layers=2, n_classes=2, pad_idx=pad_idx).to(device)
    print(f"Parameter: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss()
    for epoch in range(1, 11):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        print(f"Epoche {epoch:2d} | Test-Acc {accuracy(model, test_loader, device):.3f}")

    print("\n--- Vergleich ---")
    print(f"Transformer  Test-Acc: {accuracy(model, test_loader, device):.3f}")
    print(f"Unigramm-BoW Test-Acc: {bow_baseline(train_data, test_data):.3f}  (nahe Zufall!)")

    # Attention inspizieren: Beispiel mit Negator
    demo_tokens = ["the", "movie", "was", "not", "good"]
    x = torch.tensor([encode(demo_tokens, vocab)], dtype=torch.long).to(device)
    maps = model.attention_maps(x)                    # Liste je (1, H, T, T)
    attn = maps[-1][0].mean(0)                         # ueber Koepfe gemittelt (T, T)
    print(f"\nAttention (letzter Block, kopf-gemittelt) fuer: {' '.join(demo_tokens)}")
    header = "        " + " ".join(f"{t:>7.7}" for t in demo_tokens)
    print(header)
    for i, t in enumerate(demo_tokens):
        row = " ".join(f"{attn[i, j].item():7.3f}" for j in range(len(demo_tokens)))
        print(f"{t:>7.7} {row}")
    print("(Zeile = Query-Token, Spalte = worauf es schaut)")


if __name__ == "__main__":
    main()
