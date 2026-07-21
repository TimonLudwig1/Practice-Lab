"""Trains the mini transformer on the negation-sentiment dataset and compares it
with a unigram bag-of-words baseline (given).

    python train.py

Expectation: the transformer learns the negation interaction (test accuracy
~0.95+), the unigram baseline stays near chance (~0.5), because no single word
reveals the label. At the end the attention weights of an example sentence are
shown: the negator 'looks' at the following polarity word.
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
    """Unigram bag-of-words + logistic regression (scikit-learn)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = CountVectorizer(analyzer=lambda toks: toks)   # tokens are already given
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
    print(f"Vocabulary: {len(vocab)} tokens, train {len(train_data)}, test {len(test_data)}")

    train_loader = DataLoader(SeqDataset(train_data, vocab), batch_size=64,
                              shuffle=True, collate_fn=lambda b: collate(b, pad_idx))
    test_loader = DataLoader(SeqDataset(test_data, vocab), batch_size=64,
                             shuffle=False, collate_fn=lambda b: collate(b, pad_idx))

    model = TransformerClassifier(len(vocab), d_model=32, n_heads=4, d_ff=64,
                                  n_layers=2, n_classes=2, pad_idx=pad_idx).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        print(f"Epoch {epoch:2d} | test acc {accuracy(model, test_loader, device):.3f}")

    print("\n--- Comparison ---")
    print(f"Transformer  test acc: {accuracy(model, test_loader, device):.3f}")
    print(f"Unigram BoW  test acc: {bow_baseline(train_data, test_data):.3f}  (near chance!)")

    # Inspect attention: example with a negator
    demo_tokens = ["the", "movie", "was", "not", "good"]
    x = torch.tensor([encode(demo_tokens, vocab)], dtype=torch.long).to(device)
    maps = model.attention_maps(x)                    # list, each (1, H, T, T)
    attn = maps[-1][0].mean(0)                         # averaged over heads (T, T)
    print(f"\nAttention (last block, head-averaged) for: {' '.join(demo_tokens)}")
    header = "        " + " ".join(f"{t:>7.7}" for t in demo_tokens)
    print(header)
    for i, t in enumerate(demo_tokens):
        row = " ".join(f"{attn[i, j].item():7.3f}" for j in range(len(demo_tokens)))
        print(f"{t:>7.7} {row}")
    print("(row = query token, column = what it looks at)")


if __name__ == "__main__":
    main()
