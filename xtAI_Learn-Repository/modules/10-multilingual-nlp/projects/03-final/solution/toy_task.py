"""Synthetische Toy-Aufgabe: Sequenz-Umkehr (vorgegeben, sehr billig).

Quelle = zufällige Symbolfolge; Ziel = dieselbe Folge **rückwärts**. Diese Aufgabe ist
nur lösbar, wenn die **Cross-Attention** funktioniert (Decoder-Schritt j muss auf
Quellposition L-1-j schauen). Sie trainiert in Sekunden auf der CPU und ist damit ein
billiger, aber echter Nachweis, dass der Encoder-Decoder-Transformer korrekt ist.

Vokabular: PAD=0, BOS=1, EOS=2, Inhaltssymbole ab 3.
"""
import torch

PAD, BOS, EOS = 0, 1, 2
FIRST = 3


def make_batch(batch_size, n_symbols=12, min_len=4, max_len=9, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    lengths = torch.randint(min_len, max_len + 1, (batch_size,), generator=g)
    L = int(lengths.max())
    src = torch.full((batch_size, L), PAD, dtype=torch.long)
    tgt = torch.full((batch_size, L + 2), PAD, dtype=torch.long)   # + BOS + EOS
    for i, n in enumerate(lengths.tolist()):
        seq = torch.randint(FIRST, FIRST + n_symbols, (n,), generator=g)
        src[i, :n] = seq
        tgt[i, 0] = BOS
        tgt[i, 1:n + 1] = torch.flip(seq, dims=[0])
        tgt[i, n + 1] = EOS
    return src, tgt
