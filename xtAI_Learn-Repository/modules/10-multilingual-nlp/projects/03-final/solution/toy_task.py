"""Synthetic toy task: sequence reversal (given, very cheap).

Source = a random symbol sequence; target = the same sequence **reversed**. This task is
only solvable if the **cross-attention** works (decoder step j must look at source position
L-1-j). It trains in seconds on the CPU and is thus a cheap but genuine proof that the
encoder-decoder transformer is correct.

Vocabulary: PAD=0, BOS=1, EOS=2, content symbols from 3 on.
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
