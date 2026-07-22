"""Encoder-decoder transformer with cross-attention — from scratch.

The full NMT architecture (script 4.3): a bidirectional encoder, a causal decoder and the
**cross-attention** that connects the two (query=decoder, key/value=encoder). We *train only
on a tiny synthetic task* (sequence reversal) that can be solved in seconds on the CPU and
succeeds **only** with working cross-attention — a cheap but complete proof that the
architecture is correct. Expensive training on real translation data would be too costly for
a laptop; the IBM Model 1 file provides the real (statistical) translation instead.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate query/key-value inputs (for cross-attn)."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_attn = None

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, q_in, kv_in, mask=None):
        B = q_in.size(0)
        Q, K, V = self._split(self.Wq(q_in)), self._split(self.Wk(kv_in)), self._split(self.Wv(kv_in))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        self.last_attn = attn
        out = (self.drop(attn) @ V).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.Wo(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class _FF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = _FF(d_model, d_ff)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        x = self.n1(x + self.attn(x, x, src_mask))
        x = self.n2(x + self.ff(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = _FF(d_model, d_ff)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, enc, tgt_mask, cross_mask):
        x = self.n1(x + self.self_attn(x, x, tgt_mask))        # causal self-attention
        x = self.n2(x + self.cross_attn(x, enc, cross_mask))   # cross-attention onto the encoder
        x = self.n3(x + self.ff(x))
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=2, d_ff=128,
                 n_enc=2, n_dec=2, dropout=0.0, pad_idx=PAD):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec)])
        self.out = nn.Linear(d_model, vocab_size)

    def src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(1)         # (B,1,1,S)

    def tgt_mask(self, tgt):
        T = tgt.size(1)
        pad = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(1)         # (B,1,1,T)
        causal = torch.tril(torch.ones(T, T, device=tgt.device)).bool().view(1, 1, T, T)
        return pad & causal

    def encode(self, src, src_mask):
        x = self.pos(self.embed(src) * math.sqrt(self.d_model))
        for l in self.enc_layers:
            x = l(x, src_mask)
        return x

    def decode(self, tgt, enc, tgt_mask, cross_mask):
        x = self.pos(self.embed(tgt) * math.sqrt(self.d_model))
        for l in self.dec_layers:
            x = l(x, enc, tgt_mask, cross_mask)
        return x

    def forward(self, src, tgt):
        sm = self.src_mask(src)
        enc = self.encode(src, sm)
        dec = self.decode(tgt, enc, self.tgt_mask(tgt), sm)
        return self.out(dec)

    @torch.no_grad()
    def greedy(self, src, max_len=25):
        self.eval()
        sm = self.src_mask(src)
        enc = self.encode(src, sm)
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            logits = self.out(self.decode(ys, enc, self.tgt_mask(ys), sm))
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            if (nxt == EOS).all():
                break
        return ys
