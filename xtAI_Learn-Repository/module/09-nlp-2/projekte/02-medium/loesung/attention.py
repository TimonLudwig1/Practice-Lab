"""Scaled-Dot-Product- und Multi-Head-Attention — von Hand.

Der Kern eines Transformers. Wir implementieren beides so, dass es gegen PyTorchs
Referenz (`F.scaled_dot_product_attention`, `nn.MultiheadAttention`) geprueft werden
kann (siehe test_attention.py).

Masken-Konvention (durchgaengig): `mask` ist ein bool/0-1-Tensor, der auf die
Score-Matrix (..., T_q, T_k) *broadcastbar* ist. `True`/`1` = Position beachten,
`False`/`0` = ausblenden (Score -> -inf vor dem Softmax).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    r"""Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V.

    Q: (..., T_q, d_k)   K: (..., T_k, d_k)   V: (..., T_k, d_v)
    Rueckgabe: (out, attn) mit out (..., T_q, d_v) und attn (..., T_q, T_k).
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)      # (..., T_q, T_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)                        # ueber die Keys
    out = attn @ V                                          # (..., T_q, d_v)
    return out, attn


class MultiHeadAttention(nn.Module):
    r"""Multi-Head-Self-Attention.

    Projiziert die Eingabe in ``n_heads`` Unterraeume der Dimension d_k = d_model/n_heads,
    rechnet dort *parallel* Scaled-Dot-Product-Attention und fuegt die Koepfe wieder
    zusammen (Vaswani et al. 2017, Gl. 1-2).
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model muss durch n_heads teilbar sein"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.last_attn = None   # zum Inspizieren nach dem Forward

    def _split_heads(self, x):
        # (B, T, d_model) -> (B, n_heads, T, d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        # (B, n_heads, T, d_k) -> (B, T, d_model)
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

    def forward(self, x, mask=None):
        # 1) lineare Projektionen und in Koepfe aufteilen
        Q = self._split_heads(self.W_q(x))     # (B, H, T, d_k)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))
        # 2) Attention pro Kopf (mask muss auf (B,H,T,T) broadcastbar sein)
        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        self.last_attn = attn                  # (B, H, T, T)
        # 3) Koepfe zusammenfuehren und Ausgangsprojektion
        out = self._merge_heads(out)           # (B, T, d_model)
        return self.W_o(out)
