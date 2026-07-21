"""Scaled dot-product and multi-head attention — by hand.

The core of a transformer. We implement both so that it can be checked against
PyTorch's reference (`F.scaled_dot_product_attention`, `nn.MultiheadAttention`)
(see test_attention.py).

Mask convention (throughout): `mask` is a bool/0-1 tensor that is *broadcastable*
onto the score matrix (..., T_q, T_k). `True`/`1` = attend to the position,
`False`/`0` = hide it (score -> -inf before the softmax).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    r"""Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V.

    Q: (..., T_q, d_k)   K: (..., T_k, d_k)   V: (..., T_k, d_v)
    Returns: (out, attn) with out (..., T_q, d_v) and attn (..., T_q, T_k).
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)      # (..., T_q, T_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)                        # over the keys
    out = attn @ V                                          # (..., T_q, d_v)
    return out, attn


class MultiHeadAttention(nn.Module):
    r"""Multi-head self-attention.

    Projects the input into ``n_heads`` subspaces of dimension d_k = d_model/n_heads,
    computes scaled dot-product attention there *in parallel* and merges the heads
    back together (Vaswani et al. 2017, eq. 1-2).
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.last_attn = None   # for inspection after the forward pass

    def _split_heads(self, x):
        # (B, T, d_model) -> (B, n_heads, T, d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        # (B, n_heads, T, d_k) -> (B, T, d_model)
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

    def forward(self, x, mask=None):
        # 1) linear projections and split into heads
        Q = self._split_heads(self.W_q(x))     # (B, H, T, d_k)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))
        # 2) attention per head (mask must broadcast onto (B,H,T,T))
        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        self.last_attn = attn                  # (B, H, T, T)
        # 3) merge the heads and apply the output projection
        out = self._merge_heads(out)           # (B, T, d_model)
        return self.W_o(out)
