"""Scaled dot-product and multi-head attention — by hand.

The core of a transformer. Implement both so that it passes against PyTorch's
reference (`F.scaled_dot_product_attention`) and the properties in
test_attention.py.

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

    TODO:
      1. scores = Q @ K^T / sqrt(d_k)          (transpose K over the last two axes)
      2. if mask is set: set scores to -inf where mask == 0
         (hint: scores.masked_fill(mask == 0, float("-inf")))
      3. attn = softmax(scores) over the KEY axis (dim=-1)
      4. out  = attn @ V
      5. return out, attn
    """
    raise NotImplementedError("Task 1: implement scaled_dot_product_attention")


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
        r"""x: (B, T, d_model) -> (B, T, d_model).

        TODO (use self._split_heads / self._merge_heads and your SDPA):
          1. project Q, K, V via self.W_q/W_k/W_v and bring them into
             (B, H, T, d_k) with _split_heads.
          2. out, attn = scaled_dot_product_attention(Q, K, V, mask)
             (mask already broadcasts onto (B, H, T, T)).
          3. self.last_attn = attn   (remember it for the tests/visualization)
          4. merge the heads with _merge_heads, then apply self.W_o and
             return the result.
        """
        raise NotImplementedError("Task 2: implement MultiHeadAttention.forward")
