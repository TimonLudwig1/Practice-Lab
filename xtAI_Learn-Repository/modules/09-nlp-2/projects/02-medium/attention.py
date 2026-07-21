"""Scaled-Dot-Product- und Multi-Head-Attention — von Hand.

Der Kern eines Transformers. Implementiere beides so, dass es gegen PyTorchs Referenz
(`F.scaled_dot_product_attention`) und die Eigenschaften in test_attention.py besteht.

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

    TODO:
      1. scores = Q @ K^T / sqrt(d_k)          (K ueber die letzten zwei Achsen transponieren)
      2. falls mask gesetzt: scores dort auf -inf setzen, wo mask == 0
         (Tipp: scores.masked_fill(mask == 0, float("-inf")))
      3. attn = softmax(scores) ueber die KEY-Achse (dim=-1)
      4. out  = attn @ V
      5. return out, attn
    """
    raise NotImplementedError("Aufgabe 1: scaled_dot_product_attention implementieren")


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
        r"""x: (B, T, d_model) -> (B, T, d_model).

        TODO (nutze self._split_heads / self._merge_heads und deine SDPA):
          1. Q, K, V per self.W_q/W_k/W_v projizieren und mit _split_heads in
             (B, H, T, d_k) bringen.
          2. out, attn = scaled_dot_product_attention(Q, K, V, mask)
             (mask ist bereits auf (B, H, T, T) broadcastbar).
          3. self.last_attn = attn   (fuer die Tests/Visualisierung merken)
          4. Koepfe mit _merge_heads zusammenfuehren, dann self.W_o anwenden und
             zurueckgeben.
        """
        raise NotImplementedError("Aufgabe 2: MultiHeadAttention.forward implementieren")
