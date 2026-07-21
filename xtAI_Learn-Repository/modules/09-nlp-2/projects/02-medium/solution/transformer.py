"""Mini transformer encoder built on the hand-made attention.

Contains:
  - sinusoidal_positional_encoding : fixed positional encoding (Vaswani eq. 5-6)
  - TransformerEncoderBlock         : MHA + residual/LayerNorm + FFN + residual/LayerNorm
  - TransformerClassifier           : embedding + PE + N blocks + mean pool + head
"""
import math
import torch
import torch.nn as nn

from attention import MultiHeadAttention


def sinusoidal_positional_encoding(seq_len, d_model):
    r"""Fixed sinusoidal positional encoding.

    PE[pos, 2i]   = sin(pos / 10000^{2i/d_model})
    PE[pos, 2i+1] = cos(pos / 10000^{2i/d_model})

    Returns: tensor (seq_len, d_model).
    """
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1).float()                   # (seq_len, 1)
    div = torch.exp(torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model))                  # (d_model/2,)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerEncoderBlock(nn.Module):
    """An encoder block in the *post-LayerNorm* style (original transformer).

    x -> MHA -> +x -> LayerNorm -> FFN -> +. -> LayerNorm
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sublayer 1: multi-head self-attention with residual + LayerNorm
        attn_out = self.mha(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Sublayer 2: position-wise feed-forward with residual + LayerNorm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerClassifier(nn.Module):
    """Encoder-only transformer for sequence classification."""

    def __init__(self, vocab_size, d_model=32, n_heads=4, d_ff=64, n_layers=2,
                 n_classes=2, pad_idx=0, max_len=64, dropout=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.register_buffer("pe", sinusoidal_positional_encoding(max_len, d_model))
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, T) token IDs
        pad_keep = (x != self.pad_idx)                      # (B, T) True = real token
        emb = self.embedding(x) * math.sqrt(self.d_model)   # scaling as in the paper
        emb = emb + self.pe[:x.size(1)].unsqueeze(0)        # add the positional encoding
        h = self.dropout(emb)
        # Key padding mask: broadcastable onto (B, H, T_q, T_k)
        attn_mask = pad_keep.unsqueeze(1).unsqueeze(1)      # (B, 1, 1, T)
        for blk in self.blocks:
            h = blk(h, attn_mask)
        # masked mean over real tokens (padding does not count)
        m = pad_keep.unsqueeze(-1).float()                  # (B, T, 1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)   # (B, d_model)
        return self.fc(pooled)

    @torch.no_grad()
    def attention_maps(self, x):
        """Runs a forward pass and returns the attention weights per block."""
        self.eval()
        _ = self.forward(x)
        return [blk.mha.last_attn for blk in self.blocks]   # each (B, H, T, T)
