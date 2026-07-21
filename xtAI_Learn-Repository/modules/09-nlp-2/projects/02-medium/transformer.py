"""Mini-Transformer-Encoder auf Basis der selbstgebauten Attention.

Enthaelt:
  - sinusoidal_positional_encoding : feste Positions-Codierung   <- DU (Aufgabe 3)
  - TransformerEncoderBlock         : MHA + Residual/LayerNorm ... <- DU (Aufgabe 3)
  - TransformerClassifier           : Embedding + PE + Bloecke + Pool + Kopf (vorgegeben)
"""
import math
import torch
import torch.nn as nn

from attention import MultiHeadAttention


def sinusoidal_positional_encoding(seq_len, d_model):
    r"""Feste sinusoidale Positions-Codierung (Vaswani Gl. 5-6).

    PE[pos, 2i]   = sin(pos / 10000^{2i/d_model})
    PE[pos, 2i+1] = cos(pos / 10000^{2i/d_model})

    Rueckgabe: Tensor (seq_len, d_model).

    TODO:
      - pe als Nullmatrix (seq_len, d_model) anlegen.
      - pos = arange(seq_len) als Spalte (seq_len, 1).
      - div = exp(arange(0, d_model, 2) * (-log(10000)/d_model))   # (d_model/2,)
      - pe[:, 0::2] = sin(pos * div);  pe[:, 1::2] = cos(pos * div)
      - return pe
    """
    raise NotImplementedError("Aufgabe 3a: sinusoidal_positional_encoding implementieren")


class TransformerEncoderBlock(nn.Module):
    """Ein Encoder-Block im *Post-LayerNorm*-Stil (Original-Transformer).

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
        r"""TODO — zwei Sub-Schichten, jeweils mit Residual-Verbindung + LayerNorm:

          1. attn_out = self.mha(x, mask)
             x = self.norm1(x + self.dropout(attn_out))
          2. ff_out = self.ff(x)
             x = self.norm2(x + self.dropout(ff_out))
          3. return x
        """
        raise NotImplementedError("Aufgabe 3b: TransformerEncoderBlock.forward implementieren")


class TransformerClassifier(nn.Module):
    """Encoder-only-Transformer fuer Sequenzklassifikation (vorgegeben)."""

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
        # x: (B, T) Token-IDs
        pad_keep = (x != self.pad_idx)                      # (B, T) True = echtes Token
        emb = self.embedding(x) * math.sqrt(self.d_model)   # Skalierung wie im Paper
        emb = emb + self.pe[:x.size(1)].unsqueeze(0)        # Positions-Codierung addieren
        h = self.dropout(emb)
        # Key-Padding-Maske: auf (B, H, T_q, T_k) broadcastbar
        attn_mask = pad_keep.unsqueeze(1).unsqueeze(1)      # (B, 1, 1, T)
        for blk in self.blocks:
            h = blk(h, attn_mask)
        # maskiertes Mittel ueber echte Tokens (Padding zaehlt nicht)
        m = pad_keep.unsqueeze(-1).float()                  # (B, T, 1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)   # (B, d_model)
        return self.fc(pooled)

    @torch.no_grad()
    def attention_maps(self, x):
        """Fuehrt einen Forward aus und gibt die Attention-Gewichte je Block zurueck."""
        self.eval()
        _ = self.forward(x)
        return [blk.mha.last_attn for blk in self.blocks]   # je (B, H, T, T)
