"""Testsuite fuer Attention & Mini-Transformer.

    python test_attention.py

Prueft die selbstgebauten Bausteine gegen PyTorch-Referenzen und ihre
mathematischen Eigenschaften. Alle Tests sind schnell (kein Training).
"""
import math
import torch
import torch.nn.functional as F

from attention import scaled_dot_product_attention, MultiHeadAttention
from transformer import (sinusoidal_positional_encoding,
                         TransformerEncoderBlock, TransformerClassifier)

torch.manual_seed(0)


def test_sdpa_matches_pytorch():
    B, H, T, d = 2, 3, 5, 8
    Q, K, V = torch.randn(B, H, T, d), torch.randn(B, H, T, d), torch.randn(B, H, T, d)
    out, attn = scaled_dot_product_attention(Q, K, V)
    ref = F.scaled_dot_product_attention(Q, K, V)          # PyTorch-Referenz
    assert torch.allclose(out, ref, atol=1e-5), "SDPA weicht von F.scaled_dot_product_attention ab"
    assert out.shape == (B, H, T, d)
    print("  SDPA == F.scaled_dot_product_attention ... OK")


def test_sdpa_rows_sum_to_one():
    Q, K, V = torch.randn(4, 6, 8), torch.randn(4, 6, 8), torch.randn(4, 6, 8)
    _, attn = scaled_dot_product_attention(Q, K, V)
    assert torch.allclose(attn.sum(-1), torch.ones(4, 6), atol=1e-5), \
        "Attention-Zeilen muessen sich zu 1 summieren (Softmax ueber Keys)"
    print("  Attention-Zeilen summieren zu 1 ... OK")


def test_causal_mask_zeroes_future():
    T, d = 5, 4
    Q, K, V = torch.randn(1, T, d), torch.randn(1, T, d), torch.randn(1, T, d)
    causal = torch.tril(torch.ones(T, T)).view(1, T, T)     # untere Dreiecksmatrix
    _, attn = scaled_dot_product_attention(Q, K, V, mask=causal)
    upper = attn[0].triu(diagonal=1)                        # obere Dreieckswerte
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), \
        "Maskierte (zukuenftige) Positionen muessen 0 Attention bekommen"
    assert torch.allclose(attn.sum(-1), torch.ones(1, T), atol=1e-5)
    print("  Causal-Maske blendet Zukunft aus ... OK")


def test_mha_shapes_and_heads():
    B, T, d_model, H = 2, 7, 16, 4
    mha = MultiHeadAttention(d_model, H)
    x = torch.randn(B, T, d_model)
    out = mha(x)
    assert out.shape == (B, T, d_model)
    assert mha.last_attn.shape == (B, H, T, T)
    assert torch.allclose(mha.last_attn.sum(-1), torch.ones(B, H, T), atol=1e-5)
    print("  MultiHeadAttention Formen & Kopfzahl ... OK")


def test_mha_padding_mask():
    B, T, d_model, H = 1, 5, 8, 2
    mha = MultiHeadAttention(d_model, H)
    x = torch.randn(B, T, d_model)
    # letzte 2 Positionen sind Padding -> Keys ausblenden
    keep = torch.tensor([[1, 1, 1, 0, 0]]).view(B, 1, 1, T)
    _ = mha(x, mask=keep)
    attn = mha.last_attn                                     # (B, H, T, T)
    assert torch.allclose(attn[..., 3:], torch.zeros_like(attn[..., 3:]), atol=1e-6), \
        "Padding-Keys duerfen keine Attention erhalten"
    print("  MHA Padding-Maske ... OK")


def test_positional_encoding():
    pe = sinusoidal_positional_encoding(10, 16)
    assert pe.shape == (10, 16)
    # PE[pos,0] = sin(pos), PE[pos,1] = cos(pos)  (i=0 -> Frequenz 1)
    assert torch.allclose(pe[0, 0], torch.tensor(0.0), atol=1e-6)   # sin(0)=0
    assert torch.allclose(pe[0, 1], torch.tensor(1.0), atol=1e-6)   # cos(0)=1
    assert abs(pe[1, 0].item() - math.sin(1.0)) < 1e-5
    assert abs(pe[1, 1].item() - math.cos(1.0)) < 1e-5
    print("  Sinusoidale Positions-Codierung ... OK")


def test_encoder_block_preserves_shape():
    B, T, d_model = 2, 6, 16
    blk = TransformerEncoderBlock(d_model, n_heads=4, d_ff=32)
    x = torch.randn(B, T, d_model)
    out = blk(x)
    assert out.shape == (B, T, d_model)
    print("  Encoder-Block erhaelt Form ... OK")


def test_classifier_forward_and_padding_invariance():
    torch.manual_seed(1)
    vocab_size = 20
    clf = TransformerClassifier(vocab_size, d_model=16, n_heads=4, d_ff=32,
                                n_layers=2, n_classes=2, pad_idx=0)
    clf.eval()
    # Zwei identische Saetze, einer mit angehaengtem Padding -> gleiche Ausgabe,
    # weil die Padding-Maske die Fuellstellen aus Attention UND Mean-Pool nimmt.
    a = torch.tensor([[3, 4, 5, 6]])
    b = torch.tensor([[3, 4, 5, 6, 0, 0]])
    with torch.no_grad():
        ya, yb = clf(a), clf(b)
    assert ya.shape == (1, 2)
    assert torch.allclose(ya, yb, atol=1e-5), \
        "Angehaengtes Padding darf die Vorhersage nicht veraendern"
    print("  Classifier-Forward & Padding-Invarianz ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Starte {len(tests)} Tests ...")
    for t in tests:
        t()
    print("Alle Tests bestanden.")
