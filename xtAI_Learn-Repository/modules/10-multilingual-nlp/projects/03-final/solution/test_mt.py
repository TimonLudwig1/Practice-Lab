"""Test suite (fast — no real training).

    python test_mt.py

Checks IBM Model 1 (EM on a toy corpus with a known solution), BLEU and the transformer
building blocks (shapes, masks, cross-attention causality).
"""
import torch

import ibm_model1 as ibm
from bleu import corpus_bleu
from transformer import Seq2SeqTransformer, MultiHeadAttention, PAD, BOS, EOS

torch.manual_seed(0)


# ---- IBM Model 1 -----------------------------------------------------------
def test_ibm_recovers_alignment():
    toy = [("das haus", "the house"), ("das buch", "the book"), ("ein buch", "a book"),
           ("ein haus", "a house"), ("das haus", "the house"), ("das buch", "the book")]
    t = ibm.train(toy, n_iter=20, verbose=False)
    # After convergence: each content word is translated unambiguously.
    assert t["house"]["haus"] > 0.95, t["house"]["haus"]
    assert t["book"]["buch"] > 0.95
    assert t["the"]["das"] > 0.9
    assert t["a"]["ein"] > 0.9
    print("  IBM1 learns the correct translation table ... OK")


def test_ibm_likelihood_increases():
    # EM must never lower the (log) likelihood. We track it across the iterations.
    import ibm_model1
    toy = [("ich sehe den hund", "i see the dog"), ("der hund läuft", "the dog runs"),
           ("ich sehe die katze", "i see the cat"), ("die katze schläft", "the cat sleeps")]
    logs = []
    orig = ibm_model1._safe_log
    # train iteratively and read the likelihood from the printout? Simpler: recompute.
    t = ibm.train(toy, n_iter=1, verbose=False)
    ll1 = _loglik(t, toy)
    t = ibm.train(toy, n_iter=6, verbose=False)
    ll6 = _loglik(t, toy)
    assert ll6 >= ll1, (ll1, ll6)
    print("  IBM1 EM increases the likelihood ... OK")


def _loglik(t, pairs):
    from math import log
    ll = 0.0
    for de, en in pairs:
        f = [ibm.NULL] + ibm.tokenize(de)
        for ew in ibm.tokenize(en):
            s = sum(t.get(ew, {}).get(fw, 0.0) for fw in f)
            ll += log(s) if s > 0 else -50
    return ll


def test_ibm_alignment_api():
    # A richer toy -> content words uniquely alignable (a function word/NULL naturally
    # stays ambiguous in mini corpora, so we check the content words).
    toy = [("das haus", "the house"), ("das buch", "the book"), ("ein buch", "a book"),
           ("ein haus", "a house")] * 3
    t = ibm.train(toy, n_iter=20, verbose=False)
    al = ibm.align(t, "das haus", "the house")
    assert len(al) == 2                       # "the", "house"
    assert ("house", "haus") in al            # content word aligned correctly
    print("  IBM1 align() ... OK")


# ---- BLEU ------------------------------------------------------------------
def test_bleu_perfect_and_penalty():
    ref = [["the", "cat", "sat", "on", "the", "mat"]]
    assert abs(corpus_bleu(ref, ref) - 100.0) < 1e-6, "identical sentences -> BLEU 100"
    # too short a hypothesis -> the brevity penalty pushes BLEU down
    short = [["the", "cat"]]
    assert corpus_bleu(short, ref) < 60.0
    print("  BLEU: perfect=100, brevity penalty works ... OK")


def test_bleu_clipping():
    # Repetition is clipped: 'the the the the' against 'the cat' -> p1 = 1/4 (not 4/4)
    hyp = [["the", "the", "the", "the"]]
    ref = [["the", "cat"]]
    b = corpus_bleu(hyp, ref, max_n=1)
    assert abs(b - 100 * (1 / 4)) < 1e-6, b
    print("  BLEU: clipped n-gram precision ... OK")


# ---- Transformer -----------------------------------------------------------
def _model(V=15):
    return Seq2SeqTransformer(V, d_model=32, n_heads=2, d_ff=64, n_enc=2, n_dec=2)


def test_forward_shape():
    m = _model()
    src = torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]])       # 0 = PAD
    tgt = torch.tensor([[BOS, 4, 5, EOS], [BOS, 7, EOS, PAD]])
    out = m(src, tgt)
    assert out.shape == (2, 4, 15)
    print("  Transformer forward shape ... OK")


def test_cross_attention_uses_encoder():
    m = _model()
    src = torch.tensor([[3, 4, 5, 6, 7]])
    tgt = torch.tensor([[BOS, 8, 9]])
    _ = m(src, tgt)
    ca = m.dec_layers[-1].cross_attn.last_attn                 # (B, h, Tq, S)
    assert ca.shape[-1] == src.size(1), "cross-attn keys must have the source length"
    assert ca.shape[-2] == tgt.size(1), "cross-attn queries must have the target length"
    print("  Cross-attention: query=target, key=source ... OK")


def test_decoder_causal_mask():
    # Changing a later target token must NOT change the logits of earlier positions.
    m = _model()
    m.eval()
    src = torch.tensor([[3, 4, 5]])
    a = torch.tensor([[BOS, 6, 7, 8]])
    b = torch.tensor([[BOS, 6, 7, 9]])                         # only the last token differs
    with torch.no_grad():
        oa, ob = m(src, a), m(src, b)
    assert torch.allclose(oa[:, :3], ob[:, :3], atol=1e-6), "causal mask violated!"
    assert not torch.allclose(oa[:, 3], ob[:, 3], atol=1e-6)
    print("  Decoder causal mask (no look into the future) ... OK")


def test_causal_self_attention_zeroes_future():
    mha = MultiHeadAttention(16, 2)
    x = torch.randn(1, 5, 16)
    T = 5
    causal = torch.tril(torch.ones(T, T)).bool().view(1, 1, T, T)
    _ = mha(x, x, causal)
    upper = mha.last_attn[0].triu(diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)
    print("  Causal self-attention hides the future ... OK")


def test_greedy_runs_and_stops():
    m = _model()
    src = torch.tensor([[3, 4, 5]])
    out = m.greedy(src, max_len=10)
    assert out.dim() == 2 and out[0, 0].item() == BOS
    print("  Greedy decode runs ... OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} tests ...")
    for t in tests:
        t()
    print("All tests passed.")
