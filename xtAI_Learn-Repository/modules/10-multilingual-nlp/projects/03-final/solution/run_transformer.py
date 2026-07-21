"""Trainiert den Encoder-Decoder-Transformer auf der Sequenz-Umkehr-Toy-Aufgabe.

    python run_transformer.py

Läuft in ~15-30 s auf der CPU (bewusst klein & billig). Zeigt, dass die Architektur
inkl. Cross-Attention korrekt lernt: nach dem Training kehrt das Modell ungesehene
Sequenzen exakt um, und die Cross-Attention-Matrix zeigt die erwartete Anti-Diagonale.
"""
import torch
import torch.nn as nn

from transformer import Seq2SeqTransformer, PAD, BOS, EOS
from toy_task import make_batch, FIRST


def exact_match(model, n=200, n_symbols=12):
    src, tgt = make_batch(n, n_symbols=n_symbols, seed=999)
    pred = model.greedy(src, max_len=tgt.size(1))
    ok = 0
    for i in range(n):
        gold = [t for t in tgt[i].tolist() if t not in (PAD, BOS, EOS)]
        out = [t for t in pred[i].tolist() if t not in (PAD, BOS, EOS)]
        ok += (gold == out)
    return ok / n


def main():
    torch.manual_seed(0)
    device = "cpu"                    # winzig -> CPU ist hier am schnellsten & kühl
    V = FIRST + 12
    model = Seq2SeqTransformer(V, d_model=96, n_heads=4, d_ff=192, n_enc=2, n_dec=2).to(device)
    print(f"Parameter: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)

    steps = 1600
    for step in range(1, steps + 1):
        model.train()
        src, tgt = make_batch(64)
        logits = model(src, tgt[:, :-1])                 # Teacher Forcing
        loss = crit(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"Schritt {step:4d} | Loss {loss.item():.3f} | Exact-Match {exact_match(model):.3f}")

    print(f"\nFinale Exact-Match auf ungesehenen Sequenzen: {exact_match(model):.3f}")
    # Beispiel + Cross-Attention-Anti-Diagonale zeigen
    src, tgt = make_batch(1, seed=7)
    pred = model.greedy(src, max_len=tgt.size(1))
    print("Quelle   :", [t for t in src[0].tolist() if t != PAD])
    print("Vorhersage:", [t for t in pred[0].tolist() if t not in (BOS, EOS, PAD)])
    print("Erwartet  :", [t for t in tgt[0].tolist() if t not in (BOS, EOS, PAD)])
    ca = model.dec_layers[-1].cross_attn.last_attn[0].mean(0)   # (Tq, S) über Köpfe
    print("\nCross-Attention (letzter Block, kopf-gemittelt) — sollte anti-diagonal sein:")
    for row in ca.tolist():
        print("  " + " ".join(f"{v:.2f}" for v in row))


if __name__ == "__main__":
    main()
