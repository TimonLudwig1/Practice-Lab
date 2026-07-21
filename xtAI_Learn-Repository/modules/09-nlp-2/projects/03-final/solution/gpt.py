"""Reference solution — a character-level GPT from scratch.

A decoder-only transformer (GPT architecture, Radford et al. 2018) on the
character level, trained autoregressively on *Tiny Shakespeare*. Deliberately kept
compact so that training runs in a few minutes on CPU/MPS/CUDA, but
architecturally complete:

  token embedding + learned positional embeddings
  -> N x [ pre-LayerNorm -> causal multi-head self-attention -> residual
           pre-LayerNorm -> feed-forward (GELU) -> residual ]
  -> final LayerNorm -> linear projection onto the vocabulary (weight tying)

Training objective: next-character prediction (cross entropy). Generation:
autoregressive sampling with temperature and optional top-k.

    python gpt.py            # trains and generates a text sample
    python gpt.py --help     # hyperparameters

Core link to project 02: the same scaled dot-product attention, now with a
**causal mask** (every token may only look to the left) — that makes the model
autoregressive.
"""
import argparse
import math
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Data: Tiny Shakespeare (character level)
# --------------------------------------------------------------------------- #
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets", "tinyshakespeare.txt")


def load_text():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print("Downloading Tiny Shakespeare ...")
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8")
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(DATA_PATH, encoding="utf-8") as f:
            text = f.read()
    return text


class CharTokenizer:
    """Bijective character <-> integer mapping (character-level vocabulary)."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (look only to the left)."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.d_k = n_embd // n_head
        # Q, K, V in a single projection (efficiency), plus the output projection
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # lower triangular matrix as the (non-learned) causal mask
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        # scaled dot-product attention with causal masking
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)     # (B, n_head, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                              # (B, n_head, T, d_k)
        y = y.transpose(1, 2).contiguous().view(B, T, C)         # merge the heads
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    """Transformer block in the pre-LayerNorm style (GPT-2)."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # residual around attention (pre-LN)
        x = x + self.mlp(self.ln2(x))      # residual around the feed-forward
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4,
                 block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)   # learned positions
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight            # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, n_embd)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)                                  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling. idx: (B, T0) context IDs."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]              # crop the context to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature           # only the last time step
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")   # everything except top-k -> -inf
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters=50):
    out = {}
    model.eval()
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    text = load_text()
    tok = CharTokenizer(text)
    print(f"Characters in the corpus: {len(text):,}   vocabulary: {tok.vocab_size}")
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    model = GPT(tok.vocab_size, n_embd=args.n_embd, n_head=args.n_head,
                n_layer=args.n_layer, block_size=args.block_size).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for it in range(1, args.iters + 1):
        xb, yb = get_batch(splits["train"], args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == 1:
            l = estimate_loss(model, splits, args.block_size, args.batch_size, device)
            print(f"Iter {it:5d} | train {l['train']:.3f} | val {l['val']:.3f}")

    # Generate
    print("\n" + "=" * 60 + "\nGenerated text sample (temperature 0.8, top-k 40):\n" + "=" * 60)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)   # start: character 0 (newline)
    out = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
