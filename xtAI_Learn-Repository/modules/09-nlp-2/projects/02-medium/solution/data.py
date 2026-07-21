"""Synthetic 'negation sentiment' dataset  (given).

Purpose: construct a task that is solvable **only through word interaction /
order** and on which a bag-of-words model (unigrams) *fails*.

Every sentence contains exactly ONE polarity word (positive/negative). With
probability 0.5 a negator ('not'/'never') stands immediately before it and
**flips** the label. The polarity word (together with a possible negator) is
placed at a random position among filler words.

Consequence for unigram BoW: 'good' appears in roughly as many positive as
negative sentences (sometimes with, sometimes without 'not'), and so does 'not'
-> **no** single-word evidence. Only the bigram interaction 'not good' solves the
task. That is exactly what attention can learn (the negator 'looks' at the
following polarity word), a unigram model cannot. This connects directly to
project 01 ('not ... good').
"""
import random

POS = ["good", "great", "love", "nice", "excellent", "amazing", "wonderful", "fun"]
NEG = ["bad", "awful", "hate", "terrible", "poor", "boring", "dull", "worst"]
NEGATORS = ["not", "never"]
FILLER = ["the", "movie", "was", "a", "really", "i", "this", "and",
          "it", "quite", "so", "very", "film", "is", "with"]

PAD, UNK = "<pad>", "<unk>"


def _make_sentence(rng):
    """Builds (tokens, label). label 1 = positive, 0 = negative."""
    positive = rng.random() < 0.5
    word = rng.choice(POS if positive else NEG)
    label = 1 if positive else 0
    chunk = [word]
    if rng.random() < 0.5:                 # negator -> the label flips
        chunk = [rng.choice(NEGATORS), word]
        label = 1 - label
    # Filler words around it; insert the polarity chunk at a random position
    n_before = rng.randint(0, 4)
    n_after = rng.randint(0, 4)
    tokens = ([rng.choice(FILLER) for _ in range(n_before)]
              + chunk
              + [rng.choice(FILLER) for _ in range(n_after)])
    return tokens, label


def make_dataset(n, seed=0):
    rng = random.Random(seed)
    return [_make_sentence(rng) for _ in range(n)]


def build_vocab(dataset):
    vocab = {PAD: 0, UNK: 1}
    for tokens, _ in dataset:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def encode(tokens, vocab):
    return [vocab.get(t, vocab[UNK]) for t in tokens]


if __name__ == "__main__":
    ds = make_dataset(6, seed=1)
    for toks, y in ds:
        print(y, " ".join(toks))
