"""Synthetischer 'Negations-Sentiment'-Datensatz  (vorgegeben).

Zweck: eine Aufgabe konstruieren, die **nur ueber Wort-Interaktion / Reihenfolge**
loesbar ist und an der ein Bag-of-Words-Modell (Unigramme) *scheitert*.

Jeder Satz enthaelt genau EIN Polaritaetswort (positiv/negativ). Mit
Wahrscheinlichkeit 0.5 steht unmittelbar davor ein Negator ('not'/'never'), der das
Label **umdreht**. Das Polaritaetswort (samt evtl. Negator) wird an eine zufaellige
Stelle zwischen Fuellwoertern gesetzt.

Folge fuer Unigramm-BoW: 'good' erscheint in ~gleich vielen positiven wie negativen
Saetzen (mal mit, mal ohne 'not'), ebenso 'not' -> **keine** einzelne-Wort-Evidenz.
Erst die Bigramm-Interaktion 'not good' loest die Aufgabe. Genau das kann Attention
lernen (der Negator 'schaut' auf das folgende Polaritaetswort), ein Unigramm-Modell
nicht. Das schliesst direkt an Projekt 01 an ('not ... good').
"""
import random

POS = ["good", "great", "love", "nice", "excellent", "amazing", "wonderful", "fun"]
NEG = ["bad", "awful", "hate", "terrible", "poor", "boring", "dull", "worst"]
NEGATORS = ["not", "never"]
FILLER = ["the", "movie", "was", "a", "really", "i", "this", "and",
          "it", "quite", "so", "very", "film", "is", "with"]

PAD, UNK = "<pad>", "<unk>"


def _make_sentence(rng):
    """Baut (tokens, label). label 1 = positiv, 0 = negativ."""
    positive = rng.random() < 0.5
    word = rng.choice(POS if positive else NEG)
    label = 1 if positive else 0
    chunk = [word]
    if rng.random() < 0.5:                 # Negator -> Label dreht sich um
        chunk = [rng.choice(NEGATORS), word]
        label = 1 - label
    # Fuellwoerter drumherum; Polaritaets-Chunk an zufaelliger Position einsetzen
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
