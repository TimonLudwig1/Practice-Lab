"""Trains IBM Model 1 on Tatoeba DE->EN and shows alignments, translations, BLEU.

    python run_ibm.py

Purely count-based (EM) — runs in a few seconds on the CPU, no GPU/no neural training.
Shows the strength (word alignment) and the weakness (no reordering/language model) of
Model 1 alone.
"""
import random

import ibm_model1 as ibm
from data import load_pairs, tokenize
from bleu import corpus_bleu


def main():
    random.seed(1)
    pairs = load_pairs()
    random.shuffle(pairs)
    # short sentences -> cleaner word alignments for the demonstration
    pairs = [(de, en) for de, en in pairs
             if 1 <= len(tokenize(de)) <= 8 and 1 <= len(tokenize(en)) <= 8]
    train = pairs[:20000]
    test = pairs[20000:20500]
    print(f"Train {len(train)}  Test {len(test)}")

    print("\nEM training (IBM Model 1):")
    t = ibm.train(train, n_iter=5)

    print("\nWord alignments (Viterbi):")
    for de, en in test[:5]:
        al = ibm.align(t, de, en)
        pretty = ", ".join(f"{e}←{f}" for e, f in al)
        print(f"  DE: {de}\n  EN: {en}\n  {pretty}\n")

    print("Word-for-word translation (DE->EN) + BLEU:")
    hyps = [ibm.translate(t, de).split() for de, en in test]
    refs = [tokenize(en) for de, en in test]
    print(f"  Corpus BLEU: {corpus_bleu(hyps, refs):.2f}")
    for de, en in test[:6]:
        print(f"  DE: {de}\n   ->: {ibm.translate(t, de)}\n   ref: {en}\n")


if __name__ == "__main__":
    main()
