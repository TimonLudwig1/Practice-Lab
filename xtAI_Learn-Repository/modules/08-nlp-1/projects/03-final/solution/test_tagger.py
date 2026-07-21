"""The acceptance test for the HMM POS tagger (the reference solution).

    python test_tagger.py

It downloads UD English-EWT (once) and takes a few seconds.
"""
from data import read_conllu
from hmm_tagger import HMMTagger, signature
from evaluate import evaluate


def test_signature():
    assert signature("42") == "<NUM>"
    assert signature("Berlin") == "<CAP>"
    assert signature("running") == "<~ing>"
    assert signature("quickly") == "<~ly>"
    print("  Signatures ................. OK")


def test_viterbi_toy():
    # A tiny, unambiguous corpus
    train = [
        [("the", "DET"), ("dog", "NOUN"), ("runs", "VERB")],
        [("the", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
        [("a", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
    ]
    tg = HMMTagger().fit(train)
    assert tg.viterbi(["the", "cat", "runs"]) == ["DET", "NOUN", "VERB"]
    print("  Viterbi (toy) .............. OK")


def test_ewt_accuracy():
    train = read_conllu("train")
    test = read_conllu("test")
    tg = HMMTagger().fit(train)
    known = set(w for s in train for w, _ in s)
    res = evaluate(tg, test, known)
    assert res["acc"] > 0.88, res["acc"]
    # a standard sentence tagged cleanly
    ex = tg.viterbi("The quick brown fox jumps over the lazy dog".split())
    assert ex[0] == "DET" and ex[-1] == "NOUN"
    print(f"  EWT accuracy > 0.88 ........ OK ({res['acc']:.4f}, unk {res['unk_acc']:.3f})")


if __name__ == "__main__":
    print("Tests:")
    test_signature()
    test_viterbi_toy()
    test_ewt_accuracy()
    print("\nAll tests passed.")
