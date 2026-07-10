"""Abnahmetest fuer den HMM-POS-Tagger (Musterloesung).

    python test_tagger.py

Laedt (einmalig) UD English-EWT und dauert einige Sekunden.
"""
from data import read_conllu
from hmm_tagger import HMMTagger, signature
from evaluate import evaluate


def test_signature():
    assert signature("42") == "<NUM>"
    assert signature("Berlin") == "<CAP>"
    assert signature("running") == "<~ing>"
    assert signature("quickly") == "<~ly>"
    print("  Signaturen ................. OK")


def test_viterbi_toy():
    # Winziges, eindeutiges Korpus
    train = [
        [("the", "DET"), ("dog", "NOUN"), ("runs", "VERB")],
        [("the", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
        [("a", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
    ]
    tg = HMMTagger().fit(train)
    assert tg.viterbi(["the", "cat", "runs"]) == ["DET", "NOUN", "VERB"]
    print("  Viterbi (Spielzeug) ........ OK")


def test_ewt_accuracy():
    train = read_conllu("train")
    test = read_conllu("test")
    tg = HMMTagger().fit(train)
    known = set(w for s in train for w, _ in s)
    res = evaluate(tg, test, known)
    assert res["acc"] > 0.88, res["acc"]
    # Standardsatz sauber getaggt
    ex = tg.viterbi("The quick brown fox jumps over the lazy dog".split())
    assert ex[0] == "DET" and ex[-1] == "NOUN"
    print(f"  EWT-Accuracy > 0.88 ........ OK ({res['acc']:.4f}, unk {res['unk_acc']:.3f})")


if __name__ == "__main__":
    print("Tests:")
    test_signature()
    test_viterbi_toy()
    test_ewt_accuracy()
    print("\nAlle Tests bestanden.")
