"""The test suite for the naive Bayes classifier.

    python test_nb.py

The last test downloads 20 newsgroups (once) and takes a few seconds.
"""
import math
from naive_bayes import MultinomialNaiveBayes


def test_tiny_deterministic():
    # Two clearly separated classes -> a perfect separation is expected.
    docs = [
        ["puck", "goal", "ice", "hockey"],
        ["hockey", "goal", "goal"],
        ["rocket", "orbit", "space", "moon"],
        ["space", "rocket", "launch"],
    ]
    y = ["sport", "sport", "space", "space"]
    nb = MultinomialNaiveBayes(alpha=1.0).fit(docs, y)
    assert nb.predict([["puck", "ice", "goal"]]) == ["sport"]
    assert nb.predict([["orbit", "launch", "moon"]]) == ["space"]
    # an OOV word must not crash
    assert nb.predict([["quidditch", "puck"]]) == ["sport"]
    print("  Mini data set (deterministic) ..... OK")


def test_log_prior_and_likelihood():
    docs = [["a", "a", "b"], ["b", "b"], ["a", "c"]]
    y = ["x", "y", "x"]
    nb = MultinomialNaiveBayes(alpha=1.0).fit(docs, y)
    # The prior: 2 of the 3 documents are class x
    assert abs(nb.log_prior["x"] - math.log(2 / 3)) < 1e-12
    # Does the likelihood normalize correctly? sum_w P(w|c) over the vocabulary = 1
    for c in nb.classes:
        s = sum(math.exp(nb.log_likelihood[c][w]) for w in nb.vocab)
        assert abs(s - 1.0) < 1e-9, (c, s)
    print("  Log prior & normalized likelihood . OK")


def test_real_data_smoke():
    from data import load, tokenize
    tr_docs, y_tr, te_docs, y_te, names = load()
    nb = MultinomialNaiveBayes().fit([tokenize(d) for d in tr_docs], y_tr)
    pred = nb.predict([tokenize(d) for d in te_docs])
    acc = sum(p == t for p, t in zip(pred, y_te)) / len(y_te)
    assert acc > 0.80, acc
    print(f"  20 newsgroups accuracy > 0.80 ..... OK ({acc:.3f})")


if __name__ == "__main__":
    print("Tests:")
    test_tiny_deterministic()
    test_log_prior_and_likelihood()
    test_real_data_smoke()
    print("\nAll tests passed.")
