"""Testsuite fuer den Naive-Bayes-Klassifikator.

    python test_nb.py

Der letzte Test laedt (einmalig) 20 Newsgroups und dauert ein paar Sekunden.
"""
import math
from naive_bayes import MultinomialNaiveBayes


def test_tiny_deterministic():
    # Zwei klar getrennte Klassen -> perfekte Trennung erwartet.
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
    # OOV-Wort darf nicht crashen
    assert nb.predict([["quidditch", "puck"]]) == ["sport"]
    print("  Mini-Datensatz (deterministisch) ... OK")


def test_log_prior_and_likelihood():
    docs = [["a", "a", "b"], ["b", "b"], ["a", "c"]]
    y = ["x", "y", "x"]
    nb = MultinomialNaiveBayes(alpha=1.0).fit(docs, y)
    # Prior: 2 von 3 Dokumenten sind Klasse x
    assert abs(nb.log_prior["x"] - math.log(2 / 3)) < 1e-12
    # Likelihood normalisiert korrekt? sum_w P(w|c) ueber Vokabular = 1
    for c in nb.classes:
        s = sum(math.exp(nb.log_likelihood[c][w]) for w in nb.vocab)
        assert abs(s - 1.0) < 1e-9, (c, s)
    print("  Log-Prior & normierte Likelihood ... OK")


def test_real_data_smoke():
    from data import load, tokenize
    tr_docs, y_tr, te_docs, y_te, names = load()
    nb = MultinomialNaiveBayes().fit([tokenize(d) for d in tr_docs], y_tr)
    pred = nb.predict([tokenize(d) for d in te_docs])
    acc = sum(p == t for p, t in zip(pred, y_te)) / len(y_te)
    assert acc > 0.80, acc
    print(f"  20-Newsgroups Accuracy > 0.80 ..... OK ({acc:.3f})")


if __name__ == "__main__":
    print("Tests:")
    test_tiny_deterministic()
    test_log_prior_and_likelihood()
    test_real_data_smoke()
    print("\nAlle Tests bestanden.")
