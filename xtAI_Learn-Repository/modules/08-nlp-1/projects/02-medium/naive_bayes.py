"""A multinomial naive Bayes text classifier — implemented by hand.

YOUR TASK: fill in `fit`, `predict_one` (and thereby `predict`).

The model (script 2.1):  c* = argmax_c [ log P(c) + sum_i log P(w_i | c) ]
with Laplace smoothing  P(w|c) = (C(w,c) + alpha) / (sum_w' C(w',c) + alpha*|V|).
Compute in log space (a sum of logs instead of a product) against underflow.
"""
import math
from collections import Counter


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = []
        self.log_prior = {}             # {c: log P(c)}
        self.log_likelihood = {}        # {c: {word: log P(w|c)}}
        self.default_ll = {}            # {c: log P(a word unseen in the vocabulary|c)}
        self.vocab = set()

    def fit(self, docs, labels):
        """docs = a list of token lists, labels = a list of classes.

        The steps:
          1) self.classes = the sorted unique labels; self.vocab = all words.
          2) Per class c: the word counts C(w,c) (a Counter over all tokens of the
             documents with label c) and the number of documents.
          3) log_prior[c] = log( #docs(c) / #docs ).
          4) With total_c = sum_w C(w,c) and denom = total_c + alpha*|V|:
                 log_likelihood[c][w] = log( (C(w,c)+alpha) / denom )  for all w in vocab.
             default_ll[c] = log( alpha / denom )   (in case you do NOT ignore OOV).
        """
        # TODO
        raise NotImplementedError

    def predict_one(self, tokens):
        """Return the class with the maximum log P(c) + sum_{w in tokens} log P(w|c).
        Ignore words that are not in self.vocab (OOV)."""
        # TODO
        raise NotImplementedError

    def predict(self, docs):
        return [self.predict_one(t) for t in docs]
