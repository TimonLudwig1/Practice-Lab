"""A multinomial naive Bayes text classifier — implemented by hand.

The model (script 2.1):  c* = argmax_c [ log P(c) + sum_i log P(w_i | c) ]
with Laplace smoothing P(w|c) = (C(w,c) + alpha) / (sum_w' C(w',c) + alpha*|V|).
The computation happens in log space against underflow.
"""
import math
from collections import Counter, defaultdict


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = []
        self.log_prior = {}             # {c: log P(c)}
        self.log_likelihood = {}        # {c: {word: log P(w|c)}}
        self.default_ll = {}            # {c: log P(a word unseen in training|c)}
        self.vocab = set()

    def fit(self, docs, labels):
        """docs = a list of token lists, labels = a list of classes."""
        self.classes = sorted(set(labels))
        n_docs = len(docs)
        word_counts = {c: Counter() for c in self.classes}     # C(w,c)
        class_doc_count = Counter()                            # #documents per class
        for tokens, c in zip(docs, labels):
            class_doc_count[c] += 1
            word_counts[c].update(tokens)
            self.vocab.update(tokens)

        V = len(self.vocab)
        for c in self.classes:
            self.log_prior[c] = math.log(class_doc_count[c] / n_docs)
            total_c = sum(word_counts[c].values())
            denom = total_c + self.alpha * V
            self.log_likelihood[c] = {
                w: math.log((word_counts[c][w] + self.alpha) / denom)
                for w in self.vocab
            }
            self.default_ll[c] = math.log(self.alpha / denom)  # for words unseen in training
        return self

    def predict_one(self, tokens):
        best_c, best_score = None, -math.inf
        for c in self.classes:
            score = self.log_prior[c]
            ll = self.log_likelihood[c]
            for w in tokens:
                if w in self.vocab:                # ignore OOV words
                    score += ll[w]
            if score > best_score:
                best_score, best_c = score, c
        return best_c

    def predict(self, docs):
        return [self.predict_one(t) for t in docs]
