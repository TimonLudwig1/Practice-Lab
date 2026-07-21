"""Multinomialer Naive-Bayes-Textklassifikator — von Hand implementiert.

DEINE AUFGABE: Fuelle `fit`, `predict_one` (und damit `predict`).

Modell (Skript 2.1):  c* = argmax_c [ log P(c) + sum_i log P(w_i | c) ]
mit Laplace-Glaettung  P(w|c) = (C(w,c) + alpha) / (sum_w' C(w',c) + alpha*|V|).
Rechne im Log-Raum (Summe von Logs statt Produkt) gegen Underflow.
"""
import math
from collections import Counter


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = []
        self.log_prior = {}             # {c: log P(c)}
        self.log_likelihood = {}        # {c: {wort: log P(w|c)}}
        self.default_ll = {}            # {c: log P(im-Vokabular-ungesehenes-Wort|c)}
        self.vocab = set()

    def fit(self, docs, labels):
        """docs = Liste von Token-Listen, labels = Liste von Klassen.

        Schritte:
          1) self.classes = sortierte eindeutige Labels; self.vocab = alle Woerter.
          2) Pro Klasse c: Wortzaehlungen C(w,c) (Counter ueber alle Tokens der
             Dokumente mit Label c) und Dokumentzahl.
          3) log_prior[c] = log( #Dok(c) / #Dok ).
          4) Mit total_c = sum_w C(w,c) und denom = total_c + alpha*|V|:
                 log_likelihood[c][w] = log( (C(w,c)+alpha) / denom )  fuer alle w in vocab.
             default_ll[c] = log( alpha / denom )   (falls du OOV NICHT ignorierst).
        """
        # TODO
        raise NotImplementedError

    def predict_one(self, tokens):
        """Gib die Klasse mit maximalem log P(c) + sum_{w in tokens} log P(w|c).
        Ignoriere Woerter, die nicht in self.vocab sind (OOV)."""
        # TODO
        raise NotImplementedError

    def predict(self, docs):
        return [self.predict_one(t) for t in docs]
