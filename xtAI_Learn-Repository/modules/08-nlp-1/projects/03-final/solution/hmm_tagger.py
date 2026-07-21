"""An HMM POS tagger with Viterbi decoding — from scratch (script 3.1).

The model:  argmax_t  prod_i  P(w_i | t_i) * P(t_i | t_{i-1})
Emission and transition by MLE from a tagged corpus, with smoothing.
Unknown words are handled via a SIGNATURE (suffix/form) — that raises the
accuracy on unknown words considerably.
"""
import math
from collections import Counter, defaultdict

START = "<s>"


def signature(word):
    """A coarse morphological signature for rare/unknown words."""
    if any(ch.isdigit() for ch in word):
        return "<NUM>"
    if word[:1].isupper():
        return "<CAP>"
    low = word.lower()
    for suf in ("ing", "ed", "ly", "ment", "tion", "ness", "ous", "ive",
                "al", "ic", "est", "er", "es", "s"):
        if low.endswith(suf) and len(low) > len(suf) + 1:
            return "<~" + suf + ">"
    if "-" in word:
        return "<HYPH>"
    return "<UNK>"


class HMMTagger:
    def __init__(self, k_trans=0.01, rare_threshold=1):
        self.k_trans = k_trans
        self.rare_threshold = rare_threshold
        self.tags = []
        self.log_trans = {}         # {(t_prev, t): log P(t|t_prev)}
        self.log_emit = {}          # {(t, word_or_signature): log P(.|t)}
        self.emit_vocab = set()     # the known emission symbols

    def fit(self, sentences):
        word_freq = Counter(w for s in sentences for w, _ in s)

        def sym(w):
            return w if word_freq[w] > self.rare_threshold else signature(w)

        tag_set = set()
        trans = defaultdict(Counter)       # trans[t_prev][t]
        emit = defaultdict(Counter)        # emit[t][symbol]
        tag_count = Counter()
        for s in sentences:
            prev = START
            for w, t in s:
                tag_set.add(t)
                trans[prev][t] += 1
                emit[t][sym(w)] += 1
                tag_count[t] += 1
                prev = t
        self.tags = sorted(tag_set)
        T = len(self.tags)

        # Transitions: add-k over the tags (including the start context)
        for tp in [START] + self.tags:
            total = sum(trans[tp].values())
            denom = total + self.k_trans * T
            for t in self.tags:
                self.log_trans[(tp, t)] = math.log((trans[tp][t] + self.k_trans) / denom)

        # Emissions: add-k over the emission vocabulary (words + signatures)
        self.emit_vocab = set(sym(w) for w in word_freq)
        Ve = len(self.emit_vocab)
        k_e = 1e-3
        for t in self.tags:
            total = sum(emit[t].values())
            denom = total + k_e * Ve
            for symbol in self.emit_vocab:
                self.log_emit[(t, symbol)] = math.log((emit[t][symbol] + k_e) / denom)
            self.log_emit[(t, None)] = math.log(k_e / denom)   # the fallback
        return self

    def _emission(self, t, word):
        if word in self.emit_vocab:
            return self.log_emit[(t, word)]
        sig = signature(word)
        if sig in self.emit_vocab:
            return self.log_emit.get((t, sig), self.log_emit[(t, None)])
        return self.log_emit[(t, None)]

    def viterbi(self, words):
        """The best tag sequence for a list of words (script 3.1 / module 07)."""
        if not words:
            return []
        V = [{}]          # V[i][tag] = the best log probability
        back = [{}]
        for t in self.tags:
            V[0][t] = self.log_trans[(START, t)] + self._emission(t, words[0])
            back[0][t] = START
        for i in range(1, len(words)):
            V.append({}); back.append({})
            for t in self.tags:
                best_prev, best_score = None, -math.inf
                emit = self._emission(t, words[i])
                for tp in self.tags:
                    score = V[i-1][tp] + self.log_trans[(tp, t)] + emit
                    if score > best_score:
                        best_score, best_prev = score, tp
                V[i][t] = best_score
                back[i][t] = best_prev
        # Backtracking
        last = max(self.tags, key=lambda t: V[-1][t])
        tags = [last]
        for i in range(len(words) - 1, 0, -1):
            last = back[i][last]
            tags.append(last)
        return list(reversed(tags))

    def predict(self, sentences_words):
        return [self.viterbi(ws) for ws in sentences_words]
