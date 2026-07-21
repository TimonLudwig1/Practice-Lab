"""HMM-POS-Tagger mit Viterbi-Dekodierung — von Grund auf (Skript 3.1).

Modell:  argmax_t  prod_i  P(w_i | t_i) * P(t_i | t_{i-1})
Emission und Transition per MLE aus getaggtem Korpus, mit Glaettung.
Unbekannte Woerter werden ueber eine SIGNATUR (Suffix/Form) behandelt — das hebt
die Genauigkeit auf unbekannten Woertern deutlich.
"""
import math
from collections import Counter, defaultdict

START = "<s>"


def signature(word):
    """Grobe morphologische Signatur fuer seltene/unbekannte Woerter."""
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
        self.log_emit = {}          # {(t, wort_oder_signatur): log P(.|t)}
        self.emit_vocab = set()     # bekannte Emissionssymbole

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

        # Transitionen: Add-k ueber Tags (inkl. Start-Kontext)
        for tp in [START] + self.tags:
            total = sum(trans[tp].values())
            denom = total + self.k_trans * T
            for t in self.tags:
                self.log_trans[(tp, t)] = math.log((trans[tp][t] + self.k_trans) / denom)

        # Emissionen: Add-k ueber das Emissionsvokabular (Woerter + Signaturen)
        self.emit_vocab = set(sym(w) for w in word_freq)
        Ve = len(self.emit_vocab)
        k_e = 1e-3
        for t in self.tags:
            total = sum(emit[t].values())
            denom = total + k_e * Ve
            for symbol in self.emit_vocab:
                self.log_emit[(t, symbol)] = math.log((emit[t][symbol] + k_e) / denom)
            self.log_emit[(t, None)] = math.log(k_e / denom)   # Fallback
        return self

    def _emission(self, t, word):
        if word in self.emit_vocab:
            return self.log_emit[(t, word)]
        sig = signature(word)
        if sig in self.emit_vocab:
            return self.log_emit.get((t, sig), self.log_emit[(t, None)])
        return self.log_emit[(t, None)]

    def viterbi(self, words):
        """Beste Tag-Folge fuer eine Wortliste (Skript 3.1 / Modul 07)."""
        if not words:
            return []
        V = [{}]          # V[i][tag] = beste Log-Wahrscheinlichkeit
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
        # Rueckverfolgung
        last = max(self.tags, key=lambda t: V[-1][t])
        tags = [last]
        for i in range(len(words) - 1, 0, -1):
            last = back[i][last]
            tags.append(last)
        return list(reversed(tags))

    def predict(self, sentences_words):
        return [self.viterbi(ws) for ws in sentences_words]
