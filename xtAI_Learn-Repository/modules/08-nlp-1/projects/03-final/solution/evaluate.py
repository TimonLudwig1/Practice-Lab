"""Trainiert den HMM-Tagger auf UD English-EWT und evaluiert die Tag-Genauigkeit."""
from collections import Counter
from data import read_conllu
from hmm_tagger import HMMTagger


def evaluate(tagger, sentences, known_words):
    total = correct = 0
    unk_total = unk_correct = 0
    confusion = Counter()
    for sent in sentences:
        words = [w for w, _ in sent]
        gold = [t for _, t in sent]
        pred = tagger.viterbi(words)
        for w, g, p in zip(words, gold, pred):
            total += 1
            if g == p:
                correct += 1
            else:
                confusion[(g, p)] += 1
            if w not in known_words:
                unk_total += 1
                unk_correct += (g == p)
    return {
        "acc": correct / total,
        "unk_acc": (unk_correct / unk_total) if unk_total else 0.0,
        "unk_share": unk_total / total,
        "confusion": confusion,
    }


def main():
    train = read_conllu("train")
    test = read_conllu("test")
    print(f"Train: {len(train):,} Saetze | Test: {len(test):,} Saetze")

    tagger = HMMTagger().fit(train)
    print(f"Tagset ({len(tagger.tags)}): {tagger.tags}")

    known = set(w for s in train for w, _ in s)
    res = evaluate(tagger, test, known)
    print(f"\nTag-Genauigkeit gesamt      : {res['acc']:.4f}")
    print(f"  davon bekannte Woerter    : (Anteil unbekannt {res['unk_share']:.3f})")
    print(f"Genauigkeit auf UNBEKANNTEN : {res['unk_acc']:.4f}")

    print("\nHaeufigste Verwechslungen (gold -> vorhergesagt):")
    for (g, p), n in res["confusion"].most_common(8):
        print(f"  {g:6s} -> {p:6s}  {n}")

    # Beispiel-Tagging
    example = "The quick brown fox jumps over the lazy dog".split()
    print("\nBeispiel:")
    for w, t in zip(example, tagger.viterbi(example)):
        print(f"  {w:8s} {t}")


if __name__ == "__main__":
    main()
