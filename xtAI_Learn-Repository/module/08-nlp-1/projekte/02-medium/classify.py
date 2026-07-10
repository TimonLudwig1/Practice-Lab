"""Vergleich: eigener Naive Bayes vs. TF-IDF + Logistische Regression (sklearn)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

from data import load, tokenize
from naive_bayes import MultinomialNaiveBayes


def main():
    tr_docs, y_tr, te_docs, y_te, names = load()
    print(f"Kategorien: {names}")
    print(f"{len(tr_docs)} Trainings- / {len(te_docs)} Testdokumente\n")

    # ---- 1) Eigener multinomialer Naive Bayes -----------------------------
    tr_tok = [tokenize(d) for d in tr_docs]
    te_tok = [tokenize(d) for d in te_docs]
    nb = MultinomialNaiveBayes(alpha=1.0).fit(tr_tok, y_tr)
    pred_nb = nb.predict(te_tok)
    acc_nb = accuracy_score(y_te, pred_nb)
    f1_nb = f1_score(y_te, pred_nb, average="macro")
    print("=== Eigener Naive Bayes ===")
    print(f"Vokabular: {len(nb.vocab):,} Woerter")
    print(f"Accuracy {acc_nb:.3f} | macro-F1 {f1_nb:.3f}")
    print(classification_report(y_te, pred_nb, target_names=names, digits=3))

    # ---- 2) TF-IDF + Logistische Regression (sklearn) ---------------------
    clf = make_pipeline(
        TfidfVectorizer(lowercase=True, token_pattern=r"[a-z]{2,}", sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=10.0),
    )
    clf.fit(tr_docs, y_tr)
    pred_lr = clf.predict(te_docs)
    acc_lr = accuracy_score(y_te, pred_lr)
    f1_lr = f1_score(y_te, pred_lr, average="macro")
    print("=== TF-IDF + Logistische Regression ===")
    print(f"Accuracy {acc_lr:.3f} | macro-F1 {f1_lr:.3f}")
    print(classification_report(y_te, pred_lr, target_names=names, digits=3))

    print("=== Fazit ===")
    print(f"Naive Bayes : Acc {acc_nb:.3f}, F1 {f1_nb:.3f}")
    print(f"TF-IDF+LogR : Acc {acc_lr:.3f}, F1 {f1_lr:.3f}")
    fuehrend = "der handgeschriebene Naive Bayes" if f1_nb >= f1_lr else "TF-IDF + LogReg"
    print(f"Beide sind starke Baselines; hier liegt {fuehrend} leicht vorn.")
    print("Bemerkenswert: NB ist trotz der (falschen) Unabhaengigkeitsannahme")
    print("konkurrenzfaehig — schnell, robust und mit sehr wenigen Parametern.")


if __name__ == "__main__":
    main()
