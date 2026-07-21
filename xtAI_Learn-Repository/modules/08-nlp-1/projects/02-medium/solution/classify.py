"""A comparison: our own naive Bayes vs. TF-IDF + logistic regression (sklearn)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

from data import load, tokenize
from naive_bayes import MultinomialNaiveBayes


def main():
    tr_docs, y_tr, te_docs, y_te, names = load()
    print(f"Categories: {names}")
    print(f"{len(tr_docs)} training / {len(te_docs)} test documents\n")

    # ---- 1) Our own multinomial naive Bayes -------------------------------
    tr_tok = [tokenize(d) for d in tr_docs]
    te_tok = [tokenize(d) for d in te_docs]
    nb = MultinomialNaiveBayes(alpha=1.0).fit(tr_tok, y_tr)
    pred_nb = nb.predict(te_tok)
    acc_nb = accuracy_score(y_te, pred_nb)
    f1_nb = f1_score(y_te, pred_nb, average="macro")
    print("=== Our own naive Bayes ===")
    print(f"Vocabulary: {len(nb.vocab):,} words")
    print(f"Accuracy {acc_nb:.3f} | macro-F1 {f1_nb:.3f}")
    print(classification_report(y_te, pred_nb, target_names=names, digits=3))

    # ---- 2) TF-IDF + logistic regression (sklearn) ------------------------
    clf = make_pipeline(
        TfidfVectorizer(lowercase=True, token_pattern=r"[a-z]{2,}", sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=10.0),
    )
    clf.fit(tr_docs, y_tr)
    pred_lr = clf.predict(te_docs)
    acc_lr = accuracy_score(y_te, pred_lr)
    f1_lr = f1_score(y_te, pred_lr, average="macro")
    print("=== TF-IDF + logistic regression ===")
    print(f"Accuracy {acc_lr:.3f} | macro-F1 {f1_lr:.3f}")
    print(classification_report(y_te, pred_lr, target_names=names, digits=3))

    print("=== Conclusion ===")
    print(f"Naive Bayes : acc {acc_nb:.3f}, F1 {f1_nb:.3f}")
    print(f"TF-IDF+LogR : acc {acc_lr:.3f}, F1 {f1_lr:.3f}")
    leader = "the hand-written naive Bayes" if f1_nb >= f1_lr else "TF-IDF + logistic regression"
    print(f"Both are strong baselines; here {leader} is slightly ahead.")
    print("Remarkable: NB is competitive despite the (false) independence")
    print("assumption — fast, robust and with very few parameters.")


if __name__ == "__main__":
    main()
