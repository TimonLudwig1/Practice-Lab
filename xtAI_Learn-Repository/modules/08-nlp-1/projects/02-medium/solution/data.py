"""Loads and tokenizes a subset of the 20 newsgroups corpus.

On the first call, scikit-learn downloads the data set (about 14 MB) and caches
it (in ~/scikit_learn_data). After that it can be used offline.
"""
import re
from sklearn.datasets import fetch_20newsgroups

CATEGORIES = [
    "rec.sport.hockey",
    "sci.space",
    "talk.politics.guns",
    "comp.graphics",
]

_TOKEN = re.compile(r"[a-z]{2,}")

def tokenize(text):
    """Lowercase, only sequences of letters of length 2 or more."""
    return _TOKEN.findall(text.lower())


def load(categories=CATEGORIES):
    """Returns (train_docs, y_train, test_docs, y_test, target_names).
    docs = a list of raw texts; y = a list of class indices."""
    strip = ("headers", "footers", "quotes")      # only the actual text
    tr = fetch_20newsgroups(subset="train", categories=categories,
                            remove=strip, shuffle=True, random_state=0)
    te = fetch_20newsgroups(subset="test", categories=categories,
                            remove=strip, shuffle=True, random_state=0)
    return tr.data, list(tr.target), te.data, list(te.target), tr.target_names
