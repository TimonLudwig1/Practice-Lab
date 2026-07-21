"""Laedt und tokenisiert einen Ausschnitt des 20-Newsgroups-Korpus.

Beim ersten Aufruf laedt scikit-learn den Datensatz (~14 MB) herunter und cached
ihn (in ~/scikit_learn_data). Danach offline nutzbar.
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
    """Kleinschreibung, nur Buchstabenfolgen ab Laenge 2."""
    return _TOKEN.findall(text.lower())


def load(categories=CATEGORIES):
    """Gibt (train_docs, y_train, test_docs, y_test, target_names) zurueck.
    docs = Liste roher Texte; y = Liste von Klassenindizes."""
    strip = ("headers", "footers", "quotes")      # nur den eigentlichen Text
    tr = fetch_20newsgroups(subset="train", categories=categories,
                            remove=strip, shuffle=True, random_state=0)
    te = fetch_20newsgroups(subset="test", categories=categories,
                            remove=strip, shuffle=True, random_state=0)
    return tr.data, list(tr.target), te.data, list(te.target), tr.target_names
