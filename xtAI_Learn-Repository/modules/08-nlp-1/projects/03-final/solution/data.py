"""Loads and parses the Universal Dependencies corpus English-EWT (.conllu).

On the first call, train/dev/test are downloaded from GitHub and cached in
datasets/. A sentence becomes a list of (WORD, UPOS TAG) pairs.
"""
import os
import time
import urllib.request

DATA_DIR = "datasets"
BASE = ("https://raw.githubusercontent.com/UniversalDependencies/"
        "UD_English-EWT/master/")
FILES = {
    "train": "en_ewt-ud-train.conllu",
    "dev":   "en_ewt-ud-dev.conllu",
    "test":  "en_ewt-ud-test.conllu",
}


def _download(split):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, FILES[split])
    if not os.path.exists(path):
        url = BASE + FILES[split]
        print(f"Downloading {split} from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        last = None
        for attempt in range(5):                    # GitHub raw can throw a 429
            try:
                data = urllib.request.urlopen(req, timeout=30).read()
                with open(path, "wb") as f:
                    f.write(data)
                break
            except Exception as e:                  # noqa: BLE001
                last = e
                print(f"  attempt {attempt+1} failed ({e}); waiting ...")
                time.sleep(8)
        else:
            raise RuntimeError(f"downloading {url} failed: {last}")
    return path


def read_conllu(split):
    """Returns a list of sentences; every sentence = a list of (form, upos)."""
    path = _download(split)
    sentences, sent = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if sent:
                    sentences.append(sent); sent = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:      # skip multi-word/empty tokens
                continue
            form, upos = cols[1], cols[3]
            sent.append((form, upos))
    if sent:
        sentences.append(sent)
    return sentences
