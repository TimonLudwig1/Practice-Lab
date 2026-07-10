"""Laedt und parst das Universal-Dependencies-Korpus English-EWT (.conllu).

Beim ersten Aufruf werden train/dev/test von GitHub geladen und in daten/ gecached.
Ein Satz wird zu einer Liste von (WORT, UPOS-TAG)-Paaren.
"""
import os
import time
import urllib.request

DATA_DIR = "daten"
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
        print(f"Lade {split} von {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        last = None
        for attempt in range(5):                    # GitHub-raw kann 429 werfen
            try:
                data = urllib.request.urlopen(req, timeout=30).read()
                with open(path, "wb") as f:
                    f.write(data)
                break
            except Exception as e:                  # noqa: BLE001
                last = e
                print(f"  Versuch {attempt+1} fehlgeschlagen ({e}); warte ...")
                time.sleep(8)
        else:
            raise RuntimeError(f"Download von {url} fehlgeschlagen: {last}")
    return path


def read_conllu(split):
    """Gibt eine Liste von Saetzen zurueck; jeder Satz = Liste von (form, upos)."""
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
            if "-" in tok_id or "." in tok_id:      # Mehrwort-/leere Tokens ueberspringen
                continue
            form, upos = cols[1], cols[3]
            sent.append((form, upos))
    if sent:
        sentences.append(sent)
    return sentences
