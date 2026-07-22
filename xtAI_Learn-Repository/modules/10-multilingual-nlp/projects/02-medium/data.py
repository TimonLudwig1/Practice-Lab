"""Load data & test lexicon (given).

Tatoeba German-English sentence pairs + a small, hand-curated dictionary test list on
which the translation quality (precision@1) is measured.
"""
import os
import io
import re
import zipfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")
RAW = os.path.join(DATA_DIR, "deu.txt")
URL = "https://www.manythings.org/anki/deu-eng.zip"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW):
        return
    print("Downloading Tatoeba DE-EN ...")
    hdr = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
           "Accept-Language": "en-US,en;q=0.9",
           "Referer": "https://www.manythings.org/anki/"}
    raw = urllib.request.urlopen(urllib.request.Request(URL, headers=hdr), timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open("deu.txt") as src, open(RAW, "wb") as dst:
            dst.write(src.read())


def load_pairs():
    """List of (english, german) sentences."""
    download()
    lines = open(RAW, encoding="utf-8").read().strip().split("\n")
    return [tuple(ln.split("\t")[:2]) for ln in lines]


def tokenize(s):
    return re.findall(r"[a-zäöüßA-ZÄÖÜ]+", s.lower())


# Hand-curated test lexicon EN->DE (only for evaluation, not for the alignment).
# The alignment anchors are mined automatically from the parallel data (embeddings.py),
# so that training and test words are cleanly separated.
TEST_LEXICON = [
    ("water", "wasser"), ("house", "haus"), ("man", "mann"), ("woman", "frau"),
    ("child", "kind"), ("dog", "hund"), ("book", "buch"), ("car", "auto"),
    ("city", "stadt"), ("night", "nacht"), ("world", "welt"), ("money", "geld"),
    ("king", "könig"), ("moon", "mond"), ("tree", "baum"), ("river", "fluss"),
    ("horse", "pferd"), ("school", "schule"), ("father", "vater"), ("mother", "mutter"),
    ("bread", "brot"), ("wine", "wein"), ("summer", "sommer"), ("church", "kirche"),
    ("language", "sprache"), ("word", "wort"), ("music", "musik"), ("garden", "garten"),
    ("street", "straße"), ("sea", "meer"), ("mountain", "berg"), ("bird", "vogel"),
    ("fish", "fisch"), ("door", "tür"), ("window", "fenster"), ("hand", "hand"),
    ("head", "kopf"), ("eye", "auge"), ("family", "familie"), ("friend", "freund"),
]
